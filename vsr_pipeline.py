# -*- coding: utf-8 -*-
"""VSR 生产流水线:实测验证的去字幕最佳流程,作为对外服务的处理核心。

流程(每帧闭环,不需要独立验收阶段):
  1. PaddleOCR(mobile)在字幕区域内检测文字框
  2. 无框帧直接写出;有框帧:框外扩 10px 生成 mask → LAMA 修复
  3. 白字自检:原帧白字形位置上若修复帧仍呈白色 → 说明有漏擦
     → 用原帧字形膨胀生成贴合 mask → LAMA 二次补擦(当帧内完成)
  4. 音频从源视频 copy 合回

与 backend/main.py 的区别:
  - 无 GUI/进度条/临时文件包袱,模型常驻(worker 进程 import 一次可处理多条视频)
  - 白字自检内建于流水线(实测中 OCR 漏检的低对比度字幕由它兜底)
  - 差分验收的判据固化在代码里(白字判据经正反例校准,详见 docs/02-use/04)

用法:
  CLI:  python vsr_pipeline.py -i in.mp4 -o out.mp4 -c 450 1010 0 720
  库:   from vsr_pipeline import process_video
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import av
import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 可调参数(均有实测依据,见 docs/02-use/04) ----------
DEFAULT_DET_MODEL_DIR = os.path.join(BASE_DIR, 'backend', 'models', 'V5', 'ch_det_fast')
DEFAULT_DET_MODEL_NAME = 'PP-OCRv5_mobile_det'
LAMA_PT = os.path.join(BASE_DIR, 'backend', 'models', 'big-lama', 'big-lama.pt')

MASK_PAD = 10            # OCR 框外扩像素(create_mask 同款经验值)
GLYPH_DILATE = 21        # 字形 mask 膨胀核(约 10px,盖住笔画边缘)
GLYPH_NEIGHBORHOOD = 60  # 字形自检的邻域:仅限 OCR 框向外扩该像素的范围
                         # (漏擦的字总是紧挨着被检出的字行;远处白色物体不进 mask,防误伤)
WHITE_ORIG_TH = 228      # 原帧白字判据:三通道下限(经 f165 残留/f180 干净校准)
WHITE_FIXED_TH = 210     # 修复帧"仍白"判据:放宽以抗重编码灰度漂移
WHITE_RB_MAX = 25        # |R-B| 上限:排除蓝裤腿等彩色亮物
RESID_MIN_PX = 50        # 帧内残留像素超过该值才触发补擦(抗压缩噪声)
# ffmpeg:优先用系统 PATH 里的(服务器/Linux 场景),否则回退仓库自带的平台二进制
FFMPEG = shutil.which('ffmpeg') or os.path.join(BASE_DIR, 'backend', 'ffmpeg', 'macos', 'ffmpeg')


# ---------- LAMA 引擎(自包含,不依赖 backend 包/任何 GUI 栈) ----------
class LamaEngine:
    """big-lama TorchScript 推理封装。

    语义与 backend/inpaint/lama_inpaint + lama_util 完全一致:
    输入 RGB uint8 (H,W,3) + mask uint8 (H,W),输出修复后的 RGB uint8 (H,W,3)。
    """

    def __init__(self, model_path=LAMA_PT, device='auto'):
        """device: 'auto'(有 CUDA 用 GPU,否则 CPU)/ 'cuda' / 'cpu'。"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            # cudnn 卷积默认开 TF32(10 位尾数),对生成像素任务会累积误差
            # 表现为修复区发雾/涂抹;关掉强制 FP32,与 CPU 输出对齐
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        if not os.path.exists(model_path):
            # git clone 后只有分片文件(完整 .pt 不入库),首次运行自动合并
            shard_dir = os.path.dirname(model_path)
            manifest = os.path.join(shard_dir, 'fs_manifest.csv')
            if os.path.exists(manifest):
                print(f'[init] 合并模型分片: {shard_dir}')
                from fsplit.filesplit import Filesplit
                Filesplit().merge(input_dir=shard_dir)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'LAMA 模型缺失: {model_path}')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

    @staticmethod
    def _to_tensor(img, modulo=8):
        """CHW(或 HW)数组 symmetric padding 到 modulo 的倍数并转 float tensor(0~1)。

        与原 lama_util 的 get_image/pad_img_to_modulo(np.pad mode='symmetric')一致;
        2D 输入(mask)自动升维为 (1,H,W)。
        """
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        c, h, w = img.shape
        oh = (h // modulo + 1) * modulo if h % modulo else h
        ow = (w // modulo + 1) * modulo if w % modulo else w
        padded = np.pad(img.astype(np.float32),
                        ((0, 0), (0, oh - h), (0, ow - w)), mode='symmetric')
        return torch.from_numpy(np.ascontiguousarray(padded)) / 255

    @torch.inference_mode()
    def inpaint(self, image_rgb, mask):
        h, w = mask.shape[:2]
        img = image_rgb.transpose(2, 0, 1)                      # HWC→CHW
        img_t = self._to_tensor(img).unsqueeze(0).to(self.device)            # (1,3,H,W) 0~1
        mask_t = self._to_tensor((mask > 0).astype('float32')).unsqueeze(0)  # (1,1,H,W)
        mask_t = (mask_t > 0) * 1
        mask_t = mask_t.to(self.device)
        out = self.model(img_t, mask_t)                          # (1,3,H,W) 0~1
        out = out[0].permute(1, 2, 0).float().cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype('uint8')[:h, :w]
        return out


# ---------- 模型单例(worker 进程内 import 一次,处理多条视频复用) ----------
class Pipeline:
    """持有常驻模型,提供单视频处理入口。"""

    def __init__(self, det_model_dir=DEFAULT_DET_MODEL_DIR,
                 det_model_name=DEFAULT_DET_MODEL_NAME,
                 lama_pt=LAMA_PT, threads=None, device='auto'):
        if threads:
            torch.set_num_threads(threads)
        print(f'[init] 加载 OCR 检测模型: {det_model_dir}')
        from paddleocr import TextDetection
        # OCR 固定 CPU:占比小(~13%),不值得为它装 paddle-gpu
        self.ocr = TextDetection(
            model_name=det_model_name,
            model_dir=det_model_dir,
            device='cpu',
            enable_hpi=False,
        )
        print(f'[init] 加载 LAMA: {lama_pt}')
        self.inpainter = LamaEngine(lama_pt, device=device)
        print(f'[init] 模型就绪(LAMA device: {self.inpainter.device})')

    # ---- OCR 检测:返回该帧在 region 内的文字框列表 [(ymin,ymax,xmin,xmax), ...] ----
    def detect(self, frame_rgb, region):
        ymin, ymax, xmin, xmax = region
        # region 裁剪送检:局部图不触发 det 的整帧缩放,框更贴合字形(实测 y 范围约紧一半);
        # 全屏时等价于原行为。PaddleX 惯例吃 BGR
        crop = frame_rgb[ymin:ymax, xmin:xmax]
        results = self.ocr.predict(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        boxes = []
        for res in results:
            polys = res.get('dt_polys')
            if polys is None or len(polys) == 0:
                continue
            for poly in polys:
                x1, y1 = poly[:, 0].min(), poly[:, 1].min()
                x2, y2 = poly[:, 0].max(), poly[:, 1].max()
                # 坐标平移回全帧,外扩后输出
                boxes.append((max(0, int(y1) + ymin - MASK_PAD), int(y2) + ymin + MASK_PAD,
                              max(0, int(x1) + xmin - MASK_PAD), int(x2) + xmin + MASK_PAD))
        return boxes

    @staticmethod
    def boxes_to_mask(boxes, h, w):
        mask = np.zeros((h, w), dtype='uint8')
        for ymin, ymax, xmin, xmax in boxes:
            mask[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)] = 255
        return mask

    @staticmethod
    def white_glyph(frame, region):
        """原帧白字形检测(独立于 OCR,差分验收的同款判据)。"""
        glyph = np.zeros(frame.shape[:2], dtype='uint8')
        ymin, ymax, xmin, xmax = region
        r = frame[ymin:ymax, xmin:xmax].astype(np.int16)
        white = ((r[:, :, 0] > WHITE_ORIG_TH) & (r[:, :, 1] > WHITE_ORIG_TH)
                 & (r[:, :, 2] > WHITE_ORIG_TH) & (np.abs(r[:, :, 0] - r[:, :, 2]) < WHITE_RB_MAX))
        glyph[ymin:ymax, xmin:xmax] = white.astype('uint8') * 255
        return glyph

    @staticmethod
    def residual_white(fixed_rgb, glyph):
        """修复帧在原白字形位置上仍是白的像素数(=漏擦残留)。"""
        f = fixed_rgb.astype(np.int16)
        still = ((f[:, :, 0] > WHITE_FIXED_TH) & (f[:, :, 1] > WHITE_FIXED_TH)
                 & (f[:, :, 2] > WHITE_FIXED_TH) & (np.abs(f[:, :, 0] - f[:, :, 2]) < WHITE_RB_MAX))
        return int(((glyph > 0) & still).sum())

    def process_video(self, input_path, output_path, region=None,
                      white_glyph_check=True, progress=None):
        """处理单条视频。

        :param region: (ymin, ymax, xmin, xmax) 字幕区域;None = 全屏
        :param white_glyph_check: 白字自检开关(白字幕场景必开;彩色字幕场景关闭,
                                  避免把画面中的白色物体误当残留)
        :param progress: 回调 fn(done_frames, total_frames, stage)
        """
        src = av.open(input_path)
        vstream = next(s for s in src.streams if s.type == 'video')
        fps = float(vstream.average_rate)
        w, h = vstream.codec_context.width, vstream.codec_context.height
        total = vstream.duration and int(float(vstream.duration * vstream.time_base * fps)) or 0
        if region is None:
            region = (0, h, 0, w)

        tmp_out = output_path + '.tmp.mp4'
        dst = av.open(tmp_out, 'w')
        ov = dst.add_stream('libx264', rate=int(round(fps)))
        ov.width = w; ov.height = h; ov.pix_fmt = 'yuv420p'
        ov.options = {'crf': '18'}

        t0 = time.time()
        n_fixed = n_repair = n = 0
        for frame in src.decode(video=0):
            n += 1
            img = np.asarray(frame.to_image())  # RGB
            boxes = self.detect(img, region)
            if boxes:
                mask = self.boxes_to_mask(boxes, h, w)
                fixed = self.inpainter.inpaint(img, mask)
                n_fixed += 1
                # 白字自检:仅在 OCR 框邻域内找漏擦字(远离框的白色物体不误伤)
                if white_glyph_check:
                    hood = np.zeros((h, w), dtype='uint8')
                    for gy1, gy2, gx1, gx2 in boxes:
                        hood[max(0, gy1 - GLYPH_NEIGHBORHOOD):min(h, gy2 + GLYPH_NEIGHBORHOOD),
                             max(0, gx1 - GLYPH_NEIGHBORHOOD):min(w, gx2 + GLYPH_NEIGHBORHOOD)] = 255
                    glyph = self.white_glyph(img, region)
                    glyph = cv2.bitwise_and(glyph, hood)
                    resid = self.residual_white(fixed, glyph)
                    if resid > RESID_MIN_PX:
                        kernel = np.ones((GLYPH_DILATE, GLYPH_DILATE), 'uint8')
                        glyph_mask = cv2.dilate(glyph, kernel)
                        fixed = self.inpainter.inpaint(fixed, glyph_mask)
                        n_repair += 1
                        print(f'  [补擦] 帧 {n}: 残留 {resid}px 已二次修复')
                frame = av.VideoFrame.from_ndarray(fixed, format='rgb24')
            for pkt in ov.encode(frame):
                dst.mux(pkt)
            if progress and (n % 30 == 0 or n == total):
                progress(n, total, f'修复 {n_fixed} / 补擦 {n_repair}')
        for pkt in ov.encode():
            dst.mux(pkt)
        dst.close()

        # 音频从源视频 copy 合回(源无音频时直接改名)
        has_audio = any(s.type == 'audio' for s in src.streams)
        src.close()
        if has_audio:
            final = output_path + '.mux.mp4'
            subprocess.check_output([
                FFMPEG, '-y', '-i', tmp_out, '-i', input_path,
                '-map', '0:v:0', '-map', '1:a:0',
                '-c:v', 'copy', '-c:a', 'aac', '-shortest',
                '-loglevel', 'error', final])
            os.remove(tmp_out)
            os.replace(final, output_path)
        else:
            os.replace(tmp_out, output_path)
        print(f'[done] {n} 帧 | 修复 {n_fixed} | 补擦 {n_repair} | '
              f'耗时 {time.time() - t0:.0f}s → {output_path}')
        return {'frames': n, 'inpainted': n_fixed, 'repaired': n_repair,
                'seconds': time.time() - t0}


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description='去字幕生产流水线(实测验证版)')
    ap.add_argument('-i', '--input', required=True)
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('-c', '--region', nargs=4, type=int, metavar=('YMIN', 'YMAX', 'XMIN', 'XMAX'),
                    help='字幕区域;不传则全屏(建议总是显式指定)')
    ap.add_argument('--no-white-glyph-check', action='store_true',
                    help='关闭白字自检(彩色字幕场景)')
    ap.add_argument('--threads', type=int, default=None, help='torch CPU 线程数(多 worker 并发时调小)')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                    help="LAMA 推理设备:auto=有 CUDA 用 GPU(默认)")
    args = ap.parse_args()

    pipe = Pipeline(threads=args.threads, device=args.device)
    stat = pipe.process_video(
        args.input, args.output,
        region=tuple(args.region) if args.region else None,
        white_glyph_check=not args.no_white_glyph_check,
        progress=lambda d, t, s: print(f'  进度 {d}/{t} ({s})'))
    print('统计:', stat)


if __name__ == '__main__':
    main()
