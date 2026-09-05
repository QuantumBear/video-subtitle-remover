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
# 默认禁止 PaddleOCR 启动时联网检查模型源(服务器离线场景/加快启动);
# 需要联网检查时显式设 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=False
os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

# ---------- 可调参数(均有实测依据,见 docs/02-use/04) ----------
DEFAULT_DET_MODEL_DIR = os.path.join(BASE_DIR, 'backend', 'models', 'V5', 'ch_det_fast')
DEFAULT_DET_MODEL_NAME = 'PP-OCRv5_mobile_det'
LAMA_PT = os.path.join(BASE_DIR, 'backend', 'models', 'big-lama', 'big-lama.pt')

from fractions import Fraction

MASK_PAD = 4             # OCR 框外扩像素:mask 比字形宽的环带是 ProPainter
                         # 传播距离最远、质量最差的区域(白雾残影所在),
                         # 收紧外扩(4px 盖住字形抗锯齿边缘)可显著缩小环带
MASK_EXPAND_DOWN = 0     # mask 向下扩展:实测下扩 55px 会把字幕正下方的画面
                         # (鞋子等)罩进 mask 擦掉,且逐帧开关造成内容闪现。
                         # emoji/贴纸的擦除改由检测扩展或后处理承担,不走盲下扩
GLYPH_DILATE = 21        # 字形 mask 膨胀核(约 10px,盖住笔画边缘)
GLYPH_NEIGHBORHOOD = 60  # 字形自检的邻域:仅限 OCR 框向外扩该像素的范围
                         # (漏擦的字总是紧挨着被检出的字行;远处白色物体不进 mask,防误伤)
WHITE_ORIG_TH = 228      # 原帧白字判据:三通道下限(经 f165 残留/f180 干净校准)
WHITE_FIXED_TH = 210     # 修复帧"仍白"判据:放宽以抗重编码灰度漂移
WHITE_RB_MAX = 25        # |R-B| 上限:排除蓝裤腿等彩色亮物
MIN_BOX_ASPECT = 1.8     # 检出框最小宽高比(w/h):字幕行是水平长条(实测≥2.7),
                         # 近方形框是动物/物体误检(实测狗被检出 1.1:1 的框),
                         # 这类框交给修复模型会造成大面积雾块;emoji 框靠下扩覆盖
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
                 lama_pt=LAMA_PT, threads=None, device='auto', inpaint_mode='lama'):
        if threads:
            torch.set_num_threads(threads)
        self.inpaint_mode = inpaint_mode
        print(f'[init] 加载 OCR 检测模型: {det_model_dir}')
        from paddleocr import TextDetection
        # OCR 固定 CPU:占比小(~13%),不值得为它装 paddle-gpu
        self.ocr = TextDetection(
            model_name=det_model_name,
            model_dir=det_model_dir,
            device='cpu',
            enable_hpi=False,
        )
        if inpaint_mode == 'lama':
            print(f'[init] 加载 LAMA: {lama_pt}')
            self.inpainter = LamaEngine(lama_pt, device=device)
            print(f'[init] 模型就绪(LAMA device: {self.inpainter.device})')
        elif inpaint_mode == 'propainter':
            # ProPainter 时序修复:被字幕遮挡的真实像素可从相邻帧沿光流传播
            # 回来,重建质量远超单帧 LAMA(对照 kaipai 目标效果);显存大,
            # 必须 GPU,首次用到时才加载
            self._pp_device = torch.device('cuda' if (device == 'auto' and torch.cuda.is_available()) or device == 'cuda' else 'cpu')
            self.inpainter = None
            print(f'[init] ProPainter 模式(引擎将在首次修复时加载,device: {self._pp_device})')
        else:
            raise ValueError(f'未知 inpaint_mode: {inpaint_mode}')

    def _ensure_propainter(self):
        """ProPainter 惰性加载(首次修复时)。"""
        if self.inpainter is None:
            from backend.inpaint.propainter_inpaint import PropainterInpaint
            from backend.tools.model_config import ModelConfig
            self.inpainter = PropainterInpaint(
                device=self._pp_device,
                model_dir=ModelConfig().PROPAINTER_MODEL_DIR,
                sub_video_length=80,
                use_fp16=self._pp_device.type == 'cuda',
            )
            print('[init] ProPainter 已加载')

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
                bw, bh = x2 - x1, y2 - y1
                # 宽高比过滤:字幕行是水平长条;近方形框是动物/物体误检
                if bw < MIN_BOX_ASPECT * bh:
                    continue
                # 坐标平移回全帧,外扩后输出
                boxes.append((max(0, int(y1) + ymin - MASK_PAD), int(y2) + ymin + MASK_PAD,
                              max(0, int(x1) + xmin - MASK_PAD), int(x2) + xmin + MASK_PAD))
        return boxes

    @staticmethod
    def boxes_to_mask(boxes, h, w):
        mask = np.zeros((h, w), dtype='uint8')
        for ymin, ymax, xmin, xmax in boxes:
            mask[max(0, ymin):min(h, ymax + MASK_EXPAND_DOWN),
                 max(0, xmin):min(w, xmax)] = 255
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
    def filter_glyph_by_height(glyph, max_h=90):
        """按连通域高度过滤白字形:字幕单行高 ≤60px;白色衣物/大块白色物体
        是几百 px 的大连通块,必须剔除,否则 LAMA 会把人/物当字幕抹掉(实测灾难)。"""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(glyph, connectivity=8)
        out = np.zeros_like(glyph)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_HEIGHT] <= max_h:
                out[labels == i] = 255
        return out

    @staticmethod
    def residual_white(fixed_rgb, glyph):
        """修复帧在原白字形位置上仍是白的像素数(=漏擦残留)。"""
        f = fixed_rgb.astype(np.int16)
        still = ((f[:, :, 0] > WHITE_FIXED_TH) & (f[:, :, 1] > WHITE_FIXED_TH)
                 & (f[:, :, 2] > WHITE_FIXED_TH) & (np.abs(f[:, :, 0] - f[:, :, 2]) < WHITE_RB_MAX))
        return int(((glyph > 0) & still).sum())

    def auto_region(self, input_path, samples=24):
        """自动探测字幕区域:均匀采样帧做全屏 OCR,取检出框并集外扩。

        全屏检测会漏检低对比度字幕,但检出的框足以定位字幕活动带
        (漏检字总是紧挨检出字行),外扩后即可覆盖。无检出时回退 None(全屏)。
        """
        c = av.open(input_path)
        vstream = next(s for s in c.streams if s.type == 'video')
        total = vstream.duration and int(float(vstream.duration * vstream.time_base * float(vstream.average_rate))) or 0
        h = vstream.codec_context.height
        w = vstream.codec_context.width
        full = (0, h, 0, w)
        step = max(1, total // samples) if total else 1
        boxes_all = []
        n = 0
        for frame in c.decode(video=0):
            if n % step == 0:
                boxes_all += self.detect(np.asarray(frame.to_image()), full)
            n += 1
        c.close()
        if not boxes_all:
            print('[auto-region] 采样未检出任何字幕框,回退全屏')
            return None
        # 聚类过滤:按框 y 中心分簇(间隔 120px),只保留框数≥2 的簇——
        # 真实字幕在多帧持续出现形成密集带,孤立的单框多为画面误检(高光/接缝)
        centers = sorted((b[0] + b[1]) / 2 for b in boxes_all)
        clusters = [[centers[0]]]
        for cy in centers[1:]:
            if cy - clusters[-1][-1] <= 120:
                clusters[-1].append(cy)
            else:
                clusters.append([cy])
        keep_ranges = [(c[0], c[-1]) for c in clusters if len(c) >= 2]
        if not keep_ranges:
            print('[auto-region] 检出框过于孤立,回退全屏')
            return None
        boxes_all = [b for b in boxes_all
                     if any(lo <= (b[0] + b[1]) / 2 <= hi for lo, hi in keep_ranges)]
        print(f'[auto-region] 聚类过滤后保留 {len(boxes_all)}/{len(centers)} 框')
        ymin = max(0, min(b[0] for b in boxes_all) - 80)
        ymax = min(h, max(b[1] for b in boxes_all) + 80)
        xmin = max(0, min(b[2] for b in boxes_all) - 40)
        xmax = min(w, max(b[3] for b in boxes_all) + 40)
        print(f'[auto-region] 采样检出 {len(boxes_all)} 框 → region: {(ymin, ymax, xmin, xmax)}')
        return (ymin, ymax, xmin, xmax)

    @staticmethod
    def _boxes_overlap(b1, b2, pad=60):
        """两框(外扩 pad)是否重叠:字幕随镜头轻微移动仍视为同一字幕。"""
        return not (b1[3] < b2[2] - pad or b1[2] > b2[3] + pad
                    or b1[1] < b2[0] - pad or b1[0] > b2[1] + pad)

    @classmethod
    def filter_boxes_by_continuity(cls, all_boxes, max_gap=5):
        """检出框的位置连续性过滤:剔除孤立错位的误检框。

        字幕在时间上连续,检出框应与前后检出帧的框位置衔接。一个框若与
        前后(帧距 ≤max_gap)检出帧的所有框均不重叠,则是画面误检
        (高光/动物被当文字),交给 ProPainter 会传播来错误内容
        (实测 f271 雾块)。删除后该帧交由最近邻继承。
        """
        hits = [i for i, b in enumerate(all_boxes) if b]
        out = [list(b) for b in all_boxes]

        def overlap_any(b, others):
            return any(cls._boxes_overlap(b, pb) for pb in others)

        for i in hits:
            prev = max((j for j in hits if j < i), default=None)
            nxt = min((j for j in hits if j > i), default=None)
            prev_ok = prev is not None and i - prev <= max_gap
            nxt_ok = nxt is not None and nxt - i <= max_gap
            if not prev_ok and not nxt_ok:
                continue  # 邻居太远无法判定,保守保留
            kept = []
            for b in all_boxes[i]:
                fail_prev = prev_ok and not overlap_any(b, all_boxes[prev])
                fail_nxt = nxt_ok and not overlap_any(b, all_boxes[nxt])
                if prev_ok and nxt_ok:
                    if fail_prev and fail_nxt:
                        continue  # 两侧都脱节:孤立误检,剔除
                elif fail_prev or fail_nxt:
                    continue  # 单侧可判定且脱节:剔除
                kept.append(b)
            out[i] = kept
        return out

    @classmethod
    def expand_timeline(cls, all_boxes, merge_gap=10):
        """字幕时间线区间化 + 最近邻传播。

        目标是防字幕闪现(逐帧独立检测时字幕'忽检出忽漏检',擦与不擦交替):
        帧号间隔 ≤merge_gap 的检出帧合并为同一区间,区间内无检出框的帧
        继承时间上最近的检出帧的框。

        注意不能用区间内全部检出框的并集:动态字幕(位置随镜头移动)的
        并集会横跨整条移动带,把人物/背景大面积罩进 mask,LAMA 会把
        画面重绘成模糊涂抹(实测灾难)。最近邻帧的框最贴近该帧字幕的
        真实位置。
        """
        all_boxes = cls.filter_boxes_by_continuity(all_boxes)
        n = len(all_boxes)
        hits = [i for i, b in enumerate(all_boxes) if b]
        if not hits:
            return [[] for _ in range(n)]
        # 按帧号聚类成区间
        ranges = [[hits[0], hits[0]]]
        for i in hits[1:]:
            if i - ranges[-1][1] <= merge_gap:
                ranges[-1][1] = i
            else:
                ranges.append([i, i])
        expanded = [list(b) for b in all_boxes]
        for lo, hi in ranges:
            frames_with = [i for i in range(lo, hi + 1) if all_boxes[i]]
            for i in range(lo, hi + 1):
                # 局部窗口(±2帧)并集:补跨帧漏检(某帧漏检的字行,常被相邻帧
                # 检出)。窗口小,字幕移动量有限,并集不会横跨移动带(对比:
                # 全区间并集会把整条移动带罩住,无真值可抄→白雾)
                near = [j for j in frames_with if abs(j - i) <= 2]
                if near:
                    union = []
                    for j in near:
                        for b in all_boxes[j]:
                            if b not in union:
                                union.append(b)
                    expanded[i] = union
                    continue
                # 窗口内无检出(漏检串>5帧):继承最近检出帧的框
                prev = max((j for j in frames_with if j < i), default=None)
                nxt = min((j for j in frames_with if j > i), default=None)
                if prev is None:
                    expanded[i] = all_boxes[nxt]
                elif nxt is None:
                    expanded[i] = all_boxes[prev]
                else:
                    expanded[i] = all_boxes[prev if i - prev <= nxt - i else nxt]
        return expanded

    def process_video(self, input_path, output_path, region=None,
                      white_glyph_check=True, progress=None):
        """处理单条视频(两遍:先全片检测+时间线区间化,再逐帧修复)。

        :param region: (ymin, ymax, xmin, xmax) 字幕区域;None = 自动探测
                       (采样帧 OCR 定位字幕活动带,失败回退全屏)
        :param white_glyph_check: 白字自检开关(白字幕场景必开;彩色字幕场景关闭,
                                  避免把画面中的白色物体误当残留)
        :param progress: 回调 fn(done_frames, total_frames, stage)
        """
        if region is None:
            region = self.auto_region(input_path)
        src = av.open(input_path)
        vstream = next(s for s in src.streams if s.type == 'video')
        fps = float(vstream.average_rate)
        w, h = vstream.codec_context.width, vstream.codec_context.height
        total = vstream.duration and int(float(vstream.duration * vstream.time_base * fps)) or 0
        if region is None:
            region = (0, h, 0, w)

        global _FRAME_TB
        _FRAME_TB = Fraction(1, int(round(fps)))
        tmp_out = output_path + '.tmp.mp4'
        dst = av.open(tmp_out, 'w')
        ov = dst.add_stream('libx264', rate=int(round(fps)))
        ov.width = w; ov.height = h; ov.pix_fmt = 'yuv420p'
        # bf=0 禁用 B 帧:B 帧延迟队列导致 encode() flush 时吐出无效 packet
        # (本机 mux EINVAL 崩溃、服务器上被静默丢帧,输出少 2~4 帧)
        ov.options = {'crf': '18', 'bf': '0'}

        t0 = time.time()
        # 第一遍:全片逐帧 OCR 检测(只记录框,不修复)
        print('[pass1] 全片字幕检测...')
        all_boxes = []
        for frame in src.decode(video=0):
            all_boxes.append(self.detect(np.asarray(frame.to_image()), region))
            if progress and (len(all_boxes) % 30 == 0 or len(all_boxes) == total):
                progress(len(all_boxes), total, '检测中')
        src.close()
        # 时间线区间化:无检出帧继承最近检出帧的框(防字幕闪现,详见方法注释)
        all_boxes = self.expand_timeline(all_boxes)

        # 第二遍:修复 + 写出
        src = av.open(input_path)
        n_fixed = n_repair = n = 0

        if self.inpaint_mode == 'propainter':
            # ---- ProPainter 分支:按连续字幕段批处理(时序模型,不可逐帧) ----
            self._ensure_propainter()
            seg_frames, seg_masks = [], []   # BGR 帧 + 每帧 mask

            def flush_segment():
                nonlocal seg_frames, seg_masks, n_fixed
                if not seg_frames:
                    return
                # 逐帧精确 mask 是 ProPainter 的正确用法:字幕移动时,某帧被遮
                # 的位置在相邻帧是未遮的真实背景,传播来的才是真实像素。若用
                # 段并集 mask,整条移动带在所有帧都被标记为遮蔽——无真值可抄,
                # 模型只能生成白色雾块(实测楼梯/裤腿上的白雾残影)
                comps = self.inpainter.inpaint(seg_frames, seg_masks)  # BGR 输出
                for comp in comps:
                    frame = av.VideoFrame.from_ndarray(
                        cv2.cvtColor(comp, cv2.COLOR_BGR2RGB), format='rgb24')
                    frame.pts = seg_pts[0]
                    frame.time_base = _FRAME_TB
                    for pkt in ov.encode(frame):
                        dst.mux(pkt)
                    seg_pts.pop(0)
                n_fixed += len(seg_frames)
                seg_frames, seg_masks = [], []

            seg_pts = []
            for frame in src.decode(video=0):
                n += 1
                img = np.asarray(frame.to_image())  # RGB
                boxes = all_boxes[n - 1] if n - 1 < len(all_boxes) else []
                if boxes:
                    seg_masks.append(self.boxes_to_mask(boxes, h, w))
                    seg_frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    seg_pts.append(n - 1)
                    if len(seg_frames) >= 400:   # 段长 100→400:段边界是实测最大的
                        flush_segment()          # 闪烁源(边界跳变 4.96x);ProPainter
                                                 # 内部自带 sub_video_length 分块,显存可控
                else:
                    flush_segment()
                    frame.pts = n - 1
                    frame.time_base = _FRAME_TB
                    for pkt in ov.encode(frame):
                        dst.mux(pkt)
                if progress and (n % 30 == 0 or n == total):
                    progress(n, total, f'ProPainter 修复 {n_fixed}')
            flush_segment()
            for pkt in ov.encode():
                dst.mux(pkt)
            dst.close()
        else:
            # ---- LAMA 分支:逐帧修复 + 白字自检 + 补擦 + 防闪混合 ----
            for frame in src.decode(video=0):
                n += 1
                img = np.asarray(frame.to_image())  # RGB
                boxes = all_boxes[n - 1] if n - 1 < len(all_boxes) else []
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
                        glyph = self.filter_glyph_by_height(glyph)
                        resid = self.residual_white(fixed, glyph)
                        if resid > RESID_MIN_PX:
                            kernel = np.ones((GLYPH_DILATE, GLYPH_DILATE), 'uint8')
                            glyph_mask = cv2.dilate(glyph, kernel)
                            fixed = self.inpainter.inpaint(fixed, glyph_mask)
                            n_repair += 1
                            print(f'  [补擦] 帧 {n}: 残留 {resid}px 已二次修复')
                    # 帧间防闪:mask 外严格保留原帧像素(模型对 mask 外的输出有逐帧
                    # 随机细微差,整帧替换会造成全画面轻微闪烁)
                    m3 = cv2.dilate(mask, np.ones((7, 7), 'uint8')).astype(np.float32)[:, :, None] / 255
                    blended = (img.astype(np.float32) * (1 - m3) + fixed.astype(np.float32) * m3)
                    frame = av.VideoFrame.from_ndarray(blended.astype('uint8'), format='rgb24')
                # 显式 pts:PyAV 对 VideoStream.encode 的自动 pts 分配在长序列上会
                # 产生乱序包(实测 flush 时 pts 跳回 3 导致 mux EINVAL/服务器丢帧),
                # 按帧号单调递增是标准做法,时间戳完全可控
                frame.pts = n - 1
                frame.time_base = _FRAME_TB
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
                    help="推理设备:auto=有 CUDA 用 GPU(默认)")
    ap.add_argument('--inpaint-mode', default='lama', choices=['lama', 'propainter'],
                    help='lama=单帧快速;propainter=时序修复(质量高,需 GPU,显存大)')
    args = ap.parse_args()

    pipe = Pipeline(threads=args.threads, device=args.device, inpaint_mode=args.inpaint_mode)
    stat = pipe.process_video(
        args.input, args.output,
        region=tuple(args.region) if args.region else None,
        white_glyph_check=not args.no_white_glyph_check,
        progress=lambda d, t, s: print(f'  进度 {d}/{t} ({s})'))
    print('统计:', stat)


if __name__ == '__main__':
    main()
