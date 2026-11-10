"""SAM3 full-frame concept masks, kept on disk between detection and inpainting."""
from contextlib import nullcontext
import gc
import os
from pathlib import Path
import sys

import av
import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = str(BASE_DIR / 'backend/models/sam3/sam3.pt')


class MaskTimeline:
    """Lossless per-frame masks without retaining a whole video in RAM."""

    def __init__(self, directory, shape):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shape = tuple(shape)
        self.count = 0

    def append(self, mask):
        if mask.shape != self.shape or mask.dtype != np.uint8:
            raise ValueError('SAM3 mask 尺寸或类型与视频不一致')
        if not np.isin(mask, [0, 255]).all():
            raise ValueError('SAM3 mask 必须为 0/255 二值')
        path = self.directory / f'{self.count:08d}.png'
        if not cv2.imwrite(str(path), mask):
            raise OSError(f'无法写入 SAM3 mask: {path}')
        self.count += 1

    def __getitem__(self, frame):
        if not 0 <= frame < self.count:
            raise IndexError(f'SAM3 mask 帧索引越界: {frame}/{self.count}')
        path = self.directory / f'{frame:08d}.png'
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != self.shape:
            raise ValueError(f'SAM3 mask 缓存损坏或缺失: {path}')
        return mask


class Sam3MaskDetector:
    """Official SAM3 image model; prompts share one image embedding."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, code_dir=None,
                 device='auto', prompts=None, score=0.5, dilate=0):
        self.prompts = tuple(prompts or ('caption',))
        if not self.prompts or any(not p.strip() for p in self.prompts):
            raise ValueError('SAM3 prompt 不能为空')
        if not 0 < score < 1:
            raise ValueError('SAM3 score 必须介于 0 和 1 之间')
        if not isinstance(dilate, int) or dilate < 0:
            raise ValueError('SAM3 dilate 必须为非负整数')
        self.dilate = dilate
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'SAM3 权重不存在: {checkpoint}；请设置 SAM3_CHECKPOINT 或 --sam3-checkpoint')
        if code_dir:
            code_dir = Path(code_dir).expanduser().resolve()
            if not (code_dir / 'sam3/model_builder.py').is_file():
                raise FileNotFoundError(f'SAM3 源码目录无效: {code_dir}')
            sys.path.insert(0, str(code_dir))
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise ImportError('缺少 SAM3 或依赖；请使用已验证的 SAM3 环境并设置 SAM3_CODE_DIR') from exc
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if device == 'auto' else device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('SAM3 请求 CUDA，但 CUDA 不可用')
        self.amp_dtype = None
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = build_sam3_image_model(device='cpu', checkpoint_path=None,
                                      load_from_HF=False, compile=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = state.get('model', state)
        weights = {k.removeprefix('detector.'): v for k, v in state.items()
                   if k.startswith('detector.')}
        if not weights:
            raise ValueError('预期 SAM3 官方 detector 检查点；不接受 SAM3.1 或其它模型格式')
        unused = set(weights) - set(model.state_dict())
        if any(not k.startswith('backbone.vision_backbone.sam2_convs.') for k in unused):
            raise ValueError(f'SAM3 图像检查点存在未知参数: {sorted(unused)[:5]}')
        model.load_state_dict({k: v for k, v in weights.items() if k not in unused}, strict=True)
        del state, weights
        self.model = model.eval().to(self.device)
        self.processor = Sam3Processor(self.model, device=str(self.device), confidence_threshold=score)
        print(f'[sam3] device={self.device} autocast={self.amp_dtype} '
              f'prompts={self.prompts} score={score} dilate={dilate}px checkpoint={checkpoint}')

    def predict(self, image_rgb):
        from PIL import Image

        amp = (torch.autocast('cuda', dtype=self.amp_dtype)
               if self.amp_dtype is not None else nullcontext())
        union = np.zeros(image_rgb.shape[:2], dtype=bool)
        with torch.inference_mode(), amp:
            state = self.processor.set_image(Image.fromarray(image_rgb))
            for prompt in self.prompts:
                self.processor.reset_all_prompts(state)
                result = self.processor.set_text_prompt(prompt=prompt, state=state)
                masks = result['masks'].squeeze(1)
                if masks.ndim != 3 or tuple(masks.shape[1:]) != union.shape:
                    raise ValueError('SAM3 输出 mask 尺寸不正确')
                if not torch.isfinite(result['scores']).all():
                    raise ValueError('SAM3 输出非有限置信度')
                union |= masks.any(dim=0).cpu().numpy().astype(bool)
        mask = union.astype(np.uint8) * 255
        if self.dilate:
            size = 2 * self.dilate + 1
            mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
        return mask

    def close(self):
        self.processor = None
        self.model = None
        gc.collect()
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()


def detect_timeline(input_path, region, directory, *, checkpoint=DEFAULT_CHECKPOINT,
                    code_dir=None, device='auto', prompts=None, score=0.5, dilate=0,
                    progress=None, scene_thumb_size=32, scene_mean_diff=18.0):
    """Infer every original frame; boxes serve scheduling/diagnostics only."""
    y1, y2, x1, x2 = region
    boxes, cuts, store, previous = [], [], None, None
    detector = Sam3MaskDetector(checkpoint=checkpoint, code_dir=code_dir, device=device,
                               prompts=prompts, score=score, dilate=dilate)
    try:
        with av.open(os.fspath(input_path)) as src:
            estimate = src.streams.video[0].frames or 0
            for n, frame in enumerate(src.decode(video=0)):
                image = frame.to_ndarray(format='rgb24')
                if store is None:
                    store = MaskTimeline(directory, image.shape[:2])
                # Always infer the original full frame. A manual ROI clips only the output.
                mask = detector.predict(image)
                clipped = np.zeros_like(mask)
                clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                store.append(clipped)
                contours, _ = cv2.findContours(clipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                frame_boxes = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    frame_boxes.append((y, y + h, x, x + w))
                boxes.append(frame_boxes)
                thumb = cv2.resize(image[y1:y2, x1:x2], (scene_thumb_size, scene_thumb_size))
                if previous is not None and np.abs(thumb.astype(np.float32) - previous).mean() > scene_mean_diff:
                    cuts.append(n)
                previous = thumb.astype(np.float32)
                if (n + 1) % 30 == 0 or n + 1 == estimate:
                    print(f'[sam3] frames={n + 1}/{estimate or "?"}')
                    if progress:
                        progress(n + 1, estimate, 'SAM3 mask 检测')
    finally:
        detector.close()
    if store is None:
        raise ValueError('输入视频没有可解码帧')
    return boxes, {'ocr_calls': 0, 'sampled': len(boxes), 'refined': 0,
                   'tracks': 0, 'discarded': 0, 'scene_change_frames': cuts,
                   'sam3_calls': len(boxes), 'sam3_prompt_calls': len(boxes) * len(prompts or ('caption',))}, store
