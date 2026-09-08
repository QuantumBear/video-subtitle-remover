# -*- coding: utf-8 -*-
"""本地开放词汇贴纸/emoji 检测（GroundingDINO）。

与 DashScope VLM 路径（``vsr_pipeline.locate_stickers_vlm``）等价可换：
两者都返回 ``{帧号: [(ymin, ymax, xmin, xmax), ...]}`` 的未外扩原始框，
外扩与时序关联统一由 ``backend.subtitle_tracking`` 承担。

相对 VLM 的取舍：无 API Key 依赖、无单视频调用预算上限、可密集采样；
代价是需本地权重（约 658MB）与一次显存占用。

判据标定（720×560 crop）：
开放词汇检测器会把真狗、鞋子、人腿一并标成 ``cartoon icon``，
但误检框面积中位 9.2 万 px、最大 39.6 万 px，而 emoji 框仅 693–898 px，
两者相差两个数量级 —— 面积上限是把它们分开的关键判据，缺了它精度只有 13%。

召回按"单体"计（每帧 3 个 emoji 检出几个），而非"帧级"（该帧是否检出任一）：
帧级口径会高估效果，实测帧级 100% 时单体只有 56%，画面上仍有 emoji 残留。
两个阈值都随素材的 emoji 显示尺寸与样式变化，换素材需重新标定。
"""
import os
from typing import Dict, List, Sequence, Tuple

Box = Tuple[int, int, int, int]

# 经实测标定的默认判据，详见模块文档字符串与 docs/02-use/04。
DEFAULT_MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
# 短语以 . 分隔是 GroundingDINO 的输入约定。收窄为 "emoji. sticker." 会让
# 单体召回明显下降，说明具体短语对召回影响很大，因此本串按素材可调
# （CLI --sticker-prompt）。
DEFAULT_PROMPT = 'emoji. sticker. cartoon icon. dog face icon.'
DEFAULT_SCORE_THRESHOLD = 0.25    # 单体召回 85%(0.30 只有 56%,实测会漏擦同排 emoji)。
                                  # 代价是少量远离字幕的小物体误检(实测 5/60 帧命中
                                  # 同一车身零件),这类框由 associate_sticker_hits
                                  # 的"须靠近字幕框"约束兜底
DEFAULT_MAX_AREA_RATIO = 0.003    # 占 crop 面积上限；720×560 下约 1200px，emoji 实测最大 898px
DEFAULT_MAX_FRAMES = 200          # 本地推理无 API 成本，采样密度只受算力约束


def filter_detections(
    detections: Sequence[dict],
    crop_area: int,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    max_area_ratio: float = DEFAULT_MAX_AREA_RATIO,
) -> List[Tuple[float, float, float, float]]:
    """按置信度与面积上限筛出贴纸框，返回 crop 坐标系的 (x1, y1, x2, y2)。

    面积上限是本方案能用的前提：开放词汇检测器对真实物体的高分误检
    全部是整幅级大框，仅靠置信度无法与 emoji 分开。
    """
    max_area = max(1.0, float(crop_area) * float(max_area_ratio))
    kept = []
    for det in detections:
        if float(det['score']) < score_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in det['box'])
        if x1 >= x2 or y1 >= y2:
            continue
        if (x2 - x1) * (y2 - y1) > max_area:
            continue
        kept.append((x1, y1, x2, y2))
    return kept


def to_frame_box(crop_box: Sequence[float], region: Sequence[int]) -> Box:
    """crop 坐标系的 (x1, y1, x2, y2) → 全帧 (ymin, ymax, xmin, xmax)，不外扩。

    外扩留给 ``vsr_pipeline`` 统一施加 ``STICKER_MASK_PAD``，
    与 VLM 路径保持同一套语义。
    """
    x1, y1, x2, y2 = crop_box
    ymin, ymax, xmin, xmax = region
    return (max(ymin, int(y1) + ymin), min(ymax, int(y2) + ymin),
            max(xmin, int(x1) + xmin), min(xmax, int(x2) + xmin))


class GroundingDinoStickerDetector:
    """常驻的 GroundingDINO 推理封装；worker 进程内加载一次可处理多条视频。"""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = 'auto'):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()

    def detect_crop(self, crop, prompt: str, score_threshold: float,
                    max_area_ratio: float) -> List[Tuple[float, float, float, float]]:
        """对单张 RGB crop 推理，返回已过滤的 crop 坐标框。"""
        import torch
        from PIL import Image

        pil = Image.fromarray(crop)
        inputs = self.processor(images=pil, text=prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        # 后处理阈值取得比判据低，细筛统一交给 filter_detections，
        # 便于离线复算判据时无需重跑推理。
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs['input_ids'],
            box_threshold=min(0.15, score_threshold),
            text_threshold=min(0.15, score_threshold),
            target_sizes=[pil.size[::-1]],
        )[0]
        detections = [{'score': float(s), 'box': [float(v) for v in b]}
                      for s, b in zip(results['scores'], results['boxes'])]
        return filter_detections(detections, crop.shape[0] * crop.shape[1],
                                 score_threshold, max_area_ratio)

    def locate(self, video_path, region: Sequence[int], sample_frames: Sequence[int],
               prompt: str = DEFAULT_PROMPT,
               score_threshold: float = DEFAULT_SCORE_THRESHOLD,
               max_area_ratio: float = DEFAULT_MAX_AREA_RATIO) -> Dict[int, List[Box]]:
        """按采样帧定位贴纸，返回 ``{帧号: [未外扩全帧框]}``。

        本地推理不会像 API 那样失败，因此每个采样帧都会留下条目（无贴纸即空列表）。
        ``associate_sticker_hits`` 依赖"成功但为空"的采样来终止轨迹，
        条目齐全反而让时序关联比 VLM 路径更准。
        """
        import av
        import numpy as np

        wanted = sorted({int(i) for i in sample_frames if int(i) >= 0})
        if not wanted:
            return {}
        ymin, ymax, xmin, xmax = region
        hits: Dict[int, List[Box]] = {}
        target = set(wanted)
        with av.open(os.fspath(video_path)) as src:
            for n, frame in enumerate(src.decode(video=0)):
                if n > wanted[-1]:
                    break
                if n not in target:
                    continue
                crop = np.asarray(frame.to_image())[ymin:ymax, xmin:xmax]
                boxes = self.detect_crop(crop, prompt, score_threshold, max_area_ratio)
                frame_boxes = [to_frame_box(b, region) for b in boxes]
                hits[n] = list(dict.fromkeys(
                    b for b in frame_boxes if b[0] < b[1] and b[2] < b[3]))
        print(f'[sticker-gdino] frames={len(hits)} hits={sum(map(len, hits.values()))}')
        return hits