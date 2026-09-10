# -*- coding: utf-8 -*-
"""本地开放词汇贴纸/emoji 检测（GroundingDINO）。

兼容的 ``locate`` 返回采样原始框；生产流水线使用 ``detect_candidates``
保留分数，由 ``backend.sticker_tracking`` 做逐帧外观核验和反馈补查。
两条路径都输出未外扩框，遮罩外扩由流水线统一处理。

相对 VLM 的取舍：无 API Key 依赖、无 API 费用，模型调用仍受帧预算约束；
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
from dataclasses import dataclass
from math import isfinite
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
DEFAULT_MAX_AREA_PX = 1200       # 绝对像素上限；emoji 实测最大 898px，留 ~30% 余量。
                                  # 用绝对值而非相对 crop 的比例：ROI 自适应后尺度会变，
                                  # 全屏下比例判据会漂到 2 倍以上，导致误检框漏过；
                                  # 绝对像素不随裁剪尺寸变化，判据一致性更好
DEFAULT_MAX_FRAMES = 200          # 本地推理无 API 成本，采样密度只受算力约束


@dataclass(frozen=True)
class StickerCandidate:
    """未外扩的全帧整数框与原始检测置信度。"""

    box: Box
    score: float


def filter_detections(
    detections: Sequence[dict],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    max_area_px: float = DEFAULT_MAX_AREA_PX,
) -> List[Tuple[float, float, float, float]]:
    """按置信度与面积上限筛出贴纸框，返回 crop 坐标系的 (x1, y1, x2, y2)。

    面积上限是本方案能用的前提：开放词汇检测器对真实物体的高分误检
    全部是整幅级大框，仅靠置信度无法与 emoji 分开。
    面积用绝对像素而非相对比例：ROI 自适应后裁剪尺寸变化较大，
    比例判据会漂移；绝对像素不随裁剪尺寸变化，判据一致性更好。
    """
    return [box for _, box in _filter_scored_detections(
        detections, score_threshold, max_area_px)]


def _filter_scored_detections(detections, score_threshold, max_area_px):
    max_area = max(1.0, float(max_area_px))
    kept = []
    for det in detections:
        score = float(det['score'])
        if not isfinite(score) or score < score_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in det['box'])
        if not all(isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        if x1 >= x2 or y1 >= y2:
            continue
        if (x2 - x1) * (y2 - y1) > max_area:
            continue
        kept.append((score, (x1, y1, x2, y2)))
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


def filter_candidates(
    detections: Sequence[dict],
    region: Sequence[int],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    max_area_px: float = DEFAULT_MAX_AREA_PX,
) -> List[StickerCandidate]:
    """保留弱检测置信度，按分数降序去重后返回未外扩的全帧候选框。"""
    candidates = []
    for score, crop_box in _filter_scored_detections(
            detections, min(0.15, score_threshold), max_area_px):
        box = to_frame_box(crop_box, region)
        if box[0] < box[1] and box[2] < box[3]:
            candidates.append(StickerCandidate(box, score))
    kept = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if not any(_box_iou(candidate.box, other.box) >= 0.6 for other in kept):
            kept.append(candidate)
    return kept


def _box_iou(first: Box, second: Box) -> float:
    top = max(first[0], second[0])
    bottom = min(first[1], second[1])
    left = max(first[2], second[2])
    right = min(first[3], second[3])
    intersection = max(0, bottom - top) * max(0, right - left)
    first_area = (first[1] - first[0]) * (first[3] - first[2])
    second_area = (second[1] - second[0]) * (second[3] - second[2])
    return intersection / (first_area + second_area - intersection)


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

    def _infer_detections(self, crop, prompt: str, score_threshold: float) -> List[dict]:
        import torch
        from PIL import Image

        pil = Image.fromarray(crop)
        inputs = self.processor(images=pil, text=prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        # 后处理保留弱检测，由调用接口选择候选筛选或强阈值筛选。
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs['input_ids'],
            box_threshold=min(0.15, score_threshold),
            text_threshold=min(0.15, score_threshold),
            target_sizes=[pil.size[::-1]],
        )[0]
        return [{'score': float(s), 'box': [float(v) for v in b]}
                for s, b in zip(results['scores'], results['boxes'])]

    def detect_crop(self, crop, prompt: str, score_threshold: float,
                    max_area_px: float) -> List[Tuple[float, float, float, float]]:
        """对单张 RGB crop 推理，返回已过滤的 crop 坐标框。"""
        return filter_detections(
            self._infer_detections(crop, prompt, score_threshold),
            score_threshold, max_area_px)

    def detect_candidates(self, crop, region: Sequence[int], prompt: str,
                          score_threshold: float,
                          max_area_px: float) -> List[StickerCandidate]:
        """对单张 RGB crop 推理一次，返回保留置信度的全帧候选框。"""
        return filter_candidates(
            self._infer_detections(crop, prompt, score_threshold),
            region, score_threshold, max_area_px)

    def locate(self, video_path, region: Sequence[int], sample_frames: Sequence[int],
               prompt: str = DEFAULT_PROMPT,
               score_threshold: float = DEFAULT_SCORE_THRESHOLD,
               max_area_px: float = DEFAULT_MAX_AREA_PX) -> Dict[int, List[Box]]:
        """按采样帧定位贴纸，返回 ``{帧号: [未外扩全帧框]}``。

        兼容旧接口，成功但无贴纸留下空列表，推理失败抛出异常。
        此接口不做外观续跟；生产路径使用 ``detect_candidates``。
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
                boxes = self.detect_crop(crop, prompt, score_threshold, max_area_px)
                frame_boxes = [to_frame_box(b, region) for b in boxes]
                hits[n] = list(dict.fromkeys(
                    b for b in frame_boxes if b[0] < b[1] and b[2] < b[3]))
        print(f'[sticker-gdino] frames={len(hits)} hits={sum(map(len, hits.values()))}')
        return hits
