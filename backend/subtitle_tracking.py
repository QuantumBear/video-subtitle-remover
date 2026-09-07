"""无模型依赖的字幕框采样、跨帧轨迹和区间处理工具。

本模块只处理整数坐标和帧号，因此可以在没有视频/OCR 推理依赖的环境中单独测试。
框坐标统一为 ``(ymin, ymax, xmin, xmax)``。
"""

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class BoxTrack:
    track_id: int
    frames: List[int]
    boxes: List[Box]

    def __post_init__(self):
        if len(self.frames) != len(self.boxes):
            raise ValueError("frames and boxes must have the same length")


def _valid_box(box) -> Optional[Box]:
    try:
        y1, y2, x1, x2 = (int(value) for value in box)
    except (TypeError, ValueError, OverflowError):
        return None
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1:
        return None
    return y1, y2, x1, x2


def _track_boxes(
    sampled: Dict[int, Sequence[Box]],
    total_frames: int,
    max_gap: int,
    match_score: Callable[[Box, Box], float],
    scene_change_frames: Sequence[int] = (),
    break_on_miss: bool = False,
) -> List[BoxTrack]:
    tracks: List[BoxTrack] = []
    active = []
    scene_changes = sorted(set(scene_change_frames))
    for frame_no in sorted(sampled):
        if frame_no < 0 or frame_no >= total_frames:
            continue
        scene = bisect_right(scene_changes, frame_no)
        active = [
            idx for idx in active
            if frame_no - tracks[idx].frames[-1] <= max_gap
            and bisect_right(scene_changes, tracks[idx].frames[-1]) == scene
        ]
        boxes = [valid for box in sampled[frame_no] if (valid := _valid_box(box))]
        candidates = []
        for box_idx, box in enumerate(boxes):
            for track_idx in active:
                score = match_score(box, tracks[track_idx].boxes[-1])
                if score >= 0:
                    candidates.append((-score, track_idx, box_idx))
        used_tracks, used_boxes = set(), set()
        for _, track_idx, box_idx in sorted(candidates):
            if track_idx in used_tracks or box_idx in used_boxes:
                continue
            tracks[track_idx].frames.append(frame_no)
            tracks[track_idx].boxes.append(boxes[box_idx])
            used_tracks.add(track_idx)
            used_boxes.add(box_idx)
        if break_on_miss:
            active = [idx for idx in active if idx in used_tracks]
        for box_idx, box in enumerate(boxes):
            if box_idx not in used_boxes:
                idx = len(tracks)
                tracks.append(BoxTrack(idx, [frame_no], [box]))
                active.append(idx)
    return tracks


def plan_ocr_frames(total_frames: int, stride: int = 5) -> List[int]:
    """返回从 0 开始、包含最后一帧的确定性 OCR 采样帧号。"""
    if total_frames <= 0:
        return []
    stride = max(1, int(stride))
    return sorted(set(range(0, total_frames, stride)) | {total_frames - 1})


def track_text_boxes(
    sampled: Dict[int, Sequence[Box]],
    total_frames: int,
    min_hits: int = 2,
    max_gap: int = 10,
    max_center_delta: float = 90.0,
    scene_change_frames: Sequence[int] = (),
) -> List[BoxTrack]:
    """按中心移动、尺寸比例和时间间隙贪心匹配采样框。

    每个采样帧中的一个框最多归属于一条轨迹；孤立命中会被丢弃。
    ``total_frames`` 用于忽略视频范围外的帧号，避免外部数据污染轨迹。
    """
    if total_frames <= 0:
        return []
    min_hits = max(1, int(min_hits))
    max_gap = max(0, int(max_gap))
    max_center_delta = max(0.0, float(max_center_delta))
    def match_score(box, previous):
        y1, y2, x1, x2 = box
        py1, py2, px1, px2 = previous
        width, height = x2 - x1, y2 - y1
        previous_width, previous_height = px2 - px1, py2 - py1
        if not (0.5 <= width / previous_width <= 2.0
                and 0.5 <= height / previous_height <= 2.0):
            return -1.0
        overlap_x = min(x2, px2) - max(x1, px1)
        overlap_y = min(y2, py2) - max(y1, py1)
        if (overlap_x <= 0 or overlap_y <= 0
                or overlap_x < 0.2 * min(width, previous_width)
                or overlap_y < 0.2 * min(height, previous_height)):
            return -1.0
        dx = (x1 + x2 - px1 - px2) / 2
        dy = (y1 + y2 - py1 - py2) / 2
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > max_center_delta or abs(dy) > 0.75 * max(height, previous_height):
            return -1.0
        return max_center_delta - distance

    return [
        track for track in _track_boxes(
            sampled, total_frames, max_gap, match_score, scene_change_frames
        ) if len(track.frames) >= min_hits
    ]


def materialize_tracks(
    tracks: Sequence[BoxTrack],
    total_frames: int,
    max_interpolation_gap: int = 10,
) -> List[List[Box]]:
    """把轨迹线性插值为逐帧框；超过间隙上限的帧保持空列表。"""
    total_frames = max(0, int(total_frames))
    max_interpolation_gap = max(0, int(max_interpolation_gap))
    timeline: List[List[Box]] = [[] for _ in range(total_frames)]
    for track in tracks:
        for frame_no, box in zip(track.frames, track.boxes):
            if 0 <= frame_no < total_frames:
                timeline[frame_no].append(tuple(int(v) for v in box))
        for left, right in zip(range(len(track.frames) - 1), range(1, len(track.frames))):
            start = track.frames[left]
            end = track.frames[right]
            gap = end - start
            if gap <= 1 or gap > max_interpolation_gap:
                continue
            a = track.boxes[left]
            b = track.boxes[right]
            for frame_no in range(start + 1, end):
                ratio = (frame_no - start) / gap
                interpolated = tuple(round(a[i] + (b[i] - a[i]) * ratio) for i in range(4))
                if 0 <= frame_no < total_frames:
                    timeline[frame_no].append(interpolated)
    return timeline


def _timeline_mapping(text_timeline, total_frames: int) -> Dict[int, List[Box]]:
    items = text_timeline.items() if isinstance(text_timeline, Mapping) else enumerate(text_timeline)
    return {
        frame_no: [valid for box in boxes if (valid := _valid_box(box))]
        for frame_no, boxes in items if 0 <= frame_no < total_frames and boxes
    }


def plan_vlm_frames(
    total_frames: int,
    text_timeline: Sequence[Sequence[Box]],
    max_calls: int = 32,
    base_step: int = 30,
    scene_change_frames: Sequence[int] = (),
) -> List[int]:
    """按开头、字幕轨迹首尾、中点、均匀采样的优先级返回候选帧。

    ``base_step`` 表示约一秒的帧数；开头两秒每半秒采样一次。
    返回值刻意不按帧号排序，以便调用预算不足时保留高优先级帧。
    """
    total_frames, max_calls = int(total_frames), int(max_calls)
    if total_frames <= 0 or max_calls <= 0:
        return []
    base_step = max(1, int(base_step))
    result, seen = [], set()

    def add(frame_no):
        if len(result) < max_calls and 0 <= frame_no < total_frames and frame_no not in seen:
            seen.add(frame_no)
            result.append(frame_no)

    for half_second in range(5):
        add(round(half_second * base_step / 2))
    sampled = _timeline_mapping(text_timeline, total_frames)
    tracks = track_text_boxes(
        sampled, total_frames, min_hits=1, max_gap=1, scene_change_frames=scene_change_frames
    )
    for track in tracks:
        add(track.frames[0])
        add(track.frames[-1])
    for track in tracks:
        add((track.frames[0] + track.frames[-1]) // 2)

    remaining = max_calls - len(result)
    base_count = (total_frames - 1) // base_step + 1
    uniform_count = min(base_count, remaining + 1)
    for index in range(uniform_count):
        grid_index = round(index * (base_count - 1) / max(1, uniform_count - 1))
        add(grid_index * base_step)
    add(total_frames - 1)
    return result


def sticker_match_score(b1: Box, b2: Box) -> float:
    """用原始框的尺寸、中心距离与重叠率匹配贴纸，排除相邻目标。"""
    a, b = _valid_box(b1), _valid_box(b2)
    if a is None or b is None:
        return -1.0
    y1a, y2a, x1a, x2a = a
    y1b, y2b, x1b, x2b = b
    wa, ha = x2a - x1a, y2a - y1a
    wb, hb = x2b - x1b, y2b - y1b
    if not (0.4 <= wa / wb <= 2.5 and 0.4 <= ha / hb <= 2.5):
        return -1.0
    if (abs(x1a + x2a - x1b - x2b) / 2 > 0.35 * (wa + wb)
            or abs(y1a + y2a - y1b - y2b) / 2 > 0.35 * (ha + hb)):
        return -1.0
    inter_w = max(0, min(x2a, x2b) - max(x1a, x1b))
    inter_h = max(0, min(y2a, y2b) - max(y1a, y1b))
    overlap = inter_w * inter_h / min(wa * ha, wb * hb)
    return overlap if overlap >= 0.15 else 0.01


def group_sticker_boxes(
    hits: Dict[int, Sequence[Box]], min_samples: int = 2
) -> List[Tuple[Box, List[int]]]:
    """兼容旧调用方的稳定贴纸分组接口；返回首框和命中帧号。"""
    total_frames = max(hits, default=-1) + 1
    return [
        (track.boxes[0], track.frames)
        for track in _track_boxes(hits, total_frames, total_frames, sticker_match_score)
        if len(track.frames) >= max(1, int(min_samples))
    ]


def _box_distance(a: Box, b: Box) -> float:
    dy = max(0, a[0] - b[1], b[0] - a[1])
    dx = max(0, a[2] - b[3], b[2] - a[3])
    return (dx * dx + dy * dy) ** 0.5


def associate_sticker_hits(
    hits: Dict[int, Sequence[Box]],
    text_timeline: Sequence[Sequence[Box]],
    total_frames: int,
    max_distance: float = 120,
    single_hit_radius: int = 6,
    max_gap: int = 60,
    scene_change_frames: Sequence[int] = (),
) -> Dict[int, List[Box]]:
    """将未外扩贴纸框关联到字幕附近，输出局部插值的逐帧框。

    ``hits`` 必须保留成功但未发现贴纸的空列表，失败请求则不应进入
    ``hits``。任一成功采样中的目标缺失都会终止该目标轨迹。单次命中
    只传播指定半径；稳定轨迹向最近缺失采样的中点扩展，最多扩展
    ``max_gap`` 帧。匹配和传播均限制在同一场景，每个输出框仍须靠近
    同场景、同帧或时间半径内的字幕。
    """
    total_frames = int(total_frames)
    if total_frames <= 0:
        return {}
    radius = max(0, int(single_hit_radius))
    max_gap = max(0, int(max_gap))
    max_distance = max(0.0, float(max_distance))
    sampled = {frame: boxes for frame, boxes in hits.items() if 0 <= frame < total_frames}
    sample_frames = sorted(sampled)
    text = _timeline_mapping(text_timeline, total_frames)
    text_frames = sorted(text)
    if not sample_frames or not text_frames:
        return {}
    scene_boundaries = [0] + sorted(set(
        frame for frame in scene_change_frames if 0 < frame < total_frames
    )) + [total_frames]

    def scene_bounds(frame_no):
        scene = bisect_right(scene_boundaries, frame_no)
        return scene_boundaries[scene - 1], scene_boundaries[scene] - 1

    def near_text(frame_no, box):
        scene_start, scene_end = scene_bounds(frame_no)
        left = bisect_left(text_frames, max(scene_start, frame_no - radius))
        right = bisect_right(text_frames, min(scene_end, frame_no + radius))
        return any(
            _box_distance(box, text_box) <= max_distance
            for nearby_frame in text_frames[left:right] for text_box in text[nearby_frame]
        )

    tracks = _track_boxes(
        sampled, total_frames, max_gap, sticker_match_score,
        scene_change_frames=scene_change_frames, break_on_miss=True
    )
    result: Dict[int, List[Box]] = {}
    for track in tracks:
        first, last = track.frames[0], track.frames[-1]
        scene_start, scene_end = scene_bounds(first)
        single_hit = len(track.frames) == 1
        if single_hit and not near_text(first, track.boxes[0]):
            continue
        first_sample = bisect_left(sample_frames, first)
        last_sample = bisect_right(sample_frames, last)
        previous_sample = sample_frames[first_sample - 1] if first_sample else None
        next_sample = sample_frames[last_sample] if last_sample < len(sample_frames) else None
        start, end = max(0, first - radius), min(total_frames - 1, last + radius)
        if previous_sample is not None and previous_sample >= scene_start:
            was_absent = not any(
                sticker_match_score(track.boxes[0], box) >= 0
                for box in sampled[previous_sample]
            )
            start = max(start, previous_sample + 1)
            if not single_hit and was_absent:
                start = max(0, previous_sample + 1, first - max_gap, (previous_sample + first) // 2)
        if next_sample is not None and next_sample <= scene_end:
            was_absent = not any(
                sticker_match_score(track.boxes[-1], box) >= 0
                for box in sampled[next_sample]
            )
            end = min(end, next_sample - 1)
            if not single_hit and was_absent:
                end = min(total_frames - 1, next_sample - 1, last + max_gap, (last + next_sample) // 2)
        for frame_no in range(max(start, scene_start), min(end, scene_end) + 1):
            right = bisect_right(track.frames, frame_no)
            if right == 0:
                box = track.boxes[0]
            elif right == len(track.frames):
                box = track.boxes[-1]
            else:
                a, b = track.boxes[right - 1], track.boxes[right]
                ratio = (frame_no - track.frames[right - 1]) / (
                    track.frames[right] - track.frames[right - 1]
                )
                box = tuple(round(a[i] + (b[i] - a[i]) * ratio) for i in range(4))
            if near_text(frame_no, box):
                result.setdefault(frame_no, []).append(box)
    return result


def merge_closed_ranges(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """合并重叠或相邻的闭区间。"""
    ordered = sorted((int(lo), int(hi)) for lo, hi in ranges if lo <= hi)
    merged: List[Tuple[int, int]] = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def refine_ranges(
    total_frames: int,
    track_frames: Sequence[int],
    scene_change_frames: Sequence[int],
    radius: int = 15,
) -> List[Tuple[int, int]]:
    """围绕轨迹命中和场景变化点生成逐帧补查范围。"""
    total_frames = int(total_frames)
    if total_frames <= 0:
        return []
    radius = max(0, int(radius))
    points = sorted(set(track_frames) | set(scene_change_frames))
    windows = [
        (max(0, point - radius), min(total_frames - 1, point + radius))
        for point in points
        if 0 <= point < total_frames
    ]
    return merge_closed_ranges(windows)


def merge_residual_runs(
    frame_numbers: Sequence[int],
    total: int,
    context: int = 5,
    max_runs: int = 1,
) -> List[Tuple[int, int]]:
    """合并残留帧运行段，扩展上下文并限制最多返回的窗口数。"""
    total = int(total)
    context = max(0, int(context))
    max_runs = int(max_runs)
    if total <= 0 or max_runs <= 0:
        return []
    points = sorted(set(i for i in frame_numbers if 0 <= i < total))
    if not points:
        return []
    runs: List[Tuple[int, int]] = []
    start = previous = points[0]
    for point in points[1:]:
        if point > previous + 1:
            runs.append((max(0, start - context), min(total - 1, previous + context)))
            start = point
        previous = point
    runs.append((max(0, start - context), min(total - 1, previous + context)))
    return merge_closed_ranges(runs)[:max_runs]
