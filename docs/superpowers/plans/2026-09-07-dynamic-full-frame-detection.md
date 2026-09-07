# 动态全屏检测与保守双层遮罩实施计划

## 当前执行状态（2026-09-07）

用户最新修订优先于下文初始代码示例：使用检测反馈调整间隔，稳定后
`1 → 2 → 4 → 5` 递增，有变化就回到逐帧并回查跳过帧。
不采用“每个命中都生成 ±15 帧补查窗口”的旧示例。

- [x] Task 1：纯函数字幕轨迹、独立插值、场景隔离及边界测试。
- [x] Task 2：反馈式 OCR、默认全屏/手动 ROI、CLI 参数和日志。
- [x] Task 3：贴纸采样上限、轨迹优先级、单次邻近例外、独立匹配和失败降级。
- [x] Task 4：白字核心/灰边遮罩、白色大物体过滤、ROI 裁剪。
- [x] Task 5：首轮结果上的局部残留复修，失败保留首轮。
- [x] Task 6（本地）：文档、最终全量验证、需求审查与代码质量审查完成。
- [ ] Task 6（服务器）：GPU 画质/显存回归待目标服务器验证。

最终本地验证：在 Python 3.12 的 `vsr` 环境中运行全量测试，结果为
`77 passed`；使用真实 OpenCV 和 PyAV，修复模型使用测试替身。
`py_compile` 通过。回归包含切景时结束 ProPainter 分段，以及单帧修复段
复制后送入模型、再裁回原帧数的保护。

真实 TikSave 视频的检测阶段实测为 687 帧、428 次 OCR（主动采样 342 次，
变化后补查 86 次），CPU 耗时 419.6 秒，较逐帧调用减少约 38%。
尚未运行真实 DashScope 与 CUDA ProPainter 全流程；0–2 秒贴纸、4–5 秒
背景画质及 24GB GPU 峰值显存不属于已完成验收。

目前未创建提交。下方步骤中的提交命令和旧示例保留作计划记录，
执行情况以上述状态及最终验证记录为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不传 `-c` 时以每 5 帧为默认步长进行全屏 OCR，通过字幕轨迹、DashScope 贴纸关联和保守双层遮罩，减少 0–2 秒 emoji 残留并避免 4–5 秒背景误擦。

**Architecture:** 将跨帧检测从 `vsr_pipeline.py` 中拆成无模型依赖的轨迹/采样模块；流水线第一遍采用“每 5 帧采样 + 不确定区间逐帧补查”，再把轨迹物化为逐帧框。ProPainter 继续使用 40+20 帧显存窗口，但改为双层遮罩，并在首轮结果后对短窗口残留做一次受限复修。

**Tech Stack:** Python 3.12、PyAV、PaddleOCR PP-OCRv5 mobile、NumPy/OpenCV、DashScope OpenAI-compatible API、PyTorch ProPainter、pytest。

---

## 文件边界

- 创建 `backend/subtitle_tracking.py`：纯 Python 的采样计划、文字轨迹、贴纸关联；不导入 PyAV、PaddleOCR、Torch 或 DashScope。
- 修改 `vsr_pipeline.py`：接入采样/轨迹模块，调整无 `-c` 的 ROI 语义，增加双层遮罩、ProPainter 局部残留复修和 CLI 参数/日志。
- 修改 `tests/sticker_tracking_test.py`：保留既有贴纸回归，并补充采样上限、单帧贴纸邻近关联测试。
- 创建 `tests/subtitle_tracking_test.py`：覆盖采样计划、轨迹匹配、间隙补齐和误检过滤。
- 创建 `tests/mask_layers_test.py`：覆盖白字核心/边缘遮罩和背景保护。
- 修改 `docs/02-use/02-怎么用API或命令处理视频.md`、`docs/02-use/04-实战去字幕流程-本机验证版.md`：更新默认无 `-c` 用法、参数和性能预期。

### Task 1: 建立纯函数采样与字幕轨迹模块

**Files:**
- Create: `backend/subtitle_tracking.py`
- Create: `tests/subtitle_tracking_test.py`

- [ ] **Step 1: 写失败测试，固定公共接口**

```python
from backend.subtitle_tracking import (
    BoxTrack,
    materialize_tracks,
    plan_ocr_frames,
    track_text_boxes,
)


def test_plan_ocr_frames_uses_five_frame_stride_and_last_frame():
    assert plan_ocr_frames(19, stride=5) == [0, 5, 10, 15, 18]


def test_track_text_boxes_keeps_two_non_adjacent_regions_separate():
    sampled = {
        0: [(100, 130, 80, 280), (900, 930, 90, 300)],
        5: [(102, 132, 82, 282), (902, 932, 92, 302)],
        10: [(104, 134, 84, 284), (904, 934, 94, 304)],
    }
    tracks = track_text_boxes(sampled, total_frames=11, max_gap=10)
    assert len(tracks) == 2
    assert all(track.frames == [0, 5, 10] for track in tracks)


def test_materialize_tracks_fills_short_gap_without_global_union():
    sampled = {0: [(100, 130, 80, 280)], 10: [(110, 140, 100, 300)]}
    tracks = track_text_boxes(sampled, total_frames=11, max_gap=10)
    timeline = materialize_tracks(tracks, total_frames=11, max_interpolation_gap=10)
    assert timeline[5]
    assert timeline[5][0][0] == 105
    assert timeline[5][0][2] == 90


def test_isolated_box_is_not_materialized():
    tracks = track_text_boxes({20: [(400, 430, 100, 300)]}, total_frames=40,
                              min_hits=2, max_gap=10)
    assert tracks == []
    assert materialize_tracks(tracks, total_frames=40) == [[] for _ in range(40)]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest -q tests/subtitle_tracking_test.py`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.subtitle_tracking'`.

- [ ] **Step 3: 实现最小纯函数模块**

在 `backend/subtitle_tracking.py` 中定义以下类型和函数：

```python
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

Box = Tuple[int, int, int, int]  # ymin, ymax, xmin, xmax


@dataclass(frozen=True)
class BoxTrack:
    track_id: int
    frames: List[int]
    boxes: List[Box]


def plan_ocr_frames(total_frames: int, stride: int = 5) -> List[int]:
    """返回从 0 开始、包含最后一帧的确定性 OCR 采样帧号。"""
    if total_frames <= 0:
        return []
    stride = max(1, int(stride))
    return sorted(set(range(0, total_frames, stride)) | {total_frames - 1})


def track_text_boxes(sampled: Dict[int, Sequence[Box]], total_frames: int,
                     min_hits: int = 2, max_gap: int = 10,
                     max_center_delta: float = 90.0) -> List[BoxTrack]:
    """按中心移动、尺寸比例和时间间隙贪心匹配采样框，丢弃孤立轨迹。"""
    mutable = []
    for frame_no in sorted(sampled):
        used = set()
        for box in sampled[frame_no]:
            y1, y2, x1, x2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            best = None
            best_distance = float("inf")
            for idx, item in enumerate(mutable):
                if idx in used or frame_no - item["frames"][-1] > max_gap:
                    continue
                py1, py2, px1, px2 = item["boxes"][-1]
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                pbw = max(1, px2 - px1)
                pbh = max(1, py2 - py1)
                if not (0.5 <= bw / pbw <= 2.0 and 0.5 <= bh / pbh <= 2.0):
                    continue
                distance = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                if distance <= max_center_delta and distance < best_distance:
                    best = idx
                    best_distance = distance
            if best is None:
                mutable.append({"frames": [frame_no], "boxes": [box]})
                used.add(len(mutable) - 1)
            else:
                mutable[best]["frames"].append(frame_no)
                mutable[best]["boxes"].append(box)
                used.add(best)
    return [BoxTrack(i, item["frames"], item["boxes"])
            for i, item in enumerate(mutable)
            if len(item["frames"]) >= min_hits]


def materialize_tracks(tracks: Sequence[BoxTrack], total_frames: int,
                       max_interpolation_gap: int = 10) -> List[List[Box]]:
    """把轨迹线性插值为逐帧框；超过间隙上限的帧保持空列表。"""
    timeline = [[] for _ in range(max(0, total_frames))]
    for track in tracks:
        for frame_no, box in zip(track.frames, track.boxes):
            if 0 <= frame_no < total_frames:
                timeline[frame_no].append(box)
        for left, right in zip(range(len(track.frames) - 1),
                               range(1, len(track.frames))):
            start = track.frames[left]
            end = track.frames[right]
            gap = end - start
            if gap <= 1 or gap > max_interpolation_gap:
                continue
            a = track.boxes[left]
            b = track.boxes[right]
            for frame_no in range(start + 1, end):
                ratio = (frame_no - start) / gap
                interpolated = tuple(
                    round(a[i] + (b[i] - a[i]) * ratio) for i in range(4))
                timeline[frame_no].append(interpolated)
    return timeline


def merge_closed_ranges(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ordered = sorted((lo, hi) for lo, hi in ranges if lo <= hi)
    merged = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def refine_ranges(total_frames: int, track_frames: Sequence[int],
                  scene_change_frames: Sequence[int], radius: int = 15
                  ) -> List[Tuple[int, int]]:
    points = sorted(set(track_frames) | set(scene_change_frames))
    if total_frames <= 0:
        return []
    radius = max(0, int(radius))
    windows = [(max(0, point - radius),
                min(total_frames - 1, point + radius))
               for point in points if 0 <= point < total_frames]
    return merge_closed_ranges(windows)


def merge_residual_runs(frame_numbers: Sequence[int], total: int,
                        context: int = 5, max_runs: int = 1
                        ) -> List[Tuple[int, int]]:
    points = sorted(set(i for i in frame_numbers if 0 <= i < total))
    if not points or total <= 0 or max_runs <= 0:
        return []
    runs = []
    start = previous = points[0]
    for point in points[1:]:
        if point > previous + 1:
            runs.append((max(0, start - context),
                         min(total - 1, previous + context)))
            start = point
        previous = point
    runs.append((max(0, start - context),
                 min(total - 1, previous + context)))
    return merge_closed_ranges(runs)[:max_runs]
```

实现时使用整数坐标、对每个轨迹独立插值，禁止将不同轨迹的框取空间并集。`BoxTrack` 的 `frames` 和 `boxes` 长度必须相同，空视频返回空列表。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest -q tests/subtitle_tracking_test.py`

Expected: 4 tests PASS。

- [ ] **Step 5: 提交轨迹模块**

```bash
git add backend/subtitle_tracking.py tests/subtitle_tracking_test.py
git commit -m "feat: 添加字幕采样与时序轨迹模块"
```

### Task 2: 接入每 5 帧 OCR 和不确定区间补查

**Files:**
- Modify: `vsr_pipeline.py:40-80, 470-525, 610-740, 810-835`
- Modify: `tests/subtitle_tracking_test.py`

- [ ] **Step 1: 先为补查范围写失败测试**

```python
from backend.subtitle_tracking import refine_ranges


def test_refine_ranges_covers_track_edges_and_scene_change():
    ranges = refine_ranges(
        total_frames=100,
        track_frames=[20, 25, 30],
        scene_change_frames=[60],
        radius=15,
    )
    assert ranges == [(5, 75)]


def test_refine_ranges_merges_overlapping_windows():
    assert refine_ranges(100, [20, 30], [35], radius=10) == [(10, 45)]
```

`refine_ranges()` 必须返回闭区间并合并重叠范围，不能产生超过 `[0, total_frames-1]` 的坐标。

- [ ] **Step 2: 运行失败测试**

Run: `pytest -q tests/subtitle_tracking_test.py -k refine_ranges`

Expected: FAIL with `ImportError: cannot import name 'refine_ranges'`。

- [ ] **Step 3: 实现补查范围并加入 OCR 计划**

在 `backend/subtitle_tracking.py` 增加：

```python
def refine_ranges(total_frames: int, track_frames: Sequence[int],
                  scene_change_frames: Sequence[int], radius: int = 15
                  ) -> List[Tuple[int, int]]:
    points = sorted(set(track_frames) | set(scene_change_frames))
    windows = [(max(0, p - radius), min(total_frames - 1, p + radius))
               for p in points if 0 <= p < total_frames]
    return merge_closed_ranges(windows)
```

在 `vsr_pipeline.py` 增加常量 `OCR_STRIDE = 5`、`OCR_REFINE_RADIUS = 15`、`SCENE_THUMB_SIZE = 32`、`SCENE_MEAN_DIFF = 18.0`。第一遍处理改为：

1. 打开输入视频，按 `plan_ocr_frames(total, OCR_STRIDE)` 采样；所有帧仍正常解码，但只在采样帧调用 `self.detect()`。
2. 将帧缩放到 `32x32` 后计算相邻采样帧的均值绝对差，超过 `18.0` 时生成场景切换候选；将采样 OCR 命中帧和场景切换帧传入 `refine_ranges()`。
3. 重新打开输入视频，只对补查范围内的帧调用 OCR；将采样结果和补查结果合并后交给 `track_text_boxes()` 和 `materialize_tracks()`。
4. 输出日志：`[detect] mode=full-frame stride=5 sampled=<N> refined=<M> tracks=<K>`。

手动 `-c` 时仍把用户 ROI 传给 `detect()`；不传 `-c` 时直接使用 `(0, h, 0, w)`，不再调用 `auto_region()` 生成整体活动带。检测模式和 ROI 需要在日志中明确打印。

- [ ] **Step 4: 增加可选 CLI 参数并保持默认值**

在 `main()` 增加：

```python
ap.add_argument('--ocr-stride', type=int, default=5,
                help='自动模式 OCR 采样步长(帧),默认 5')
ap.add_argument('--ocr-refine-radius', type=int, default=15,
                help='场景/轨迹不确定点前后逐帧补查半径,默认 15')
ap.add_argument('--vlm-max-calls', type=int, default=32,
                help='单视频 DashScope 最大请求数,默认 32')
```

将三个值传入 `Pipeline.process_video()`；`-c` 模式也接受参数但不改变 ROI 语义。对步长和半径使用 `max(1, value)` 校验，VLM 上限至少为 1。

- [ ] **Step 5: 运行轨迹和静态检查**

Run: `pytest -q tests/subtitle_tracking_test.py && python -m py_compile vsr_pipeline.py backend/subtitle_tracking.py`

Expected: 所有轨迹测试 PASS，编译命令无输出且退出码为 0。

- [ ] **Step 6: 提交自适应 OCR**

```bash
git add vsr_pipeline.py backend/subtitle_tracking.py tests/subtitle_tracking_test.py
git commit -m "feat: 使用五帧步长和不确定区间补查OCR"
```

### Task 3: 改造 DashScope 全屏采样与贴纸关联

**Files:**
- Modify: `vsr_pipeline.py:238-320, 650-690`
- Modify: `backend/subtitle_tracking.py`
- Modify: `tests/sticker_tracking_test.py`

- [ ] **Step 1: 写失败测试固定采样上限和单帧邻近规则**

```python
from backend.subtitle_tracking import (
    associate_sticker_hits,
    plan_vlm_frames,
)


def test_plan_vlm_frames_is_bounded_and_covers_early_window():
    frames = plan_vlm_frames(687, text_timeline=[[(500, 540, 250, 500)]] * 687,
                              max_calls=32)
    assert len(frames) <= 32
    assert {0, 15, 30, 45, 60}.issubset(frames)


def test_single_sticker_hit_near_text_is_limited_to_short_window():
    text = [[(500, 540, 250, 500)] for _ in range(40)]
    hits = {10: [(545, 575, 300, 330)]}
    out = associate_sticker_hits(hits, text, total_frames=40)
    assert out
    assert min(out) >= 4 and max(out) <= 16


def test_single_sticker_hit_far_from_text_is_dropped():
    text = [[(500, 540, 250, 500)] for _ in range(40)]
    assert associate_sticker_hits({10: [(100, 130, 20, 50)]}, text, 40) == {}
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest -q tests/sticker_tracking_test.py -k "vlm_frames or single_sticker"`

Expected: FAIL because the new helpers do not exist。

- [ ] **Step 3: 实现贴纸采样和关联**

在 `backend/subtitle_tracking.py` 增加：

```python
def plan_vlm_frames(total_frames: int, text_timeline: Sequence[Sequence[Box]],
                    max_calls: int = 32, base_step: int = 30) -> List[int]:
    # 基础每秒采样 + 轨迹首尾/中点 + 前 60 帧每 15 帧采样；按优先级去重，
    # 最后截断到 max_calls，保证早期窗口优先保留。
    if total_frames <= 0 or max_calls <= 0:
        return []
    base_step = max(1, int(base_step))
    priority = list(range(0, min(total_frames, 61), 15))
    priority.extend(range(0, total_frames, base_step))
    hit_frames = [i for i, boxes in enumerate(text_timeline) if boxes]
    if hit_frames:
        priority.extend([hit_frames[0], hit_frames[-1],
                         (hit_frames[0] + hit_frames[-1]) // 2])
    ordered = []
    seen = set()
    for frame_no in priority:
        if 0 <= frame_no < total_frames and frame_no not in seen:
            ordered.append(frame_no)
            seen.add(frame_no)
    return ordered[:max_calls]


def associate_sticker_hits(hits: Dict[int, Sequence[Box]],
                           text_timeline: Sequence[Sequence[Box]],
                           total_frames: int, max_distance: int = 120,
                           single_hit_radius: int = 6
                           ) -> Dict[int, List[Box]]:
    # 稳定轨迹沿用 _group_sticker_boxes；单次命中只有在时间/空间邻近文字时
    # 才允许在命中帧前后 single_hit_radius 帧传播。
    result = {}
    for frame_no, boxes in sorted(hits.items()):
        if not (0 <= frame_no < total_frames):
            continue
        text_boxes = text_timeline[frame_no] if frame_no < len(text_timeline) else []
        for box in boxes:
            y1, y2, x1, x2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            nearby = False
            for ty1, ty2, tx1, tx2 in text_boxes:
                tx = min(abs(cx - tx1), abs(cx - tx2),
                         0 if tx1 <= cx <= tx2 else abs(cx - (tx1 + tx2) / 2))
                ty = min(abs(cy - ty1), abs(cy - ty2),
                         0 if ty1 <= cy <= ty2 else abs(cy - (ty1 + ty2) / 2))
                if (tx * tx + ty * ty) ** 0.5 <= max_distance:
                    nearby = True
                    break
            if not nearby:
                continue
            for target in range(max(0, frame_no - single_hit_radius),
                                min(total_frames, frame_no + single_hit_radius + 1)):
                result.setdefault(target, []).append(box)
    return result
```

相邻 emoji 必须继续使用现有 `_sticker_match_score()` 的独立匹配逻辑。`locate_stickers_vlm()` 增加 `max_calls` 参数，在请求前按计划帧过滤，达到上限立即停止采样并打印 `[sticker-vlm] calls=<N>/<max>`。

- [ ] **Step 4: 接入流水线**

在 `process_video()` 中增加 `ocr_stride=OCR_STRIDE`、`ocr_refine_radius=OCR_REFINE_RADIUS`、`vlm_max_calls=32` 参数；将 `all_boxes` 替换为轨迹物化后的逐帧文字框，再把该时间线传给 `plan_vlm_frames()`。自动模式的 `region` 是整帧，手动 ROI 继续裁剪。VLM 命中先经过 `associate_sticker_hits()`，再与文字框合并；不得再用“仅在 OCR 命中帧 ±3 帧”这一单一门槛过滤。

无 API key、超时、HTTP 非 2xx 或 JSON 解析失败时返回空贴纸结果并继续 OCR；每个异常打印一次帧号和错误类型，不打印密钥。

- [ ] **Step 5: 运行贴纸回归**

Run: `pytest -q tests/sticker_tracking_test.py`

Expected: 既有相邻 emoji、坐标换算、ProPainter 配置测试和新增关联测试全部 PASS。

- [ ] **Step 6: 提交 DashScope 关联**

```bash
git add vsr_pipeline.py backend/subtitle_tracking.py tests/sticker_tracking_test.py
git commit -m "feat: 限制DashScope采样并关联字幕轨迹"
```

### Task 4: 实现白字幕双层遮罩

**Files:**
- Modify: `vsr_pipeline.py:45-65, 440-470, 410-440, 720-760`
- Create: `tests/mask_layers_test.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
from vsr_pipeline import Pipeline


def test_dual_layer_mask_covers_antialias_but_not_gap():
    pipe = Pipeline.__new__(Pipeline)
    frame = np.zeros((80, 240, 3), dtype=np.uint8)
    frame[30:42, 20:55] = 255       # 核心白字
    frame[28:44, 55:58] = 205       # 抗锯齿/描边边缘
    frame[30:42, 90:125] = 255
    mask = pipe.propainter_boxes_to_mask([(24, 48, 15, 135)], frame,
                                         (0, 80, 0, 240))
    assert mask[35, 30] == 255
    assert mask[35, 56] == 255       # 较低亮度边缘被纳入
    assert mask[35, 75] == 0         # 字间背景仍保留


def test_dual_layer_mask_does_not_expand_white_object_outside_text_box():
    pipe = Pipeline.__new__(Pipeline)
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    frame[10:110, 170:230] = 255     # 高大的白色衣物
    mask = pipe.propainter_boxes_to_mask([(35, 60, 20, 120)], frame,
                                         (0, 120, 0, 240))
    assert mask[:, 180].sum() == 0
```

- [ ] **Step 2: 运行测试确认当前实现失败**

Run: `pytest -q tests/mask_layers_test.py`

Expected: 第二个断言或抗锯齿断言 FAIL，证明当前严格字形遮罩没有新边缘层。

- [ ] **Step 3: 实现核心层和边缘层**

新增常量：

```python
WHITE_EDGE_TH = 180
WHITE_EDGE_DILATE = 3
```

将 `white_glyph(frame, region, threshold=WHITE_ORIG_TH)` 参数化；在 `propainter_boxes_to_mask()` 中对每个明显横向白字幕框计算：

```python
core = self.filter_glyph_by_height(self.white_glyph(frame_rgb, region,
                                                    WHITE_ORIG_TH))
loose = self.filter_glyph_by_height(self.white_glyph(frame_rgb, region,
                                                     WHITE_EDGE_TH))
near_core = cv2.dilate(core, np.ones((7, 7), np.uint8))
edge = cv2.bitwise_and(loose, near_core)
edge = cv2.dilate(edge, np.ones((WHITE_EDGE_DILATE, WHITE_EDGE_DILATE), np.uint8))
text_mask = cv2.bitwise_or(core, edge)
```

只把 `text_mask` 在当前 OCR 框的局部区域写入最终遮罩；彩色字幕和近方形贴纸继续使用各自小矩形。不要恢复整行矩形或 `MASK_EXPAND_DOWN`。

- [ ] **Step 4: 运行遮罩测试和已有回归**

Run: `pytest -q tests/mask_layers_test.py tests/sticker_tracking_test.py`

Expected: 新双层测试和既有遮罩/贴纸测试全部 PASS。

- [ ] **Step 5: 提交双层遮罩**

```bash
git add vsr_pipeline.py tests/mask_layers_test.py
git commit -m "feat: 为ProPainter加入保守白字幕双层遮罩"
```

### Task 5: 加入 ProPainter 局部残留复核

**Files:**
- Modify: `vsr_pipeline.py:690-735`
- Modify: `tests/mask_layers_test.py`

- [ ] **Step 1: 写失败测试固定残留窗口合并**

```python
from backend.subtitle_tracking import merge_residual_runs


def test_merge_residual_runs_adds_context_and_caps_one_run():
    assert merge_residual_runs([10, 11, 13, 40], total=60,
                               context=5, max_runs=1) == [(5, 18)]
```

- [ ] **Step 2: 运行失败测试**

Run: `pytest -q tests/mask_layers_test.py -k residual`

Expected: FAIL with missing helper。

- [ ] **Step 3: 实现受限二次修复**

在 ProPainter 分支为 `seg_frames`、`seg_masks` 同步保存 `seg_boxes`。把写出逻辑抽为：

```python
def _repair_propainter_segment(self, frames_bgr, masks, boxes, pts):
    first = self.inpainter.inpaint(frames_bgr, masks)
    residual = [self._residual_mask(first[i], boxes[i]) for i in range(len(first))]
    runs = merge_residual_runs(
        [i for i, mask in enumerate(residual)
         if int((mask > 0).sum()) >= RESID_MIN_PX],
        total=len(first), context=5, max_runs=1)
    repairs = 0
    for lo, hi in runs:
        local_masks = [residual[i] for i in range(lo, hi + 1)]
        try:
            second = self.inpainter.inpaint(frames_bgr[lo:hi + 1], local_masks)
        except RuntimeError as exc:
            print(f'[propainter] 局部残留复修失败,保留首轮结果: {type(exc).__name__}')
            continue
        for i, repaired in enumerate(second, start=lo):
            m = cv2.dilate(residual[i], np.ones((5, 5), np.uint8))[:, :, None] > 0
            first[i] = np.where(m, repaired, first[i])
        repairs += 1
    return first, repairs
```

`_residual_mask()` 先将 ProPainter 的 BGR 结果转为 RGB，再只在 `boxes[i]` 内调用白字判据，不能扫描整帧；`merge_residual_runs()` 合并相邻残留帧、扩展前后文并最多返回一个窗口。局部二次调用失败时保留首轮结果并继续输出，首轮 ProPainter 失败仍按原有逻辑报告错误。

- [ ] **Step 4: 更新统计和进度日志**

`flush_segment()` 接收并裁剪 `seg_boxes`，写出前调用 `_repair_propainter_segment()`；将返回的 `repairs` 累加到 `n_repair`。完成日志继续输出修复帧数和局部补擦窗口数。

- [ ] **Step 5: 运行单元测试和编译**

Run: `pytest -q tests/mask_layers_test.py && python -m py_compile vsr_pipeline.py`

Expected: 测试 PASS，编译退出码 0。

- [ ] **Step 6: 提交局部残留复核**

```bash
git add vsr_pipeline.py tests/mask_layers_test.py
git commit -m "feat: 为ProPainter增加局部残留复核"
```

### Task 6: 更新文档、CLI 帮助和服务器回归

**Files:**
- Modify: `docs/02-use/02-怎么用API或命令处理视频.md`
- Modify: `docs/02-use/04-实战去字幕流程-本机验证版.md`
- Modify: `tests/sticker_tracking_test.py`

- [ ] **Step 1: 更新用户命令示例**

将默认示例改为：

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_auto_dynamic.mp4 \
  --inpaint-mode propainter
```

同时保留手动 ROI 对照示例，并说明：无 `-c` 为全屏自动检测，每 5 帧 OCR；`-c` 只用于已知固定区域的视频。

- [ ] **Step 2: 运行本地测试和静态检查**

Run:

```bash
pytest -q
python -m py_compile vsr_pipeline.py backend/subtitle_tracking.py
git diff --check
```

Expected: 全部测试 PASS；编译和 diff 检查退出码为 0。

- [ ] **Step 3: 在服务器运行无框回归**

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_auto_dynamic.mp4 \
  --inpaint-mode propainter \
  --ocr-stride 5 \
  --vlm-max-calls 32
```

记录并核对：

- `[detect] mode=full-frame stride=5`，OCR 总调用约 150–250；
- `[sticker-vlm] calls=<N>/32`；
- 0–2 秒 emoji 无明显残留；
- 4–5 秒楼梯、裤腿和车身没有大块擦除或闪烁；
- ProPainter 峰值显存低于 24GB，未发生 OOM。

- [ ] **Step 4: 用 ffprobe 和关键帧验收输出**

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,nb_frames,duration \
  -of default=nw=1 test_auto_dynamic.mp4
```

核对输出帧数、尺寸、时长和音频轨道与输入一致，再人工检查 0–2 秒和 4–5 秒关键帧。若 DashScope 失败，确认视频仍正常输出且日志明确说明贴纸层被跳过。

- [ ] **Step 5: 提交文档和最终回归结果**

```bash
git add docs/02-use/02-怎么用API或命令处理视频.md \
        docs/02-use/04-实战去字幕流程-本机验证版.md
git commit -m "docs: 更新自动全屏检测使用说明"
```

## 计划自检

- 设计中的默认每 5 帧 OCR、不确定区间逐帧补查：Task 1–2。
- 无 `-c` 全屏、手动 ROI 兼容：Task 2、Task 6。
- DashScope 全屏采样、早期加密、32 次上限、单帧邻近例外：Task 3。
- 白字幕核心/边缘双层遮罩、彩色/贴纸矩形：Task 4。
- ProPainter 局部残留复核、显存失败降级：Task 5。
- 日志、测试、输出帧数/音频/显存验收：Task 2、Task 5、Task 6。
- 计划中没有占位标记或未定义的文件/函数名称；每个修改步骤均给出目标文件、测试命令和预期结果。
