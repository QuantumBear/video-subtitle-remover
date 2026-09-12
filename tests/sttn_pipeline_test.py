"""vsr_pipeline 的 STTN 并列修复模式。

STTN 是带级时序模型：整条修复带被压到 432x240 再放大回来，笔画级精度会在
横向 1.67 倍压缩中丢失。因此本模式走整框矩形遮罩，不复用 ProPainter 的
字形级遮罩，且所有字形相关开关必须显式报错而非静默降级——静默忽略会让
调用方以为字形保护生效。

STTN 的 __call__ 只接单张遮罩（不是逐帧列表），所以每段送入的是段内所有
检出框的并集。
"""

from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("cv2")

from vsr_pipeline import STTN_SEG_LEN, Pipeline

REGION = (10, 40, 5, 60)
BOX = (20, 30, 10, 50)


def make_video(path, values, rate=30):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "rgb24"
        stream.options = {"crf": "0", "bf": "0"}
        for i, value in enumerate(values):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), value, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_pipe(boxes_for=lambda img, region: [BOX]):
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "sttn"
    pipe.detect = boxes_for
    return pipe


def record_calls(pipe):
    """把 STTN 引擎替换成记录器，返回 (帧数, mask) 调用列表。"""
    calls = []

    def engine(frames, mask):
        calls.append((len(frames), mask.copy()))
        return [f.copy() for f in frames]

    pipe.inpainter = engine
    pipe._ensure_sttn = lambda: None
    return calls


# ---- 字形相关开关必须显式报错 ----

@pytest.mark.parametrize("option", ["white_glyph_check", "template_refine", "temporal_glyphs"])
def test_glyph_options_rejected_in_sttn_mode(tmp_path, option):
    # 三者都依赖字形级遮罩，报错须指向 propainter。temporal_glyphs 由既有的
    # 通用守卫拦下，另两个由 sttn 专属守卫拦下，对调用方是同一种拒绝
    pipe = make_pipe()
    with pytest.raises(ValueError, match="propainter"):
        pipe.process_video(tmp_path / "missing.mp4", tmp_path / "out.mp4", **{option: True})


def test_non_default_subtitle_strength_rejected_in_sttn_mode(tmp_path):
    pipe = make_pipe()
    with pytest.raises(ValueError, match="propainter"):
        pipe.process_video(tmp_path / "missing.mp4", tmp_path / "out.mp4",
                           subtitle_strength="conservative")


def test_glyph_options_still_allowed_for_propainter(tmp_path):
    """报错只针对 STTN，不能连带收紧 ProPainter 的既有能力。"""
    pipe = make_pipe()
    pipe.inpaint_mode = "propainter"
    # 走到打开视频才失败，说明开关校验已放行
    with pytest.raises(Exception) as exc:
        pipe.process_video(tmp_path / "missing.mp4", tmp_path / "out.mp4",
                           white_glyph_check=True, subtitle_strength="conservative")
    assert "sttn" not in str(exc.value)


# ---- 遮罩粒度与批处理 ----

def test_sttn_receives_box_level_rect_mask(tmp_path):
    """整框矩形：框内全部置 255，不做字形抠取。"""
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 12)
    pipe = make_pipe()
    calls = record_calls(pipe)
    pipe.process_video(source, output, region=REGION, locate_stickers=False)

    assert calls, "STTN 引擎未被调用"
    _, mask = calls[0]
    ymin, ymax, xmin, xmax = BOX
    assert mask[ymin:ymax, xmin:xmax].min() == 255
    outside = mask.copy()
    outside[ymin:ymax, xmin:xmax] = 0
    assert outside.max() == 0


def test_mask_handed_to_sttn_keeps_255_scale(tmp_path):
    # STTNDetInpaint 内部经 div(255) 归一后判 > 0.5，遮罩必须是 0/255
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 8)
    pipe = make_pipe()
    calls = record_calls(pipe)
    pipe.process_video(source, output, region=REGION, locate_stickers=False)
    assert calls[0][1].max() == 255


def test_segment_mask_is_union_of_member_boxes(tmp_path):
    """单张遮罩服务整段，必须覆盖段内每一帧的框。"""
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    # 帧值需逐帧变化才能让检测在两个框之间交替；相邻差 1 远低于场景切换阈值
    make_video(source, list(range(10)))
    moving = [(20, 30, 10, 30), (20, 30, 35, 55)]
    pipe = make_pipe(lambda img, region: [moving[int(img[0, 0, 0]) % 2]])
    calls = record_calls(pipe)
    pipe.process_video(source, output, region=REGION, locate_stickers=False)

    union = np.zeros_like(calls[0][1])
    for _, mask in calls:
        union = np.maximum(union, mask)
    for ymin, ymax, xmin, xmax in moving:
        assert union[ymin:ymax, xmin:xmax].min() == 255


def test_segments_respect_max_load_and_never_exceed_it(tmp_path):
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * (STTN_SEG_LEN * 2 + 7))
    pipe = make_pipe()
    calls = record_calls(pipe)
    pipe.process_video(source, output, region=REGION, locate_stickers=False)

    sizes = [n for n, _ in calls]
    assert sizes and max(sizes) <= STTN_SEG_LEN
    assert sum(sizes) == STTN_SEG_LEN * 2 + 7


def test_segments_do_not_cross_scene_changes(tmp_path):
    source, output = tmp_path / "scenes.mp4", tmp_path / "out.mp4"
    make_video(source, [0] * 23 + [120] * 20)
    pipe = make_pipe()
    seen = []

    def engine(frames, mask):
        values = {int(f[0, 0, 0]) for f in frames}
        assert len(values) == 1, "批次跨越了场景边界"
        seen.append(len(frames))
        return [f.copy() for f in frames]

    pipe.inpainter = engine
    pipe._ensure_sttn = lambda: None
    stats = pipe.process_video(source, output, locate_stickers=False)
    assert seen == [23, 20] and stats["frames"] == 43


# ---- 输出完整性 ----

def test_frame_count_and_pts_preserved(tmp_path):
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 83)
    pipe = make_pipe()
    record_calls(pipe)
    stats = pipe.process_video(source, output, region=REGION, locate_stickers=False)

    assert stats["frames"] == 83
    with av.open(str(output)) as result:
        frames = list(result.decode(video=0))
        assert len(frames) == 83
        assert [round(float(f.pts * f.time_base) * 30) for f in frames] == list(range(83))


def test_pixels_outside_roi_are_untouched(tmp_path):
    """引擎返回全白也不能污染 ROI 之外，与 ProPainter 分支同一约定。"""
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 6)
    pipe = make_pipe()
    pipe.inpainter = lambda frames, mask: [np.full_like(f, 255) for f in frames]
    pipe._ensure_sttn = lambda: None
    pipe.process_video(source, output, region=REGION, locate_stickers=False)

    ry1, ry2, rx1, rx2 = REGION
    with av.open(str(output)) as result:
        img = np.asarray(next(result.decode(video=0)).to_image())
    assert img[:ry1].max() < 200 and img[ry2:].max() < 200


def test_no_detection_skips_model_load(tmp_path):
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 15)
    pipe = make_pipe(lambda img, region: [])
    pipe._ensure_sttn = lambda: pytest.fail("无字幕视频不应加载 STTN 权重")
    stats = pipe.process_video(source, output, region=REGION, locate_stickers=False)
    assert stats["frames"] == 15 and stats["inpainted"] == 0


def test_stats_report_sttn_mode(tmp_path):
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 12)
    pipe = make_pipe()
    record_calls(pipe)
    stats = pipe.process_video(source, output, region=REGION, locate_stickers=False)
    # 字形相关统计在本模式下无意义，必须标记为未启用而不是伪造 0
    assert stats["inpainted"] == 12
    assert stats["residual_check_enabled"] is False
    assert stats["temporal_glyphs_enabled"] is False
