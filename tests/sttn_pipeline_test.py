"""vsr_pipeline 的 STTN 并列修复模式。

STTN 是带级时序模型：整条修复带被压到 432x240 再放大回来，笔画级精度会在
横向 1.67 倍压缩中丢失。因此本模式走整框矩形遮罩，不复用 ProPainter 的
字形级遮罩，且所有字形相关开关必须显式报错而非静默降级——静默忽略会让
调用方以为字形保护生效。

STTN 的 __call__ 接收与帧序列等长的逐帧遮罩；修复带的几何范围可覆盖
当前段的全部字幕位置，但不能把该范围当作每帧的遮罩。
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
    """把 STTN 引擎替换成记录器，返回 (帧数, 逐帧 mask) 调用列表。"""
    calls = []

    def engine(frames, mask):
        if isinstance(mask, np.ndarray):
            saved_mask = mask.copy()
        else:
            saved_mask = [m.copy() for m in mask]
        calls.append((len(frames), saved_mask))
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
    _, masks = calls[0]
    mask = masks[0]
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
    assert max(mask.max() for mask in calls[0][1]) == 255


def test_sttn_receives_per_frame_masks_without_segment_union(tmp_path):
    """每帧只接收自己的矩形 mask，不能把段内框并集传播给其他帧。"""
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    # 帧值需逐帧变化才能让检测在两个框之间交替；相邻差 1 远低于场景切换阈值
    make_video(source, list(range(10)))
    moving = [(20, 30, 10, 30), (20, 30, 35, 55)]
    pipe = make_pipe(lambda img, region: [moving[int(img[0, 0, 0]) % 2]])
    calls = record_calls(pipe)
    pipe.process_video(source, output, region=REGION, locate_stickers=False)

    _, masks = calls[0]
    assert len(masks) == 10
    for i, (ymin, ymax, xmin, xmax) in enumerate(moving * 5):
        mask = masks[i]
        assert mask[ymin:ymax, xmin:xmax].min() == 255
        other = moving[(i + 1) % 2]
        oy1, oy2, ox1, ox2 = other
        assert mask[oy1:oy2, ox1:ox2].max() == 0


def test_segments_respect_max_load_and_never_exceed_it(tmp_path):
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * (STTN_SEG_LEN * 2 + 7))
    pipe = make_pipe()
    calls = record_calls(pipe)
    pipe.process_video(source, output, region=REGION, locate_stickers=False)

    sizes = [n for n, _ in calls]
    assert sizes and max(sizes) <= STTN_SEG_LEN
    assert sum(sizes) == STTN_SEG_LEN * 2 + 7


def test_segment_flush_releases_cuda_cache(tmp_path, monkeypatch):
    """每段处理完必须清理 CUDA 缓存池，否则 reserved 单调爬升不释放。

    实测：687 帧(~23秒)视频跑完后 reserved 涨到 11+GiB，而 allocated
    全程稳定在 0.07GiB——PyTorch 的缓存池从未归还给驱动。这不是内存
    泄漏(allocated 没有增长)，但长视频/高并发场景下会耗尽 free 显存
    触发 OOM。backend/inpaint/sttn_auto_inpaint.py 的既有实现和
    _release_sticker_detector 都在这个位置做 gc.collect+empty_cache，
    ProPainter 分支和本模式改动前的 sttn 分支都遗漏了这一步。
    """
    import vsr_pipeline

    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * (STTN_SEG_LEN + 5))
    pipe = make_pipe()
    record_calls(pipe)

    # cuda_memory_snapshot 在 is_available()=True 时还会调 reset_peak_memory_stats/
    # memory_allocated/memory_reserved/max_memory_*/mem_get_info；CPU-only 的
    # torch 构建里这些真实调用会触发 _lazy_init 断言，逐一打桩避免误报
    cuda = vsr_pipeline.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(cuda, "mem_get_info", lambda: (0, 1))
    calls = []
    monkeypatch.setattr(cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(vsr_pipeline.gc, "collect", lambda: calls.append("gc.collect"))

    pipe.process_video(source, output, region=REGION, locate_stickers=False)

    # 两段(STTN_SEG_LEN+5 帧按 STTN_SEG_LEN 切分产生 2 段)，每段各清理一次
    assert calls.count("empty_cache") == 2
    assert calls.count("gc.collect") == 2


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


# ---- 残留检测(仅标记,不自动修复) ----
#
# 实测证明 STTN 在已覆盖区域内的生成质量本身不稳定(偶发发白/发糊)，
# 不是遮罩形状问题：真实素材上两种独立的后处理修复尝试都不能稳定改善
# ——cv2.inpaint 空间补反而更白，用相邻干净帧做时间复制在一个案例上改善
# 却在另一个案例上变差，找不出决定成败的变量。因此 sttn 分支只做检测和
# 计数，不尝试自动修复：把有缺陷的段交给使用者复核或换 propainter 重跑，
# 而不是无声地把缺陷结果当成功交付。

def test_residual_probe_checks_every_frame_with_a_box(tmp_path):
    """探测行为本身:逐帧调用探测，用真实原帧/输出帧作证据。

    _residual_mask 判据本身（同极性笔画重合、抗新增白背景误报等）已由
    tests/glyph_residual_test.py 独立覆盖，这里只验证 flush_sttn 是否
    正确接入了探测——每个有检出框的帧都应被探测一次，参数顺序与
    ProPainter 分支一致：(引擎输出, 原帧, 该帧检出框)。
    """
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 6)
    pipe = make_pipe()
    pipe.inpainter = lambda frames, mask: [f.copy() for f in frames]
    pipe._ensure_sttn = lambda: None

    probed = []
    original_probe = Pipeline._residual_mask

    def spy_probe(self, fixed_bgr, original_bgr, boxes):
        probed.append((np.array_equal(fixed_bgr, original_bgr), boxes))
        return original_probe(self, fixed_bgr, original_bgr, boxes)

    pipe._residual_mask = spy_probe.__get__(pipe, Pipeline)
    pipe.process_video(source, output, region=REGION, locate_stickers=False)

    assert len(probed) == 6, "应对每个有检出框的帧都调用探测"
    assert all(is_identity for is_identity, _ in probed), "引擎原样返回时不应报告缺陷"
    assert all(boxes == [BOX] for _, boxes in probed), "探测须用该帧自己的框，不是段级并集"


def test_residual_probe_flags_frame_with_visible_defect(tmp_path):
    """复用 _residual_mask（已验证对发白/发糊两类真实缺陷都敏感）逐帧探测。"""
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 6)
    pipe = make_pipe()

    def engine(frames, mask):
        # 制造 _residual_mask 能识别的结构性缺陷:原帧笔画状高对比条纹在
        # 输出中被铣平——这正是 glyph_residual_test.py 里验证过的判据形态
        out = []
        for f in frames:
            comp = f.copy()
            comp[mask > 0] = 160
            out.append(comp)
        return out

    def stub_residual(self, fixed_bgr, original_bgr, boxes):
        # 只验证计数/累加逻辑，不重复验证 _residual_mask 内部像素判据
        return np.full(fixed_bgr.shape[:2], 255, dtype='uint8')

    pipe.inpainter = engine
    pipe._ensure_sttn = lambda: None
    pipe._residual_mask = stub_residual.__get__(pipe, Pipeline)
    stats = pipe.process_video(source, output, region=REGION, locate_stickers=False)
    assert stats["unresolved"] == 6


def test_residual_probe_does_not_alter_output_pixels(tmp_path):
    """检测只计数，不修改任何输出像素——这是本次范围的核心约束。"""
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 6)
    pipe = make_pipe()

    def engine(frames, mask):
        out = []
        for f in frames:
            comp = f.copy()
            comp[BOX[0]:BOX[1], BOX[2]:BOX[3]] = 250  # 明显异常，理应被探测到但不应被改写
            out.append(comp)
        return out

    def stub_residual(self, fixed_bgr, original_bgr, boxes):
        return np.full(fixed_bgr.shape[:2], 255, dtype='uint8')  # 强制判定为缺陷

    pipe.inpainter = engine
    pipe._ensure_sttn = lambda: None
    pipe._residual_mask = stub_residual.__get__(pipe, Pipeline)
    stats = pipe.process_video(source, output, region=REGION, locate_stickers=False)
    assert stats["unresolved"] == 6  # 确认探测确实判定了缺陷，不是没触发就通过
    with av.open(str(output)) as result:
        frame = next(result.decode(video=0))
        img = np.asarray(frame.to_image())
    ymin, ymax, xmin, xmax = BOX
    # H.264 有损编码会引入个位数噪声，不能断言精确相等
    assert img[ymin:ymax, xmin:xmax].mean() > 235


def test_residual_probe_runs_without_white_glyph_check_flag(tmp_path):
    """探测在 sttn 分支内置生效，不依赖被本分支拒绝的 white_glyph_check 开关。"""
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 6)
    pipe = make_pipe()
    record_calls(pipe)
    stats = pipe.process_video(source, output, region=REGION, locate_stickers=False)
    assert "unresolved" in stats
    assert stats["residual_check_enabled"] is False  # 开关本身仍是字形专属，标记不属于它


def test_residual_probe_skipped_when_no_detection(tmp_path):
    source, output = tmp_path / "in.mp4", tmp_path / "out.mp4"
    make_video(source, [90] * 8)
    pipe = make_pipe(lambda img, region: [])
    pipe._ensure_sttn = lambda: pytest.fail("无字幕视频不应加载 STTN 权重")
    stats = pipe.process_video(source, output, region=REGION, locate_stickers=False)
    assert stats["unresolved"] == 0
