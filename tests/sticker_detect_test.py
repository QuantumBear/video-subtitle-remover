# -*- coding: utf-8 -*-
"""本地 GroundingDINO 贴纸检测的判据与后端分流测试。

判据数值取自实测标定：emoji 框 693–898 px，物体误检框 9.2 万–39.6 万 px。
测试锁定"面积上限把两者分开"这一关键行为，以及两个后端在流水线里
可互换、互不越界。
"""
import pytest

from backend import sticker_detect


EMOJI = {'score': 0.48, 'box': [343.3, 50.8, 371.1, 77.8]}      # 实测 ~751 px
BIG_OBJECT = {'score': 0.49, 'box': [10.0, 10.0, 500.0, 480.0]}  # 实测量级的整幅误检


def test_area_limit_rejects_high_score_whole_frame_detections():
    """核心判据：物体误检分数比 emoji 还高，只有面积上限能挡住它。"""
    kept = sticker_detect.filter_detections([EMOJI, BIG_OBJECT])
    assert kept == [tuple(EMOJI['box'])]


def test_without_area_limit_object_misdetections_survive():
    """反证：放开面积上限后误检混入，说明该判据不可省。"""
    kept = sticker_detect.filter_detections([EMOJI, BIG_OBJECT], max_area_px=1_000_000)
    assert len(kept) == 2


def test_score_threshold_drops_low_confidence_boxes():
    faint = {'score': 0.21, 'box': [343.0, 50.0, 371.0, 78.0]}
    assert sticker_detect.filter_detections([faint]) == []
    assert sticker_detect.filter_detections([faint], score_threshold=0.2)


def test_default_threshold_keeps_weak_emoji_that_strict_cutoff_would_miss():
    """默认阈值定在 0.25 而非 0.30：同排 emoji 里分数偏低的那个必须留住。

    实测每帧 3 个 emoji，0.30 只能捞回 56%，画面上仍有残留；
    0.25 提到 85%。0.26–0.29 这段分数区间正是差距所在。
    """
    weak = {'score': 0.27, 'box': [313.0, 47.0, 341.0, 77.0]}   # 实测同排 emoji 的弱检出
    assert sticker_detect.filter_detections([weak]) == [(313.0, 47.0, 341.0, 77.0)]
    assert sticker_detect.filter_detections([weak], score_threshold=0.30) == []


def test_degenerate_boxes_are_discarded():
    boxes = [
        {'score': 0.9, 'box': [10.0, 10.0, 10.0, 20.0]},   # 零宽
        {'score': 0.9, 'box': [10.0, 20.0, 30.0, 20.0]},   # 零高
        {'score': 0.9, 'box': [30.0, 30.0, 10.0, 10.0]},   # 反向
    ]
    assert sticker_detect.filter_detections(boxes) == []


def test_area_limit_is_absolute_pixels_not_relative():
    """面积上限用绝对像素而非相对 crop 比例：ROI 尺寸变化时判据不漂移。

    同一个 900px 的框，在 720×560 crop 和 320×240 crop 下都应保留
    （都在 1200px 绝对上限内），因为 emoji 的物理像素尺寸不随裁剪变化。
    若用比例判据，320×240 下限额会缩到 230px，真 emoji 反而被误杀。
    """
    box = {'score': 0.9, 'box': [0.0, 0.0, 30.0, 30.0]}     # 900px，与实测 emoji 同量级
    assert sticker_detect.filter_detections([box]) == [(0.0, 0.0, 30.0, 30.0)]
    # 同一个框，换个"crop 尺寸"概念，判据不变（filter_detections 根本不吃 crop_area）
    assert sticker_detect.filter_detections([box]) == [(0.0, 0.0, 30.0, 30.0)]


def test_to_frame_box_offsets_by_region_without_padding():
    """坐标换算不外扩：外扩由 vsr_pipeline 统一施加，与 VLM 路径同语义。"""
    assert sticker_detect.to_frame_box((10.0, 20.0, 40.0, 60.0),
                                       (450, 1010, 0, 720)) == (470, 510, 10, 40)


def test_to_frame_box_clips_to_region_bounds():
    assert sticker_detect.to_frame_box((-5.0, -5.0, 900.0, 900.0),
                                       (450, 1010, 0, 720)) == (450, 1010, 0, 720)


# ---- 后端分流 ----

def test_unknown_sticker_backend_is_rejected():
    import vsr_pipeline

    pipe = vsr_pipeline.Pipeline.__new__(vsr_pipeline.Pipeline)
    with pytest.raises(ValueError, match='sticker_backend'):
        vsr_pipeline.Pipeline.__init__(pipe, sticker_backend='nope')


def test_gdino_backend_never_calls_dashscope(monkeypatch):
    """走本地后端时不得触碰 API 路径，否则 Key 缺失会静默退化。"""
    import vsr_pipeline

    monkeypatch.setattr(vsr_pipeline, 'locate_stickers_vlm',
                        lambda *a, **kw: pytest.fail('gdino 后端不应调用 VLM'))
    captured = {}

    class FakeDetector:
        def locate(self, video_path, region, sample_frames, **kwargs):
            captured.update(kwargs)
            captured['frames'] = list(sample_frames)
            return {}

    hits = vsr_pipeline.locate_stickers_gdino(
        'x.mp4', (0, 48, 0, 64), [0, 5], FakeDetector())
    assert hits == {}
    assert captured['frames'] == [0, 5]
    # 未显式传参时应落到实测标定的默认判据
    assert captured['score_threshold'] == sticker_detect.DEFAULT_SCORE_THRESHOLD
    assert captured['max_area_px'] == sticker_detect.DEFAULT_MAX_AREA_PX
    assert captured['prompt'] == sticker_detect.DEFAULT_PROMPT


def test_gdino_backend_forwards_explicit_overrides():
    import vsr_pipeline

    captured = {}

    class FakeDetector:
        def locate(self, video_path, region, sample_frames, **kwargs):
            captured.update(kwargs)
            return {}

    vsr_pipeline.locate_stickers_gdino(
        'x.mp4', (0, 48, 0, 64), [0], FakeDetector(),
        prompt='emoji.', score_threshold=0.5, max_area_px=500)
    assert captured == {'prompt': 'emoji.', 'score_threshold': 0.5,
                        'max_area_px': 500}


def test_locate_returns_empty_dict_for_empty_schedule():
    detector = sticker_detect.GroundingDinoStickerDetector.__new__(
        sticker_detect.GroundingDinoStickerDetector)
    assert detector.locate('missing.mp4', (0, 48, 0, 64), []) == {}
