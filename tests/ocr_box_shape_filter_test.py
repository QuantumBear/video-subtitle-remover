"""OCR 检测框的形状合法性过滤。

字幕框是扁长的文字行。当 OCR 把画面内容（车辆、宠物、反光等）误检成文本时，
框会明显超出文字行的高度。这类框一旦进入 mask，STTN 会把整块区域换成
低分辨率生成结果，形成大面积块状模糊。

阈值依据本仓库 TikSave 样片全片 OCR 实测（196 个采样帧、613 个框）：
框高 p50=29px、p90=33px、p99=36px（占帧高 2.8%），
随后直接跳到两个异常框 178px(13.9%) 与 337px(26.3%)，中间没有分布。
按帧高 8% 取阈值，对正常框留有约 2.8 倍余量。
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from backend.tools.subtitle_detect import SubtitleDetect

FRAME_H, FRAME_W = 1280, 720
# 全片实测：正常字幕框高 11..36px，误检块 178px 与 337px
NORMAL_BOX = (188, 523, 841, 916)      # xmin, xmax, ymin, ymax，高 75px
SQUARE_BLOB = (152, 527, 478, 815)     # f279 实测误检块，375x337，宽高比 1.11
TALL_SLIVER = (421, 445, 523, 701)     # f645 实测误检块，24x178


def test_normal_subtitle_box_is_kept():
    assert SubtitleDetect.is_plausible_text_box(*NORMAL_BOX, FRAME_H)


def test_square_blob_is_rejected():
    # f279：375x337 近方形。宽高比 1.11 落在正常字幕范围(0.67~8.77)内，
    # 只有高度判据能识别它
    assert not SubtitleDetect.is_plausible_text_box(*SQUARE_BLOB, FRAME_H)


def test_tall_sliver_is_rejected():
    assert not SubtitleDetect.is_plausible_text_box(*TALL_SLIVER, FRAME_H)


@pytest.mark.parametrize("height", [11, 25, 29, 33, 36, 40])
def test_observed_real_subtitle_heights_all_pass(height):
    # 全片实测的真实字幕框高区间，含 p99=36 并留一档余量
    assert SubtitleDetect.is_plausible_text_box(100, 400, 800, 800 + height, FRAME_H)


@pytest.mark.parametrize("height", [178, 250, 337])
def test_observed_bogus_heights_all_rejected(height):
    assert not SubtitleDetect.is_plausible_text_box(100, 400, 400, 400 + height, FRAME_H)


def test_threshold_scales_with_frame_height():
    # 同样的绝对高度，在小尺寸帧里应被判为超高
    box = (100, 400, 100, 190)  # 高 90px
    assert SubtitleDetect.is_plausible_text_box(*box, 1280)
    assert not SubtitleDetect.is_plausible_text_box(*box, 480)


def test_degenerate_box_is_rejected():
    assert not SubtitleDetect.is_plausible_text_box(100, 100, 800, 830, FRAME_H)
    assert not SubtitleDetect.is_plausible_text_box(100, 400, 800, 800, FRAME_H)


def test_detect_subtitle_drops_oversized_box(monkeypatch):
    """过滤必须落在 detect_subtitle 内，使所有修复模式共享同一道防线。"""
    detector = SubtitleDetect.__new__(SubtitleDetect)
    detector.sub_areas = []

    def fake_predict(img):
        def poly(xmin, xmax, ymin, ymax):
            return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

        return [{"dt_polys": np.array([poly(*NORMAL_BOX), poly(*SQUARE_BLOB)], dtype=np.float32)}]

    monkeypatch.setattr(
        SubtitleDetect, "text_detector", property(lambda self: type("D", (), {"predict": staticmethod(fake_predict)})())
    )

    boxes = detector.detect_subtitle(np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8))
    assert boxes == [NORMAL_BOX]
