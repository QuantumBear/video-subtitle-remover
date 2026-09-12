"""vsr_pipeline.detect 的检出框高度上限。

MIN_BOX_ASPECT 只排除近方形与竖条框，挡不住"宽且高"的误检块：
600x300 的宽扁块宽高比 2.0，高于 1.8 的下限，能通过宽高比判据。
这类框进入 propainter_boxes_to_mask 后，若框内白色字形不足
（有色物体的常见情形），会走整框矩形擦除分支，把整块画面重绘。

阈值依据样片全片 OCR 实测：正常字幕框高 p99 占帧高 2.8%，
误检块占 13.9% 与 26.3%，取 8% 留约 2.8 倍余量。
相对阈值与绝对下限取大：纯相对值在小分辨率下会收得比真实文字行还紧
（60px 高的帧算出 4.8px），必须由 MAX_BOX_HEIGHT_FLOOR 兜底。
"""

import numpy as np
import pytest

pytest.importorskip("cv2")

from vsr_pipeline import MASK_PAD, MAX_BOX_HEIGHT_FLOOR, MAX_BOX_HEIGHT_PERMILLE, Pipeline


def quad(xmin, ymin, xmax, ymax):
    """构造 OCR 风格的四点检测框（坐标相对 region 裁剪图）。"""
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)


def make_pipe(polys):
    pipe = Pipeline.__new__(Pipeline)
    pipe.ocr = type("FakeOCR", (), {"predict": staticmethod(lambda img: [{"dt_polys": np.array(polys)}])})()
    return pipe


def detect_on(polys, frame_h=1280, frame_w=720):
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    return make_pipe(polys).detect(frame, (0, frame_h, 0, frame_w))


def test_normal_subtitle_line_is_kept():
    # f600 实测字幕框 335x75
    boxes = detect_on([quad(188, 841, 523, 916)])
    assert boxes == [(841 - MASK_PAD, 916 + MASK_PAD, 188 - MASK_PAD, 523 + MASK_PAD)]


def test_wide_flat_block_is_rejected_by_height():
    # 600x300：宽高比 2.0 通过 MIN_BOX_ASPECT，只有高度判据能拦住
    assert detect_on([quad(60, 400, 660, 700)]) == []


def test_square_blob_still_rejected_by_aspect():
    # f279 实测误检块 375x337，宽高比 1.11
    assert detect_on([quad(152, 478, 527, 815)]) == []


def test_height_limit_scales_with_frame_height():
    # 同一个 700x150 的框：2160 高的帧内合规(阈值 172px)，1280 高的帧内超限(阈值 102px)
    tall = quad(10, 400, 710, 550)
    assert detect_on([tall], frame_h=2160) != []
    assert detect_on([tall], frame_h=1280) == []


@pytest.mark.parametrize("frame_h,box_h", [(60, 20), (80, 10), (360, 40)])
def test_small_frames_fall_back_to_absolute_floor(frame_h, box_h):
    # 纯相对阈值在小帧上会收得比真实文字行还紧(60px 帧的 8% 仅 4.8px)，
    # 必须由绝对下限兜底，否则合成小样本与低分辨率视频的字幕全被剔除。
    # 宽度按 box_h 放大到 4:1，确保这里检验的是高度判据而非宽高比判据
    box_w = box_h * 4
    assert box_h <= MAX_BOX_HEIGHT_FLOOR
    assert detect_on([quad(5, 10, 5 + box_w, 10 + box_h)],
                     frame_h=frame_h, frame_w=box_w + 20) != []


def test_oversized_block_dropped_without_losing_sibling_subtitle():
    # 同帧混合：误检块被剔除，真实字幕行保留
    boxes = detect_on([quad(60, 400, 660, 700), quad(188, 841, 523, 916)])
    assert boxes == [(841 - MASK_PAD, 916 + MASK_PAD, 188 - MASK_PAD, 523 + MASK_PAD)]


@pytest.mark.parametrize("height", [11, 29, 36, 75])
def test_observed_real_subtitle_heights_pass(height):
    assert detect_on([quad(100, 800, 500, 800 + height)]) != []


@pytest.mark.parametrize("height", [178, 300, 337])
def test_observed_bogus_heights_rejected(height):
    # 宽度取够大以确保通过宽高比判据，隔离出高度判据的作用
    assert detect_on([quad(20, 300, 700, 300 + height)]) == []


def test_threshold_constant_matches_backend_default():
    # 与 backend/config.py 的 subtitleMaxHeightPermille 保持同一标定
    assert MAX_BOX_HEIGHT_PERMILLE == 80
