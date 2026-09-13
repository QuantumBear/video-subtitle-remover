"""sttn-det 修复带的几何与合成契约。

STTN 模型输入固定 432x240,修复带必须与该宽高比一致才不会各向异性压缩;
合成必须只落在 mask 命中的像素上,否则整条修复带都会带上缩放往返的模糊。
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")

from backend.inpaint.sttn_det_inpaint import STTNDetInpaint

MODEL_W, MODEL_H = 432, 240
FRAME_W, FRAME_H = 720, 1280
# 假修复结果用中性灰,避免 BGR/RGB 通道次序干扰"哪些像素被替换"的判据
COMP_VALUE = 77


def make_inpainter(recorder=None):
    """绕过权重加载,只保留 __call__ 需要的几何参数。"""
    inpainter = STTNDetInpaint.__new__(STTNDetInpaint)
    inpainter.model_input_width = MODEL_W
    inpainter.model_input_height = MODEL_H
    inpainter.neighbor_stride = 5
    inpainter.ref_length = 10

    def fake_inpaint(frames, masks):
        if recorder is not None:
            recorder["masks"] = [np.array(m).copy() for m in masks]
            recorder["frames"] = [np.array(f).copy() for f in frames]
        return [np.full((MODEL_H, MODEL_W, 3), COMP_VALUE, dtype=np.uint8) for _ in frames]

    inpainter.inpaint = fake_inpaint
    return inpainter


def textured_frame():
    return np.random.RandomState(0).randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


def subtitle_mask(ymin=700, ymax=760, xmin=100, xmax=600):
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    mask[ymin:ymax, xmin:xmax] = 255
    return mask


def test_split_h_matches_model_aspect_ratio():
    # 修复带宽高比必须等于模型输入宽高比,否则纵横两轴缩放倍率不同,画面被拉扁
    split_h = STTNDetInpaint.compute_split_h(FRAME_W, FRAME_H, MODEL_W, MODEL_H)
    assert split_h == 400
    assert FRAME_W / split_h == pytest.approx(MODEL_W / MODEL_H)


def test_split_h_does_not_depend_on_frame_orientation():
    # 竖屏不应走另一套公式:带高只由帧宽和模型宽高比决定
    portrait = STTNDetInpaint.compute_split_h(720, 1280, MODEL_W, MODEL_H)
    landscape = STTNDetInpaint.compute_split_h(720, 480, MODEL_W, MODEL_H)
    assert portrait == landscape == 400


def test_split_h_clamped_to_frame_height():
    # 超宽画幅下等比带高会超出画面,必须夹到帧高而不是越界
    assert STTNDetInpaint.compute_split_h(2560, 1080, MODEL_W, MODEL_H) == 1080


def test_pixels_outside_mask_are_untouched():
    # 只有 mask 命中的像素才允许被 432x240 往返的低分辨率结果替换,
    # 否则整条修复带(实测 906 行)清晰度掉到原片的 13%~23%
    frame = textured_frame()
    mask = subtitle_mask()
    out = make_inpainter()([frame], mask)[0]

    untouched = mask == 0
    assert np.array_equal(out[untouched], frame[untouched])


def test_pixels_inside_mask_take_the_inpainted_result():
    frame = textured_frame()
    mask = subtitle_mask()
    out = make_inpainter()([frame], mask)[0]

    assert np.all(out[mask > 0] == COMP_VALUE)


def test_mask_handed_to_model_keeps_255_scale():
    # ToTorchFormatTensor 会 div(255);mask 若归一化成 0/1,
    # masks_tensor > 0.5 会整体为假,模型认不出空洞,退化成恒等重建
    recorder = {}
    make_inpainter(recorder)([textured_frame()], subtitle_mask())
    assert recorder["masks"], "inpaint 未被调用"
    assert recorder["masks"][0].max() == 255


def test_per_frame_masks_are_forwarded_independently():
    """同一批次内不同帧的字幕位置不能被并集成一张模型 mask。"""
    recorder = {}
    frames = [textured_frame(), textured_frame()]
    left = subtitle_mask(xmin=80, xmax=260)
    right = subtitle_mask(xmin=360, xmax=540)
    out = make_inpainter(recorder)(frames, [left, right])

    handed = recorder["masks"]
    assert len(handed) == 2
    assert not np.array_equal(handed[0], handed[1])
    assert np.count_nonzero((handed[0] > 0) & (handed[1] > 0)) == 0
    assert np.array_equal(out[0][left == 0], frames[0][left == 0])
    assert np.array_equal(out[1][right == 0], frames[1][right == 0])


def test_zero_mask_returns_frames_unchanged():
    frame = textured_frame()
    empty = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    out = make_inpainter()([frame], empty)[0]
    assert np.array_equal(out, frame)


def test_composite_mask_limits_final_writeback_to_narrow_region():
    frame = textured_frame()
    model_mask = subtitle_mask(xmin=100, xmax=600)
    composite_mask = subtitle_mask(xmin=260, xmax=440)
    recorder = {}
    out = make_inpainter(recorder)([frame], [model_mask],
                                   composite_mask=[composite_mask])[0]

    assert recorder['masks'][0].max() == 255
    assert np.count_nonzero(recorder['masks'][0]) > 0
    assert np.all(out[composite_mask > 0] == COMP_VALUE)
    assert np.array_equal(out[(model_mask > 0) & (composite_mask == 0)],
                          frame[(model_mask > 0) & (composite_mask == 0)])
    assert np.array_equal(out[model_mask == 0], frame[model_mask == 0])


def test_composite_mask_is_intersected_with_model_mask():
    frame = textured_frame()
    model_mask = subtitle_mask(xmin=200, xmax=400)
    composite_mask = subtitle_mask(xmin=100, xmax=500)
    out = make_inpainter()([frame], [model_mask],
                           composite_mask=[composite_mask])[0]
    assert np.all(out[model_mask > 0] == COMP_VALUE)
    assert np.array_equal(out[(composite_mask > 0) & (model_mask == 0)],
                          frame[(composite_mask > 0) & (model_mask == 0)])


def test_composite_mask_length_and_shape_are_validated():
    frame = textured_frame()
    model_mask = subtitle_mask()
    with pytest.raises(ValueError, match='合成 mask'):
        make_inpainter()([frame], [model_mask], composite_mask=[])
    with pytest.raises(ValueError, match='合成 mask'):
        make_inpainter()([frame], [model_mask],
                         composite_mask=[np.zeros((FRAME_H - 1, FRAME_W), np.uint8)])


@pytest.mark.parametrize('x_bounds', [(72, 360), (0, 288), (432, 720)])
def test_horizontal_roi_increases_mask_resolution_and_preserves_pixels(x_bounds):
    frame = textured_frame()
    x0, x1 = x_bounds
    mask = subtitle_mask(xmin=x0 + 48, xmax=x1 - 48)
    full, cropped = {}, {}
    make_inpainter(full)([frame], mask)
    out = make_inpainter(cropped)([frame], mask, x_bounds=x_bounds)[0]

    # 同一字幕在固定模型输入中占更多像素，同时逐像素保留所有非 mask 区域。
    assert np.count_nonzero(cropped['masks'][0]) > np.count_nonzero(full['masks'][0])
    assert out.shape == frame.shape
    assert np.all(out[mask > 0] == COMP_VALUE)
    assert np.array_equal(out[mask == 0], frame[mask == 0])


def test_horizontal_roi_uses_same_crop_for_clean_context_and_core(monkeypatch):
    from backend.inpaint import sttn_det_inpaint

    calls = []
    original = sttn_det_inpaint.get_inpaint_area_by_mask

    def record_areas(width, height, split_h, mask):
        areas = original(width, height, split_h, mask)
        calls.append((width, height, split_h, areas))
        return areas

    monkeypatch.setattr(sttn_det_inpaint, 'get_inpaint_area_by_mask', record_areas)
    frames = [textured_frame() for _ in range(3)]
    masks = [np.zeros((FRAME_H, FRAME_W), dtype=np.uint8),
             subtitle_mask(xmin=280, xmax=400),
             subtitle_mask(xmin=320, xmax=460)]
    recorder = {}
    out = make_inpainter(recorder)(frames, masks, x_bounds=(240, 528))

    width, height, split_h, areas = calls[-1]
    assert (width, height, split_h) == (288, FRAME_H, 160)
    y0, y1 = areas[0][:2]
    expected = cv2.resize(frames[0][y0:y1, 240:528], (MODEL_W, MODEL_H))
    assert np.array_equal(recorder['frames'][0], expected)
    assert not recorder['masks'][0].any()
    assert not np.array_equal(recorder['masks'][1], recorder['masks'][2])
    for i, mask in enumerate(masks):
        assert np.array_equal(out[i][mask == 0], frames[i][mask == 0])
        assert np.all(out[i][mask > 0] == COMP_VALUE)


def test_narrow_roi_never_clips_top_or_bottom_of_tall_mask():
    frame = textured_frame()
    mask = subtitle_mask(ymin=600, ymax=900, xmin=250, xmax=350)
    out = make_inpainter()([frame], mask, x_bounds=(202, 398))[0]
    assert np.all(out[mask > 0] == COMP_VALUE)
    assert np.array_equal(out[mask == 0], frame[mask == 0])


def test_multiple_vertical_regions_are_all_repaired_in_horizontal_roi():
    frame = textured_frame()
    mask = subtitle_mask(ymin=20, ymax=80, xmin=250, xmax=350)
    mask |= subtitle_mask(ymin=1200, ymax=1260, xmin=300, xmax=400)
    out = make_inpainter()([frame], mask, x_bounds=(202, 448))[0]
    assert np.all(out[mask > 0] == COMP_VALUE)
    assert np.array_equal(out[mask == 0], frame[mask == 0])


def test_full_width_roi_matches_legacy_call():
    frame, mask = textured_frame(), subtitle_mask()
    engine = make_inpainter()
    assert np.array_equal(engine([frame], mask)[0],
                          engine([frame], mask, x_bounds=(0, FRAME_W))[0])


@pytest.mark.parametrize('bounds', [(300, 200), (FRAME_W, FRAME_W + 20), (200, 250)])
def test_horizontal_roi_rejects_invalid_or_mask_clipping_bounds(bounds):
    with pytest.raises(ValueError, match='ROI'):
        make_inpainter()([textured_frame()], subtitle_mask(), x_bounds=bounds)
