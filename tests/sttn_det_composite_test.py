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


def test_zero_mask_returns_frames_unchanged():
    frame = textured_frame()
    empty = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    out = make_inpainter()([frame], empty)[0]
    assert np.array_equal(out, frame)
