"""批量预处理必须保持 STTN 的颜色、量化边界和输入保护约定。"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint import sttn_det_inpaint
from benchmarks.sttn_preprocess import legacy_preprocess


@pytest.mark.parametrize('frame_count', [1, 7])
@pytest.mark.parametrize('layout', ['contiguous', 'roi', 'reversed'])
def test_all_uint8_values_match_legacy_without_modifying_inputs(frame_count, layout):
    rng = np.random.default_rng(19)
    frames = [rng.integers(0, 256, (16, 32, 3), dtype=np.uint8)
              for _ in range(frame_count)]
    masks = [np.arange(512, dtype=np.uint16).astype(np.uint8).reshape(16, 32)
             for _ in frames]
    # 确保所有 256 种帧值均被验证，防止改变除法顺序带来的舍入差。
    frames[0][:8, :, 0] = np.arange(256, dtype=np.uint8).reshape(8, 32)
    if layout == 'roi':
        frames = [frame[:, 1:-1] for frame in frames]
        masks = [mask[:, 1:-1] for mask in masks]
    elif layout == 'reversed':
        frames = [frame[::-1, ::-1] for frame in frames]
        masks = [mask[::-1, ::-1] for mask in masks]
    originals = [item.copy() for item in frames + masks]
    original_ids = [id(item) for item in frames + masks]
    expected = legacy_preprocess(frames, masks)

    actual = sttn_det_inpaint._preprocess_sttn(frames, masks)

    for tensor, reference in zip(actual[:2], expected[:2]):
        assert tensor.dtype == torch.float32
        assert tensor.device.type == 'cpu'
        assert tensor.is_contiguous()
        torch.testing.assert_close(tensor, reference, rtol=0, atol=0)
    for mask, reference in zip(actual[2], expected[2]):
        np.testing.assert_array_equal(mask, reference)
    assert [id(item) for item in frames + masks] == original_ids
    for item, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(item, original)
    # 输出不能与调用方输入共用可写内存，否则后续处理或重试会污染原帧。
    actual[0].zero_()
    actual[1].zero_()
    for mask in actual[2]:
        mask.fill(0)
    for item, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(item, original)


def test_rgb_channels_and_two_mask_thresholds_are_preserved():
    frame = np.tile(np.array([0, 127, 255], dtype=np.uint8), (1, 6, 1))
    mask = np.array([[0, 1, 127, 128, 254, 255]], dtype=np.uint8)

    feats, model_mask, binary_masks = sttn_det_inpaint._preprocess_sttn([frame], [mask])

    assert feats.shape == (1, 1, 3, 1, 6)
    # BGR [0,127,255] 必须成为 RGB [255,127,0]，并按 /255、*2、-1 归一化。
    torch.testing.assert_close(feats[0, 0, :, 0, 0],
                               torch.tensor([1.0, -0.003921568393707275, -1.0]),
                               rtol=0, atol=0)
    torch.testing.assert_close(model_mask,
                               torch.tensor([[[[[0., 0., 0., 1., 1., 1.]]]]]),
                               rtol=0, atol=0)
    np.testing.assert_array_equal(binary_masks[0],
                                  np.array([[[0], [1], [1], [1], [1], [1]]],
                                           dtype=np.uint8))
