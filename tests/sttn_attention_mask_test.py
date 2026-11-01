"""删除未使用的注意力 mask 必须保持特征、完整修复结果与权重兼容。"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint.sttn.network_sttn import InpaintGenerator, MultiHeadedAttention
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from benchmarks.sttn_attention_mask import legacy_attention_forward, use_attention


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model():
    torch.manual_seed(43)
    model = InpaintGenerator(init_weights=False).eval()
    for block in model.transformer:
        block.attention.patchsize = [(8, 8), (4, 4), (2, 2), (1, 1)]
    return model


@pytest.mark.parametrize('cached', [False, True])
def test_inference_no_longer_allocates_quadratic_attention_masks(cached):
    model = small_model()
    features = torch.randn(3, 256, 8, 8)
    masks = torch.rand(3, 1, 32, 32)
    with torch.inference_mode():
        qkv = model.project_first_qkv(features) if cached else None
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
            actual = model.infer(features, masks, qkv_cache=qkv)
    assert torch.isfinite(actual).all()
    # 旧路径每层四路构造 T*N × T*N 的 bool repeat，本测试约束这项分配消失。
    assert not any(event.key == 'aten::repeat' for event in trace.key_averages())


@pytest.mark.parametrize('batch,frames,cached', [(1, 1, False), (1, 3, True), (2, 3, False)])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
def test_attention_features_match_legacy_for_different_masks(batch, frames, cached, device):
    model = small_model().transformer[0].attention.to(device)
    features = torch.randn(batch * frames, 256, 8, 8, device=device)
    with torch.inference_mode():
        qkv = model.project_qkv(features) if cached else None
        reference = legacy_attention_forward(model, features,
                                             torch.rand(batch * frames, 1, 8, 8, device=device),
                                             batch, 256, qkv)
        for mask in (torch.zeros(batch * frames, 1, 8, 8, device=device),
                     torch.ones(batch * frames, 1, 8, 8, device=device)):
            actual = model(features, mask, batch, 256, qkv)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize('frames,cached', [(1, False), (7, False), (7, True)])
def test_complete_band_matches_legacy_and_preserves_repair_masks(frames, cached):
    engine = STTNDetInpaint.__new__(STTNDetInpaint)
    engine.device = torch.device('cpu')
    engine.precision = engine.requested_precision = 'fp32'
    engine.precision_fallbacks = 0
    engine.model = small_model()
    engine.model_input_width = engine.model_input_height = 32
    engine.neighbor_stride, engine.ref_length = 2, 3
    engine.cache_first_qkv = cached
    rng = np.random.default_rng(73)
    images = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(frames)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in images]
    masks[0][:] = 255
    for index in range(2, frames):
        masks[index][8:24, index:16 + index] = 255
    originals = [item.copy() for item in images + masks]
    with use_attention(legacy_attention_forward):
        expected = engine.inpaint(images, masks)
    actual = engine.inpaint(images, masks)
    for image, result, reference, mask in zip(images, actual, expected, masks):
        np.testing.assert_array_equal(result, reference)
        np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
    assert np.any(actual[0] != images[0]), '修复 mask 仍须让模型处理空洞'
    for item, original in zip(images + masks, originals):
        np.testing.assert_array_equal(item, original)


def test_benchmark_restores_attention_after_exception():
    previous = MultiHeadedAttention.forward
    with pytest.raises(RuntimeError, match='injected'):
        with use_attention(legacy_attention_forward):
            raise RuntimeError('injected')
    assert MultiHeadedAttention.forward is previous
