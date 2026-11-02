"""生产 Q/K/V 合并须兼容原权重、训练、设备迁移和完整修复带。"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint import sttn_det_inpaint
from backend.inpaint.sttn.network_sttn import InpaintGenerator, MultiHeadedAttention
from benchmarks import sttn_qkv_fusion as benchmark


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def convolution_calls(profile):
    return sum(event.count for event in profile.key_averages() if event.key == 'aten::conv2d')


def test_prepared_projection_matches_hand_derived_qkv_with_one_convolution():
    attention = MultiHeadedAttention([(1, 1)], 4).eval()
    with torch.no_grad():
        for layer, scale, bias in [(attention.query_embedding, 1, 1),
                                   (attention.key_embedding, 2, 3),
                                   (attention.value_embedding, -1, 5)]:
            layer.weight.zero_()
            layer.weight[:, :, 0, 0].copy_(torch.eye(4) * scale)
            layer.bias.fill_(bias)
    keys = set(attention.state_dict())
    attention.prepare_fused_qkv()
    assert set(attention.state_dict()) == keys
    # .to() 必须同时迁移推理缓存，且重复使用不能再次拼接。
    attention.double()
    features = torch.arange(32, dtype=torch.float64).reshape(2, 4, 2, 2)
    for _ in range(2):
        with torch.no_grad(), torch.profiler.profile() as profile:
            query, key, value = attention.project_qkv(features)
        assert convolution_calls(profile) == 1
        assert not any(event.key == 'aten::cat' for event in profile.key_averages())
        torch.testing.assert_close(query, features + 1, rtol=0, atol=0)
        torch.testing.assert_close(key, features * 2 + 3, rtol=0, atol=0)
        torch.testing.assert_close(value, -features + 5, rtol=0, atol=0)


def test_loading_original_checkpoint_invalidates_packed_weights():
    attention = MultiHeadedAttention([(1, 1)], 4).eval()
    attention.prepare_fused_qkv()
    state = {key: value.clone() for key, value in attention.state_dict().items()}
    state['query_embedding.weight'].zero_()
    state['query_embedding.bias'].fill_(7)
    attention.load_state_dict(state, strict=True)
    with torch.no_grad():
        query, _, _ = attention.project_qkv(torch.randn(2, 4, 2, 2))
    torch.testing.assert_close(query, torch.full_like(query, 7), rtol=0, atol=0)
    attention.prepare_fused_qkv()
    with torch.no_grad():
        query, _, _ = attention.project_qkv(torch.randn(2, 4, 2, 2))
    torch.testing.assert_close(query, torch.full_like(query, 7), rtol=0, atol=0)


def test_training_updates_original_weights_and_repreparation_uses_new_values():
    attention = MultiHeadedAttention([(1, 1)], 4).eval()
    attention.prepare_fused_qkv()
    attention.train()
    optimizer = torch.optim.SGD(attention.parameters(), lr=0.1)
    features = torch.ones(2, 4, 2, 2)
    outputs = attention.project_qkv(features)
    sum(value.sum() for value in outputs).backward()
    for layer in (attention.query_embedding, attention.key_embedding, attention.value_embedding):
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad) == layer.weight.numel()
    optimizer.step()
    attention.eval()
    with torch.no_grad():
        expected = attention.project_qkv(features)
        attention.prepare_fused_qkv()
        actual = attention.project_qkv(features)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_grad_enabled_eval_uses_original_parameters():
    attention = MultiHeadedAttention([(1, 1)], 4).eval()
    attention.prepare_fused_qkv()
    query, key, value = attention.project_qkv(torch.ones(2, 4, 2, 2))
    (query.sum() + key.sum() + value.sum()).backward()
    assert all(layer.weight.grad is not None for layer in
               (attention.query_embedding, attention.key_embedding, attention.value_embedding))


def test_explicit_separate_engine_does_not_retain_packed_weights(monkeypatch):
    model = InpaintGenerator(init_weights=False)
    monkeypatch.setattr(sttn_det_inpaint, 'InpaintGenerator', lambda: model)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'netG': model.state_dict()})
    engine = sttn_det_inpaint.STTNDetInpaint('cpu', 'legacy.pth', precision='fp32', fuse_qkv=False)
    with torch.no_grad(), torch.profiler.profile() as profile:
        engine.model.project_first_qkv(torch.ones(2, 256, 2, 2))
    assert convolution_calls(profile) == 3
    assert all(value is None for name, value in engine.model.named_buffers()
               if name.endswith(('_qkv_weight', '_qkv_bias')))


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
def test_default_engine_loads_legacy_checkpoint_and_uses_fused_projection(monkeypatch, cached, device):
    torch.manual_seed(19)
    model = InpaintGenerator(init_weights=False).eval()
    for block in model.transformer:
        block.attention.patchsize = [(8, 8), (4, 4), (2, 2), (1, 1)]
    state = {key: value.clone() for key, value in model.state_dict().items()}
    monkeypatch.setattr(sttn_det_inpaint, 'InpaintGenerator', lambda: model)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {'netG': state})
    engine = sttn_det_inpaint.STTNDetInpaint(device, 'legacy.pth',
                                           precision='fp16' if device == 'cuda' else 'fp32',
                                           cache_first_qkv=cached)
    engine.model_input_width = engine.model_input_height = 32
    engine.neighbor_stride, engine.ref_length = 2, 3
    assert set(engine.model.state_dict()) == set(state)
    rng = np.random.default_rng(19)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    for mask in masks:
        mask[8:24, 8:24] = 255
    calls = []
    # 实际算子计数验证默认生产路径，无需进入实验上下文。
    features = torch.randn(3, 256, 8, 8, device=device)
    with torch.no_grad(), engine._autocast_context(engine.precision), torch.profiler.profile() as profile:
        engine.model.project_first_qkv(features)
    assert convolution_calls(profile) == 1
    original = engine.model.transformer[0].attention.project_qkv

    def record(features):
        calls.append(features.shape[0])
        return original(features)

    engine.model.transformer[0].attention.project_qkv = record
    actual = engine.inpaint(frames, masks)
    assert calls == ([5] if cached else [4, 5, 4])
    # separate 基准必须强制三卷积，不能因生产默认值改变而测成 fused/fused。
    with benchmark.qkv_execution(engine.model, mode='separate'):
        with torch.no_grad(), engine._autocast_context(engine.precision), torch.profiler.profile() as profile:
            engine.model.project_first_qkv(features)
        assert convolution_calls(profile) == 3
        expected = engine.inpaint(frames, masks)
    for result, reference, frame, mask in zip(actual, expected, frames, masks):
        np.testing.assert_array_equal(result, reference)
        np.testing.assert_array_equal(result[mask == 0], frame[mask == 0])
    assert engine.precision_fallbacks == 0
