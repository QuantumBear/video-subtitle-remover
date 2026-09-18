"""STTN 精度选择、完整修复带重试与数值边界。"""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from backend.inpaint import sttn_det_inpaint
from backend.inpaint.sttn.network_sttn import Attention, InpaintGenerator
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def engine_factory(monkeypatch):
    """真实八层网络，仅缩小图像/分块；用随机权重避免文件依赖。"""
    torch.manual_seed(31)
    model = InpaintGenerator(init_weights=False).eval()
    for block in model.transformer:
        block.attention.patchsize = [(8, 8), (4, 4), (2, 2), (1, 1)]
    monkeypatch.setattr(sttn_det_inpaint, 'InpaintGenerator', lambda: model)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {'netG': model.state_dict()})

    def make(**kwargs):
        engine = STTNDetInpaint(device='cpu', model_path='unused.pth', **kwargs)
        engine.model_input_width = engine.model_input_height = 32
        engine.neighbor_stride, engine.ref_length = 2, 3
        return engine

    return make


@pytest.fixture
def inputs():
    rng = np.random.default_rng(73)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(7)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    for mask in masks:
        mask[8:24, 8:24] = 255
    return frames, masks


def test_default_is_fp32_and_cpu_fp16_falls_back(engine_factory, inputs, capsys):
    engine = engine_factory(profile=True)
    assert engine.requested_precision == engine.precision == 'fp32'
    expected = engine.inpaint(*inputs)
    engine = engine_factory(precision='fp16', profile=True)
    assert engine.requested_precision == 'fp16'
    assert engine.precision == 'fp32'
    assert engine.precision_fallbacks == 1
    actual = engine.inpaint(*inputs)
    for output, reference in zip(actual, expected):
        np.testing.assert_array_equal(output, reference)
    assert all(p.dtype == torch.float32 for p in engine.model.parameters())
    assert engine.last_profile['requested_precision'] == 'fp16'
    assert engine.last_profile['precision'] == 'fp32'
    assert engine.last_profile['precision_fallbacks'] == 1
    log = capsys.readouterr().out
    assert '[sttn-precision-fallback]' in log
    assert 'requested_precision=fp16 precision=fp32 precision_fallbacks=1' in log


def test_invalid_precision_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(sttn_det_inpaint, 'InpaintGenerator',
                        lambda: pytest.fail('must not load model'))
    with pytest.raises(ValueError, match='precision'):
        STTNDetInpaint('cpu', 'unused.pth', precision='bf16')


@pytest.mark.parametrize('masked', [False, True])
def test_attention_supports_half_without_changing_existing_math(masked):
    # 原实现 masked_fill 的返回值被丢弃。本次保持已部署的 attention 语义。
    query = torch.tensor([[[0.1, 0.2], [0.3, -0.4]]])
    mask = torch.full((1, 2, 2), masked)
    expected = Attention()(query, query, query, mask)
    actual = Attention()(query.half(), query.half(), query.half(), mask)
    for output, reference in zip(actual, expected):
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, rtol=1e-3, atol=1e-3)


@pytest.fixture
def amp_engine(engine_factory, monkeypatch):
    """CPU 实网检查重试；仅替换 autocast 上下文，CUDA 算子另行验证。"""
    engine = engine_factory()
    engine.requested_precision = engine.precision = 'fp16'
    scopes = []

    @contextmanager
    def context(precision):
        scopes.append(precision)
        try:
            yield
        finally:
            scopes.pop()

    monkeypatch.setattr(engine, '_autocast_context', context)
    return engine, scopes


@pytest.mark.parametrize('failure', ['kernel', 'not_implemented', 'nan', 'inf'])
def test_failure_restarts_entire_band_and_keeps_future_calls_fp32(
        amp_engine, inputs, capsys, failure):
    engine, scopes = amp_engine
    expected = engine._inpaint_once(*inputs, 'fp32')
    original_frames = [f.copy() for f in inputs[0]]
    cache_calls = []
    decodes = []

    def encoder_hook(module, args):
        cache_calls.append(scopes[-1])

    def decoder_hook(module, args, output):
        decodes.append(scopes[-1])
        # 第二个窗口失败，确保首窗口的半精度结果不会混入 FP32 重试。
        if scopes[-1] == 'fp16' and len(decodes) == 2:
            if failure == 'kernel':
                raise RuntimeError('operator does not support Half')
            if failure == 'not_implemented':
                raise NotImplementedError('Half kernel unavailable')
            return torch.full_like(output, float(failure))
        if scopes[-1] == 'fp16':
            return output + 0.5
        return output

    engine.model.encoder.register_forward_pre_hook(encoder_hook)
    engine.model.decoder.register_forward_hook(decoder_hook)
    for _ in range(2):
        actual = engine.inpaint(*inputs)
        for output, reference in zip(actual, expected):
            np.testing.assert_array_equal(output, reference)
    assert cache_calls == ['fp16', 'fp32', 'fp32']
    assert scopes == []
    assert engine.precision == 'fp32'
    assert engine.precision_fallbacks == 1
    assert capsys.readouterr().out.count('[sttn-precision-fallback]') == 1
    for frame, original in zip(inputs[0], original_frames):
        np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize('error', [torch.cuda.OutOfMemoryError('OOM'),
                                  RuntimeError('CUDA out of memory')])
def test_oom_propagates_without_fp32_retry(amp_engine, inputs, error):
    engine, scopes = amp_engine
    calls = []

    def fail(module, args):
        calls.append(scopes[-1])
        raise error

    engine.model.encoder.register_forward_pre_hook(fail)
    with pytest.raises(type(error)) as exc:
        engine.inpaint(*inputs)
    assert exc.value is error
    assert calls == ['fp16']
    assert engine.precision == 'fp16'
    assert engine.precision_fallbacks == 0
    assert scopes == []


def test_nonfinite_fp32_retry_propagates(amp_engine, inputs):
    engine, scopes = amp_engine
    calls = []

    def fail(module, args, output):
        calls.append(scopes[-1])
        return torch.full_like(output, float('nan'))

    engine.model.decoder.register_forward_hook(fail)
    with pytest.raises(RuntimeError, match='non-finite'):
        engine.inpaint(*inputs)
    assert calls == ['fp16', 'fp32']
    assert scopes == []


@pytest.mark.parametrize('precision', [None, 'fp32', 'fp16'])
def test_cli_precision_reaches_lazy_engine(monkeypatch, precision):
    import sys
    import vsr_pipeline
    from backend.tools import model_config

    real_pipeline = vsr_pipeline.Pipeline
    seen = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def process_video(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(vsr_pipeline, 'Pipeline', FakePipeline)
    argv = ['vsr_pipeline.py', '-i', 'in.mp4', '-o', 'out.mp4', '--inpaint-mode', 'sttn']
    if precision:
        argv += ['--sttn-precision', precision]
    monkeypatch.setattr(sys, 'argv', argv)
    vsr_pipeline.main()
    assert seen['sttn_precision'] == (precision or 'fp32')
    monkeypatch.setitem(sys.modules, 'paddleocr', SimpleNamespace(TextDetection=lambda **kw: None))
    monkeypatch.setattr(model_config, 'ModelConfig', lambda: SimpleNamespace(STTN_DET_MODEL_PATH='unused.pth'))
    calls = []

    def make_engine(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(sttn_det_inpaint, 'STTNDetInpaint', make_engine)
    pipe = real_pipeline(device='cpu', inpaint_mode='sttn',
                         sttn_precision=seen['sttn_precision'])
    assert not calls
    pipe._ensure_sttn()
    pipe._ensure_sttn()
    assert len(calls) == 1
    assert pipe.inpainter.precision == (precision or 'fp32')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('cached', [False, True])
def test_cuda_real_network_fp16_scoped_and_finite(engine_factory, inputs, cached):
    engine = engine_factory(cache_first_qkv=cached, profile=True)
    engine.device = torch.device('cuda')
    engine.model.to(engine.device)
    engine.requested_precision = engine.precision = 'fp16'
    observed = {}
    for name in ('encoder', 'decoder'):
        getattr(engine.model, name).register_forward_hook(
            lambda module, args, output, name=name: observed.setdefault(name, []).append(output.dtype))
    actual = engine.inpaint(*inputs)
    assert engine.precision == 'fp16'
    assert engine.precision_fallbacks == 0
    assert not torch.is_autocast_enabled()
    assert all(p.dtype == torch.float32 for p in engine.model.parameters())
    assert all(dtype == torch.float16 for values in observed.values() for dtype in values)
    for output, frame, mask in zip(actual, *inputs):
        assert np.isfinite(output).all()
        np.testing.assert_array_equal(output[mask == 0], frame[mask == 0])
    assert engine.last_profile['precision'] == 'fp16'
