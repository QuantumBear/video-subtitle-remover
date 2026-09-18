"""DINO 混合精度的设备选择、数值边界和回退行为。"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from backend.sticker_detect import DEFAULT_PRECISION, GroundingDinoStickerDetector


@pytest.fixture
def fake_loader(monkeypatch):
    import transformers

    loaded = []

    class Model:
        def to(self, device):
            loaded.append(device)
            return self

        def eval(self):
            return self

    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained',
                        lambda *a, **kw: object())
    monkeypatch.setattr(transformers.AutoModelForZeroShotObjectDetection,
                        'from_pretrained', lambda *a, **kw: Model())
    return loaded


@pytest.mark.parametrize('precision', ['fp16', 'bf16'])
def test_cpu_uses_fp32_with_explicit_fallback(fake_loader, capsys, precision):
    detector = GroundingDinoStickerDetector(device='cpu', precision=precision, profile=True)
    report = detector.profile_summary()
    assert report['requested_precision'] == precision
    assert report['precision'] == 'fp32'
    assert report['precision_fallbacks'] == 1
    log = capsys.readouterr().out
    assert '[gdino-precision-fallback]' in log
    assert f'requested={precision}' in log
    assert 'device=cpu' in log


@pytest.mark.parametrize('supported', [True, False])
def test_bf16_checks_the_selected_cuda_device(fake_loader, monkeypatch, supported):
    devices = []

    @contextmanager
    def device_scope(device):
        devices.append(device)
        yield

    monkeypatch.setattr(torch.cuda, 'device', device_scope)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: supported)
    detector = GroundingDinoStickerDetector(device='cuda:1', precision='bf16')
    assert devices == [torch.device('cuda:1')]
    assert detector.precision == ('bf16' if supported else 'fp32')
    assert detector.precision_fallbacks == (0 if supported else 1)
    assert fake_loader == [torch.device('cuda:1')]


def test_default_precision_and_invalid_precision(fake_loader, capsys):
    detector = GroundingDinoStickerDetector(device='cpu')
    assert detector.requested_precision == DEFAULT_PRECISION == 'fp16'
    assert detector.precision == 'fp32'
    assert detector.precision_fallbacks == 1
    assert '[gdino-precision-fallback]' in capsys.readouterr().out
    with pytest.raises(ValueError, match='precision'):
        GroundingDinoStickerDetector(device='cpu', precision='int8')
    assert len(fake_loader) == 1


@pytest.fixture
def amp_detector(monkeypatch):
    """用 CPU 张量隔离精度调度；真实 CUDA 算子另行测试。"""
    detector = GroundingDinoStickerDetector.__new__(GroundingDinoStickerDetector)
    detector.device = torch.device('cuda')
    detector.requested_precision = detector.precision = 'fp16'
    scopes = []

    @contextmanager
    def autocast(device_type, dtype=None, enabled=True):
        assert device_type == 'cuda'
        scopes.append((enabled, dtype))
        try:
            yield
        finally:
            scopes.pop()

    monkeypatch.setattr(torch, 'autocast', autocast)
    return detector, scopes


def output(dtype=torch.float16):
    # GroundingDINO 用 -inf 填充文本 padding，不能将其视作数值异常。
    return SimpleNamespace(logits=torch.tensor([[[0.5, -float('inf')]]], dtype=dtype),
                           pred_boxes=torch.tensor([[[0.5, 0.5, 0.2, 0.2]]], dtype=dtype))


@pytest.mark.parametrize('precision,dtype', [('fp16', torch.float16), ('bf16', torch.bfloat16)])
def test_amp_is_scoped_to_forward_and_outputs_are_fp32(amp_detector, precision, dtype):
    detector, scopes = amp_detector
    detector.precision = detector.requested_precision = precision
    tokens = torch.tensor([[1, 2]], dtype=torch.int64)
    weight = torch.ones(1, dtype=torch.float32)

    def model(**inputs):
        assert scopes == [(True, dtype)]
        assert torch.is_inference_mode_enabled()
        assert inputs['input_ids'] is tokens
        assert inputs['input_ids'].dtype == torch.int64
        assert weight.dtype == torch.float32
        return output(dtype)

    detector.model = model
    result = detector._forward({'input_ids': tokens})
    assert scopes == []
    assert result.logits.dtype == result.pred_boxes.dtype == torch.float32
    assert torch.isneginf(result.logits[0, 0, 1])
    assert detector.precision == precision
    assert detector.precision_fallbacks == 0


@pytest.mark.parametrize('failure', ['kernel', 'nan', 'inf', 'all_negative_inf', 'box'])
def test_amp_failure_retries_once_then_stays_fp32(amp_detector, capsys, failure):
    detector, scopes = amp_detector
    detector.profile = True
    calls = []

    def model(**inputs):
        enabled, _ = scopes[-1]
        calls.append(enabled)
        result = output(torch.float32)
        if enabled:
            if failure == 'kernel':
                raise RuntimeError('operator does not support Half')
            if failure == 'box':
                result.pred_boxes[0, 0, 0] = float('nan')
            else:
                result.logits[0, 0, 0] = {
                    'nan': float('nan'), 'inf': float('inf'),
                    'all_negative_inf': -float('inf'),
                }[failure]
        return result

    detector.model = model
    result = detector._forward({})
    assert torch.isfinite(result.pred_boxes).all()
    detector._forward({})
    assert calls == [True, False, False]
    assert detector.precision == 'fp32'
    report = detector.profile_summary()
    assert report['requested_precision'] == 'fp16'
    assert report['precision_fallbacks'] == 1
    assert capsys.readouterr().out.count('[gdino-precision-fallback]') == 1


@pytest.mark.parametrize('error', [torch.cuda.OutOfMemoryError('OOM'),
                                   RuntimeError('CUDA out of memory')])
def test_oom_is_left_to_batch_fallback(amp_detector, error):
    detector, _ = amp_detector

    def model(**inputs):
        raise error

    detector.model = model
    with pytest.raises(type(error)) as exc:
        detector._forward({})
    assert exc.value is error
    assert detector.precision == 'fp16'
    assert detector.precision_fallbacks == 0


def test_failed_fp32_retry_propagates(amp_detector):
    detector, _ = amp_detector
    calls = []

    def model(**inputs):
        calls.append(1)
        result = output()
        result.pred_boxes.fill_(float('nan'))
        return result

    detector.model = model
    with pytest.raises(RuntimeError, match='non-finite'):
        detector._forward({})
    assert len(calls) == 2


@pytest.mark.parametrize('precision', [None, 'fp16', 'bf16'])
def test_precision_cli_reaches_lazy_detector(monkeypatch, precision):
    import vsr_pipeline

    seen = {}
    real_pipeline = vsr_pipeline.Pipeline

    class FakePipeline:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def process_video(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(vsr_pipeline, 'Pipeline', FakePipeline)
    argv = ['vsr_pipeline.py', '-i', 'in.mp4', '-o', 'out.mp4', '--sticker-backend', 'gdino']
    if precision is not None:
        argv.extend(['--sticker-precision', precision])
    monkeypatch.setattr('sys.argv', argv)
    vsr_pipeline.main()
    assert seen['sticker_precision'] == (precision or DEFAULT_PRECISION)

    # 实际 Pipeline 构造与惰性加载都要转发精度，OCR 用替身避免加载模型。
    import sys
    monkeypatch.setitem(sys.modules, 'paddleocr', SimpleNamespace(TextDetection=lambda **kw: None))
    pipe = real_pipeline(sticker_backend='gdino', inpaint_mode='sttn', device='cpu',
                         sticker_precision=seen['sticker_precision'])
    options = {}

    def detector_factory(**kwargs):
        options.update(kwargs)
        return SimpleNamespace(device='cpu')

    monkeypatch.setattr(vsr_pipeline.sticker_detect, 'GroundingDinoStickerDetector', detector_factory)
    pipe._ensure_sticker_detector()
    assert options['precision'] == (precision or DEFAULT_PRECISION)


def test_cli_rejects_invalid_precision_before_loading_models(monkeypatch):
    import vsr_pipeline

    monkeypatch.setattr(vsr_pipeline, 'Pipeline', lambda **kw: pytest.fail('must not load models'))
    monkeypatch.setattr('sys.argv', ['vsr_pipeline.py', '-i', 'in.mp4', '-o', 'out.mp4',
                                     '--sticker-precision', 'int8'])
    with pytest.raises(SystemExit) as exc:
        vsr_pipeline.main()
    assert exc.value.code == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('precision,dtype', [('fp16', torch.float16), ('bf16', torch.bfloat16)])
def test_cuda_autocast_through_batch_and_single_candidate_paths(precision, dtype):
    import numpy as np

    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA device does not support BF16')
    observed = []

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 4)

        def forward(self, pixel_values, input_ids):
            assert input_ids.dtype == torch.int64
            assert torch.is_inference_mode_enabled()
            boxes = self.projection(pixel_values).sigmoid().unsqueeze(1)
            observed.append(boxes.dtype)
            logits = torch.zeros(len(boxes), 1, 2, dtype=boxes.dtype, device=boxes.device)
            logits[:, :, 1] = -float('inf')
            return SimpleNamespace(logits=logits, pred_boxes=boxes)

    class Processor:
        def __call__(self, images=None, text=None, **kwargs):
            if text is not None:
                return {'input_ids': torch.tensor([[1, 2]])}
            return {'pixel_values': torch.ones(len(images), 3)}

        def post_process_grounded_object_detection(self, outputs, input_ids, **kwargs):
            assert outputs.logits.dtype == outputs.pred_boxes.dtype == torch.float32
            assert not torch.is_autocast_enabled()
            return [{'scores': [0.8], 'boxes': [[10, 10, 30, 30]]}
                    for _ in kwargs['target_sizes']]

    detector = GroundingDinoStickerDetector.__new__(GroundingDinoStickerDetector)
    detector.device = torch.device('cuda')
    detector.precision = detector.requested_precision = precision
    detector.model = Model().to(detector.device).eval()
    detector.processor = Processor()
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    batch = detector.detect_candidates_batch([crop, crop], (0, 100, 0, 100), 'emoji.', 0.25, 1200)
    single = detector.detect_candidates(crop, (0, 100, 0, 100), 'emoji.', 0.25, 1200)
    assert batch == [single, single]
    assert len(single) == 1
    assert observed == [dtype, dtype]
    assert all(p.dtype == torch.float32 for p in detector.model.parameters())
    assert detector.precision_fallbacks == 0
