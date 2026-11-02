"""Decoder 布局实验须隔离修改、计入转换，并拒绝无效性能比较。"""

from contextlib import nullcontext
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint import sttn_det_inpaint
from backend.inpaint.sttn.network_sttn import InpaintGenerator
from benchmarks import sttn_decoder_layout as benchmark


@pytest.fixture
def engine():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(81)
    result = sttn_det_inpaint.STTNDetInpaint.__new__(sttn_det_inpaint.STTNDetInpaint)
    result.device = torch.device('cpu')
    result.precision = result.requested_precision = 'fp32'
    result.precision_fallbacks = 0
    result.profile = False
    result.model = InpaintGenerator(init_weights=False).eval()
    for block in result.model.transformer:
        block.attention.patchsize = [(8, 8), (4, 4), (2, 2), (1, 1)]
    result.model.prepare_fused_qkv()
    result.model_input_width = result.model_input_height = 32
    result.neighbor_stride, result.ref_length = 2, 3
    result.cache_first_qkv = True
    yield result
    torch.set_num_threads(previous)


def band():
    rng = np.random.default_rng(81)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(5)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    masks[1].fill(255)
    masks[2][8:24, 8:24] = 255
    masks[3][16:, :] = 255
    return frames, masks


@pytest.mark.parametrize('mode', ['contiguous', 'channels-last'])
@pytest.mark.parametrize('raise_inside', [False, True])
def test_decoder_only_layout_and_output_conversion_restore_even_on_error(engine, mode, raise_inside):
    model = engine.model
    weights = {name: (param, param.detach().clone(), param.stride())
               for name, param in model.named_parameters()}
    original_forward = model.decoder.forward
    model.decoder.forward = original_forward  # 已有实例覆盖也必须恢复。
    x = torch.randn(3, 256, 8, 8)
    with torch.no_grad():
        expected = model.decoder(x).clone()
    observed = []

    def observe(module, args, output):
        observed.append((args[0].is_contiguous(memory_format=torch.channels_last),
                         module.weight.is_contiguous(memory_format=torch.channels_last)))

    handles = [layer.register_forward_hook(observe) for layer in model.decoder.modules()
               if isinstance(layer, torch.nn.Conv2d)]
    try:
        with pytest.raises(RuntimeError, match='injected') if raise_inside else nullcontext():
            with benchmark.decoder_layout(model, mode=mode) as state:
                for name, param in model.named_parameters():
                    if not name.startswith('decoder.'):
                        assert param is weights[name][0]
                        assert param.stride() == weights[name][2]
                with torch.no_grad():
                    actual = model.decoder(x)
                assert actual.is_contiguous()
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                assert observed == [(mode == 'channels-last', mode == 'channels-last')] * 4
                assert state['decoder_calls'] == 1 and state['decoder_frame_visits'] == 3
                if raise_inside:
                    raise RuntimeError('injected')
    finally:
        for handle in handles:
            handle.remove()
    for name, param in model.named_parameters():
        assert param is weights[name][0]
        assert param.stride() == weights[name][2]
        torch.testing.assert_close(param, weights[name][1], rtol=0, atol=0)
    assert model.decoder.__dict__['forward'] is original_forward


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('device,precision', [('cpu', 'fp32'),
    pytest.param('cuda', 'fp16', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA'))])
def test_band_uses_current_download_and_compares_both_layouts(engine, monkeypatch, cached, device, precision):
    engine.device = torch.device(device)
    engine.model.to(engine.device)
    engine.cache_first_qkv = cached
    engine.precision = engine.requested_precision = precision
    frames, masks = band()
    originals = [item.copy() for item in frames + masks]
    expected = np.stack(engine.inpaint(frames, masks)).astype(np.float32)
    settings = benchmark.backend_settings()
    process = sttn_det_inpaint._postprocess_sttn
    downloaded = []

    def observe(prediction, *args):
        downloaded.append(prediction.dtype)
        return process(prediction, *args)

    monkeypatch.setattr(sttn_det_inpaint, '_postprocess_sttn', observe)
    baseline, reference = benchmark.measure_layout(engine, frames, masks, mode='contiguous',
                                                   warmup=1, repeats=1)
    np.testing.assert_array_equal(reference, expected)
    candidate, returned = benchmark.measure_layout(engine, frames, masks, mode='channels-last',
                                                    warmup=1, repeats=1, reference=reference)
    assert returned is reference
    assert baseline['outputs_match'] is True
    assert downloaded == [torch.uint8 if device == 'cuda' else torch.float32] * 18
    for report in (baseline, candidate):
        assert report['decoder_calls'] == 9 and report['decoder_frame_visits'] == 33
        assert report['decoder_conv_layers'] == 4 and report['qkv_fused_modules'] == 8
        assert report['backend']['benchmark'] is False
        assert report['precision_fallbacks'] == 0
        assert report['layout_setup_seconds'] > 0 and report['warm_median_seconds'] > 0
        assert [row['phase'] for row in report['runs']] == ['first', 'warmup', 'measured']
        assert report['outputs_match'] == (report['max_pixel_difference'] == 0)
        if device == 'cuda':
            assert report['peak_allocated_gib'] > 0
            assert report['peak_reserved_gib'] >= report['peak_allocated_gib']
            assert report['layout_setup_peak_allocated_gib'] > 0
        else:
            assert report['peak_allocated_gib'] is None
            assert report['layout_setup_peak_allocated_gib'] is None
    assert benchmark.backend_settings() == settings
    assert 'forward' not in engine.model.decoder.__dict__
    for actual, frame, mask in zip(reference, frames, masks):
        np.testing.assert_array_equal(actual[mask == 0], frame[mask == 0])
    for item, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(item, original)


@pytest.mark.parametrize('failure', ['pixels', 'fallback', 'nonfinite', 'inference'])
def test_bad_measurement_restores_decoder_and_backend(engine, failure):
    frames, masks = band()
    reference = np.stack(engine.inpaint(frames, masks)).astype(np.float32)
    if failure == 'pixels':
        reference[0, 0, 0, 0] += 0.25  # 防止额外 uint8 转换隐藏小数差异。
    else:
        original = engine.inpaint

        def invalid(images, holes):
            if failure == 'inference':
                raise RuntimeError('injected')
            output = original(images, holes)
            if failure == 'fallback':
                engine.precision_fallbacks += 1
            else:
                output[0] = output[0].astype(np.float32)
                output[0][0, 0, 0] = np.nan
            return output

        engine.inpaint = invalid
    strides = {name: param.stride() for name, param in engine.model.named_parameters()}
    settings = benchmark.backend_settings()
    if failure == 'pixels':
        report, _ = benchmark.measure_layout(engine, frames, masks, mode='channels-last',
                                               warmup=0, repeats=1, reference=reference)
        assert report['outputs_match'] is False
        assert report['max_pixel_difference'] >= 0.25
    else:
        with pytest.raises(RuntimeError, match='精度回退|非有限|injected'):
            benchmark.measure_layout(engine, frames, masks, mode='channels-last',
                                     warmup=0, repeats=1, reference=reference)
    assert 'forward' not in engine.model.decoder.__dict__
    assert strides == {name: param.stride() for name, param in engine.model.named_parameters()}
    assert settings == benchmark.backend_settings()


def test_candidate_requires_independent_contiguous_reference(engine):
    with pytest.raises(ValueError, match='reference'):
        benchmark.measure_layout(engine, *band(), mode='channels-last', warmup=0, repeats=1)


@pytest.mark.parametrize('gpu,match,speedup', [(False, True, None), (True, False, None),
                                             (True, True, 2)])
def test_summary_never_reports_speedup_for_cpu_or_pixel_difference(gpu, match, speedup):
    sessions = [dict(mode=mode, outputs_match=match, first_seconds=seconds + 1,
                     layout_setup_seconds=0.01, peak_allocated_gib=1, peak_reserved_gib=2,
                     runs=[dict(phase='first', wall_seconds=100),
                           dict(phase='measured', wall_seconds=seconds)])
                for mode, seconds in [('contiguous', 2), ('channels-last', 1)]]
    report = benchmark.summarize(sessions, gpu_comparison=gpu)
    assert report['warm_speedup'] == speedup
    assert report['warm_time_saved_percent'] == (50 if speedup else None)


def test_cli_reversed_independent_workers_save_fractional_difference_before_failing(tmp_path, monkeypatch):
    torch.manual_seed(91)
    model = tmp_path / 'sttn.pth'
    torch.save({'netG': InpaintGenerator(init_weights=False).state_dict()}, model)
    output = tmp_path / 'result'
    run = benchmark.subprocess.run
    perturbed = False

    def run_and_perturb(*args, **kwargs):
        nonlocal perturbed
        result = run(*args, **kwargs)
        if result.returncode == 0 and not perturbed:
            path = output / 'reference.npy'
            reference = np.load(path, allow_pickle=False)
            reference[0, 0, 0, 0] += 0.25
            np.save(path, reference, allow_pickle=False)
            perturbed = True
        return result

    monkeypatch.setattr(benchmark.subprocess, 'run', run_and_perturb)
    with pytest.raises(SystemExit, match='像素不一致'):
        benchmark.main(['--device', 'cpu', '--precision', 'fp32', '--model', str(model),
                        '--frames', '2', '--warmup', '0', '--repeats', '1',
                        '--sessions', '2', '--output-dir', str(output)])
    report = json.loads((output / 'report.json').read_text())
    assert report['outputs_match'] is False and report['max_pixel_difference'] >= 0.25
    assert report['gpu_comparison'] is False and report['warm_speedup'] is None
    sessions = report['sessions']
    assert [row['mode'] for row in sessions] == ['contiguous', 'channels-last', 'channels-last', 'contiguous']
    assert len({row['environment']['pid'] for row in sessions} | {os.getpid()}) == 5
    for row in sessions:
        assert row['backend'] == sessions[0]['backend']
        assert row['precision_fallbacks'] == 0 and row['qkv_fused_modules'] == 8
        assert row['decoder_calls'] == 2 and row['decoder_frame_visits'] == 4
        assert (output / row['log']).is_file()
