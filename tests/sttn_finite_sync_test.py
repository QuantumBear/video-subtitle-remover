"""有限值检查延后读取仍须拒绝坏窗口，并完整重试修复带。"""

from contextlib import contextmanager
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint import sttn_det_inpaint
from backend.inpaint.sttn.network_sttn import InpaintGenerator

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))]


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


@pytest.mark.parametrize('failure', ['inf', 'nan'])
@pytest.mark.parametrize('device', DEVICES)
def test_deferred_decoder_check_survives_tanh_saturation(engine, failure, device):
    from benchmarks import sttn_finite_sync as benchmark

    engine.device = torch.device(device)
    engine.model.to(engine.device)
    # Inf 经过 tanh 后是有限值，只检查归一化预测会漏报。
    engine.model.decoder.register_forward_hook(
        lambda module, args, output: torch.full_like(output, float(failure)))
    with benchmark.finite_execution(engine, mode='deferred'):
        with pytest.raises(RuntimeError, match='non-finite decoder output'):
            engine.inpaint(*band())


@pytest.mark.parametrize('device', DEVICES)
def test_deferred_prediction_check_runs_before_compositing(engine, monkeypatch, device):
    from benchmarks import sttn_finite_sync as benchmark

    engine.device = torch.device(device)
    engine.model.to(engine.device)
    original_tanh = torch.tanh
    monkeypatch.setattr(torch, 'tanh', lambda value: original_tanh(value) * float('nan'))
    with benchmark.finite_execution(engine, mode='deferred'):
        with pytest.raises(RuntimeError, match='STTN non-finite output'):
            engine.inpaint(*band())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('failure', ['decoder_inf', 'activation_kernel'])
def test_second_window_failure_discards_earlier_results_and_reuses_flag_buffer(
        engine, monkeypatch, device, failure):
    from benchmarks import sttn_finite_sync as benchmark

    engine.device = torch.device(device)
    engine.model.to(engine.device)
    frames, masks = band()
    expected = engine.inpaint(frames, masks)
    engine.precision = engine.requested_precision = 'fp16'
    scopes, attempts, decodes = [], [], []

    @contextmanager
    def autocast(precision):
        scopes.append(precision)
        attempts.append(precision)
        try:
            yield
        finally:
            scopes.pop()

    monkeypatch.setattr(engine, '_autocast_context', autocast)

    def inject(module, args, output):
        decodes.append(scopes[-1])
        if scopes[-1] == 'fp16':
            if len(decodes) == 1:
                return output + 0.5
            if failure == 'decoder_inf':
                return torch.full_like(output, float('inf'))
        return output

    original_tanh = torch.tanh

    def tanh(value):
        if failure == 'activation_kernel' and scopes[-1] == 'fp16' and len(decodes) == 2:
            # 第一个标志已提交异步拷贝，预测下载尚未发生。
            raise RuntimeError('injected activation failure')
        return original_tanh(value)

    monkeypatch.setattr(torch, 'tanh', tanh)
    engine.model.decoder.register_forward_hook(inject)
    with benchmark.finite_execution(engine, mode='deferred'):
        actual = engine.inpaint(frames, masks)
        subsequent = engine.inpaint(frames, masks)
    assert attempts == ['fp16', 'fp32', 'fp32']
    assert engine.precision == 'fp32'
    assert engine.precision_fallbacks == 1
    for result, later, reference in zip(actual, subsequent, expected):
        np.testing.assert_array_equal(result, reference)
        np.testing.assert_array_equal(later, reference)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('device', DEVICES)
def test_full_band_matches_default_and_keeps_inputs_and_backend(engine, cached, device):
    from benchmarks import sttn_finite_sync as benchmark

    engine.device = torch.device(device)
    engine.model.to(engine.device)
    engine.precision = engine.requested_precision = 'fp16' if device == 'cuda' else 'fp32'
    engine.cache_first_qkv = cached
    frames, masks = band()
    originals = [item.copy() for item in frames + masks]
    expected = engine.inpaint(frames, masks)
    previous_check = sttn_det_inpaint._check_sttn_finite
    previous_post = sttn_det_inpaint._postprocess_sttn
    settings = benchmark.backend_settings()
    immediate, reference = benchmark.measure_finite(engine, frames, masks, mode='immediate',
                                                    warmup=0, repeats=1)
    deferred, returned = benchmark.measure_finite(engine, frames, masks, mode='deferred',
                                                 warmup=0, repeats=1, reference=reference)
    assert returned is reference
    for row in (immediate, deferred):
        assert row['outputs_match'] is True
        assert row['max_pixel_difference'] == row['mean_pixel_difference'] == 0
        assert row['precision_fallbacks'] == 0
        assert row['decoder_checks'] == row['prediction_checks'] == row['completed_windows'] == 6
        assert row['qkv_fused_modules'] == 8
        assert row['backend']['benchmark'] is False
    assert immediate['early_scalar_reads'] == 12
    assert deferred['early_scalar_reads'] == 0
    assert deferred['flag_d2h_copies'] == (12 if device == 'cuda' else 0)
    assert deferred['pinned_flag_bytes'] == (2 if device == 'cuda' else 0)
    assert sttn_det_inpaint._check_sttn_finite is previous_check
    assert sttn_det_inpaint._postprocess_sttn is previous_post
    assert benchmark.backend_settings() == settings
    for actual, default, frame, mask in zip(reference, expected, frames, masks):
        np.testing.assert_array_equal(actual, default)
        np.testing.assert_array_equal(actual[mask == 0], frame[mask == 0])
    for item, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(item, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_deferred_cuda_uses_download_instead_of_device_scalar_reads(engine):
    from benchmarks import sttn_finite_sync as benchmark

    engine.device = torch.device('cuda')
    engine.model.to(engine.device)
    engine.precision = engine.requested_precision = 'fp16'
    frames, masks = band()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), benchmark.finite_execution(engine, mode='deferred'):
        # 不使用默认 stream，验证检查标志与现有下载的顺序关系。
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            result = engine.inpaint(frames, masks)
    assert all(np.isfinite(frame).all() for frame in result)
    assert not any(event.key == 'aten::_local_scalar_dense' for event in profile.key_averages())


@pytest.mark.parametrize('failure', ['kernel', 'oom'])
def test_context_restores_helpers_when_inference_aborts_before_download(engine, failure):
    from benchmarks import sttn_finite_sync as benchmark

    previous_check = sttn_det_inpaint._check_sttn_finite
    previous_post = sttn_det_inpaint._postprocess_sttn
    error = torch.cuda.OutOfMemoryError('OOM') if failure == 'oom' else RuntimeError('injected')

    def fail(module, args, output):
        raise error

    engine.model.decoder.register_forward_hook(fail)
    with pytest.raises(type(error), match='OOM|injected'):
        with benchmark.finite_execution(engine, mode='deferred'):
            engine.inpaint(*band())
    assert sttn_det_inpaint._check_sttn_finite is previous_check
    assert sttn_det_inpaint._postprocess_sttn is previous_post
    assert engine.precision_fallbacks == 0


@pytest.mark.parametrize('failure', ['pixels', 'fallback'])
def test_measurement_rejects_bad_results_and_restores_helpers(engine, failure):
    from benchmarks import sttn_finite_sync as benchmark

    frames, masks = band()
    reference = np.stack(engine.inpaint(frames, masks)).astype(np.float32)
    previous_check = sttn_det_inpaint._check_sttn_finite
    previous_post = sttn_det_inpaint._postprocess_sttn
    if failure == 'pixels':
        reference[0, 0, 0, 0] += 0.25
        report, _ = benchmark.measure_finite(engine, frames, masks, mode='deferred',
                                             warmup=0, repeats=1, reference=reference)
        assert report['outputs_match'] is False
        assert report['max_pixel_difference'] == 0.25
    else:
        original = engine.inpaint

        def fallback(images, holes):
            result = original(images, holes)
            engine.precision_fallbacks += 1
            return result

        engine.inpaint = fallback
        with pytest.raises(RuntimeError, match='精度回退'):
            benchmark.measure_finite(engine, frames, masks, mode='deferred',
                                     warmup=0, repeats=1, reference=reference)
    assert sttn_det_inpaint._check_sttn_finite is previous_check
    assert sttn_det_inpaint._postprocess_sttn is previous_post


@pytest.mark.parametrize('gpu,match,speedup', [(False, True, None), (True, False, None),
                                             (True, True, 2)])
def test_summary_suppresses_speedup_for_cpu_or_pixel_differences(gpu, match, speedup):
    from benchmarks import sttn_finite_sync as benchmark

    sessions = [dict(mode=mode, outputs_match=match, first_seconds=seconds + 1,
                     setup_seconds=0.01, peak_allocated_gib=1, peak_reserved_gib=2,
                     runs=[dict(phase='first', wall_seconds=100),
                           dict(phase='measured', wall_seconds=seconds)])
                for mode, seconds in [('immediate', 2), ('deferred', 1)]]
    result = benchmark.summarize(sessions, gpu_comparison=gpu)
    assert result['warm_speedup'] == speedup
    assert result['warm_time_saved_percent'] == (50 if speedup else None)


def test_cli_independent_processes_reversed_order_and_pixel_failure_report(tmp_path, monkeypatch):
    from benchmarks import sttn_finite_sync as benchmark

    model = tmp_path / 'sttn.pth'
    torch.save({'netG': InpaintGenerator(init_weights=False).state_dict()}, model)
    output = tmp_path / 'result'
    run = benchmark.subprocess.run
    perturbed = False

    def run_and_perturb(*args, **kwargs):
        nonlocal perturbed
        result = run(*args, **kwargs)
        if result.returncode == 0 and not perturbed:
            reference_path = output / 'reference.npy'
            reference = np.load(reference_path, allow_pickle=False)
            reference[0, 0, 0, 0] += 0.25
            np.save(reference_path, reference, allow_pickle=False)
            perturbed = True
        return result

    monkeypatch.setattr(benchmark.subprocess, 'run', run_and_perturb)
    with pytest.raises(SystemExit, match='像素不一致'):
        benchmark.main(['--device', 'cpu', '--precision', 'fp32', '--model', str(model),
                        '--frames', '2', '--warmup', '0', '--repeats', '1',
                        '--sessions', '2', '--output-dir', str(output)])
    report = json.loads((output / 'report.json').read_text())
    assert report['outputs_match'] is False
    assert report['max_pixel_difference'] >= 0.25
    assert report['gpu_comparison'] is False
    assert report['warm_speedup'] is None
    sessions = report['sessions']
    assert [row['mode'] for row in sessions] == ['immediate', 'deferred', 'deferred', 'immediate']
    assert len({row['environment']['pid'] for row in sessions} | {os.getpid()}) == 5
    for row in sessions:
        assert row['backend'] == sessions[0]['backend']
        assert row['precision_fallbacks'] == 0
        assert row['qkv_fused_modules'] == 8
        assert row['decoder_checks'] == row['prediction_checks'] == row['completed_windows'] == 2
        assert row['early_scalar_reads'] == (4 if row['mode'] == 'immediate' else 0)
        assert (output / row['log']).is_file()
