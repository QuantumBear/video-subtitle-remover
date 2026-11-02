"""设备端量化只改变回传 dtype，不能改变量化、mask 或窗口递推。"""

from contextlib import contextmanager, nullcontext
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


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', ['fp16', 'fp32'])
def test_quantization_boundaries_match_numpy_without_mutating_strided_input(device, dtype):
    from benchmarks import sttn_quantized_download as benchmark
    if dtype == 'fp16':
        # 穷举 [0, 1] 内所有正 FP16 位模式，包括次正规数。
        values = np.arange(0x3c01, dtype=np.uint16).view(np.float16)
    else:
        boundaries = np.arange(256, dtype=np.float32) / np.float32(255)
        values = np.concatenate([boundaries, np.nextafter(boundaries, np.float32(0)),
                                 np.nextafter(boundaries, np.float32(1))])
    padded = torch.from_numpy(np.repeat(values, 2).copy()).to(device)
    prediction = padded[::2].view(1, 1, 1, -1).expand(1, 3, 1, -1)
    assert not prediction.is_contiguous()
    before = prediction.clone()
    expected = (prediction.float().cpu().numpy() * 255).astype(np.uint8)
    actual = benchmark.quantized_download(prediction)
    assert actual.dtype == torch.uint8 and actual.device.type == 'cpu'
    assert actual.shape == prediction.shape
    np.testing.assert_array_equal(actual.numpy(), expected)
    torch.testing.assert_close(prediction, before, rtol=0, atol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_three_overlaps_keep_uint8_before_mask_and_recursive_half_mix(device):
    from benchmarks import sttn_quantized_download as benchmark
    frames = [np.full((1, 2, 3), 123, dtype=np.uint8)[:, ::-1]]
    masks = [np.array([[[1], [0]]], dtype=np.uint8)]
    result = [None]
    with benchmark.download_execution(mode='uint8') as state:
        for value, expected in [(20.75, 20), (41.75, 30.5), (80.75, 55.25)]:
            prediction = torch.full((1, 3, 1, 2), value / 255, device=device)
            downloaded = sttn_det_inpaint._download_sttn_prediction(prediction)
            sttn_det_inpaint._postprocess_sttn(downloaded, [0], frames, masks, result)
            np.testing.assert_array_equal(result[0][0, 0], [expected] * 3)
            np.testing.assert_array_equal(result[0][0, 1], [123] * 3)
        assert state['download_payload_bytes'] == 18
        assert state['download_calls'] == state['download_frame_visits'] == 3
    np.testing.assert_array_equal(frames[0], np.full((1, 2, 3), 123))


@pytest.mark.parametrize('raise_inside', [False, True])
@pytest.mark.parametrize('mode', ['fp32', 'uint8'])
def test_context_restores_both_helpers_on_success_or_error(mode, raise_inside):
    from benchmarks import sttn_quantized_download as benchmark
    download = sttn_det_inpaint._download_sttn_prediction
    process = sttn_det_inpaint._postprocess_sttn
    with pytest.raises(RuntimeError, match='injected') if raise_inside else nullcontext():
        with benchmark.download_execution(mode=mode):
            if raise_inside:
                raise RuntimeError('injected')
    assert sttn_det_inpaint._download_sttn_prediction is download
    assert sttn_det_inpaint._postprocess_sttn is process


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('cached', [False, True])
def test_band_keeps_default_fp32_download_and_matches_quantized_path(engine, monkeypatch, device, cached):
    from benchmarks import sttn_quantized_download as benchmark
    engine.device = torch.device(device)
    engine.model.to(engine.device)
    engine.precision = engine.requested_precision = 'fp16' if device == 'cuda' else 'fp32'
    engine.cache_first_qkv = cached
    frames, masks = band()
    originals = [item.copy() for item in frames + masks]
    process = sttn_det_inpaint._postprocess_sttn
    dtypes = []

    def observe(prediction, *args):
        dtypes.append((prediction.dtype, prediction.device.type))
        return process(prediction, *args)

    with monkeypatch.context() as patch:
        patch.setattr(sttn_det_inpaint, '_postprocess_sttn', observe)
        default_output = engine.inpaint(frames, masks)
    assert dtypes == [(torch.float32, 'cpu')] * 3
    baseline, reference = benchmark.measure_download(engine, frames, masks, mode='fp32',
                                                     warmup=0, repeats=1)
    selected, returned = benchmark.measure_download(engine, frames, masks, mode='uint8',
                                                     warmup=0, repeats=1, reference=reference)
    assert returned is reference
    for report, bytes_per_value in [(baseline, 4), (selected, 1)]:
        assert report['outputs_match'] is True
        assert report['max_pixel_difference'] == report['mean_pixel_difference'] == 0
        assert report['precision_fallbacks'] == 0
        assert report['backend']['benchmark'] is False
        assert report['qkv_fused_modules'] == 8
        assert report['download_calls'] == 6
        assert report['download_frame_visits'] == 22
        assert report['download_payload_bytes'] == 22 * 3 * 32 * 32 * bytes_per_value
        assert [row['phase'] for row in report['runs']] == ['first', 'measured']
        if device == 'cuda':
            assert report['peak_allocated_gib'] > 0
            assert report['peak_reserved_gib'] >= report['peak_allocated_gib']
        else:
            assert report['peak_allocated_gib'] is None
    np.testing.assert_array_equal(np.stack(default_output), reference)
    for actual, frame, mask in zip(reference, frames, masks):
        np.testing.assert_array_equal(actual[mask == 0], frame[mask == 0])
    for item, before in zip(frames + masks, originals):
        np.testing.assert_array_equal(item, before)


@pytest.mark.parametrize('failure', ['decoder_inf', 'decoder_nan', 'prediction_nan'])
def test_nonfinite_is_rejected_before_quantization(engine, monkeypatch, failure):
    from benchmarks import sttn_quantized_download as benchmark
    if failure.startswith('decoder'):
        engine.model.decoder.register_forward_hook(lambda module, args, output:
            torch.full_like(output, float('inf' if failure == 'decoder_inf' else 'nan')))
    else:
        tanh = torch.tanh
        monkeypatch.setattr(torch, 'tanh', lambda value: tanh(value) * float('nan'))
    with benchmark.download_execution(mode='uint8') as state:
        with pytest.raises(RuntimeError, match='non-finite'):
            engine.inpaint(*band())
        assert state['download_calls'] == 0


def test_second_window_failure_retries_entire_band_without_partial_composites(engine):
    from benchmarks import sttn_quantized_download as benchmark
    frames, masks = band()
    expected = np.stack(engine.inpaint(frames, masks))
    scopes, attempts = [], []

    @contextmanager
    def autocast(precision):
        scopes.append(precision)
        attempts.append(precision)
        try:
            yield
        finally:
            scopes.pop()

    engine.precision = engine.requested_precision = 'fp16'
    engine._autocast_context = autocast
    decodes = 0

    def fail_second_decode(module, args, output):
        nonlocal decodes
        decodes += 1
        if scopes[-1] == 'fp16':
            if decodes == 1:
                return output + 0.5  # 第一窗必须不同，才能发现重试错误复用了部分结果。
            if decodes == 2:
                return torch.full_like(output, float('inf'))

    handle = engine.model.decoder.register_forward_hook(fail_second_decode)
    try:
        with benchmark.download_execution(mode='uint8') as state:
            actual = np.stack(engine.inpaint(frames, masks))
            assert state['download_calls'] == 4  # 丢弃第一带已完成的一窗，再完整三窗。
    finally:
        handle.remove()
    assert attempts == ['fp16', 'fp32']
    assert engine.precision == 'fp32' and engine.precision_fallbacks == 1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('failure', ['pixels', 'fallback', 'nonfinite', 'inference'])
def test_invalid_measurement_restores_helpers_and_backend(engine, failure):
    from benchmarks import sttn_quantized_download as benchmark
    frames, masks = band()
    reference = np.stack(engine.inpaint(frames, masks)).astype(np.float32)
    if failure == 'pixels':
        reference[0, 0, 0, 0] += 0.25
    else:
        inpaint = engine.inpaint

        def invalid(images, holes):
            if failure == 'inference':
                raise RuntimeError('injected')
            result = inpaint(images, holes)
            if failure == 'fallback':
                engine.precision_fallbacks += 1
            else:
                result[0] = result[0].astype(np.float32)
                result[0][0, 0, 0] = np.nan
            return result

        engine.inpaint = invalid
    previous = (sttn_det_inpaint._download_sttn_prediction, sttn_det_inpaint._postprocess_sttn)
    settings = benchmark.backend_settings()
    if failure == 'pixels':
        report, _ = benchmark.measure_download(engine, frames, masks, mode='uint8',
                                                warmup=0, repeats=1, reference=reference)
        assert report['outputs_match'] is False
        assert report['max_pixel_difference'] == 0.25
    else:
        with pytest.raises(RuntimeError, match='精度回退|非有限|injected'):
            benchmark.measure_download(engine, frames, masks, mode='uint8',
                                       warmup=0, repeats=1, reference=reference)
    assert previous == (sttn_det_inpaint._download_sttn_prediction, sttn_det_inpaint._postprocess_sttn)
    assert settings == benchmark.backend_settings()


@pytest.mark.parametrize('gpu,match,speedup', [(False, True, None), (True, False, None),
                                             (True, True, 2)])
def test_summary_suppresses_speedup_for_cpu_or_differences(gpu, match, speedup):
    from benchmarks import sttn_quantized_download as benchmark
    sessions = [dict(mode=mode, outputs_match=match, first_seconds=seconds + 1,
                     setup_seconds=0.01, peak_allocated_gib=1, peak_reserved_gib=2,
                     runs=[dict(phase='first', wall_seconds=100),
                           dict(phase='measured', wall_seconds=seconds)])
                for mode, seconds in [('fp32', 2), ('uint8', 1)]]
    report = benchmark.summarize(sessions, gpu_comparison=gpu)
    assert report['warm_speedup'] == speedup
    assert report['warm_time_saved_percent'] == (50 if speedup else None)


def test_cli_independent_reversed_workers_save_fractional_differences_before_failing(tmp_path, monkeypatch):
    from benchmarks import sttn_quantized_download as benchmark
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
    assert [row['mode'] for row in sessions] == ['fp32', 'uint8', 'uint8', 'fp32']
    assert len({row['environment']['pid'] for row in sessions} | {os.getpid()}) == 5
    for row in sessions:
        assert row['backend'] == sessions[0]['backend']
        assert row['precision_fallbacks'] == 0 and row['qkv_fused_modules'] == 8
        assert row['download_calls'] == 2 and row['download_frame_visits'] == 4
        assert row['download_payload_bytes'] == 4 * 3 * 240 * 432 * (4 if row['mode'] == 'fp32' else 1)
        assert (output / row['log']).is_file()
