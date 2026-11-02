"""后处理 A/B 必须保留量化、逐帧 mask 与重叠窗口递推语义。"""

import json
import os
import numpy as np
import pytest
from contextlib import nullcontext

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint import sttn_det_inpaint
from backend.inpaint.sttn.network_sttn import InpaintGenerator


def frozen_postprocess(prediction, ids, frames, masks, composites):
    """抽出入口之前的生产逻辑，作为独立回归参考。"""
    pixels = prediction.permute(0, 2, 3, 1).numpy() * 255
    for index, frame_id in enumerate(ids):
        image = (pixels[index].astype(np.uint8) * masks[frame_id]
                 + frames[frame_id] * (1 - masks[frame_id]))
        if composites[frame_id] is None:
            composites[frame_id] = image
        else:
            composites[frame_id] = (composites[frame_id].astype(np.float32) * 0.5
                                    + image.astype(np.float32) * 0.5)


@pytest.mark.parametrize('implementation', ['production', 'batched'])
def test_quantization_masks_and_strided_inputs_match_frozen_production(implementation):
    if implementation == 'production':
        process = sttn_det_inpaint._postprocess_sttn
    else:
        from benchmarks.sttn_postprocess import batched_postprocess as process
    rng = np.random.default_rng(81)
    # 包含所有 uint8 值、量化边界两侧，以及非连续 NCHW tensor。
    values = np.arange(256, dtype=np.float32) / np.float32(255)
    boundary = np.concatenate([values, np.nextafter(values, np.float32(0)),
                               np.nextafter(values, np.float32(1))]).reshape(16, 48)
    predictions = torch.from_numpy(np.broadcast_to(boundary, (3, 3, 16, 48)).copy())
    predictions = predictions[:, :, :, ::2]
    frames = [rng.integers(0, 256, (16, 48, 3), dtype=np.uint8)[:, ::2, ::-1]
              for _ in range(3)]
    masks = [np.zeros((16, 48, 1), dtype=np.uint8)[:, ::2] for _ in frames]
    masks[1].fill(1)
    masks[2][::2, ::2] = 1
    inputs = [item.copy() for item in frames + masks]
    tensor_before = predictions.clone()
    expected, actual = [None] * 3, [None] * 3
    for ids, prediction in [([2, 0, 1], predictions), ([0, 2], predictions[:2]),
                            ([2, 1, 0], predictions.flip(0))]:
        frozen_postprocess(prediction, ids, frames, masks, expected)
        process(prediction, ids, frames, masks, actual)
        for result, reference in zip(actual, expected):
            assert result.dtype == reference.dtype
            np.testing.assert_array_equal(result, reference)
    for result, frame, mask in zip(actual, frames, masks):
        np.testing.assert_array_equal(result[mask[..., 0] == 0], frame[mask[..., 0] == 0])
        assert not any(np.shares_memory(result, item) for item in frames + masks)
    torch.testing.assert_close(predictions, tensor_before, rtol=0, atol=0)
    for item, original in zip(frames + masks, inputs):
        np.testing.assert_array_equal(item, original)


@pytest.mark.parametrize('implementation', ['production', 'batched'])
def test_three_overlaps_keep_recursive_half_mix_and_uint8_before_compositing(implementation):
    if implementation == 'production':
        process = sttn_det_inpaint._postprocess_sttn
    else:
        from benchmarks.sttn_postprocess import batched_postprocess as process
    frames = [np.zeros((1, 1, 3), dtype=np.uint8)]
    masks = [np.ones((1, 1, 1), dtype=np.uint8)]
    result = [None]
    for value, expected in [(20.75, 20), (41.75, 30.5), (80.75, 55.25)]:
        process(torch.full((1, 3, 1, 1), value / 255), [0], frames, masks, result)
        np.testing.assert_array_equal(result[0], np.full((1, 1, 3), expected))


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


@pytest.mark.parametrize('raise_inside', [False, True])
def test_benchmark_restores_helper_on_normal_or_exception_exit(raise_inside):
    from benchmarks import sttn_postprocess as benchmark
    previous = sttn_det_inpaint._postprocess_sttn
    with pytest.raises(RuntimeError, match='injected') if raise_inside else nullcontext():
        with benchmark.postprocess_execution(mode='batched'):
            assert sttn_det_inpaint._postprocess_sttn is not previous
            if raise_inside:
                raise RuntimeError('injected')
    assert sttn_det_inpaint._postprocess_sttn is previous


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
def test_full_band_preserves_fused_qkv_precision_inputs_and_unmasked_pixels(engine, cached, device):
    from benchmarks import sttn_postprocess as benchmark
    engine.device = torch.device(device)
    engine.model.to(engine.device)
    engine.precision = engine.requested_precision = 'fp16' if device == 'cuda' else 'fp32'
    engine.cache_first_qkv = cached
    frames, masks = band()
    originals = [item.copy() for item in frames + masks]
    previous = sttn_det_inpaint._postprocess_sttn
    legacy, reference = benchmark.measure_postprocess(engine, frames, masks, mode='legacy',
                                                      warmup=0, repeats=1)
    batched, returned = benchmark.measure_postprocess(engine, frames, masks, mode='batched',
                                                      warmup=0, repeats=1, reference=reference)
    for report in (legacy, batched):
        assert report['outputs_match'] is True
        assert report['max_pixel_difference'] == report['mean_pixel_difference'] == 0
        assert report['precision_fallbacks'] == 0
        assert report['backend']['benchmark'] is False
        assert report['qkv_fused_modules'] == 8
        assert report['postprocess_calls'] == 6  # 两轮各三个窗口。
        assert report['postprocess_frame_visits'] == 22  # 各窗口 3/5/3 帧。
        assert report['process_peak_rss_gib'] > 0
        assert [row['phase'] for row in report['runs']] == ['first', 'measured']
        if device == 'cuda':
            assert report['peak_allocated_gib'] > 0
            assert report['peak_reserved_gib'] >= report['peak_allocated_gib']
        else:
            assert report['peak_allocated_gib'] is None
    assert returned is reference
    assert sttn_det_inpaint._postprocess_sttn is previous
    for actual, frame, mask in zip(reference, frames, masks):
        np.testing.assert_array_equal(actual[mask == 0], frame[mask == 0])
    for item, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(item, original)


@pytest.mark.parametrize('failure', ['pixels', 'fallback', 'nonfinite', 'inference'])
def test_invalid_band_is_rejected_and_helper_restored(engine, failure):
    from benchmarks import sttn_postprocess as benchmark
    frames, masks = band()
    reference = np.stack(engine.inpaint(frames, masks)).astype(np.float32)
    if failure == 'pixels':
        reference[0, 0, 0, 0] += 0.25
    else:
        original = engine.inpaint

        def invalid(images, holes):
            if failure == 'inference':
                raise RuntimeError('injected')
            outputs = original(images, holes)
            if failure == 'fallback':
                engine.precision_fallbacks += 1
            else:
                outputs[0] = outputs[0].astype(np.float32)
                outputs[0][0, 0, 0] = np.nan
            return outputs

        engine.inpaint = invalid
    previous = sttn_det_inpaint._postprocess_sttn
    settings = benchmark.backend_settings()
    if failure == 'pixels':
        report, _ = benchmark.measure_postprocess(engine, frames, masks, mode='batched',
                                                  warmup=0, repeats=1, reference=reference)
        assert report['outputs_match'] is False
        assert report['max_pixel_difference'] == 0.25
    else:
        with pytest.raises(RuntimeError, match='精度回退|非有限|injected'):
            benchmark.measure_postprocess(engine, frames, masks, mode='batched',
                                           warmup=0, repeats=1, reference=reference)
    assert sttn_det_inpaint._postprocess_sttn is previous
    assert benchmark.backend_settings() == settings


@pytest.mark.parametrize('gpu,match,speedup', [(False, True, None), (True, False, None),
                                             (True, True, 2)])
def test_summary_suppresses_speedup_for_cpu_or_pixel_differences(gpu, match, speedup):
    from benchmarks import sttn_postprocess as benchmark
    sessions = [dict(mode=mode, outputs_match=match, first_seconds=seconds + 1,
                     setup_seconds=0.01, peak_allocated_gib=1, peak_reserved_gib=2,
                     process_peak_rss_gib=3,
                     runs=[dict(phase='first', wall_seconds=100),
                           dict(phase='measured', wall_seconds=seconds)])
                for mode, seconds in [('legacy', 2), ('batched', 1)]]
    result = benchmark.summarize(sessions, gpu_comparison=gpu)
    assert result['warm_speedup'] == speedup
    assert result['warm_time_saved_percent'] == (50 if speedup else None)
    assert result['summary']['legacy']['warm_median_seconds'] == 2
    assert result['summary']['batched']['process_peak_rss_gib'] == 3


def test_cli_uses_independent_processes_reversed_order_and_fused_qkv(tmp_path, monkeypatch):
    from benchmarks import sttn_postprocess as benchmark
    torch.manual_seed(91)
    model = tmp_path / 'sttn.pth'
    torch.save({'netG': InpaintGenerator(init_weights=False).state_dict()}, model)
    output = tmp_path / 'result'
    # 第一个真实 worker 保存参考后注入小数像素差，确保父进程先落完整报告再失败。
    run = benchmark.subprocess.run
    perturbed = False

    def run_and_perturb_reference(*args, **kwargs):
        nonlocal perturbed
        result = run(*args, **kwargs)
        if result.returncode == 0 and not perturbed:
            reference_path = output / 'reference.npy'
            reference = np.load(reference_path, allow_pickle=False)
            reference[0, 0, 0, 0] += 0.25
            np.save(reference_path, reference, allow_pickle=False)
            perturbed = True
        return result

    monkeypatch.setattr(benchmark.subprocess, 'run', run_and_perturb_reference)
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
    assert [row['mode'] for row in sessions] == ['legacy', 'batched', 'batched', 'legacy']
    assert len({row['environment']['pid'] for row in sessions} | {os.getpid()}) == 5
    for row in sessions:
        assert row['backend'] == sessions[0]['backend']
        assert row['precision_fallbacks'] == 0
        assert row['qkv_fused_modules'] == 8
        assert row['postprocess_calls'] == 2
        assert row['postprocess_frame_visits'] == 4
        assert row['process_peak_rss_gib'] > 0
        assert (output / row['log']).is_file()
    assert (output / 'reference.npy').is_file()
