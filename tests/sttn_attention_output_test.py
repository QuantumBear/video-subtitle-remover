"""Attention 拼接实验只改变复制路径，不能改变卷积入口和最终像素。"""

from contextlib import nullcontext
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint import sttn_det_inpaint
from backend.inpaint.sttn.network_sttn import InpaintGenerator, MultiHeadedAttention


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


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('fused,cached', [(False, False), (True, False), (True, True)])
def test_direct_write_preserves_patch_channel_time_order_and_removes_cat(batch, fused, cached):
    from benchmarks import sttn_attention_output as benchmark

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(93)
        model = MultiHeadedAttention([(12, 8), (6, 4), (3, 2), (1, 1)], 8).eval()
        if fused:
            model.prepare_fused_qkv()
        x = torch.randn(batch * 3, 8, 8, 24)[..., ::2]
        before = x.clone()
        masks = torch.randn(batch * 3, 1, 8, 12)
        entries = []

        def capture(module, args):
            tensor = args[0]
            entries.append((tensor.clone(), tensor.stride(), tensor.dtype, tensor.is_contiguous()))

        handle = model.output_linear.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                projected = model.project_qkv(x) if cached else None
                results = []
                for mode in ('cat', 'direct'):
                    with benchmark.attention_output_execution(model, mode=mode) as state:
                        with torch.profiler.profile() as profile:
                            results.append(model(x, masks, batch, 8, projected_qkv=projected))
                        assert sum(event.count for event in profile.key_averages()
                                   if event.key == 'aten::cat') == (1 if mode == 'cat' else 0)
                        assert state['multihead_calls'] == 1
                        assert state['multihead_frame_visits'] == batch * 3
                        assert state['attention_branch_calls'] == 4
                        assert state['output_cat_calls'] == (1 if mode == 'cat' else 0)
                        assert state['direct_branch_copy_calls'] == (4 if mode == 'direct' else 0)
                torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
        finally:
            handle.remove()
        assert entries[0][1:] == entries[1][1:]
        assert entries[1][3] is True
        torch.testing.assert_close(entries[0][0], entries[1][0], rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        assert 'forward' not in model.__dict__
    finally:
        torch.set_num_threads(previous_threads)


def test_autocast_output_buffer_uses_attention_dtype_with_fp32_input():
    from benchmarks import sttn_attention_output as benchmark

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(94)
        model = MultiHeadedAttention([(8, 8), (4, 4), (2, 2), (1, 1)], 8).eval()
        model.prepare_fused_qkv()
        x = torch.randn(2, 8, 8, 8)
        masks = torch.zeros(2, 1, 8, 8)
        entries = []
        handle = model.output_linear.register_forward_pre_hook(
            lambda module, args: entries.append(args[0].clone()))
        try:
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                results = []
                for mode in ('cat', 'direct'):
                    with benchmark.attention_output_execution(model, mode=mode):
                        results.append(model(x, masks, 1, 8))
        finally:
            handle.remove()
        assert x.dtype == torch.float32
        assert entries[0].dtype == entries[1].dtype == torch.bfloat16
        assert entries[0].stride() == entries[1].stride()
        torch.testing.assert_close(entries[0], entries[1], rtol=0, atol=0)
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('mode', ['cat', 'direct'])
@pytest.mark.parametrize('raise_inside', [False, True])
def test_context_restores_existing_forward_weights_and_fused_buffers(engine, mode, raise_inside):
    from benchmarks import sttn_attention_output as benchmark

    first = engine.model.transformer[0].attention
    previous = first.forward
    first.forward = previous
    parameters = {name: (param, param.detach().clone(), param.stride())
                  for name, param in engine.model.named_parameters()}
    buffers = [(block.attention._qkv_weight, block.attention._qkv_bias)
               for block in engine.model.transformer]
    with pytest.raises(RuntimeError, match='injected') if raise_inside else nullcontext():
        with benchmark.attention_output_execution(engine.model, mode=mode):
            with torch.no_grad():
                first(torch.randn(2, 256, 8, 8), torch.zeros(2, 1, 8, 8), 1, 256)
            if raise_inside:
                raise RuntimeError('injected')
    assert first.__dict__['forward'] is previous
    assert not any('forward' in block.attention.__dict__ for block in engine.model.transformer[1:])
    for name, param in engine.model.named_parameters():
        assert param is parameters[name][0] and param.stride() == parameters[name][2]
        torch.testing.assert_close(param, parameters[name][1], rtol=0, atol=0)
    for block, (weight, bias) in zip(engine.model.transformer, buffers):
        assert block.attention._qkv_weight is weight and block.attention._qkv_bias is bias


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('device,precision', [('cpu', 'fp32'),
    pytest.param('cuda', 'fp16', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA'))])
def test_band_preserves_pixels_current_download_and_counter_paths(engine, monkeypatch, cached, device, precision):
    from benchmarks import sttn_attention_output as benchmark

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
    baseline, reference = benchmark.measure_output(engine, frames, masks, mode='cat', warmup=0, repeats=1)
    candidate, returned = benchmark.measure_output(engine, frames, masks, mode='direct',
                                                    warmup=0, repeats=1, reference=reference)
    assert returned is reference
    np.testing.assert_array_equal(reference, expected)
    assert downloaded == [torch.uint8 if device == 'cuda' else torch.float32] * 12
    for report in (baseline, candidate):
        assert report['outputs_match'] is True
        assert report['max_pixel_difference'] == report['mean_pixel_difference'] == 0
        # 三窗输入帧数为 4、5、4；每带八层，首次和正式各一带。
        assert report['multihead_calls'] == 48 and report['multihead_frame_visits'] == 208
        assert report['attention_branch_calls'] == 192
        assert report['attention_modules'] == report['qkv_fused_modules'] == 8
        assert report['precision_fallbacks'] == 0 and report['backend']['benchmark'] is False
        assert report['setup_seconds'] > 0 and report['warm_median_seconds'] > 0
        assert [row['phase'] for row in report['runs']] == ['first', 'measured']
        if device == 'cuda':
            assert report['peak_allocated_gib'] > 0
            assert report['peak_reserved_gib'] >= report['peak_allocated_gib']
        else:
            assert report['peak_allocated_gib'] is None
    assert baseline['output_cat_calls'] == 48 and baseline['direct_branch_copy_calls'] == 0
    assert candidate['output_cat_calls'] == 0 and candidate['direct_branch_copy_calls'] == 192
    assert benchmark.backend_settings() == settings
    assert not any('forward' in block.attention.__dict__ for block in engine.model.transformer)
    for actual, frame, mask in zip(reference, frames, masks):
        np.testing.assert_array_equal(actual[mask == 0], frame[mask == 0])
    for item, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(item, original)


def test_cached_shape_error_and_execution_error_restore_forward(engine):
    from benchmarks import sttn_attention_output as benchmark

    attention = engine.model.transformer[0].attention
    with pytest.raises(ValueError, match='shape'):
        with benchmark.attention_output_execution(engine.model, mode='direct'), torch.no_grad():
            attention(torch.randn(2, 256, 8, 8), torch.zeros(2, 1, 8, 8), 1, 256,
                      projected_qkv=(torch.zeros(1, 256, 8, 8),) * 3)
    assert not any('forward' in block.attention.__dict__ for block in engine.model.transformer)


@pytest.mark.parametrize('failure', ['pixels', 'fallback', 'nonfinite', 'inference'])
def test_invalid_measurement_cannot_pass_or_leave_wrappers(engine, failure):
    from benchmarks import sttn_attention_output as benchmark

    frames, masks = band()
    reference = np.stack(engine.inpaint(frames, masks)).astype(np.float32)
    if failure == 'pixels':
        reference[0, 0, 0, 0] += 0.25
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
    settings = benchmark.backend_settings()
    if failure == 'pixels':
        report, _ = benchmark.measure_output(engine, frames, masks, mode='direct',
                                               warmup=0, repeats=1, reference=reference)
        assert report['outputs_match'] is False and report['max_pixel_difference'] == 0.25
    else:
        with pytest.raises(RuntimeError, match='精度回退|非有限|injected'):
            benchmark.measure_output(engine, frames, masks, mode='direct',
                                     warmup=0, repeats=1, reference=reference)
    assert settings == benchmark.backend_settings()
    assert not any('forward' in block.attention.__dict__ for block in engine.model.transformer)


def test_direct_mode_needs_independent_reference(engine):
    from benchmarks import sttn_attention_output as benchmark

    with pytest.raises(ValueError, match='reference'):
        benchmark.measure_output(engine, *band(), mode='direct', warmup=0, repeats=1)


@pytest.mark.parametrize('gpu,match,speedup', [(False, True, None), (True, False, None), (True, True, 2)])
def test_summary_only_reports_speedup_for_matching_cuda_outputs(gpu, match, speedup):
    from benchmarks import sttn_attention_output as benchmark

    sessions = [dict(mode=mode, outputs_match=match, first_seconds=seconds + 1,
                     setup_seconds=0.01, peak_allocated_gib=1, peak_reserved_gib=2,
                     runs=[dict(phase='first', wall_seconds=100),
                           dict(phase='measured', wall_seconds=seconds)])
                for mode, seconds in [('cat', 2), ('direct', 1)]]
    report = benchmark.summarize(sessions, gpu_comparison=gpu)
    assert report['warm_speedup'] == speedup
    assert report['warm_time_saved_percent'] == (50 if speedup else None)


def test_cli_four_independent_reversed_workers_save_pixel_difference_before_failure(tmp_path, monkeypatch):
    from benchmarks import sttn_attention_output as benchmark

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
    assert report['outputs_match'] is False and report['max_pixel_difference'] == 0.25
    assert report['gpu_comparison'] is False and report['warm_speedup'] is None
    sessions = report['sessions']
    assert [row['mode'] for row in sessions] == ['cat', 'direct', 'direct', 'cat']
    assert len({row['environment']['pid'] for row in sessions} | {os.getpid()}) == 5
    for row in sessions:
        assert row['backend'] == sessions[0]['backend']
        assert row['precision_fallbacks'] == 0 and row['qkv_fused_modules'] == 8
        assert row['multihead_calls'] == 16 and row['multihead_frame_visits'] == 32
        assert row['attention_branch_calls'] == 64
        assert row['output_cat_calls'] == (16 if row['mode'] == 'cat' else 0)
        assert row['direct_branch_copy_calls'] == (64 if row['mode'] == 'direct' else 0)
        assert (output / row['log']).is_file()
