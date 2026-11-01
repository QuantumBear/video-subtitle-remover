"""FFN 编译实验必须执行编译图、检查整带输出，并恢复生产实例。"""

import json
import os
from contextlib import nullcontext

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint.sttn.network_sttn import InpaintGenerator
from backend.inpaint.sttn.network_sttn import FeedForward
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from benchmarks import sttn_ffn_compile as benchmark

requires_compile = pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(), reason='torch.compile unsupported by this Python/PyTorch')


@pytest.fixture
def engine():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(81)
    result = STTNDetInpaint.__new__(STTNDetInpaint)
    result.device = torch.device('cpu')
    result.precision = result.requested_precision = 'fp32'
    result.precision_fallbacks = 0
    result.profile = False
    result.model = InpaintGenerator(init_weights=False).eval()
    for block in result.model.transformer:
        block.attention.patchsize = [(8, 8), (4, 4), (2, 2), (1, 1)]
    result.model_input_width = result.model_input_height = 32
    result.neighbor_stride, result.ref_length = 2, 3
    result.cache_first_qkv = True
    yield result
    torch.set_num_threads(previous_threads)


def make_bands():
    rng = np.random.default_rng(91)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    for mask in masks:
        mask[8:24, 8:24] = 255
    return {3: (frames, masks), 2: (frames[:2], masks[:2])}


@pytest.mark.parametrize('raise_inside', [False, True])
def test_eager_baseline_restores_instance_forward_even_on_failure(engine, raise_inside):
    ffns = [block.feed_forward for block in engine.model.transformer]
    original = ffns[0].forward
    ffns[0].forward = original
    weights = {name: (weight, weight.detach().clone(), weight.stride())
               for name, weight in engine.model.named_parameters()}
    flags = benchmark.backend_settings()
    with pytest.raises(RuntimeError, match='injected') if raise_inside else nullcontext():
        with benchmark.ffn_execution(engine.model, mode='eager', backend='eager') as state:
            with torch.no_grad():
                for ffn in ffns:
                    ffn(torch.randn(2, 256, 8, 8))
            assert state['ffn_calls'] == 8
            assert state['compiled_calls'] == 0
            if raise_inside:
                raise RuntimeError('injected')
    assert ffns[0].__dict__['forward'] is original
    assert not any('forward' in ffn.__dict__ for ffn in ffns[1:])
    assert benchmark.backend_settings() == flags
    for name, param in engine.model.named_parameters():
        assert param is weights[name][0]
        assert param.stride() == weights[name][2]
        torch.testing.assert_close(param, weights[name][1], rtol=0, atol=0)


@requires_compile
def test_compiled_bands_execute_graphs_check_shapes_and_reuse_on_return(engine):
    torch._dynamo.reset()
    bands = make_bands()
    originals = [array.copy() for frames, masks in bands.values() for array in frames + masks]
    _, references = benchmark.measure_compile(engine, bands, mode='eager', backend='eager',
                                              warmup=0, repeats=1)
    flags = benchmark.backend_settings()
    cache_limit = torch._dynamo.config.cache_size_limit if hasattr(torch._dynamo.config, 'cache_size_limit') else None
    report, _ = benchmark.measure_compile(engine, bands, mode='compiled', backend='eager',
                                          warmup=1, repeats=2, references=references)
    assert report['outputs_match'] is True
    assert report['max_pixel_difference'] == 0
    assert report['ffn_modules'] == 8
    assert report['backend_compilations'] > 0
    assert report['compiled_calls'] == report['ffn_calls'] > 0
    assert [(row['visit'], row['frames']) for row in report['bands']] == [
        ('initial', 3), ('shape', 2), ('revisit', 3)]
    assert report['bands'][0]['runs'][0]['backend_compilations'] > 0
    assert report['bands'][-1]['backend_compilations'] == 0
    for band in report['bands']:
        assert [row['phase'] for row in band['runs']] == ['first', 'warmup', 'measured', 'measured']
        assert band['warm_median_seconds'] > 0
        for run in band['runs']:
            assert run['ffn_calls'] == (16 if band['frames'] == 3 else 8)
            assert run['compiled_calls'] == run['ffn_calls']
            assert run['peak_allocated_gib'] is None
        for result, frame, mask in zip(references[band['frames']], *bands[band['frames']]):
            np.testing.assert_array_equal(result[mask == 0], frame[mask == 0])
    for array, original in zip((array for frames, masks in bands.values()
                               for array in frames + masks), originals):
        np.testing.assert_array_equal(array, original)
    assert not any('forward' in block.feed_forward.__dict__ for block in engine.model.transformer)
    assert benchmark.backend_settings() == flags
    if cache_limit is not None:
        assert torch._dynamo.config.cache_size_limit == cache_limit


@pytest.mark.parametrize('failure', ['pixels', 'fallback', 'nonfinite'])
def test_invalid_band_outputs_are_rejected_and_wrappers_restored(engine, failure):
    bands = make_bands()
    _, references = benchmark.measure_compile(engine, bands, mode='eager', backend='eager',
                                              warmup=0, repeats=1)
    if failure == 'pixels':
        references[3][0, 12, 12, 0] = 1000.5
    else:
        original = engine.inpaint

        def invalid(frames, masks):
            result = original(frames, masks)
            if failure == 'fallback':
                engine.precision_fallbacks += 1
            else:
                result[0] = result[0].astype(np.float32)
                result[0][12, 12, 0] = np.nan
            return result

        engine.inpaint = invalid
    flags = benchmark.backend_settings()
    if failure == 'pixels':
        report, _ = benchmark.measure_compile(engine, bands, mode='eager', backend='eager',
                                              warmup=0, repeats=1, references=references)
        assert report['outputs_match'] is False
        assert report['max_pixel_difference'] >= 745.5
    else:
        with pytest.raises(RuntimeError, match='精度回退|非有限'):
            benchmark.measure_compile(engine, bands, mode='eager', backend='eager',
                                      warmup=0, repeats=1, references=references)
    assert not any('forward' in block.feed_forward.__dict__ for block in engine.model.transformer)
    assert benchmark.backend_settings() == flags


@requires_compile
def test_compile_failure_is_not_silently_accepted_and_restores_forward(engine, monkeypatch):
    from torch._dynamo.backends import registry

    def broken_backend(graph, inputs):
        raise RuntimeError('injected compiler failure')

    monkeypatch.setattr(registry, 'lookup_backend', lambda name: broken_backend)
    bands = make_bands()
    _, references = benchmark.measure_compile(engine, bands, mode='eager', backend='eager',
                                              warmup=0, repeats=1)
    torch._dynamo.reset()
    with pytest.raises(RuntimeError, match='injected compiler failure'):
        benchmark.measure_compile(engine, bands, mode='compiled', backend='eager',
                                  warmup=0, repeats=1, references=references)
    assert not any('forward' in block.feed_forward.__dict__ for block in engine.model.transformer)


@requires_compile
def test_inductor_executes_ffn_and_restores_weights_and_forward():
    torch._dynamo.reset()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    ffn = FeedForward(4).eval()
    x = torch.randn(2, 4, 8, 8)
    weights = {name: (param, param.detach().clone()) for name, param in ffn.named_parameters()}
    try:
        with torch.no_grad():
            expected = ffn(x)
            with benchmark.ffn_execution(ffn, mode='compiled', backend='inductor') as state:
                actual = ffn(x)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                ffn(x)
                assert state['compiled_calls'] == state['ffn_calls'] == 2
                assert state['backend_compilations'] == 1
    finally:
        torch.set_num_threads(previous_threads)
    assert 'forward' not in ffn.__dict__
    for name, param in ffn.named_parameters():
        assert param is weights[name][0]
        torch.testing.assert_close(param, weights[name][1], rtol=0, atol=0)


@pytest.mark.parametrize('case', ['valid', 'pixels', 'warm_compile', 'cpu'])
def test_summary_only_reports_speedup_for_stable_matching_gpu_bands(case):
    sessions = []
    for mode, seconds in [('eager', 2.0), ('compiled', 1.0)]:
        runs = [dict(phase='first', wall_seconds=20.0, backend_compilations=8,
                     peak_allocated_gib=1.0, peak_reserved_gib=2.0),
                dict(phase='measured', wall_seconds=seconds,
                     backend_compilations=1 if mode == 'compiled' and case == 'warm_compile' else 0,
                     peak_allocated_gib=1.0, peak_reserved_gib=2.0)]
        sessions.append(dict(mode=mode, outputs_match=case != 'pixels', compile_backend='inductor',
                             bands=[dict(visit='initial', frames=50, first_seconds=20.0,
                                         backend_compilations=sum(row['backend_compilations'] for row in runs),
                                         runs=runs)]))
    summary = benchmark.summarize(sessions, gpu_comparison=case != 'cpu')
    assert summary[0]['warm_stable'] is (case != 'warm_compile')
    assert summary[0]['warm_speedup'] == (2.0 if case == 'valid' else None)
    assert summary[0]['warm_time_saved_percent'] == (50.0 if case == 'valid' else None)


@requires_compile
def test_cli_compares_independent_processes_and_saves_shape_references(tmp_path):
    torch.manual_seed(91)
    model = tmp_path / 'sttn.pth'
    torch.save({'netG': InpaintGenerator(init_weights=False).state_dict()}, model)
    output = tmp_path / 'result'
    try:
        benchmark.main(['--device', 'cpu', '--precision', 'fp32', '--compile-backend', 'eager',
                        '--model', str(model), '--frames', '3', '--shape-frames', '2',
                        '--warmup', '0', '--repeats', '1', '--sessions', '1',
                        '--output-dir', str(output)])
    except SystemExit as exc:
        assert '像素不一致' in str(exc)
    report = json.loads((output / 'report.json').read_text())
    assert report['gpu_comparison'] is False
    assert report['summary'][0]['warm_speedup'] is None
    sessions = report['sessions']
    assert [row['mode'] for row in sessions] == ['eager', 'compiled']
    assert len({row['environment']['pid'] for row in sessions} | {os.getpid()}) == 3
    assert sessions[0]['backend'] == sessions[1]['backend']
    assert sessions[0]['backend']['benchmark'] is False
    assert sessions[1]['backend_compilations'] > 0
    for row in sessions:
        assert row['precision_fallbacks'] == 0
        assert (output / row['log']).is_file()
        assert str(output) in row['environment']['inductor_cache_dir']
    with np.load(output / 'reference.npz', allow_pickle=False) as reference:
        assert set(reference.files) == {'3', '2'}
