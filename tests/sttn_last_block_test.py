"""末层裁剪必须保留全部参考上下文，并只省去不被解码的逐帧计算。"""

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


def band(count=5):
    rng = np.random.default_rng(81)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(count)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    masks[1].fill(255)
    masks[-1][8:24, 8:24] = 255
    return frames, masks


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
def test_only_last_output_and_ffn_drop_reference_frames(engine, cached, device):
    from benchmarks import sttn_last_block as benchmark
    engine.device = torch.device(device)
    engine.model.to(engine.device)
    engine.precision = engine.requested_precision = 'fp16' if device == 'cuda' else 'fp32'
    engine.cache_first_qkv = cached
    frames, masks = band()
    originals = [value.copy() for value in frames + masks]
    default_output = engine.inpaint(frames, masks)
    baseline, reference = benchmark.measure_last_block(engine, frames, masks, mode='full',
                                                       warmup=0, repeats=1)
    seen = {'early': [], 'qkv': [], 'output': [], 'ffn': [], 'queries': [], 'keys': []}
    last = engine.model.transformer[-1]
    handles = []
    for name, module in [('early', engine.model.transformer[-2].feed_forward),
                         ('qkv', last.attention),
                         ('output', last.attention.output_linear[0]), ('ffn', last.feed_forward)]:
        handles.append(module.register_forward_pre_hook(
            lambda module, args, name=name: seen[name].append(args[0].shape[0])))
    def attention_shapes(module, args):
        seen['queries'].append(args[0].shape)
        seen['keys'].append(args[1].shape)

    handles.append(last.attention.attention.register_forward_pre_hook(attention_shapes))
    try:
        selected, returned = benchmark.measure_last_block(
            engine, frames, masks, mode='targets', warmup=0, repeats=1, reference=reference)
    finally:
        for handle in handles:
            handle.remove()
    assert returned is reference
    np.testing.assert_array_equal(np.stack(default_output), reference)
    for report in (baseline, selected):
        assert report['outputs_match'] is True
        assert report['max_pixel_difference'] == report['mean_pixel_difference'] == 0
        assert report['precision_fallbacks'] == 0
        assert report['qkv_fused_modules'] == 8
        assert report['backend']['benchmark'] is False
        assert report['last_block_calls'] == 6
        assert report['last_attention_frame_visits'] == 26
        assert report['target_frame_visits'] == 22
    assert baseline['last_output_frame_visits'] == baseline['last_ffn_frame_visits'] == 26
    assert selected['last_output_frame_visits'] == selected['last_ffn_frame_visits'] == 22
    assert baseline['pruned_frame_visits'] == 0
    assert selected['pruned_frame_visits'] == 4
    assert seen['early'] == [4, 5, 4] * 2
    assert seen['output'] == [3, 5, 3] * 2
    assert seen['ffn'] == [3, 5, 3] * 2
    assert all(q[1] == k[1] for q, k in zip(seen['queries'], seen['keys']))
    assert len(seen['queries']) == 24
    assert seen['qkv'] == [4, 5, 4] * 2
    for item, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(item, original)
    for actual, frame, mask in zip(reference, frames, masks):
        np.testing.assert_array_equal(actual[mask == 0], frame[mask == 0])


def test_last_features_keep_reference_influence_and_match_full_before_quantization(engine):
    from benchmarks import sttn_last_block as benchmark
    features = torch.randn(4, 256, 8, 8)
    masks = torch.zeros(4, 1, 32, 32)
    with torch.no_grad():
        expected = engine.model.infer(features, masks)[:3]
        without_reference = engine.model.infer(features[:3], masks[:3])
        with benchmark.last_block_execution(engine, mode='targets'):
            assert engine.get_ref_index([0, 1, 2], 5) == [3]
            actual = engine.model.infer(features, masks)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape == (3, 256, 8, 8)
    assert not torch.allclose(actual, without_reference)


@pytest.mark.parametrize('raise_inside', [False, True])
@pytest.mark.parametrize('existing_override', [False, True])
def test_all_instance_overrides_restore_even_after_exception(engine, raise_inside, existing_override):
    from benchmarks import sttn_last_block as benchmark
    last = engine.model.transformer[-1]
    owners = [(engine, 'get_ref_index'), (engine.model, 'infer'), (last, 'forward'),
              (last.attention.output_linear, 'forward'), (last.feed_forward, 'forward')]
    if existing_override:
        for owner, name in owners:
            setattr(owner, name, getattr(owner, name))
    before = [(name in owner.__dict__, getattr(owner, name)) for owner, name in owners]
    with pytest.raises(RuntimeError, match='injected') if raise_inside else nullcontext():
        with benchmark.last_block_execution(engine, mode='targets'):
            engine.inpaint(*band())
            if raise_inside:
                raise RuntimeError('injected')
    for (owner, name), (existed, method) in zip(owners, before):
        assert (name in owner.__dict__) == existed
        assert getattr(owner, name) == method


def test_targets_requires_reference_and_a_valid_production_window(engine):
    from benchmarks import sttn_last_block as benchmark
    frames, masks = band()
    with pytest.raises(ValueError, match='独立 full'):
        benchmark.measure_last_block(engine, frames, masks, mode='targets', warmup=0, repeats=1)
    features = torch.randn(4, 256, 8, 8)
    holes = torch.zeros(4, 1, 32, 32)
    with torch.no_grad(), benchmark.last_block_execution(engine, mode='targets'):
        with pytest.raises(RuntimeError, match='生产窗口'):
            engine.model.infer(features, holes)
        engine.get_ref_index([0, 1, 2], 5)
        with pytest.raises(RuntimeError, match='形状不一致'):
            engine.model.infer(features[:3], holes[:3])


def test_no_reference_window_does_not_drop_any_frames(engine):
    from benchmarks import sttn_last_block as benchmark
    frames, masks = band(2)
    full, reference = benchmark.measure_last_block(engine, frames, masks, mode='full', warmup=0, repeats=1)
    selected, _ = benchmark.measure_last_block(engine, frames, masks, mode='targets',
                                               warmup=0, repeats=1, reference=reference)
    assert selected['outputs_match'] is True
    assert selected['pruned_frame_visits'] == 0
    assert full['last_output_frame_visits'] == selected['last_output_frame_visits'] == 4


@pytest.mark.parametrize('failure', ['pixels', 'fallback', 'nonfinite', 'inference'])
def test_invalid_measurement_is_rejected_and_methods_restored(engine, failure):
    from benchmarks import sttn_last_block as benchmark
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
    previous = engine.model.infer
    if failure == 'pixels':
        report, _ = benchmark.measure_last_block(engine, frames, masks, mode='targets',
                                                 warmup=0, repeats=1, reference=reference)
        assert report['outputs_match'] is False
        assert report['max_pixel_difference'] == 0.25
    else:
        with pytest.raises(RuntimeError, match='精度回退|非有限|injected'):
            benchmark.measure_last_block(engine, frames, masks, mode='targets',
                                          warmup=0, repeats=1, reference=reference)
    assert engine.model.infer == previous


@pytest.mark.parametrize('gpu,match,pruned,speedup', [
    (False, True, 2, None), (True, False, 2, None),
    (True, True, 0, None), (True, True, 2, 2)])
def test_summary_only_claims_speedup_for_matching_gpu_with_pruned_work(gpu, match, pruned, speedup):
    from benchmarks import sttn_last_block as benchmark
    sessions = [dict(mode=mode, outputs_match=match, first_seconds=seconds + 1,
                     setup_seconds=0.01, peak_allocated_gib=1, peak_reserved_gib=2,
                     pruned_frame_visits=pruned if mode == 'targets' else 0,
                     runs=[dict(phase='first', wall_seconds=100),
                           dict(phase='measured', wall_seconds=seconds)])
                for mode, seconds in [('full', 2), ('targets', 1)]]
    result = benchmark.summarize(sessions, gpu_comparison=gpu)
    assert result['warm_speedup'] == speedup
    assert result['warm_time_saved_percent'] == (50 if speedup else None)
    assert result['pruning_effective'] is bool(pruned)


def test_cli_separate_processes_reversed_order_and_report_before_pixel_failure(tmp_path, monkeypatch):
    from benchmarks import sttn_last_block as benchmark
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
    assert report['outputs_match'] is False
    assert report['max_pixel_difference'] >= 0.25
    assert report['warm_speedup'] is None
    sessions = report['sessions']
    assert [row['mode'] for row in sessions] == ['full', 'targets', 'targets', 'full']
    assert len({row['environment']['pid'] for row in sessions} | {os.getpid()}) == 5
    assert all(row['qkv_fused_modules'] == 8 and row['precision_fallbacks'] == 0 for row in sessions)
    assert all((output / row['log']).is_file() for row in sessions)
