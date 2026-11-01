"""cuDNN A/B 必须隔离进程，保留首次成本和数值差异，不能默许精度回退。"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint.sttn.network_sttn import InpaintGenerator
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from benchmarks import sttn_cudnn_benchmark as benchmark


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


def inputs():
    rng = np.random.default_rng(91)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    for mask in masks:
        mask[8:24, 8:24] = 255
    return frames, masks


def test_band_measurement_records_first_and_warm_runs_without_changing_inputs(engine):
    frames, masks = inputs()
    original = [array.copy() for array in frames + masks]
    expected = engine.inpaint(frames, masks)
    flags = benchmark.backend_settings()
    report, reference = benchmark.measure_mode(engine, frames, masks, enabled=not flags['benchmark'],
                                                warmup=1, repeats=2)
    assert [row['phase'] for row in report['runs']] == ['first', 'warmup', 'measured', 'measured']
    assert all(row['wall_seconds'] > 0 for row in report['runs'])
    assert report['warm_median_seconds'] > 0
    assert report['outputs_match'] is True
    assert report['max_pixel_difference'] == 0
    assert report['backend']['benchmark'] is not flags['benchmark']
    assert {key: value for key, value in report['backend'].items() if key != 'benchmark'} == {
        key: value for key, value in flags.items() if key != 'benchmark'}
    assert benchmark.backend_settings() == flags
    assert report['peak_allocated_gib'] is None
    assert report['peak_reserved_gib'] is None
    for actual, wanted, frame, mask in zip(reference, expected, frames, masks):
        np.testing.assert_array_equal(actual, wanted)
        np.testing.assert_array_equal(actual[mask == 0], frame[mask == 0])
    assert any(np.any(actual != frame) for actual, frame in zip(reference, frames))
    for array, wanted in zip(frames + masks, original):
        np.testing.assert_array_equal(array, wanted)


def test_all_phases_compare_against_external_baseline_including_fractional_pixels(engine):
    frames, masks = inputs()
    reference = np.stack(engine.inpaint(frames, masks)).astype(np.float32)
    reference[0, 12, 12, 0] += .5
    report, _ = benchmark.measure_mode(engine, frames, masks, enabled=True,
                                      warmup=1, repeats=2, reference=reference)
    assert report['outputs_match'] is False
    assert report['max_pixel_difference'] == .5
    assert all(row['max_pixel_difference'] == .5 for row in report['runs'])
    assert report['mean_pixel_difference'] == pytest.approx(.5 / reference.size)


@pytest.mark.parametrize('failure', ['exception', 'fallback', 'nonfinite'])
def test_invalid_measurements_fail_and_restore_global_settings(engine, failure):
    frames, masks = inputs()
    original = engine.inpaint

    def invalid(images, holes):
        if failure == 'exception':
            raise RuntimeError('injected')
        output = original(images, holes)
        if failure == 'fallback':
            engine.precision_fallbacks += 1
        else:
            output[0] = output[0].astype(np.float32)
            output[0][12, 12, 0] = np.nan
        return output

    engine.inpaint = invalid
    flags = benchmark.backend_settings()
    with pytest.raises(RuntimeError, match='injected|精度回退|非有限'):
        benchmark.measure_mode(engine, frames, masks, enabled=not flags['benchmark'],
                               warmup=0, repeats=1)
    assert benchmark.backend_settings() == flags


@pytest.mark.parametrize('changed_reference', [False, True])
def test_cli_runs_isolated_modes_and_rejects_pixel_changes(tmp_path, monkeypatch, changed_reference):
    # 标准 STTN 结构和真实推理，固定随机权重避免测试依赖本地模型文件。
    torch.manual_seed(91)
    model = tmp_path / 'sttn.pth'
    torch.save({'netG': InpaintGenerator(init_weights=False).state_dict()}, model)
    output = tmp_path / 'result'
    command = ['--device', 'cpu', '--precision', 'fp32', '--model', str(model),
               '--frames', '2', '--warmup', '0', '--repeats', '1',
               '--sessions', '1' if changed_reference else '2', '--output-dir', str(output)]
    if changed_reference:
        # 只在真实子进程边界注入损坏参考值；模型、计时、文件报告仍全部运行。
        launch = benchmark.subprocess.run

        def corrupt_reference_then_launch(*args, **kwargs):
            path = output / 'reference.npy'
            if path.exists():
                reference = np.load(path, allow_pickle=False)
                reference[0, 0, 0, 0] += .5
                np.save(path, reference, allow_pickle=False)
            return launch(*args, **kwargs)

        monkeypatch.setattr(benchmark.subprocess, 'run', corrupt_reference_then_launch)
        with pytest.raises(SystemExit, match='像素不一致'):
            benchmark.main(command)
    else:
        benchmark.main(command)
    report = json.loads((output / 'report.json').read_text())
    assert report['outputs_match'] is not changed_reference
    assert report['max_pixel_difference'] == (.5 if changed_reference else 0)
    assert report['gpu_comparison'] is False
    assert report['warm_speedup'] is None
    workers = report['sessions']
    assert [worker['mode'] for worker in workers] == (
        ['off', 'on'] if changed_reference else ['off', 'on', 'on', 'off'])
    assert len({worker['environment']['pid'] for worker in workers} | {os.getpid()}) == len(workers) + 1
    assert {worker['environment']['torch_threads'] for worker in workers} == {1}
    for worker in workers:
        assert worker['model_load_seconds'] > 0
        assert worker['process_wall_seconds'] > worker['model_load_seconds']
        assert worker['outputs_match'] is (not changed_reference or worker['mode'] == 'off')
        assert worker['precision_fallbacks'] == 0
        assert len(worker['runs']) == 2
        assert (output / worker['log']).exists()
    assert workers[0]['backend']['benchmark'] is False
    assert workers[1]['backend']['benchmark'] is True
    assert (output / 'reference.npy').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_measurement_records_first_and_warm_memory_peaks(engine):
    engine.device = torch.device('cuda')
    engine.model.to(engine.device)
    frames, masks = inputs()
    flags = benchmark.backend_settings()
    report, _ = benchmark.measure_mode(engine, frames, masks, enabled=True, warmup=1, repeats=2)
    assert report['peak_allocated_gib'] > 0
    assert report['peak_reserved_gib'] >= report['peak_allocated_gib']
    for run in report['runs']:
        assert run['peak_allocated_gib'] > 0
        assert run['peak_reserved_gib'] >= run['peak_allocated_gib']
    assert benchmark.backend_settings() == flags
