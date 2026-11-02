"""验证 Q/K/V 顺序、实际卷积合并、异常恢复及完整带 A/B 的拒绝规则。"""

import json
import os
from contextlib import nullcontext

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint.sttn.network_sttn import InpaintGenerator, MultiHeadedAttention
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from benchmarks import sttn_qkv_fusion as benchmark


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


def band(length=5):
    rng = np.random.default_rng(91)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(length)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    for mask in masks:
        mask[8:24, 8:24] = 255
    return frames, masks


@pytest.mark.parametrize('raise_inside', [False, True])
def test_fused_projection_runs_one_convolution_in_qkv_order_and_restores(raise_inside):
    model = MultiHeadedAttention([(1, 1)], 4).double().eval()
    with torch.no_grad():
        for layer, scale, bias in [(model.query_embedding, 1, 1),
                                   (model.key_embedding, 2, 3),
                                   (model.value_embedding, -1, 5)]:
            layer.weight.zero_()
            layer.weight[:, :, 0, 0].copy_(torch.eye(4) * scale)
            layer.bias.fill_(bias)
    weights = {name: (param, param.detach().clone(), param.stride())
               for name, param in model.named_parameters()}
    # 原有实例覆盖也须原样恢复，类定义不应变化。
    original = model.project_qkv
    model.project_qkv = original
    x = torch.arange(32, dtype=torch.float64).reshape(2, 4, 2, 2)
    for mode, convolution_count in [('separate', 3), ('fused', 1)]:
        with pytest.raises(RuntimeError, match='injected') if raise_inside else nullcontext():
            with benchmark.qkv_execution(model, mode=mode) as state:
                with torch.no_grad(), torch.profiler.profile() as profile:
                    query, key, value = model.project_qkv(x)
                assert sum(event.count for event in profile.key_averages()
                           if event.key == 'aten::conv2d') == convolution_count
                torch.testing.assert_close(query, x + 1, rtol=0, atol=0)
                torch.testing.assert_close(key, x * 2 + 3, rtol=0, atol=0)
                torch.testing.assert_close(value, -x + 5, rtol=0, atol=0)
                assert state['qkv_calls'] == 1
                assert state['qkv_conv_calls'] == convolution_count
                if raise_inside:
                    raise RuntimeError('injected')
        assert model.__dict__['project_qkv'] is original
        for name, param in model.named_parameters():
            assert param is weights[name][0]
            assert param.stride() == weights[name][2]
            torch.testing.assert_close(param, weights[name][1], rtol=0, atol=0)


def test_partial_setup_failure_restores_preceding_projections():
    model = torch.nn.ModuleList([MultiHeadedAttention([(1, 1)], 4) for _ in range(2)]).eval()
    model[1].key_embedding = torch.nn.Conv2d(4, 4, 3)
    with pytest.raises(ValueError, match='1×1'):
        with benchmark.qkv_execution(model, mode='fused'):
            pytest.fail('unsupported convolution was accepted')
    assert all('project_qkv' not in layer.__dict__ for layer in model)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
def test_full_band_fusion_preserves_inputs_cache_and_strict_reference(engine, cached, device):
    engine.device = torch.device(device)
    engine.model.to(engine.device)
    engine.cache_first_qkv = cached
    frames, masks = band()
    originals = [array.copy() for array in frames + masks]
    settings = benchmark.backend_settings()
    separate, reference = benchmark.measure_qkv(engine, frames, masks, mode='separate',
                                              warmup=1, repeats=2)
    fused, returned = benchmark.measure_qkv(engine, frames, masks, mode='fused',
                                          warmup=1, repeats=2, reference=reference)
    for report, multiplier in [(separate, 3), (fused, 1)]:
        assert report['qkv_modules'] == 8
        # 5 帧/步长 2 有三个窗口；缓存首层时 1+7*3=22，否则 8*3=24。
        assert report['qkv_calls'] == (88 if cached else 96)
        assert report['qkv_conv_calls'] == report['qkv_calls'] * multiplier
        assert report['backend']['benchmark'] is False
        assert report['setup_seconds'] > 0
        assert [row['phase'] for row in report['runs']] == ['first', 'warmup', 'measured', 'measured']
        assert report['warm_median_seconds'] > 0
        assert report['precision_fallbacks'] == 0
        assert report['outputs_match'] == (report['max_pixel_difference'] == 0)
    assert returned is reference
    for result, frame, mask in zip(reference, frames, masks):
        np.testing.assert_array_equal(result[mask == 0], frame[mask == 0])
    for array, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(array, original)
    assert not any('project_qkv' in module.__dict__ for module in engine.model.modules())
    assert benchmark.backend_settings() == settings
    if device == 'cuda':
        assert fused['peak_allocated_gib'] > 0
        assert fused['peak_reserved_gib'] >= fused['peak_allocated_gib']
        assert fused['setup_peak_allocated_gib'] > 0
    else:
        assert fused['peak_allocated_gib'] is None


@pytest.mark.parametrize('failure', ['pixels', 'fallback', 'nonfinite'])
def test_invalid_outputs_cannot_pass_and_projection_is_restored(engine, failure):
    frames, masks = band(3)
    reference = np.stack(engine.inpaint(frames, masks)).astype(np.float32)
    if failure == 'pixels':
        reference[0, 12, 12, 0] = 1000.5
    else:
        original = engine.inpaint

        def invalid(images, holes):
            result = original(images, holes)
            if failure == 'fallback':
                engine.precision_fallbacks += 1
            else:
                result[0] = result[0].astype(np.float32)
                result[0][12, 12, 0] = np.nan
            return result

        engine.inpaint = invalid
    settings = benchmark.backend_settings()
    if failure == 'pixels':
        report, _ = benchmark.measure_qkv(engine, frames, masks, mode='fused',
                                         warmup=0, repeats=1, reference=reference)
        assert report['outputs_match'] is False
        assert report['max_pixel_difference'] >= 745.5
    else:
        with pytest.raises(RuntimeError, match='精度回退|非有限'):
            benchmark.measure_qkv(engine, frames, masks, mode='fused',
                                  warmup=0, repeats=1, reference=reference)
    assert not any('project_qkv' in module.__dict__ for module in engine.model.modules())
    assert benchmark.backend_settings() == settings


@pytest.mark.parametrize('gpu,match,speedup', [(False, True, None), (True, False, None), (True, True, 2)])
def test_summary_only_reports_speedup_for_matching_gpu_outputs(gpu, match, speedup):
    sessions = [dict(mode=mode, outputs_match=match, first_seconds=seconds + 1,
                     setup_seconds=0.1, peak_allocated_gib=1, peak_reserved_gib=2,
                     runs=[dict(phase='first', wall_seconds=100),
                           dict(phase='measured', wall_seconds=seconds)])
                for mode, seconds in [('separate', 2), ('fused', 1)]]
    result = benchmark.summarize(sessions, gpu_comparison=gpu)
    assert result['warm_speedup'] == speedup
    assert result['warm_time_saved_percent'] == (50 if speedup else None)
    assert result['summary']['separate']['warm_median_seconds'] == 2


def test_cli_runs_independent_processes_in_reversed_order(tmp_path):
    torch.manual_seed(91)
    model = tmp_path / 'sttn.pth'
    torch.save({'netG': InpaintGenerator(init_weights=False).state_dict()}, model)
    output = tmp_path / 'result'
    try:
        benchmark.main(['--device', 'cpu', '--precision', 'fp32', '--model', str(model),
                        '--frames', '2', '--warmup', '0', '--repeats', '1',
                        '--sessions', '2', '--output-dir', str(output)])
    except SystemExit as exc:
        assert '像素不一致' in str(exc)
        assert json.loads((output / 'report.json').read_text())['outputs_match'] is False
    report = json.loads((output / 'report.json').read_text())
    assert report['gpu_comparison'] is False
    assert report['warm_speedup'] is None
    sessions = report['sessions']
    assert [row['mode'] for row in sessions] == ['separate', 'fused', 'fused', 'separate']
    assert len({row['environment']['pid'] for row in sessions} | {os.getpid()}) == 5
    for row in sessions:
        assert row['backend'] == sessions[0]['backend']
        assert row['precision_fallbacks'] == 0
        assert row['qkv_calls'] == 16
        assert row['qkv_conv_calls'] == (16 if row['mode'] == 'fused' else 48)
        assert (output / row['log']).is_file()
    assert (output / 'reference.npy').is_file()
