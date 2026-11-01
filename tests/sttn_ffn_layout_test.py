"""只改 FFN 布局，确保转换生效、退出恢复，完整带基准仍严格检查输出。"""

import json
import os
from contextlib import nullcontext

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint.sttn.network_sttn import InpaintGenerator
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from benchmarks import sttn_ffn_layout as benchmark


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


@pytest.mark.parametrize('raise_inside', [False, True])
def test_only_ffn_convolutions_use_channels_last_and_restore_on_exit(engine, raise_inside):
    model = engine.model
    weights = {name: (weight, weight.detach().clone(), weight.stride())
               for name, weight in model.named_parameters()}
    ffns = [block.feed_forward for block in model.transformer]
    # 已有实例 forward 覆盖也必须恢复，且不修改 FeedForward 类。
    original_forward = ffns[0].forward
    ffns[0].forward = original_forward
    observed = []
    handles = []

    def check_layout(module, args, output):
        observed.append((args[0].is_contiguous(memory_format=torch.channels_last),
                         module.weight.is_contiguous(memory_format=torch.channels_last),
                         output.is_contiguous(memory_format=torch.channels_last)))

    for ffn in ffns:
        handles.extend(layer.register_forward_hook(check_layout)
                       for layer in ffn.conv if isinstance(layer, torch.nn.Conv2d))
    try:
        with pytest.raises(RuntimeError, match='injected') if raise_inside else nullcontext():
            with benchmark.ffn_channels_last(model):
                for name, param in model.named_parameters():
                    if 'feed_forward' not in name:
                        assert param.stride() == weights[name][2]
                        assert param is weights[name][0]
                with torch.no_grad():
                    for ffn in ffns:
                        result = ffn(torch.randn(3, 256, 8, 8))
                        assert result.is_contiguous()
                assert observed == [(True, True, True)] * 16
                if raise_inside:
                    raise RuntimeError('injected')
    finally:
        for handle in handles:
            handle.remove()
    for name, param in model.named_parameters():
        assert param is weights[name][0]
        assert param.stride() == weights[name][2]
        torch.testing.assert_close(param, weights[name][1], rtol=0, atol=0)
    assert ffns[0].__dict__['forward'] is original_forward
    assert not any('forward' in ffn.__dict__ for ffn in ffns[1:])


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
def test_full_band_measures_layout_with_conversion_and_keeps_clean_pixels(engine, device):
    engine.device = torch.device(device)
    engine.model.to(engine.device)
    rng = np.random.default_rng(91)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    for mask in masks:
        mask[8:24, 8:24] = 255
    originals = [array.copy() for array in frames + masks]
    flags = benchmark.backend_settings()
    report, reference = benchmark.measure_layout(engine, frames, masks, mode='channels-last',
                                                warmup=1, repeats=2)
    assert report['mode'] == 'channels-last'
    assert report['backend']['benchmark'] is False
    assert report['ffn_modules'] == 8
    assert report['layout_setup_seconds'] > 0
    assert [row['phase'] for row in report['runs']] == ['first', 'warmup', 'measured', 'measured']
    assert report['warm_median_seconds'] > 0
    assert report['precision_fallbacks'] == 0
    assert not any('forward' in module.__dict__ for module in engine.model.modules())
    assert benchmark.backend_settings() == flags
    for result, frame, mask in zip(reference, frames, masks):
        np.testing.assert_array_equal(result[mask == 0], frame[mask == 0])
    for array, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(array, original)
    if device == 'cuda':
        assert report['peak_allocated_gib'] > 0
        assert report['peak_reserved_gib'] >= report['peak_allocated_gib']
        assert report['layout_setup_peak_allocated_gib'] > 0
    else:
        assert report['peak_allocated_gib'] is None


@pytest.mark.parametrize('failure', ['pixels', 'fallback', 'nonfinite'])
def test_bad_outputs_cannot_be_accepted_and_layout_is_restored(engine, failure):
    frames = [np.full((32, 32, 3), 127, dtype=np.uint8) for _ in range(3)]
    masks = [np.full((32, 32), 255, dtype=np.uint8) for _ in frames]
    reference = np.stack(engine.inpaint(frames, masks)).astype(np.float32)
    if failure == 'pixels':
        reference[0, 12, 12, 0] = 1000.5  # uint8 输出及半值融合都不可能达到。
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
    flags = benchmark.backend_settings()
    if failure == 'pixels':
        report, _ = benchmark.measure_layout(engine, frames, masks, mode='channels-last',
                                            warmup=0, repeats=1, reference=reference)
        assert report['outputs_match'] is False
        assert report['max_pixel_difference'] >= 745.5
    else:
        with pytest.raises(RuntimeError, match='精度回退|非有限'):
            benchmark.measure_layout(engine, frames, masks, mode='channels-last',
                                     warmup=0, repeats=1, reference=reference)
    assert not any('forward' in module.__dict__ for module in engine.model.modules())
    assert all(param.is_contiguous() for param in engine.model.parameters())
    assert benchmark.backend_settings() == flags


def test_cli_uses_independent_processes_for_both_layouts(tmp_path):
    torch.manual_seed(91)
    model = tmp_path / 'sttn.pth'
    torch.save({'netG': InpaintGenerator(init_weights=False).state_dict()}, model)
    output = tmp_path / 'result'
    # CPU 卷积换布局同样可能有数值差异；若发生，仍须落盘并拒绝。
    try:
        benchmark.main(['--device', 'cpu', '--precision', 'fp32', '--model', str(model),
                        '--frames', '2', '--warmup', '0', '--repeats', '1',
                        '--sessions', '1', '--output-dir', str(output)])
    except SystemExit as exc:
        assert '像素不一致' in str(exc)
        assert json.loads((output / 'report.json').read_text())['outputs_match'] is False
    report = json.loads((output / 'report.json').read_text())
    assert report['gpu_comparison'] is False
    assert report['warm_speedup'] is None
    sessions = report['sessions']
    assert [row['mode'] for row in sessions] == ['contiguous', 'channels-last']
    assert len({row['environment']['pid'] for row in sessions} | {os.getpid()}) == 3
    assert sessions[0]['backend'] == sessions[1]['backend']
    assert sessions[0]['backend']['benchmark'] is False
    for row in sessions:
        assert row['precision_fallbacks'] == 0
        assert len(row['runs']) == 2
        assert (output / row['log']).is_file()
    assert (output / 'reference.npy').is_file()
