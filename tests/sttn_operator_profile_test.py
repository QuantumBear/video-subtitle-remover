"""算子报告必须保留原输出，且不能重复累计嵌套算子或 CUDA kernel。"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
pytest.importorskip('cv2')

from backend.inpaint.sttn.network_sttn import InpaintGenerator
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from benchmarks import sttn_operator_profile as benchmark


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_engine():
    torch.manual_seed(81)
    engine = STTNDetInpaint.__new__(STTNDetInpaint)
    engine.device = torch.device('cpu')
    engine.precision = engine.requested_precision = 'fp32'
    engine.precision_fallbacks = 0
    engine.profile = False
    engine.model = InpaintGenerator(init_weights=False).eval()
    for block in engine.model.transformer:
        block.attention.patchsize = [(8, 8), (4, 4), (2, 2), (1, 1)]
    engine.model.prepare_fused_qkv()
    engine.model_input_width = engine.model_input_height = 32
    engine.neighbor_stride, engine.ref_length = 2, 3
    engine.cache_first_qkv = True
    return engine


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
def test_profile_writes_trace_and_keeps_complete_band_output(tmp_path, device):
    engine = small_engine()
    engine.device = torch.device(device)
    engine.model.to(engine.device)
    rng = np.random.default_rng(51)
    frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(7)]
    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in frames]
    for mask in masks:
        mask[8:24, 8:24] = 255
    originals = [array.copy() for array in frames + masks]
    expected = engine.inpaint(frames, masks)

    report = benchmark.run_profile(engine, frames, masks, tmp_path,
                                   warmup=1, repeats=2, top=10)

    assert report['outputs_match'] is True
    assert report['max_pixel_difference'] == 0
    assert len(report['baseline_seconds']) == 2
    assert all(value > 0 for value in report['baseline_seconds'])
    if device == 'cuda':
        assert report['device_timing_status'] == 'attributed'
        assert report['attributed_self_device_total_ms'] > 0
    else:
        assert report['device_timing_status'] == 'cpu_only'
        assert report['attributed_self_device_total_ms'] is None
    assert any(row['scope'] == 'transformer' and row['category'] == 'convolution'
               and row['self_cpu_ms'] > 0 for row in report['categories'])
    assert any(row['region'] == 'attention' and row['category'] == 'matmul'
               for row in report['operators'])
    assert any(row['category'] == 'finite_check' for row in report['operators'])
    assert json.loads((tmp_path / 'report.json').read_text()) == report
    trace = json.loads((tmp_path / 'trace.json').read_text())
    names = {event.get('name') for event in trace['traceEvents']}
    assert {'sttn::transformer', 'sttn::attention', 'aten::bmm'} <= names
    assert not any('forward' in module.__dict__ for module in engine.model.modules())
    for result, reference in zip(engine.inpaint(frames, masks), expected):
        np.testing.assert_array_equal(result, reference)
    for array, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(array, original)


def test_temporary_ranges_restore_instance_forward_after_exception():
    engine = small_engine()
    # 实例已有覆盖也必须原样恢复；不修改类级 forward。
    original = engine.model.encoder.forward
    engine.model.encoder.forward = original
    with pytest.raises(RuntimeError, match='injected'):
        with benchmark.model_ranges(engine.model):
            raise RuntimeError('injected')
    assert engine.model.encoder.__dict__['forward'] is original
    del engine.model.encoder.forward
    assert not any('forward' in module.__dict__ for module in engine.model.modules())


def test_report_uses_exclusive_times_and_does_not_count_kernel_twice():
    from torch.autograd.profiler_util import FunctionEvent

    def event(index, name, start, end, parent=None, device=None):
        # 新版统一 device 时间要求 use_device='cuda'；2.2 的 CUDA 字段尚未统一。
        kwargs = {'use_device': 'cuda'} if hasattr(FunctionEvent, 'device_time_total') else {}
        if device is not None:
            kwargs['device_type'] = device
        result = FunctionEvent(index, name, 1, start, end, **kwargs)
        if parent is not None:
            parent.append_cpu_child(result)
            result.cpu_parent = parent
        return result

    region = event(1, 'sttn::transformer', 0, 100)
    conv = event(2, 'aten::conv2d', 10, 70, region)
    backend = event(3, 'aten::cudnn_convolution', 20, 60, conv)
    backend.append_kernel('conv_kernel', 0, 30)
    kernel = event(4, 'conv_kernel', 20, 50, device=torch.autograd.DeviceType.CUDA)
    wait = event(5, 'cudaDeviceSynchronize', 100, 120)

    report = benchmark.summarize_events([region, conv, backend, kernel, wait], cuda=True)

    assert report['operator_self_cpu_total_ms'] == pytest.approx(.080)
    assert report['attributed_self_device_total_ms'] == pytest.approx(.030)
    assert report['device_event_total_ms'] == pytest.approx(.030)
    assert len(report['operators']) == 3
    category = next(row for row in report['categories'] if row['category'] == 'convolution')
    assert category['self_cpu_ms'] == pytest.approx(.060)
    assert category['self_device_ms'] == pytest.approx(.030)
    assert report['regions'][0]['cpu_total_ms'] == pytest.approx(.100)
    assert report['regions'][0]['device_total_ms'] == pytest.approx(.030)


@pytest.mark.parametrize('modern', [False, True])
def test_device_time_fields_support_old_and_new_torch(modern):
    fields = ({'self_device_time_total': 7., 'device_time_total': 11.} if modern else
              {'self_cuda_time_total': 7., 'cuda_time_total': 11.})
    event = SimpleNamespace(**fields)
    assert benchmark.device_time_us(event, exclusive=True) == 7.
    assert benchmark.device_time_us(event, exclusive=False) == 11.


def test_missing_cuda_events_are_reported_as_unavailable():
    report = benchmark.summarize_events([], cuda=True)
    assert report['device_timing_status'] == 'unavailable'
    assert report['attributed_self_device_total_ms'] is None
    assert report['device_event_total_ms'] is None


@pytest.mark.parametrize('failure', ['pixel_change', 'precision_fallback'])
@pytest.mark.parametrize('timing', ['profiler', 'cuda-event'])
def test_profile_rejects_changed_results_or_precision(tmp_path, failure, timing):
    engine = small_engine()
    original = engine.inpaint
    frames = [np.full((32, 32, 3), 127, dtype=np.uint8)]
    masks = [np.full((32, 32), 255, dtype=np.uint8)]
    calls = 0

    def change_after_baseline(images, holes):
        nonlocal calls
        calls += 1
        result = original(images, holes)
        if calls == 2:
            if failure == 'pixel_change':
                result[0] = result[0].copy()
                result[0][0, 0, 0] = (int(result[0][0, 0, 0]) + 1) % 256
            else:
                engine.precision_fallbacks += 1
        return result

    engine.inpaint = change_after_baseline
    with pytest.raises(RuntimeError, match='像素不一致|精度回退'):
        benchmark.run_profile(engine, frames, masks, tmp_path, warmup=0, repeats=1, timing=timing)
    assert not any('forward' in module.__dict__ for module in engine.model.modules())
    if failure == 'pixel_change':
        report = json.loads((tmp_path / 'report.json').read_text())
        assert report['outputs_match'] is False
        assert report['max_pixel_difference'] == 1


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))])
def test_region_timing_does_not_need_torch_profiler_and_preserves_output(tmp_path, monkeypatch, device):
    engine = small_engine()
    engine.device = torch.device(device)
    engine.model.to(engine.device)
    frames = [np.full((32, 32, 3), 127, dtype=np.uint8) for _ in range(3)]
    masks = [np.full((32, 32), 255, dtype=np.uint8) for _ in frames]
    expected = engine.inpaint(frames, masks)

    def no_cupti(*args, **kwargs):
        raise AssertionError('CUDA Event 模式不能调用 torch.profiler')

    monkeypatch.setattr(benchmark, 'profile', no_cupti)
    report = benchmark.run_profile(engine, frames, masks, tmp_path,
                                   warmup=0, repeats=2, timing='cuda-event')
    assert report['timing'] == 'cuda-event'
    assert report['compute_clock'] == ('cuda_event' if device == 'cuda' else 'wall')
    assert report['outputs_match'] is True
    assert report['max_pixel_difference'] == 0
    assert len(report['region_runs']) == 2
    assert len(report['baseline_seconds']) == 2
    rows = {row['name']: row for row in report['regions']}
    assert rows['attention']['calls'] == 64  # 2 窗口 × 8 层 × 4 路。
    assert rows['qkv_projection']['calls'] == 15  # 首层缓存1次 + 两窗口各7次投影。
    assert rows['transformer']['inclusive_host_ms'] > 0
    assert rows['feed_forward']['self_host_ms'] > 0
    if device == 'cuda':
        assert rows['transformer']['inclusive_stream_ms'] > 0
        assert rows['attention']['self_stream_ms'] > 0
        for run in report['region_runs']:
            band = next(row for row in run['regions'] if row['name'] == 'band')
            assert sum(row['self_stream_ms'] for row in run['regions']) == pytest.approx(
                band['inclusive_stream_ms'], abs=.01)
    else:
        assert rows['transformer']['inclusive_stream_ms'] is None
    assert report['trace'] is None
    assert not (tmp_path / 'trace.json').exists()
    assert json.loads((tmp_path / 'report.json').read_text()) == report
    assert not any('forward' in module.__dict__ for module in engine.model.modules())
    assert 'project_first_qkv' not in engine.model.__dict__
    for actual, reference in zip(engine.inpaint(frames, masks), expected):
        np.testing.assert_array_equal(actual, reference)


def test_cuda_region_timer_defers_sync_and_subtracts_children(monkeypatch):
    from benchmarks.sttn_region_timing import RegionTimer

    # 本机无 CUDA，仅替换硬件接口；真实计时器负责嵌套/去重/复用/读数顺序。
    clock = SimpleNamespace(ms=0., synchronized=False, syncs=0, events=0)

    class Stream:
        def synchronize(self):
            clock.syncs += 1
            clock.synchronized = True

    stream = Stream()

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            clock.events += 1

        def record(self, target):
            assert target is stream
            self.ms = clock.ms
            clock.synchronized = False

        def elapsed_time(self, end):
            assert clock.synchronized, '只能在整带同步后读取事件'
            return end.ms - self.ms

    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: stream)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    timer = RegionTimer('cuda')
    for _ in range(2):
        timer.reset()
        clock.ms = 0.
        with timer.region('band'):
            clock.ms = 10.
            with timer.region('transformer'):
                clock.ms = 20.
                with timer.region('feed_forward'):
                    clock.ms = 60.
                clock.ms = 80.
            clock.ms = 100.
        assert not clock.synchronized
        rows = {row['name']: row for row in timer.finish()}
        assert rows['band']['inclusive_stream_ms'] == 100.
        assert rows['band']['self_stream_ms'] == 30.
        assert rows['transformer']['inclusive_stream_ms'] == 70.
        assert rows['transformer']['self_stream_ms'] == 30.
        assert rows['feed_forward']['self_stream_ms'] == 40.
        assert sum(row['self_stream_ms'] for row in rows.values()) == 100.
    assert clock.syncs == 2
    assert clock.events == 6, '重复采集须复用已初始化的事件'


def test_region_ranges_restore_after_model_exception():
    from benchmarks.sttn_region_timing import RegionTimer

    engine = small_engine()
    original = engine.model.transformer[0].feed_forward.forward

    def fail(features):
        raise RuntimeError('injected region failure')

    engine.model.transformer[0].feed_forward.forward = fail
    timer = RegionTimer('cpu')
    with pytest.raises(RuntimeError, match='injected region failure'):
        with benchmark.model_ranges(engine.model, range_context=timer.region):
            with timer.region('band'):
                engine.inpaint([np.full((32, 32, 3), 127, dtype=np.uint8)],
                               [np.full((32, 32), 255, dtype=np.uint8)])
    assert engine.model.transformer[0].feed_forward.__dict__['forward'] is fail
    assert 'forward' not in engine.model.encoder.__dict__
    assert 'project_first_qkv' not in engine.model.__dict__
    assert timer.finish()
    engine.model.transformer[0].feed_forward.forward = original
