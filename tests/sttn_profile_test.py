"""STTN 阶段计时：CUDA Event 与主机等待分开，关闭时不触发同步。"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from backend.inpaint import sttn_profile


def test_cpu_stages_accumulate_wall_time(monkeypatch):
    ticks = iter([0.0, 1.0, 1.25, 2.0, 2.5, 3.0])
    monkeypatch.setattr(sttn_profile.time, 'perf_counter', lambda: next(ticks))
    profiler = sttn_profile.STTNProfiler(True, 'cpu')
    with profiler.stage('encoder', device_work=True):
        pass
    with profiler.stage('encoder', device_work=True):
        pass
    result = profiler.finish()
    assert result['wall_seconds'] == 3.0
    assert result['stage_seconds']['encoder'] == 0.75
    assert result['host_seconds']['encoder'] == 0.75
    assert result['compute_clock'] == 'wall'


def test_disabled_profiler_does_not_read_clock_or_touch_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('关闭性能分析时不应调用计时器或 CUDA')

    monkeypatch.setattr(sttn_profile.time, 'perf_counter', unexpected)
    monkeypatch.setattr(torch.cuda, 'Event', unexpected)
    monkeypatch.setattr(torch.cuda, 'current_stream', unexpected)
    profiler = sttn_profile.STTNProfiler(False, 'cuda:1')
    with profiler.stage('decoder', device_work=True):
        pass
    assert profiler.finish() is None


def test_cuda_events_defer_sync_and_separate_host_wait(monkeypatch):
    calls = []
    ticks = iter([0., 1., 1.8, 2., 2.4, 2.5, 2.75, 3.])
    monkeypatch.setattr(sttn_profile.time, 'perf_counter', lambda: next(ticks))

    stream = SimpleNamespace(synchronize=lambda: calls.append('sync'))

    def current_stream(device):
        assert device == torch.device('cuda:1')
        return stream

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self, selected_stream):
            assert selected_stream is stream
            calls.append('record')

        def elapsed_time(self, end):
            assert 'sync' in calls
            return 12.5  # ms；故意与主机等待时间不同。

    monkeypatch.setattr(torch.cuda, 'current_stream', current_stream)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    profiler = sttn_profile.STTNProfiler(True, 'cuda:1')
    for _ in range(2):
        with profiler.stage('download', device_work=True):
            pass
    with profiler.stage('postprocess'):
        pass
    assert calls == ['record'] * 4
    result = profiler.finish()
    assert calls.count('sync') == 1
    assert result['compute_clock'] == 'cuda_event'
    assert result['stage_seconds']['download'] == pytest.approx(0.025)
    assert result['host_seconds']['download'] == pytest.approx(1.2)
    assert result['stage_seconds']['postprocess'] == 0.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要 CUDA 验证 Event 实际执行')
def test_real_cuda_timing():
    device = torch.device('cuda')
    profiler = sttn_profile.STTNProfiler(True, device)
    with profiler.stage('matmul', device_work=True):
        value = torch.randn(256, 256, device=device)
        result = value @ value
    report = profiler.finish()
    assert torch.isfinite(result).all()
    assert report['stage_seconds']['matmul'] > 0
    assert report['wall_seconds'] > 0
