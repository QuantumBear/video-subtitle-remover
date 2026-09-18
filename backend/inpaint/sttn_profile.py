"""STTN 阶段计时；CUDA 事件区间与主机等待时间分别统计。"""

from contextlib import contextmanager
import time

import torch


class STTNProfiler:
    """一次修复带推理的计时器，关闭时不访问时钟或 CUDA。

    CUDA 阶段只记录事件，在 finish 时同步当前 stream 一次。事件区间
    可能包含 kernel 提交间隙，不等同于各个 kernel 的纯执行时间之和。
    host_seconds 可能包含等待前序 GPU 工作的时间，不能与事件时间相加。
    非 CUDA 设备只记录主机 wall time，不保证异步设备工作的完成时间。
    """

    def __init__(self, enabled, device):
        self.enabled = enabled
        if not enabled:
            return
        self._started = time.perf_counter()
        self._host_seconds = {}
        self._stage_seconds = {}
        self._events = []
        device = torch.device(device)
        self._stream = torch.cuda.current_stream(device) if device.type == 'cuda' else None

    @contextmanager
    def stage(self, name, *, device_work=False):
        if not self.enabled:
            yield
            return
        use_events = device_work and self._stream is not None
        if use_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        if use_events:
            start.record(self._stream)
        try:
            yield
        finally:
            if use_events:
                end.record(self._stream)
                self._events.append((name, start, end))
            elapsed = time.perf_counter() - started
            self._host_seconds[name] = self._host_seconds.get(name, 0.0) + elapsed
            if not use_events:
                self._stage_seconds[name] = self._stage_seconds.get(name, 0.0) + elapsed

    def finish(self):
        if not self.enabled:
            return None
        if self._events:
            self._stream.synchronize()
            for name, start, end in self._events:
                self._stage_seconds[name] = (
                    self._stage_seconds.get(name, 0.0) + start.elapsed_time(end) / 1000)
        return {
            'wall_seconds': time.perf_counter() - self._started,
            'stage_seconds': self._stage_seconds,
            'host_seconds': self._host_seconds,
            'compute_clock': 'cuda_event' if self._stream is not None else 'wall',
        }
