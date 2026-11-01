"""流水线主机计时；嵌套阶段只累计自身时间，不增加设备同步。"""

from contextlib import contextmanager
import time


class PipelineProfiler:
    def __init__(self, enabled):
        self.enabled = enabled
        if enabled:
            self._started = time.perf_counter()
            self._stack = []
            self._seconds = {}
            self._calls = {}

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        entry = [time.perf_counter(), 0.0]
        self._stack.append(entry)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - entry[0]
            self._stack.pop()
            self._seconds[name] = self._seconds.get(name, 0.0) + elapsed - entry[1]
            self._calls[name] = self._calls.get(name, 0) + 1
            if self._stack:
                self._stack[-1][1] += elapsed

    def call(self, name, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        with self.stage(name):
            return function(*args, **kwargs)

    def iterate(self, name, values):
        if not self.enabled:
            yield from values
            return
        iterator = iter(values)
        while True:
            with self.stage(name):
                try:
                    value = next(iterator)
                except StopIteration:
                    return
            # 消费者处理帧的时间不能计入解码。
            yield value

    def finish(self):
        if not self.enabled:
            return None
        wall = time.perf_counter() - self._started
        accounted = sum(self._seconds.values())
        return {
            'wall_seconds': wall,
            'stage_seconds': dict(self._seconds),
            'stage_calls': dict(self._calls),
            'accounted_seconds': accounted,
            'other_seconds': wall - accounted,
        }
