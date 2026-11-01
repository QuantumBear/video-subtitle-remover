"""基准专用的嵌套范围计时，不依赖 torch.profiler/CUPTI。"""

from contextlib import contextmanager
import statistics
import time

import torch


class RegionTimer:
    """当前 stream 上记录事件，每带末尾同步一次并扣除直接子范围。

    仅支持当前 STTN 的单 stream 顺序基准。事件区间包含 CPU 提交间隙，
    并非纯 kernel 时间。重复采集复用事件，避免每轮创建/初始化对象。
    """

    def __init__(self, device):
        device = torch.device(device)
        self.stream = torch.cuda.current_stream(device) if device.type == 'cuda' else None
        self._pool = []
        self.reset()

    def reset(self):
        self._records = []
        self._stack = []

    @contextmanager
    def region(self, name):
        index = len(self._records)
        if self.stream is not None and index == len(self._pool):
            self._pool.append((torch.cuda.Event(enable_timing=True),
                               torch.cuda.Event(enable_timing=True)))
        events = self._pool[index] if self.stream is not None else None
        row = dict(name=name, parent=self._stack[-1] if self._stack else None,
                   events=events)
        self._records.append(row)
        self._stack.append(index)
        if events is not None:
            events[0].record(self.stream)
        start = time.perf_counter()
        try:
            yield
        finally:
            row['host_ms'] = (time.perf_counter() - start) * 1000
            if events is not None:
                events[1].record(self.stream)
            self._stack.pop()

    def finish(self):
        if self._stack:
            raise RuntimeError('计时范围尚未退出')
        if self.stream is not None and self._records:
            self.stream.synchronize()
        for row in self._records:
            row['stream_ms'] = (row['events'][0].elapsed_time(row['events'][1])
                                if row['events'] is not None else None)
            row['child_host_ms'] = row['child_stream_ms'] = 0.
        for row in self._records:
            if row['parent'] is not None:
                parent = self._records[row['parent']]
                parent['child_host_ms'] += row['host_ms']
                parent['child_stream_ms'] += row['stream_ms'] or 0.
        groups = {}
        for row in self._records:
            group = groups.setdefault(row['name'], dict(
                name=row['name'], calls=0, inclusive_host_ms=0., self_host_ms=0.,
                inclusive_stream_ms=0. if self.stream is not None else None,
                self_stream_ms=0. if self.stream is not None else None))
            group['calls'] += 1
            group['inclusive_host_ms'] += row['host_ms']
            group['self_host_ms'] += row['host_ms'] - row['child_host_ms']
            if self.stream is not None:
                group['inclusive_stream_ms'] += row['stream_ms']
                group['self_stream_ms'] += row['stream_ms'] - row['child_stream_ms']
        return list(groups.values())


def capture_regions(engine, frames, masks, *, range_context, repeats, check_precision,
                    compare):
    """独立校准一带以初始化事件，再重复采集；不调用 Kineto profiler。"""
    timer = RegionTimer(engine.device)

    def run():
        timer.reset()
        start = time.perf_counter()
        with range_context(engine.model, range_context=timer.region):
            with timer.region('band'):
                output = engine.inpaint(frames, masks)
        rows = timer.finish()
        elapsed = time.perf_counter() - start
        check_precision()
        compare(output)
        return dict(wall_seconds=elapsed, regions=rows)

    run()  # 不计入结果，初始化每一对 CUDA Event。
    runs = []
    for index in range(repeats):
        runs.append(run())
        print(f'[region-capture] repeat={index + 1} seconds={runs[-1]["wall_seconds"]:.6f}',
              flush=True)
    names = [row['name'] for row in runs[0]['regions']]
    rows = []
    for name in names:
        samples = [next(row for row in run['regions'] if row['name'] == name) for run in runs]
        if len({row['calls'] for row in samples}) != 1:
            raise RuntimeError(f'重复采集中范围调用次数变化: {name}')
        row = dict(name=name, calls=samples[0]['calls'])
        for key in ('inclusive_host_ms', 'self_host_ms', 'inclusive_stream_ms', 'self_stream_ms'):
            row[key] = (statistics.median(sample[key] for sample in samples)
                        if samples[0][key] is not None else None)
        rows.append(row)
    clock = 'cuda_event' if timer.stream is not None else 'wall'
    return dict(
        compute_clock=clock, device_timing_status='stream_intervals' if timer.stream else 'cpu_only',
        region_calibration_runs=1, region_runs=runs, regions=rows,
        region_capture_median_seconds=statistics.median(run['wall_seconds'] for run in runs),
        operators=[], categories=[], device_events=[],
        operator_self_cpu_total_ms=None, attributed_self_device_total_ms=None,
        device_event_total_ms=None)
