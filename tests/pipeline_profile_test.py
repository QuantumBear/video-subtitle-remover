"""流水线计时必须排除子阶段，避免加载、补修和读写重复相加。"""

import pytest

from backend import pipeline_profile


def test_nested_and_repeated_stages_account_for_each_second_once(monkeypatch):
    ticks = iter([0., 1., 2., 5., 7., 8., 9., 10.])
    monkeypatch.setattr(pipeline_profile.time, 'perf_counter', lambda: next(ticks))
    profiler = pipeline_profile.PipelineProfiler(True)
    with profiler.stage('sticker_detect'):
        with profiler.stage('sticker_init'):
            pass
    with profiler.stage('sticker_init'):
        pass
    report = profiler.finish()
    assert report['wall_seconds'] == 10.
    assert report['stage_seconds'] == {'sticker_init': 4., 'sticker_detect': 3.}
    assert report['stage_calls'] == {'sticker_init': 2, 'sticker_detect': 1}
    assert report['accounted_seconds'] == 7.
    assert report['other_seconds'] == 3.


def test_decode_iterator_excludes_time_processing_each_frame(monkeypatch):
    elapsed = [0.]
    monkeypatch.setattr(pipeline_profile.time, 'perf_counter', lambda: elapsed[0])

    def frames():
        elapsed[0] += 2.
        yield 'frame'

    profiler = pipeline_profile.PipelineProfiler(True)
    for frame in profiler.iterate('decode', frames()):
        assert frame == 'frame'
        elapsed[0] += 5.
    report = profiler.finish()
    assert report['stage_seconds']['decode'] == 2.
    assert report['wall_seconds'] == 7.
    assert report['other_seconds'] == 5.


def test_disabled_profiler_preserves_iteration_without_reading_clock(monkeypatch):
    def unexpected():
        pytest.fail('关闭流水线分析时不应读取计时时钟')

    monkeypatch.setattr(pipeline_profile.time, 'perf_counter', unexpected)
    profiler = pipeline_profile.PipelineProfiler(False)
    with profiler.stage('ocr_timeline'):
        assert list(profiler.iterate('decode', iter([1, 2]))) == [1, 2]
    assert profiler.finish() is None
