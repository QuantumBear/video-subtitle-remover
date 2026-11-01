"""连续任务基准必须测真实视频流程，明确导入、初始化与任务计时边界。"""
import json
import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip('av')
pytest.importorskip('cv2')

from benchmarks import sttn_pipeline_reuse as benchmark
import vsr_pipeline
from backend import sticker_tracking


@pytest.fixture
def workload(tmp_path, monkeypatch):
    source = tmp_path / 'in.mp4'
    with av.open(str(source), 'w') as container:
        stream = container.add_stream('libx264rgb', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'rgb24'
        stream.options = {'crf': '0', 'bf': '0'}
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 90 + i,
                                                       dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    elapsed = [0.]
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: elapsed[0])
    monkeypatch.setattr(vsr_pipeline, 'cuda_memory_snapshot', lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoProcessor=object(), AutoModelForZeroShotObjectDetection=object()))
    monkeypatch.setattr(sticker_tracking, 'locate_tracked_stickers',
                        lambda *a, **kw: ({}, {'model_failed': 0}))

    class TestPipeline(vsr_pipeline.Pipeline):
        def __init__(self, **kwargs):
            elapsed[0] += 2.
            self.inpaint_mode = 'sttn'
            self.sttn_profile = True
            self.sticker_backend = 'gdino'
            self._sticker_detector = None
            self.inpainter = None

        def _detect_timeline(self, *args):
            elapsed[0] += 1.
            return [[(20, 30, 10, 50)]] * 3, dict(
                sampled=3, refined=0, ocr_calls=3, tracks=1, discarded=0,
                scene_change_frames=[])

        def _ensure_sticker_detector(self):
            if self._sticker_detector is None:
                elapsed[0] += 5.
                self._sticker_detector = SimpleNamespace(profile_summary=lambda: dict(
                    vsr_pipeline.sticker_detect.GroundingDinoStickerDetector._new_profile_stats(),
                    init_seconds=5., init_processor_seconds=1., init_model_seconds=4.,
                    requested_precision='fp16', precision='fp16', precision_fallbacks=0))
            return self._sticker_detector

        def _ensure_sttn(self):
            if self.inpainter is None:
                elapsed[0] += 3.

                def engine(frames, masks, x_bounds=None):
                    elapsed[0] += 2.
                    return [f.copy() for f in frames]

                self.inpainter = engine

    monkeypatch.setattr(vsr_pipeline, 'Pipeline', TestPipeline)
    return dict(inputs=[str(source)] * 2, output_dir=str(tmp_path / 'results'),
                pipeline_options={'device': 'cpu'},
                process_options={'region': (10, 40, 5, 60), 'sttn_residual_check': False})


@pytest.mark.parametrize('mode,initializations,job_seconds', [
    ('new-pipeline', 2, [13., 13.]), ('reuse-pipeline', 1, [13., 8.])])
def test_sequence_separates_initialization_and_reused_model_time(
        workload, mode, initializations, job_seconds):
    report = benchmark.run_sequence(mode=mode, **workload)
    assert report['pipeline_initializations'] == initializations
    assert [job['task_seconds'] for job in report['jobs']] == job_seconds
    assert [job['pipeline_init_seconds'] for job in report['jobs']] == (
        [2., 2.] if mode == 'new-pipeline' else [2., 0.])
    assert [job['stats']['pipeline_profile']['stage_seconds']['sttn_init']
            for job in report['jobs']] == ([3., 3.] if mode == 'new-pipeline' else [3., 0.])
    for job in report['jobs']:
        assert job['stats']['frames'] == 3
        assert job['stats']['unresolved'] is None
        assert 'residual_probe' not in job['stats']['pipeline_profile']['stage_seconds']
        # 进程复用不会让现有逐片释放的 GDINO 自动变成常驻模型。
        assert job['gdino_init']['init_model_seconds'] == 4.
        assert job['stats']['pipeline_profile']['stage_seconds']['sticker_init'] == 5.
        assert job['gdino_import_seconds'] == 0.
    signatures = [benchmark.video_signature(job['output']) for job in report['jobs']]
    assert signatures[0] == signatures[1]
    assert signatures[0]['frames'] == 3


def test_summary_distinguishes_process_wall_from_task_time():
    sessions = [dict(mode='fresh-process', process_wall_seconds=20., jobs=[dict(task_seconds=13.)]),
                dict(mode='fresh-process', process_wall_seconds=22., jobs=[dict(task_seconds=14.)]),
                dict(mode='reuse-pipeline', process_wall_seconds=26.,
                     jobs=[dict(task_seconds=13.), dict(task_seconds=8.)])]
    summary = benchmark.summarize(sessions)
    assert summary['fresh-process']['amortized_process_seconds_per_video'] == 21.
    assert summary['fresh-process']['task_median_seconds'] == 13.5
    assert summary['fresh-process']['warm_task_median_seconds'] is None
    assert summary['reuse-pipeline']['amortized_process_seconds_per_video'] == 13.
    assert summary['reuse-pipeline']['warm_task_median_seconds'] == 8.


def test_named_transformers_import_is_measured_only_when_detector_is_used(workload, monkeypatch):
    real_import = benchmark.importlib.import_module
    first_import = [True]
    clock = benchmark.time.perf_counter

    def import_module(name):
        if name == 'transformers' and first_import[0]:
            first_import[0] = False
            # 测试时钟由 fixture 的闭包持有，替换为偏移相同的连续时钟。
            monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: clock() + 7.)
        return real_import(name)

    monkeypatch.setattr(benchmark.importlib, 'import_module', import_module)
    report = benchmark.run_sequence(mode='reuse-pipeline', **workload)
    assert [job['gdino_import_seconds'] for job in report['jobs']] == [7., 0.]
    assert report['jobs'][0]['stats']['pipeline_profile']['stage_seconds']['sticker_init'] == 12.
    assert report['jobs'][0]['gdino_init']['init_seconds'] == 5.


def test_benchmark_rejects_detection_exception_swallowed_by_pipeline(workload, monkeypatch):
    def failed_detect(*args, **kwargs):
        raise RuntimeError('injected detection failure')

    monkeypatch.setattr(vsr_pipeline, 'locate_stickers_gdino', failed_detect)
    with pytest.raises(RuntimeError, match='injected detection failure'):
        benchmark.run_sequence(mode='reuse-pipeline', **workload)
    assert vsr_pipeline.locate_stickers_gdino is failed_detect


def test_benchmark_rejects_failed_model_frames_in_tracking(workload, monkeypatch):
    def partial_detect(*args, **kwargs):
        return {}, {'model_failed': 1}

    monkeypatch.setattr(sticker_tracking, 'locate_tracked_stickers', partial_detect)
    with pytest.raises(RuntimeError, match='model_failed=1'):
        benchmark.run_sequence(mode='reuse-pipeline', **workload)
    assert sticker_tracking.locate_tracked_stickers is partial_detect


def test_output_comparison_catches_pixel_or_pts_change(tmp_path):
    # 使用测试素材生成器，直接改变真实编码后的内容和帧率。
    def write(path, value, rate):
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('libx264rgb', rate=rate)
            stream.width, stream.height, stream.pix_fmt = 64, 48, 'rgb24'
            stream.options = {'crf': '0', 'bf': '0'}
            for index in range(2):
                frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), value, dtype=np.uint8),
                                                   format='rgb24')
                frame.pts = index
                frame.time_base = Fraction(1, rate)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    paths = [tmp_path / f'{i}.mp4' for i in range(3)]
    for path, value, rate in zip(paths, [90, 91, 90], [30, 30, 25]):
        write(path, value, rate)
    signatures = [benchmark.video_signature(path) for path in paths]
    assert signatures[0]['pixel_sha256'] != signatures[1]['pixel_sha256']
    assert signatures[0]['pixel_sha256'] == signatures[2]['pixel_sha256']
    assert signatures[0]['timestamp_sha256'] != signatures[2]['timestamp_sha256']


@pytest.mark.parametrize('option,value', [('--repeats', '1'), ('--threads', '0')])
def test_cli_rejects_invalid_parameters_before_creating_output(tmp_path, option, value):
    output = tmp_path / 'result'
    with pytest.raises(SystemExit) as exc:
        benchmark.main(['-i', 'missing.mp4', '--output-dir', str(output), option, value])
    assert exc.value.code == 2
    assert not output.exists()


def test_cli_preserves_existing_output_directory(tmp_path):
    source = tmp_path / 'in.mp4'
    source.write_bytes(b'not opened until workers start')
    output = tmp_path / 'results'
    output.mkdir()
    sentinel = output / 'report.json'
    sentinel.write_text(json.dumps({'keep': True}))
    with pytest.raises(SystemExit):
        benchmark.main(['-i', str(source), '--output-dir', str(output)])
    assert json.loads(sentinel.read_text()) == {'keep': True}
