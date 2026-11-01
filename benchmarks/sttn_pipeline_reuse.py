"""STTN 连续任务基准：独立进程、同进程新 Pipeline、同进程复用 Pipeline。

只测现有生命周期；GDINO 仍逐片释放，不实现模型常驻或服务调度。
父进程只用标准库，模型在测量子进程内加载。输出目录必须是新目录。
"""
import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = ('fresh-process', 'new-pipeline', 'reuse-pipeline')


def _measure_video(pipe, source, output, options):
    """在原惰性加载位置测 transformers 命名导入，不提前预热模型。"""
    report = {'gdino_import_seconds': 0., 'gdino_init': None, 'gdino_tracking': None}
    ensure = pipe._ensure_sticker_detector
    pipeline_module = sys.modules['vsr_pipeline']
    detect = pipeline_module.locate_stickers_gdino
    detection_errors = []

    def measured_ensure():
        started = time.perf_counter()
        transformers = importlib.import_module('transformers')
        # transformers 惰性导出在属性访问时也可能付出导入成本。
        getattr(transformers, 'AutoProcessor')
        getattr(transformers, 'AutoModelForZeroShotObjectDetection')
        report['gdino_import_seconds'] += time.perf_counter() - started
        detector = ensure()
        summary = detector.profile_summary()
        if summary is not None:
            report['gdino_init'] = {key: summary[key] for key in (
                'init_seconds', 'init_processor_seconds', 'init_model_seconds',
                'requested_precision', 'precision', 'precision_fallbacks')}
        return detector

    def measured_detect(*args, **kwargs):
        # 生产入口会捕获检测异常；基准必须在任务返回后拒绝这种降级结果。
        try:
            tracking_module = importlib.import_module('backend.sticker_tracking')
            tracked = tracking_module.locate_tracked_stickers

            def measured_tracking(*tracking_args, **tracking_kwargs):
                boxes, stats = tracked(*tracking_args, **tracking_kwargs)
                report['gdino_tracking'] = stats
                return boxes, stats

            tracking_module.locate_tracked_stickers = measured_tracking
            try:
                return detect(*args, **kwargs)
            finally:
                tracking_module.locate_tracked_stickers = tracked
        except Exception as exc:
            detection_errors.append(f'{type(exc).__name__}: {exc}')
            raise

    pipe._ensure_sticker_detector = measured_ensure
    pipeline_module.locate_stickers_gdino = measured_detect
    try:
        report['stats'] = pipe.process_video(source, str(output), **options)
    finally:
        # 避免记录器跨任务累积或引用上一条视频的状态。
        pipe._ensure_sticker_detector = ensure
        pipeline_module.locate_stickers_gdino = detect
    if detection_errors:
        raise RuntimeError(f'GDINO 检测失败，基准无效: {detection_errors}')
    if report['gdino_tracking'] and report['gdino_tracking']['model_failed']:
        raise RuntimeError('GDINO 检测未完成，基准无效: '
                           f'model_failed={report["gdino_tracking"]["model_failed"]}')
    return report


def run_sequence(mode, inputs, output_dir, pipeline_options=None, process_options=None):
    """一个进程内顺序处理；返回逐任务及首次导入统计。"""
    if mode not in MODES:
        raise ValueError(f'未知 mode: {mode}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    started = time.perf_counter()
    pipeline_module = importlib.import_module('vsr_pipeline')
    module_seconds = time.perf_counter() - started
    torch = pipeline_module.torch
    pipe_options = dict(inpaint_mode='sttn', sticker_backend='gdino',
                        sttn_profile=True, sticker_profile=True)
    pipe_options.update(pipeline_options or {})
    video_options = dict(sttn_residual_check=False)
    video_options.update(process_options or {})
    report = dict(mode=mode, pipeline_module_import_seconds=module_seconds,
                  pipeline_initializations=0, sequence_cleanup_seconds=0., jobs=[])
    pipe = None
    for index, source in enumerate(inputs):
        task_started = time.perf_counter()
        init_seconds = 0.
        if pipe is None:
            started = time.perf_counter()
            pipe = pipeline_module.Pipeline(**pipe_options)
            init_seconds = time.perf_counter() - started
            report['pipeline_initializations'] += 1
        output = output_dir / f'{index + 1:03d}.mp4'
        started = time.perf_counter()
        job = _measure_video(pipe, source, output, video_options)
        job.update(index=index + 1, input=str(source), output=str(output),
                   pipeline_init_seconds=init_seconds,
                   process_video_seconds=time.perf_counter() - started,
                   task_seconds=time.perf_counter() - task_started)
        stages = job['stats']['pipeline_profile']['stage_seconds']
        if video_options.get('locate_stickers', True) and 'sticker_init' in stages and job['gdino_init'] is None:
            raise RuntimeError('GDINO 初始化失败或没有可用统计，不能将跳过贴纸检测当成性能收益')
        job['sttn_precision'] = (dict(requested=pipe.inpainter.requested_precision,
                                      actual=pipe.inpainter.precision,
                                      fallbacks=pipe.inpainter.precision_fallbacks)
                                 if getattr(pipe.inpainter, 'precision', None) is not None else None)
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            job['cuda_after'] = dict(allocated_gib=torch.cuda.memory_allocated() / 1024**3,
                                     reserved_gib=torch.cuda.memory_reserved() / 1024**3,
                                     free_gib=free / 1024**3, total_gib=total / 1024**3)
        report['jobs'].append(job)
        print(f'[reuse-benchmark-job] mode={mode} job={index + 1}/{len(inputs)} '
              f'pipeline_init={init_seconds:.3f}s '
              f'process_video={job["process_video_seconds"]:.3f}s '
              f'task={job["task_seconds"]:.3f}s', flush=True)
        if mode != 'reuse-pipeline':
            started = time.perf_counter()
            pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            report['sequence_cleanup_seconds'] += time.perf_counter() - started
    report['environment'] = dict(
        pid=os.getpid(), python=sys.version, platform=platform.platform(), torch=torch.__version__,
        numpy=pipeline_module.np.__version__, opencv=pipeline_module.cv2.__version__,
        torch_threads=torch.get_num_threads(), opencv_threads=pipeline_module.cv2.getNumThreads(),
        cuda_available=torch.cuda.is_available(),
        device_name=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
    return report


def video_signature(path):
    """流式对比解码 RGB 像素和呈现时间，避免完整视频常驻内存。"""
    import av

    pixels, timestamps = hashlib.sha256(), hashlib.sha256()
    count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            array = frame.to_ndarray(format='rgb24')
            pixels.update(str(array.shape).encode())
            pixels.update(array.tobytes())
            stamp = str(frame.pts * frame.time_base) if frame.pts is not None else 'None'
            timestamps.update((stamp + '\n').encode())
            count += 1
    return dict(frames=count, pixel_sha256=pixels.hexdigest(),
                timestamp_sha256=timestamps.hexdigest())


def summarize(sessions):
    result = {}
    for mode in MODES:
        runs = [session for session in sessions if session['mode'] == mode]
        if not runs:
            continue
        jobs = [job for session in runs for job in session['jobs']]
        warm = [job['task_seconds'] for session in runs for job in session['jobs'][1:]]
        wall = sum(session['process_wall_seconds'] for session in runs)
        result[mode] = dict(videos=len(jobs), processes=len(runs), process_wall_seconds=wall,
                            amortized_process_seconds_per_video=wall / len(jobs),
                            task_median_seconds=statistics.median(job['task_seconds'] for job in jobs),
                            warm_task_median_seconds=statistics.median(warm) if warm else None)
        metrics = {}
        for key in ('pipeline_init_seconds', 'process_video_seconds', 'gdino_import_seconds'):
            values = [job[key] for job in jobs if key in job]
            if values:
                metrics[key] = statistics.median(values)
        for stage in ('sticker_init', 'ocr_timeline', 'sticker_detect', 'sttn_init', 'sttn_infer'):
            values = [job['stats']['pipeline_profile']['stage_seconds'][stage] for job in jobs
                      if stage in job.get('stats', {}).get('pipeline_profile', {}).get('stage_seconds', {})]
            if values:
                metrics[stage] = statistics.median(values)
        result[mode]['median_seconds'] = metrics
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--_worker':
        request = json.loads(Path(argv[1]).read_text())
        report = run_sequence(**request)
        Path(argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', nargs='+', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3, help='每种模式重复输入列表的次数，至少 2')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int)
    parser.add_argument('-c', '--region', nargs=4, type=int)
    parser.add_argument('--sticker-model-id', default='backend/models/grounding-dino-tiny')
    parser.add_argument('--sticker-score', type=float, default=0.22)
    parser.add_argument('--sticker-max-area-px', type=float, default=1000.)
    parser.add_argument('--sticker-batch-size', type=int, default=4)
    parser.add_argument('--no-locate-stickers', action='store_true', help='本地无贴纸模型时仅作流程验证')
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error('--repeats 至少为 2，才能比较同进程后续任务')
    if args.threads is not None and args.threads < 1:
        parser.error('--threads 必须不小于 1')
    if args.sticker_batch_size < 1:
        parser.error('--sticker-batch-size 必须不小于 1')
    if args.region and (args.region[0] >= args.region[1] or args.region[2] >= args.region[3]
                        or min(args.region) < 0):
        parser.error('--region 必须是有效的 ymin ymax xmin xmax')
    if any(not source.is_file() for source in args.input):
        parser.error('输入视频不存在')
    if args.output_dir.exists():
        parser.error('--output-dir 必须是新目录，避免覆盖已有结果')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    inputs = [str(source.resolve()) for source in args.input] * args.repeats
    pipeline_options = dict(device=args.device, threads=args.threads,
                            sticker_model_id=args.sticker_model_id,
                            sticker_batch_size=args.sticker_batch_size)
    process_options = dict(region=args.region, sticker_score=args.sticker_score,
                           sticker_max_area_px=args.sticker_max_area_px,
                           locate_stickers=not args.no_locate_stickers)
    sessions = []
    for mode in dict.fromkeys(args.modes):
        batches = [[source] for source in inputs] if mode == 'fresh-process' else [inputs]
        for index, batch in enumerate(batches, 1):
            label = f'{mode}-{index}'
            request = dict(mode=mode, inputs=batch, output_dir=str(output_dir / label),
                           pipeline_options=pipeline_options, process_options=process_options)
            request_path = output_dir / f'{label}.request.json'
            report_path = output_dir / f'{label}.json'
            log_path = output_dir / f'{label}.log'
            request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2))
            print(f'[reuse-benchmark-start] mode={mode} videos={len(batch)} log={log_path}', flush=True)
            started = time.perf_counter()
            with log_path.open('w') as log:
                completed = subprocess.run(
                    [sys.executable, '-u', str(Path(__file__).resolve()), '--_worker',
                     str(request_path), str(report_path)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.perf_counter() - started
            if completed.returncode:
                raise RuntimeError(f'{label} 失败，退出码 {completed.returncode}；查看 {log_path}')
            report = json.loads(report_path.read_text())
            report.update(process_wall_seconds=elapsed, log=str(log_path))
            sessions.append(report)
            print(f'[reuse-benchmark-session] mode={mode} videos={len(batch)} '
                  f'process_wall={elapsed:.3f}s', flush=True)

    # 父进程在所有测量结束后才读取视频，验收耗时不混入任务或进程计时。
    references, comparisons = {}, []
    for session in sessions:
        for job in session['jobs']:
            signature = video_signature(job['output'])
            reference = references.setdefault(job['input'], signature)
            comparisons.append(dict(mode=session['mode'], input=job['input'], output=job['output'],
                                    matches_reference=signature == reference, **signature))
    result = dict(scope='sequential_sttn_pipeline_lifecycle',
                  gdino_released_after_each_video=True, pipeline_options=pipeline_options,
                  process_options=process_options, repeats=args.repeats,
                  sessions=sessions, summary=summarize(sessions), comparisons=comparisons,
                  outputs_match=all(item['matches_reference'] for item in comparisons))
    (output_dir / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(summary=result['summary'], outputs_match=result['outputs_match'],
                         report=str(output_dir / 'report.json')), ensure_ascii=False, indent=2))
    if not result['outputs_match']:
        raise SystemExit('输出像素、帧数或时间戳存在差异；查看 report.json，不能只比较速度')


if __name__ == '__main__':
    main()
