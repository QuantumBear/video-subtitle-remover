"""独立进程比较 cuDNN benchmark 开/关；仅测完整修复带，不改生产设置。

父进程只用标准库。每次会话重新加载同一模型，首次调用单独计时；
后续预热和测量保持同一引擎、输入、精度和窗口。输出目录必须是新目录。
"""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def backend_settings():
    import torch

    return dict(
        benchmark=torch.backends.cudnn.benchmark,
        enabled=torch.backends.cudnn.enabled,
        deterministic=torch.backends.cudnn.deterministic,
        allow_tf32=torch.backends.cudnn.allow_tf32,
        benchmark_limit=getattr(torch.backends.cudnn, 'benchmark_limit', None),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def measure_mode(engine, frames, masks, *, enabled, warmup, repeats, reference=None):
    """顺序采集一个模式；即使推理失败也恢复全局 benchmark 开关。

    此函数不清理 cuDNN 缓存，调用方必须为每个模式启动新进程。
    reference 是 off 模式首次输出，保留浮点融合值，不能先转 uint8。
    """
    import numpy as np
    import torch

    if warmup < 0 or repeats < 1:
        raise ValueError('warmup 必须非负，repeats 必须为正')
    if engine.profile:
        raise ValueError('整带 A/B 不启用逐阶段 profile')
    expected = engine.requested_precision
    if engine.precision != expected or engine.precision_fallbacks:
        raise RuntimeError('引擎发生精度回退，基准无效')
    cuda = engine.device.type == 'cuda'
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = enabled
    runs = []
    try:
        settings = backend_settings()
        for phase in ['first'] + ['warmup'] * warmup + ['measured'] * repeats:
            if cuda:
                torch.cuda.synchronize(engine.device)
                torch.cuda.reset_peak_memory_stats(engine.device)
            started = time.perf_counter()
            output = engine.inpaint(frames, masks)
            if cuda:
                torch.cuda.synchronize(engine.device)
            seconds = time.perf_counter() - started
            if engine.precision != expected or engine.precision_fallbacks:
                raise RuntimeError('测量中发生精度回退，基准无效')
            actual = np.stack(output).astype(np.float32)
            if not np.isfinite(actual).all():
                raise RuntimeError('输出存在非有限值，基准无效')
            if reference is None:
                reference = actual.copy()
            if actual.shape != reference.shape or not np.isfinite(reference).all():
                raise RuntimeError('参考输出形状错误或存在非有限值')
            diff = np.abs(actual - reference)
            runs.append(dict(
                phase=phase, wall_seconds=seconds,
                max_pixel_difference=float(diff.max()),
                mean_pixel_difference=float(diff.mean(dtype=np.float64)),
                peak_allocated_gib=torch.cuda.max_memory_allocated(engine.device) / 1024**3 if cuda else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(engine.device) / 1024**3 if cuda else None))
            print(f'[cudnn-band] benchmark={"on" if enabled else "off"} '
                  f'phase={phase} seconds={seconds:.6f} '
                  f'max_diff={runs[-1]["max_pixel_difference"]}', flush=True)
        max_diff = max(row['max_pixel_difference'] for row in runs)
        report = dict(
            mode='on' if enabled else 'off', backend=settings, runs=runs,
            first_seconds=runs[0]['wall_seconds'],
            warm_median_seconds=statistics.median(row['wall_seconds'] for row in runs
                                                 if row['phase'] == 'measured'),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in runs) if cuda else None,
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in runs) if cuda else None,
            outputs_match=max_diff == 0, max_pixel_difference=max_diff,
            mean_pixel_difference=statistics.mean(row['mean_pixel_difference'] for row in runs),
            requested_precision=expected, precision=engine.precision,
            precision_fallbacks=engine.precision_fallbacks)
        return report, reference
    finally:
        torch.backends.cudnn.benchmark = previous


def run_worker(options, mode, reference_path):
    started = time.perf_counter()
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    from backend.inpaint.sttn_det_inpaint import STTNDetInpaint

    import_seconds = time.perf_counter() - started
    torch.set_num_threads(options['threads'])
    torch.manual_seed(42)
    device = torch.device(options['device'])
    if device.type == 'cuda' and (not torch.cuda.is_available()
                                  or not torch.backends.cudnn.is_available()
                                  or not torch.backends.cudnn.enabled):
        raise RuntimeError('CUDA/cuDNN 不可用，不能做 GPU A/B')
    started = time.perf_counter()
    engine = STTNDetInpaint(device, options['model'], precision=options['precision'],
                           cache_first_qkv=options['cache'] == 'on', profile=False)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    width, height = engine.model_input_width, engine.model_input_height
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
              for _ in range(options['frames'])]
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in frames]
    for index, mask in enumerate(masks):
        if index % 10:
            x = width // 3 + index % 8
            mask[height // 2:height // 2 + 32, x:x + 128] = 255
    reference_path = Path(reference_path)
    reference = np.load(reference_path, allow_pickle=False) if reference_path.exists() else None
    report, reference = measure_mode(engine, frames, masks, enabled=mode == 'on',
                                     warmup=options['warmup'], repeats=options['repeats'],
                                     reference=reference)
    if not reference_path.exists():
        np.save(reference_path, reference, allow_pickle=False)
    report.update(
        model_load_seconds=load_seconds, module_import_seconds=import_seconds,
        frames=len(frames), frame_shape=list(frames[0].shape),
        neighbor_stride=engine.neighbor_stride, ref_length=engine.ref_length,
        cache_active=engine.cache_first_qkv and len(frames) > engine.neighbor_stride,
        environment=dict(pid=os.getpid(), python=sys.version, platform=platform.platform(),
                         torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                         cudnn=torch.backends.cudnn.version(), torch_threads=torch.get_num_threads(),
                         device=str(device), device_name=torch.cuda.get_device_name(device)
                         if device.type == 'cuda' else 'CPU'))
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--_worker':
        request = json.loads(Path(argv[1]).read_text())
        report = run_worker(**request)
        Path(argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--model', type=Path, default=ROOT / 'backend/models/sttn-det/sttn.pth')
    parser.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--cache', choices=('on', 'off'), default='on')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--sessions', type=int, default=2,
                        help='每模式的新进程数；默认两组，按 off/on、on/off 交替')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.frames, args.repeats, args.sessions, args.threads) < 1 or args.warmup < 0:
        parser.error('frames/repeats/sessions/threads 必须为正，warmup 必须非负')
    if args.device == 'cpu' and args.precision != 'fp32':
        parser.error('CPU 冒烟需指定 --precision fp32，不能默许 FP16 回退')
    if args.frames < 2:
        parser.error('frames 至少为 2，才能包含非空修复 mask')
    args.model = args.model.resolve()
    if not args.model.is_file():
        parser.error(f'模型不存在: {args.model}')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    options = {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items() if key not in ('sessions', 'output_dir')}
    sessions = []
    for index in range(args.sessions):
        for mode in (('off', 'on') if index % 2 == 0 else ('on', 'off')):
            name = f'{index + 1:02d}-{mode}'
            request_path = output_dir / f'{name}-request.json'
            report_path = output_dir / f'{name}.json'
            log_path = output_dir / f'{name}.log'
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npy'))))
            print(f'[cudnn-session-start] session={index + 1} mode={mode} log={log_path}', flush=True)
            started = time.perf_counter()
            with log_path.open('w') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--_worker',
                                         str(request_path), str(report_path)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} 子进程失败（{result.returncode}），请查看 {log_path}')
            session = json.loads(report_path.read_text())
            session.update(session=index + 1, log=log_path.name,
                           process_wall_seconds=time.perf_counter() - started)
            sessions.append(session)
            print(f'[cudnn-session] session={index + 1} mode={mode} '
                  f'first={session["first_seconds"]:.6f}s warm={session["warm_median_seconds"]:.6f}s '
                  f'outputs_match={session["outputs_match"]}', flush=True)
    settings = [{key: value for key, value in row['backend'].items() if key != 'benchmark'}
                for row in sessions]
    if any(value != settings[0] for value in settings[1:]):
        raise RuntimeError('子进程除 benchmark 外的后端设置不一致，A/B 无效')
    summary = {}
    for mode in ('off', 'on'):
        rows = [row for row in sessions if row['mode'] == mode]
        warm = [run for row in rows for run in row['runs'] if run['phase'] == 'measured']
        summary[mode] = dict(
            first_median_seconds=statistics.median(row['first_seconds'] for row in rows),
            warm_median_seconds=statistics.median(run['wall_seconds'] for run in warm),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in rows) if args.device == 'cuda' else None,
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows) if args.device == 'cuda' else None)
    match = all(row['outputs_match'] for row in sessions)
    valid_gpu = args.device == 'cuda' and match
    off_s, on_s = (summary[mode]['warm_median_seconds'] for mode in ('off', 'on'))
    report = dict(
        scope='inpaint_band', profile=False, options=options, sessions=sessions, summary=summary,
        gpu_comparison=args.device == 'cuda', outputs_match=match,
        max_pixel_difference=max(row['max_pixel_difference'] for row in sessions),
        mean_pixel_difference=statistics.mean(row['mean_pixel_difference'] for row in sessions),
        warm_speedup=off_s / on_s if valid_gpu else None,
        warm_time_saved_percent=100 * (1 - on_s / off_s) if valid_gpu else None,
        timing_notes=[
            'Each mode/session runs in a fresh process; only cudnn.benchmark changes.',
            'First band excludes model loading, includes remaining lazy initialization and algorithm selection; not pure search time.',
            'Warm measurements use synchronized band wall time with production finite checks and compositing; no profiler.',
            'All first/warmup/measured outputs are compared with the first off output, without uint8 truncation.',
            'Peak memory is reset per band, includes model/workspace/allocator state; no empty_cache between bands.',
            'CPU runs validate the tool only; speedup is null. Pixel differences also suppress speedup.',
            'Fixed synthetic inputs and one band length do not establish real video quality, throughput or concurrency capacity.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[cudnn-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
