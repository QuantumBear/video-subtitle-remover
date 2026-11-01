"""独立进程比较 STTN eager/局部 FFN torch.compile，测首次、热态及变长段。

父进程仅用标准库；只包装 FFN 实例，退出恢复，生产引擎不变。
默认使用 Inductor；CPU 可指定 eager 后端验证图捕获及报告流程。
"""

import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.sttn_cudnn_benchmark import backend_settings


@contextmanager
def ffn_execution(model, *, mode, backend):
    """记录真正执行的编译图；fullgraph 禁止断图，编译失败不得悄悄退回 eager。"""
    import torch
    from backend.inpaint.sttn.network_sttn import FeedForward

    if mode not in ('eager', 'compiled'):
        raise ValueError(f'未知执行模式: {mode}')
    state = dict(ffn_modules=0, ffn_calls=0, compiled_calls=0,
                 backend_compilations=0, backend_compile_seconds=0.0, compilation_events=[])
    forwards = []
    if mode == 'compiled':
        from torch._dynamo.backends.registry import lookup_backend

        compiler = lookup_backend(backend)

        def counting_backend(graph, inputs):
            started = time.perf_counter()
            compiled = compiler(graph, inputs)
            seconds = time.perf_counter() - started
            state['backend_compilations'] += 1
            state['backend_compile_seconds'] += seconds
            state['compilation_events'].append(dict(
                seconds=seconds, input_shapes=[list(map(str, value.shape))
                                               for value in inputs if isinstance(value, torch.Tensor)]))

            def execute(*args):
                state['compiled_calls'] += 1
                return compiled(*args)

            return execute

    # 旧版 Dynamo 按共享 forward 代码对象计数：8 个 FFN 已占满默认 8 项，
    # 下一段长会立刻触顶。显式给变长实验留预算，仍禁止触顶后回退 eager。
    config = torch._dynamo.config
    config_values = {'suppress_errors': False}
    for names, value in [(('recompile_limit', 'cache_size_limit'), 64),
                         (('accumulated_recompile_limit', 'accumulated_cache_size_limit'), 512)]:
        key = next((name for name in names if hasattr(config, name)), None)
        if key is not None:
            config_values[key] = value
    state['compiler_config'] = config_values if mode == 'compiled' else None
    config_context = config.patch(config_values) if mode == 'compiled' else nullcontext()
    try:
        with config_context:
            # 固定遍历列表，防止实例修改改变 Module 遍历。
            for module in list(model.modules()):
                if not isinstance(module, FeedForward):
                    continue
                previous = module.forward
                forwards.append((module, 'forward' in module.__dict__, previous))
                target = (torch.compile(previous, backend=counting_backend, fullgraph=True,
                                        dynamic=None) if mode == 'compiled' else previous)

                def forward(x, target=target):
                    state['ffn_calls'] += 1
                    return target(x)

                module.forward = forward
            state['ffn_modules'] = len(forwards)
            if not forwards:
                raise ValueError('模型没有可编译的 FFN')
            yield state
    finally:
        for module, had_override, previous in reversed(forwards):
            if had_override:
                module.forward = previous
            else:
                del module.forward


def measure_compile(engine, bands, *, mode, backend, warmup, repeats, references=None):
    """同一实例持续经过主段、不同段长、主段重访；输出比较保留浮点融合值。"""
    import numpy as np
    import torch

    if not bands or warmup < 0 or repeats < 1 or engine.profile:
        raise ValueError('bands 非空、warmup 非负、repeats 为正，且不启用 profile')
    expected = engine.requested_precision
    if engine.precision != expected or engine.precision_fallbacks:
        raise RuntimeError('引擎发生精度回退，基准无效')
    if references is None:
        if mode != 'eager':
            raise ValueError('compiled 模式需要独立 eager 参考输出')
        references = {}
    elif set(references) != set(bands):
        raise ValueError('每个段长都需要 eager 参考输出')
    cuda = engine.device.type == 'cuda'
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    reports = []
    try:
        if cuda:
            torch.cuda.synchronize(engine.device)
            torch.cuda.reset_peak_memory_stats(engine.device)
        started = time.perf_counter()
        with ffn_execution(engine.model, mode=mode, backend=backend) as state:
            if cuda:
                torch.cuda.synchronize(engine.device)
            setup_seconds = time.perf_counter() - started
            setup_allocated = torch.cuda.max_memory_allocated(engine.device) / 1024**3 if cuda else None
            setup_reserved = torch.cuda.max_memory_reserved(engine.device) / 1024**3 if cuda else None
            settings = backend_settings()
            lengths = list(bands)
            visits = [('initial' if index == 0 else 'shape', length)
                      for index, length in enumerate(lengths)] + [('revisit', lengths[0])]
            for visit, length in visits:
                frames, masks = bands[length]
                runs = []
                for phase in ['first'] + ['warmup'] * warmup + ['measured'] * repeats:
                    if cuda:
                        torch.cuda.synchronize(engine.device)
                        torch.cuda.reset_peak_memory_stats(engine.device)
                    before = {key: state[key] for key in ('ffn_calls', 'compiled_calls',
                                                          'backend_compilations', 'backend_compile_seconds')}
                    started = time.perf_counter()
                    output = engine.inpaint(frames, masks)
                    if cuda:
                        torch.cuda.synchronize(engine.device)
                    seconds = time.perf_counter() - started
                    if engine.precision != expected or engine.precision_fallbacks:
                        raise RuntimeError('测量中发生精度回退，基准无效')
                    counts = {key: state[key] - value for key, value in before.items()}
                    if not counts['ffn_calls'] or (mode == 'compiled' and
                                                  counts['compiled_calls'] != counts['ffn_calls']):
                        raise RuntimeError('FFN 未全部执行编译图，基准无效')
                    actual = np.stack(output).astype(np.float32)
                    if not np.isfinite(actual).all():
                        raise RuntimeError('输出存在非有限值，基准无效')
                    if length not in references:
                        references[length] = actual.copy()
                    reference = references[length]
                    if reference.shape != actual.shape or not np.isfinite(reference).all():
                        raise RuntimeError('参考输出形状错误或存在非有限值')
                    diff = np.abs(actual - reference)
                    runs.append(dict(
                        phase=phase, wall_seconds=seconds, **counts,
                        max_pixel_difference=float(diff.max()),
                        mean_pixel_difference=float(diff.mean(dtype=np.float64)),
                        peak_allocated_gib=torch.cuda.max_memory_allocated(engine.device) / 1024**3 if cuda else None,
                        peak_reserved_gib=torch.cuda.max_memory_reserved(engine.device) / 1024**3 if cuda else None))
                    print(f'[ffn-compile-band] mode={mode} visit={visit} frames={length} phase={phase} '
                          f'seconds={seconds:.6f} graphs={counts["backend_compilations"]} '
                          f'max_diff={runs[-1]["max_pixel_difference"]}', flush=True)
                reports.append(dict(
                    visit=visit, frames=length, runs=runs, first_seconds=runs[0]['wall_seconds'],
                    warm_median_seconds=statistics.median(row['wall_seconds'] for row in runs
                                                         if row['phase'] == 'measured'),
                    backend_compilations=sum(row['backend_compilations'] for row in runs)))
            if mode == 'compiled' and not state['backend_compilations']:
                raise RuntimeError('没有实际图编译，基准无效')
            all_runs = [row for band in reports for row in band['runs']]
            max_diff = max(row['max_pixel_difference'] for row in all_runs)
            report = dict(
                mode=mode, compile_backend=backend, backend=settings, bands=reports, **state,
                setup_seconds=setup_seconds, setup_peak_allocated_gib=setup_allocated,
                setup_peak_reserved_gib=setup_reserved,
                outputs_match=max_diff == 0, max_pixel_difference=max_diff,
                mean_pixel_difference=statistics.mean(row['mean_pixel_difference'] for row in all_runs),
                requested_precision=expected, precision=engine.precision,
                precision_fallbacks=engine.precision_fallbacks)
            return report, references
    finally:
        torch.backends.cudnn.benchmark = previous


def run_worker(options, mode, reference_path):
    started = time.perf_counter()
    import numpy as np
    import torch
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
    bands = {}
    for length in [options['frames']] + options['shape_frames']:
        frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(length)]
        masks = [np.zeros((height, width), dtype=np.uint8) for _ in frames]
        for index, mask in enumerate(masks):
            if index % 10:
                x = width // 3 + index % 8
                mask[height // 2:height // 2 + 32, x:x + 128] = 255
        bands[length] = frames, masks
    reference_path = Path(reference_path)
    references = None
    if reference_path.exists():
        with np.load(reference_path, allow_pickle=False) as saved:
            references = {int(key): saved[key] for key in saved.files}
    report, references = measure_compile(
        engine, bands, mode=mode, backend=options['compile_backend'],
        warmup=options['warmup'], repeats=options['repeats'], references=references)
    if not reference_path.exists():
        np.savez(reference_path, **{str(key): value for key, value in references.items()})
    report.update(
        model_load_seconds=load_seconds, module_import_seconds=import_seconds,
        frame_shape=[height, width, 3], neighbor_stride=engine.neighbor_stride,
        ref_length=engine.ref_length, cache_first_qkv=engine.cache_first_qkv,
        environment=dict(pid=os.getpid(), python=sys.version, platform=platform.platform(),
                         torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                         cudnn=torch.backends.cudnn.version(), torch_threads=torch.get_num_threads(),
                         device=str(device), device_name=torch.cuda.get_device_name(device)
                         if device.type == 'cuda' else 'CPU',
                         inductor_cache_dir=os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
                         triton_cache_dir=os.environ.get('TRITON_CACHE_DIR')))
    return report


def summarize(sessions, *, gpu_comparison):
    """逐段长和访问顺序汇总，避免把首次编译或不同工作量混成一个加速比。"""
    match = all(row['outputs_match'] for row in sessions)
    summary = []
    for index, first in enumerate(sessions[0]['bands']):
        entry = dict(visit=first['visit'], frames=first['frames'])
        stable = True
        for mode in ('eager', 'compiled'):
            rows = [row for row in sessions if row['mode'] == mode]
            bands = [row['bands'][index] for row in rows]
            runs = [run for band in bands for run in band['runs']]
            measured = [run for run in runs if run['phase'] == 'measured']
            stable = stable and all(run['backend_compilations'] == 0 for run in measured)
            entry[mode] = dict(
                first_median_seconds=statistics.median(band['first_seconds'] for band in bands),
                warm_median_seconds=statistics.median(run['wall_seconds'] for run in runs
                                                     if run['phase'] == 'measured'),
                backend_compilations=[band['backend_compilations'] for band in bands],
                peak_allocated_gib=max(run['peak_allocated_gib'] for run in runs) if gpu_comparison else None,
                peak_reserved_gib=max(run['peak_reserved_gib'] for run in runs) if gpu_comparison else None)
        valid = gpu_comparison and match and stable and all(
            row['compile_backend'] == 'inductor' for row in sessions)
        old, new = (entry[mode]['warm_median_seconds'] for mode in ('eager', 'compiled'))
        entry.update(warm_stable=stable, warm_speedup=old / new if valid else None,
                     warm_time_saved_percent=100 * (1 - new / old) if valid else None)
        summary.append(entry)
    return summary


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
    parser.add_argument('--compile-backend', choices=('inductor', 'eager'), default='inductor',
                        help='eager 后端仅用于 CPU 图捕获冒烟，不代表编译加速')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--shape-frames', type=int, nargs='*', default=[12, 2],
                        help='在同一引擎中依次测试的其他段长，最后重访主段；默认 12 2')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--sessions', type=int, default=2, help='每模式独立进程数，交替执行顺序')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    lengths = [args.frames] + args.shape_frames
    if min(args.repeats, args.sessions, args.threads) < 1 or min(lengths) < 2 or args.warmup < 0:
        parser.error('段长至少为 2，repeats/sessions/threads 必须为正，warmup 必须非负')
    if len(set(lengths)) != len(lengths):
        parser.error('主段和 shape-frames 段长不能重复')
    if args.device == 'cpu' and args.precision != 'fp32':
        parser.error('CPU 冒烟需指定 --precision fp32，不能默许 FP16 回退')
    if args.device == 'cuda' and args.compile_backend != 'inductor':
        parser.error('GPU A/B 必须使用 --compile-backend inductor')
    args.model = args.model.resolve()
    if not args.model.is_file():
        parser.error(f'模型不存在: {args.model}')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    options = {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items() if key not in ('sessions', 'output_dir')}
    sessions = []
    for index in range(args.sessions):
        for mode in (('eager', 'compiled') if index % 2 == 0 else ('compiled', 'eager')):
            name = f'{index + 1:02d}-{mode}'
            request_path, report_path, log_path = (output_dir / f'{name}{suffix}'
                                                  for suffix in ('-request.json', '.json', '.log'))
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npz'))))
            env = os.environ.copy()
            env.update(TORCHINDUCTOR_CACHE_DIR=str(output_dir / 'compile-cache' / name / 'inductor'),
                       TRITON_CACHE_DIR=str(output_dir / 'compile-cache' / name / 'triton'),
                       TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0', TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE='0')
            print(f'[ffn-compile-start] session={index + 1} mode={mode} log={log_path}', flush=True)
            started = time.perf_counter()
            with log_path.open('w') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--_worker',
                                         str(request_path), str(report_path)],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} 子进程失败（{result.returncode}），请查看 {log_path}')
            session = json.loads(report_path.read_text())
            session.update(session=index + 1, log=log_path.name,
                           process_wall_seconds=time.perf_counter() - started)
            sessions.append(session)
            print(f'[ffn-compile-session] session={index + 1} mode={mode} '
                  f'graphs={session["backend_compilations"]} outputs_match={session["outputs_match"]}', flush=True)
    if any(row['backend'] != sessions[0]['backend'] for row in sessions[1:]):
        raise RuntimeError('子进程后端设置不一致，A/B 无效')
    match = all(row['outputs_match'] for row in sessions)
    report = dict(
        scope='inpaint_band', profile=False, options=options, sessions=sessions,
        summary=summarize(sessions, gpu_comparison=args.device == 'cuda'),
        gpu_comparison=args.device == 'cuda', outputs_match=match,
        max_pixel_difference=max(row['max_pixel_difference'] for row in sessions),
        mean_pixel_difference=statistics.mean(row['mean_pixel_difference'] for row in sessions),
        timing_notes=[
            'Fresh processes and separate Inductor/Triton disk cache directories per mode/session; remote caches disabled.',
            'Only FFN forwards are compiled with Inductor default mode, fullgraph=True, dynamic=None; other modules stay eager.',
            'cuDNN benchmark is false; no explicit weight/input layout changes; compiler may choose internal layouts/fusions.',
            'First band includes lazy graph capture, compilation and device initialization; backend_compile_seconds is only compiler delegate host wall.',
            'Setup excludes lazy compilation. No CUDA Events or profiler; lightweight FFN/compiled graph counters remain in all timed runs.',
            'Every shape has first/warmup/measured runs; revisit reuses the initial frames/masks after intervening shapes.',
            'Compiler budgets are explicit (64 per code object, 512 accumulated); measured-run compilations suppress warm speedup.',
            'Backend compilation count is compiler invocations, not an exact Dynamo guard-failure count; inputs include parameters and symbolic shapes.',
            'Synchronized band wall includes production checks, transfers and compositing; no empty_cache between bands.',
            'Each shape compares all float outputs with its first independent eager reference, without uint8 truncation.',
            'Pixel differences suppress all speedups and cause failure after report is saved; precision fallback/non-finite/compile errors fail the worker.',
            'CPU or eager backend only validates the tool; no GPU speedup claim. Synthetic lengths do not establish real-video throughput.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[ffn-compile-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
