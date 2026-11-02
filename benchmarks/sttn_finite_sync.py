"""独立进程比较 STTN 有限值检查立即读取与预测下载后读取。

生产仍立即检查；实验保留两次 GPU reduction，将两个 bool 异步拷到 pinned
CPU 缓冲区，由原预测 .float().cpu() 等待后读取。父进程只用标准库。
"""

import argparse
from contextlib import contextmanager
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

from benchmarks.sttn_cudnn_benchmark import backend_settings, measure_mode


@contextmanager
def finite_execution(engine, *, mode):
    """单线程基准专用；所有标志须在量化/合成之前验证。"""
    import torch
    from backend.inpaint import sttn_det_inpaint

    if mode not in ('immediate', 'deferred'):
        raise ValueError(f'未知有限值检查模式: {mode}')
    cuda = engine.device.type == 'cuda'
    deferred = mode == 'deferred'
    # 一次准备，按窗口复用。只固定两个 bool，不固定整幅输出或改变下载。
    flags = torch.empty(2, dtype=torch.bool, pin_memory=cuda) if deferred else None
    slots = flags.unbind() if deferred else None
    messages = ('STTN non-finite decoder output', 'STTN non-finite output')
    state = dict(decoder_checks=0, prediction_checks=0, completed_windows=0,
                 early_scalar_reads=0, flag_d2h_copies=0,
                 pinned_flag_bytes=2 if deferred and cuda else 0)
    previous_check = sttn_det_inpaint._check_sttn_finite
    previous_post = sttn_det_inpaint._postprocess_sttn
    pending = []
    pending_stream = None

    def drain():
        nonlocal pending_stream
        # 异常可能发生在异步拷贝与原 .cpu() 之间，缓冲区释放/重用前须完成拷贝。
        if pending_stream is not None:
            pending_stream.synchronize()
            pending_stream = None
        pending.clear()

    def check(tensor, message):
        nonlocal pending_stream
        if message not in messages:
            raise RuntimeError(f'未知 STTN 数值检查: {message}')
        index = messages.index(message)
        state['decoder_checks' if index == 0 else 'prediction_checks'] += 1
        if not deferred:
            state['early_scalar_reads'] += 1
            return previous_check(tensor, message)
        if index == 0 and pending:
            # inpaint 内部 FP32 重试可能早于上下文退出；丢弃失败窗口的旧标志。
            drain()
        if len(pending) != index:
            raise RuntimeError('STTN 有限值检查顺序不完整')
        result = torch.isfinite(tensor).all()
        if cuda:
            stream = torch.cuda.current_stream(engine.device)
            if pending_stream is not None and stream != pending_stream:
                raise RuntimeError('有限值检查实验仅支持每窗口单 stream')
            pending_stream = stream
        slots[index].copy_(result, non_blocking=cuda)
        pending.append(result)  # 保持 GPU 源标志存活，直到下载完成。
        state['flag_d2h_copies'] += int(cuda)

    def postprocess(prediction, ids, frames, masks, composites):
        nonlocal pending_stream
        if deferred:
            if prediction.device.type != 'cpu' or len(pending) != 2:
                raise RuntimeError('必须在原预测 CPU 下载完成后验证两项检查')
            if cuda and torch.cuda.current_stream(engine.device) != pending_stream:
                raise RuntimeError('检查标志与预测下载必须使用同一 stream')
            # 生产 .float().cpu() 已等待同 stream 的两次标志拷贝，无需新增同步。
            values = flags.tolist()
            pending_stream = None
            pending.clear()
            for valid, message in zip(values, messages):
                if not valid:
                    raise RuntimeError(message)
        previous_post(prediction, ids, frames, masks, composites)
        state['completed_windows'] += 1

    sttn_det_inpaint._check_sttn_finite = check
    sttn_det_inpaint._postprocess_sttn = postprocess
    try:
        yield state
    finally:
        # 先恢复函数，即使 CUDA 在错误收尾中抛错也不泄漏全局替换。
        sttn_det_inpaint._check_sttn_finite = previous_check
        sttn_det_inpaint._postprocess_sttn = previous_post
        drain()


def measure_finite(engine, frames, masks, *, mode, warmup, repeats, reference=None):
    if mode == 'deferred' and reference is None:
        raise ValueError('deferred 模式需要独立 immediate 参考输出')
    fused = sum(block.attention._qkv_weight is not None for block in engine.model.transformer)
    if engine.model.training or fused != len(engine.model.transformer):
        raise ValueError('数值检查 A/B 须保持各层生产 QKV 合并推理')
    started = time.perf_counter()
    with finite_execution(engine, mode=mode) as state:
        setup_seconds = time.perf_counter() - started
        report, reference = measure_mode(engine, frames, masks, enabled=False,
                                         warmup=warmup, repeats=repeats, reference=reference)
        windows = state['completed_windows']
        if not windows or state['decoder_checks'] != windows or state['prediction_checks'] != windows:
            raise RuntimeError('未执行完整的窗口数值检查，基准无效')
        report.update(mode=mode, **state, setup_seconds=setup_seconds, qkv_fused_modules=fused)
    return report, reference


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
    frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
              for _ in range(options['frames'])]
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in frames]
    for index, mask in enumerate(masks):
        if index % 10:
            x = width // 3 + index % 8
            mask[height // 2:height // 2 + 32, x:x + 128] = 255
    reference_path = Path(reference_path)
    reference = np.load(reference_path, allow_pickle=False) if reference_path.exists() else None
    report, reference = measure_finite(engine, frames, masks, mode=mode,
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


def summarize(sessions, *, gpu_comparison):
    summary = {}
    for mode in ('immediate', 'deferred'):
        rows = [row for row in sessions if row['mode'] == mode]
        warm = [run for row in rows for run in row['runs'] if run['phase'] == 'measured']
        summary[mode] = dict(
            first_median_seconds=statistics.median(row['first_seconds'] for row in rows),
            warm_median_seconds=statistics.median(run['wall_seconds'] for run in warm),
            setup_median_seconds=statistics.median(row['setup_seconds'] for row in rows),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in rows) if gpu_comparison else None,
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows) if gpu_comparison else None)
    valid = gpu_comparison and all(row['outputs_match'] for row in sessions)
    old, new = (summary[mode]['warm_median_seconds'] for mode in ('immediate', 'deferred'))
    return dict(summary=summary, warm_speedup=old / new if valid else None,
                warm_time_saved_percent=100 * (1 - new / old) if valid else None)


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
    parser.add_argument('--sessions', type=int, default=2, help='每模式独立进程数，交替执行顺序')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.repeats, args.sessions, args.threads) < 1 or args.frames < 2 or args.warmup < 0:
        parser.error('frames 至少为 2，repeats/sessions/threads 必须为正，warmup 必须非负')
    if args.device == 'cpu' and args.precision != 'fp32':
        parser.error('CPU 冒烟需指定 --precision fp32，不能默许 FP16 回退')
    args.model = args.model.resolve()
    if not args.model.is_file():
        parser.error(f'模型不存在: {args.model}')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    options = {key: str(value) if isinstance(value, Path) else value
               for key, value in vars(args).items() if key not in ('sessions', 'output_dir')}
    sessions = []
    for index in range(args.sessions):
        for mode in (('immediate', 'deferred') if index % 2 == 0 else ('deferred', 'immediate')):
            name = f'{index + 1:02d}-{mode}'
            request_path, report_path, log_path = (output_dir / f'{name}{suffix}'
                                                  for suffix in ('-request.json', '.json', '.log'))
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npy'))))
            print(f'[finite-sync-start] session={index + 1} mode={mode} log={log_path}', flush=True)
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
            print(f'[finite-sync-session] session={index + 1} mode={mode} '
                  f'first={session["first_seconds"]:.6f}s warm={session["warm_median_seconds"]:.6f}s '
                  f'outputs_match={session["outputs_match"]}', flush=True)
    if any(row['backend'] != sessions[0]['backend'] for row in sessions[1:]):
        raise RuntimeError('子进程后端设置不一致，A/B 无效')
    match = all(row['outputs_match'] for row in sessions)
    report = dict(
        scope='inpaint_band', profile=False, options=options, sessions=sessions,
        **summarize(sessions, gpu_comparison=args.device == 'cuda'),
        gpu_comparison=args.device == 'cuda', outputs_match=match,
        max_pixel_difference=max(row['max_pixel_difference'] for row in sessions),
        mean_pixel_difference=statistics.mean(row['mean_pixel_difference'] for row in sessions),
        timing_notes=[
            'Fresh sequential processes per mode/session; production fused QKV, batched CPU postprocess and cuDNN benchmark=false.',
            'Both modes keep decoder and normalized prediction finite reductions; only host reading time changes.',
            'Deferred mode enqueues two non-blocking bool copies into a reusable pinned CPU buffer on the current stream.',
            'Original prediction float().cpu() is unchanged and waits for those copies; flags are read before quantization/compositing.',
            'Pinned buffer payload is two bytes, not its allocator page footprint; allocation is setup, copies and validation are inside timed bands.',
            'No added synchronize in successful timed windows; aborted pending copies are drained only during retry/context cleanup.',
            'early_scalar_reads and flag_d2h_copies are path counters, not measured CUDA synchronization or kernel counts.',
            'First band excludes model loading/setup, includes remaining lazy initialization; synchronized band wall includes transfers/compositing.',
            'Both modes keep lightweight check/window counters; no profiler/CUDA Events; no empty_cache between bands.',
            'All first/warmup/measured outputs compare with first independent immediate reference without extra uint8 truncation.',
            'Pixel differences suppress speedups and fail after report is saved; precision fallback/non-finite/execution errors fail the worker.',
            'CPU runs validate the tool only; synthetic bands do not establish real-video throughput or concurrency capacity.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[finite-sync-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
