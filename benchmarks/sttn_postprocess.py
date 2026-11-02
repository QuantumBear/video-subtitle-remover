"""独立进程比较 STTN 原逐帧与窗口内批量 CPU 后处理。

保持生产 QKV 合并、量化与跨窗口递推顺序；父进程只用标准库。
输出目录必须不存在，所有模式/会话顺序执行。
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


def legacy_postprocess(pred_img, neighbor_ids, frames, binary_masks, comp_frames):
    """冻结生产接入之前的逐帧实现，保留独立 A/B 基线。"""
    import numpy as np

    pred_img = pred_img.permute(0, 2, 3, 1).numpy() * 255
    for i, idx in enumerate(neighbor_ids):
        img = (pred_img[i].astype(np.uint8) * binary_masks[idx]
               + frames[idx] * (1 - binary_masks[idx]))
        if comp_frames[idx] is None:
            comp_frames[idx] = img
        else:
            comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5
                                + img.astype(np.float32) * 0.5)


@contextmanager
def postprocess_execution(*, mode):
    """只在单线程基准中替换后处理，异常时也恢复原函数。"""
    from backend.inpaint import sttn_det_inpaint

    if mode not in ('legacy', 'batched'):
        raise ValueError(f'未知后处理模式: {mode}')
    previous = sttn_det_inpaint._postprocess_sttn
    target = legacy_postprocess if mode == 'legacy' else previous
    state = dict(postprocess_calls=0, postprocess_frame_visits=0)

    def process(prediction, ids, frames, masks, composites):
        target(prediction, ids, frames, masks, composites)
        state['postprocess_calls'] += 1
        state['postprocess_frame_visits'] += len(ids)

    sttn_det_inpaint._postprocess_sttn = process
    try:
        yield state
    finally:
        sttn_det_inpaint._postprocess_sttn = previous


def process_peak_rss_gib():
    """进程启动以来的 RSS 高水位；不是单带峰值或 NumPy 分配量。"""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024**3 if sys.platform == 'darwin' else peak / 1024**2


def measure_postprocess(engine, frames, masks, *, mode, warmup, repeats, reference=None):
    if mode == 'batched' and reference is None:
        raise ValueError('batched 模式需要独立 legacy 参考输出')
    fused = sum(block.attention._qkv_weight is not None for block in engine.model.transformer)
    if engine.model.training or fused != len(engine.model.transformer):
        raise ValueError('后处理 A/B 须保持各层生产 QKV 合并推理')
    rss_before = process_peak_rss_gib()
    started = time.perf_counter()
    with postprocess_execution(mode=mode) as state:
        setup_seconds = time.perf_counter() - started
        report, reference = measure_mode(engine, frames, masks, enabled=False,
                                         warmup=warmup, repeats=repeats, reference=reference)
        if not state['postprocess_calls']:
            raise RuntimeError('未执行后处理，基准无效')
        report.update(mode=mode, **state, setup_seconds=setup_seconds,
                      qkv_fused_modules=fused, process_peak_rss_before_gib=rss_before,
                      process_peak_rss_gib=process_peak_rss_gib())
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
    report, reference = measure_postprocess(engine, frames, masks, mode=mode,
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
    for mode in ('legacy', 'batched'):
        rows = [row for row in sessions if row['mode'] == mode]
        warm = [run for row in rows for run in row['runs'] if run['phase'] == 'measured']
        rss = [row['process_peak_rss_gib'] for row in rows if row['process_peak_rss_gib'] is not None]
        summary[mode] = dict(
            first_median_seconds=statistics.median(row['first_seconds'] for row in rows),
            warm_median_seconds=statistics.median(run['wall_seconds'] for run in warm),
            setup_median_seconds=statistics.median(row['setup_seconds'] for row in rows),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in rows) if gpu_comparison else None,
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows) if gpu_comparison else None,
            process_peak_rss_gib=max(rss, default=None))
    valid = gpu_comparison and all(row['outputs_match'] for row in sessions)
    old, new = (summary[mode]['warm_median_seconds'] for mode in ('legacy', 'batched'))
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
        for mode in (('legacy', 'batched') if index % 2 == 0 else ('batched', 'legacy')):
            name = f'{index + 1:02d}-{mode}'
            request_path, report_path, log_path = (output_dir / f'{name}{suffix}'
                                                  for suffix in ('-request.json', '.json', '.log'))
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npy'))))
            print(f'[postprocess-start] session={index + 1} mode={mode} log={log_path}', flush=True)
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
            print(f'[postprocess-session] session={index + 1} mode={mode} '
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
            'Fresh sequential processes per mode/session; both use production fused QKV and cuDNN benchmark=false.',
            'Only the CPU postprocess helper changes; tensors are downloaded as FP32 using the production path.',
            'Batched mode quantizes the window before applying per-frame binary masks; recursive 0.5 mixing order is unchanged.',
            'Batched NumPy allocations, frame/mask stacking, conversion and compositing are inside synchronized band wall.',
            'Setup only wraps the helper; first band excludes model loading and setup, includes remaining lazy initialization.',
            'All first/warmup/measured outputs compare with the first independent legacy reference without extra uint8 truncation.',
            'Per-band allocated/reserved peaks include GPU model/workspace/allocator state; no empty_cache between bands.',
            'process_peak_rss_gib is the process lifetime RSS high-water mark, including imports/loading and output comparisons; '
            'it cannot be reset or attributed to one band, and is null when resource is unavailable.',
            'Lightweight helper counters remain in both timed modes; no profiler/CUDA Events or sampling thread.',
            'Pixel differences suppress speedups and fail after saving report; precision fallback/non-finite values fail the worker.',
            'CPU runs validate the tool only; synthetic bands do not establish real-video throughput or concurrency capacity.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[postprocess-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
