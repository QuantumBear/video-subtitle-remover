"""独立进程比较 STTN FFN contiguous/channels_last，布局往返计入整带耗时。

仅基准中替换 FFN 实例 forward，不改生产模型；两组均关闭 cuDNN benchmark。
父进程只用标准库，输出目录必须为新目录。
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

# 复用已有的同步整带计时、浮点输出比较和精度校验；该模块顶层仅用标准库。
from benchmarks.sttn_cudnn_benchmark import backend_settings, measure_mode


@contextmanager
def ffn_channels_last(model):
    """仅 FFN 内使用 NHWC 存储，返回 NCHW 连续输出；异常也恢复权重布局。

    权重预先转换，计时区间内不重复转换权重。只记原 stride，不持有额外
    GPU 权重副本，避免人为抬高峰值。仅用于无梯度的顺序基准。
    """
    import torch
    from backend.inpaint.sttn.network_sttn import FeedForward

    forwards, weights = [], []
    try:
        for module in model.modules():
            if not isinstance(module, FeedForward):
                continue
            for layer in module.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    weight = layer.weight
                    weights.append((weight, weight.stride()))
                    weight.data = weight.detach().contiguous(memory_format=torch.channels_last)
            previous = module.forward
            forwards.append((module, 'forward' in module.__dict__, previous))

            def forward(x, original=previous):
                return original(x.contiguous(memory_format=torch.channels_last)).contiguous()

            module.forward = forward
        if not forwards:
            raise ValueError('模型没有可转换的 FFN')
        yield len(forwards)
    finally:
        for module, had_override, previous in reversed(forwards):
            if had_override:
                module.forward = previous
            else:
                del module.forward
        with torch.no_grad():
            for weight, stride in reversed(weights):
                if weight.stride() != stride:
                    restored = torch.empty_strided(weight.shape, stride,
                                                   device=weight.device, dtype=weight.dtype)
                    restored.copy_(weight)
                    weight.data = restored


def measure_layout(engine, frames, masks, *, mode, warmup, repeats, reference=None):
    """首次、预热、热态均严格对照基准；转换输入和输出的开销在 inpaint 内。"""
    import torch
    from backend.inpaint.sttn.network_sttn import FeedForward

    if mode not in ('contiguous', 'channels-last'):
        raise ValueError(f'未知 FFN 布局: {mode}')
    cuda = engine.device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
    started = time.perf_counter()
    context = ffn_channels_last(engine.model) if mode == 'channels-last' else nullcontext()
    with context:
        if cuda:
            torch.cuda.synchronize(engine.device)
        setup_seconds = time.perf_counter() - started
        setup_allocated = torch.cuda.max_memory_allocated(engine.device) / 1024**3 if cuda else None
        setup_reserved = torch.cuda.max_memory_reserved(engine.device) / 1024**3 if cuda else None
        report, reference = measure_mode(engine, frames, masks, enabled=False,
                                         warmup=warmup, repeats=repeats, reference=reference)
    report.update(mode=mode, ffn_modules=sum(isinstance(module, FeedForward)
                                           for module in engine.model.modules()),
                  layout_setup_seconds=setup_seconds,
                  layout_setup_peak_allocated_gib=setup_allocated,
                  layout_setup_peak_reserved_gib=setup_reserved)
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
    report, reference = measure_layout(engine, frames, masks, mode=mode,
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
    parser.add_argument('--sessions', type=int, default=2, help='每布局的新进程数，交替执行顺序')
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
        order = ('contiguous', 'channels-last') if index % 2 == 0 else ('channels-last', 'contiguous')
        for mode in order:
            name = f'{index + 1:02d}-{mode}'
            request_path = output_dir / f'{name}-request.json'
            report_path = output_dir / f'{name}.json'
            log_path = output_dir / f'{name}.log'
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npy'))))
            print(f'[ffn-session-start] session={index + 1} mode={mode} log={log_path}', flush=True)
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
            print(f'[ffn-session] session={index + 1} mode={mode} '
                  f'first={session["first_seconds"]:.6f}s warm={session["warm_median_seconds"]:.6f}s '
                  f'outputs_match={session["outputs_match"]}', flush=True)
    if any(row['backend'] != sessions[0]['backend'] for row in sessions[1:]):
        raise RuntimeError('子进程后端设置不一致，A/B 无效')
    summary = {}
    for mode in ('contiguous', 'channels-last'):
        rows = [row for row in sessions if row['mode'] == mode]
        warm = [run for row in rows for run in row['runs'] if run['phase'] == 'measured']
        summary[mode] = dict(
            first_median_seconds=statistics.median(row['first_seconds'] for row in rows),
            warm_median_seconds=statistics.median(run['wall_seconds'] for run in warm),
            layout_setup_median_seconds=statistics.median(row['layout_setup_seconds'] for row in rows),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in rows) if args.device == 'cuda' else None,
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows) if args.device == 'cuda' else None)
    match = all(row['outputs_match'] for row in sessions)
    valid_gpu = args.device == 'cuda' and match
    old_s, new_s = (summary[mode]['warm_median_seconds'] for mode in ('contiguous', 'channels-last'))
    report = dict(
        scope='inpaint_band', profile=False, options=options, sessions=sessions, summary=summary,
        gpu_comparison=args.device == 'cuda', outputs_match=match,
        max_pixel_difference=max(row['max_pixel_difference'] for row in sessions),
        mean_pixel_difference=statistics.mean(row['mean_pixel_difference'] for row in sessions),
        warm_speedup=old_s / new_s if valid_gpu else None,
        warm_time_saved_percent=100 * (1 - new_s / old_s) if valid_gpu else None,
        timing_notes=[
            'Fresh processes per layout/session; cuDNN benchmark is false in both layouts.',
            'Only FFN Conv2d weights and inputs use channels_last; FFN output returns to contiguous NCHW.',
            'Input/output layout conversion is inside every timed band; one-time weight setup is reported separately.',
            'First band excludes model loading and weight setup; includes remaining lazy initialization.',
            'Synchronized wall time includes production finite checks, transfers and compositing; no profiler.',
            'All phases compare float outputs against the first contiguous output, without uint8 truncation.',
            'Per-band peak memory includes model/workspace/allocator state; no empty_cache between bands.',
            'CPU runs only validate the tool. Speedup is null on CPU or if pixels differ.',
            'Synthetic frames and one band length do not establish real video quality or concurrency capacity.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[ffn-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
