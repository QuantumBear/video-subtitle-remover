"""独立进程比较 STTN decoder contiguous/channels_last，布局往返计入整带。

只在基准上下文中改变 decoder；保留生产 uint8 CUDA 下载和立即有限值检查。
父进程只用标准库，输出目录必须为新目录。
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

MODES = ('contiguous', 'channels-last')


@contextmanager
def decoder_layout(model, *, mode):
    """只改 decoder 权重及输入，输出恢复 NCHW 连续；退出恢复实例与原 stride。

    仅供无梯度顺序实验。只保存原 stride，不长期持有额外 GPU 权重副本。
    两模式都包装调用和帧次计数；基线直接调用原 decoder。
    """
    import torch

    if mode not in MODES:
        raise ValueError(f'未知 decoder 布局: {mode}')
    decoder = model.decoder
    had_override, previous = 'forward' in decoder.__dict__, decoder.forward
    weights = []
    state = dict(decoder_calls=0, decoder_frame_visits=0)
    try:
        if mode == 'channels-last':
            for layer in decoder.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    weight = layer.weight
                    weights.append((weight, weight.stride()))
                    weight.data = weight.detach().contiguous(memory_format=torch.channels_last)
            if not weights:
                raise ValueError('decoder 没有可转换的 Conv2d')

        def forward(x):
            state['decoder_calls'] += 1
            state['decoder_frame_visits'] += x.shape[0]
            if mode == 'channels-last':
                return previous(x.contiguous(memory_format=torch.channels_last)).contiguous()
            return previous(x)

        decoder.forward = forward
        yield state
    finally:
        if had_override:
            decoder.forward = previous
        elif 'forward' in decoder.__dict__:
            del decoder.forward
        with torch.no_grad():
            for weight, stride in reversed(weights):
                if weight.stride() != stride:
                    restored = torch.empty_strided(weight.shape, stride,
                                                   device=weight.device, dtype=weight.dtype)
                    restored.copy_(weight)
                    weight.data = restored


def measure_layout(engine, frames, masks, *, mode, warmup, repeats, reference=None):
    """测完整修复带；候选必须与独立 contiguous 输出比较，不能自建参考。"""
    import torch

    if mode not in MODES:
        raise ValueError(f'未知 decoder 布局: {mode}')
    if mode == 'channels-last' and reference is None:
        raise ValueError('channels-last 需要独立 contiguous reference')
    cuda = engine.device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
    started = time.perf_counter()
    with decoder_layout(engine.model, mode=mode) as state:
        if cuda:
            torch.cuda.synchronize(engine.device)
        setup_seconds = time.perf_counter() - started
        setup_allocated = torch.cuda.max_memory_allocated(engine.device) / 1024**3 if cuda else None
        setup_reserved = torch.cuda.max_memory_reserved(engine.device) / 1024**3 if cuda else None
        report, reference = measure_mode(engine, frames, masks, enabled=False,
                                         warmup=warmup, repeats=repeats, reference=reference)
    report.update(
        mode=mode, **state,
        decoder_conv_layers=sum(isinstance(layer, torch.nn.Conv2d)
                                for layer in engine.model.decoder.modules()),
        qkv_fused_modules=sum(getattr(layer, '_qkv_weight', None) is not None
                              for layer in engine.model.modules()),
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
        prediction_download_dtype='uint8' if device.type == 'cuda' else 'fp32',
        environment=dict(pid=os.getpid(), python=sys.version, platform=platform.platform(),
                         torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                         cudnn=torch.backends.cudnn.version(), torch_threads=torch.get_num_threads(),
                         device=str(device), device_name=torch.cuda.get_device_name(device)
                         if device.type == 'cuda' else 'CPU'))
    return report


def summarize(sessions, *, gpu_comparison):
    summary = {}
    for mode in MODES:
        rows = [row for row in sessions if row['mode'] == mode]
        warm = [run for row in rows for run in row['runs'] if run['phase'] == 'measured']
        summary[mode] = dict(
            first_median_seconds=statistics.median(row['first_seconds'] for row in rows),
            warm_median_seconds=statistics.median(run['wall_seconds'] for run in warm),
            layout_setup_median_seconds=statistics.median(row['layout_setup_seconds'] for row in rows),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in rows) if gpu_comparison else None,
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows) if gpu_comparison else None)
    valid = gpu_comparison and all(row['outputs_match'] for row in sessions)
    old, new = (summary[mode]['warm_median_seconds'] for mode in MODES)
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
    parser.add_argument('--sessions', type=int, default=2, help='每布局独立进程数，交替执行顺序')
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
        for mode in (MODES if index % 2 == 0 else tuple(reversed(MODES))):
            name = f'{index + 1:02d}-{mode}'
            request_path, report_path, log_path = (output_dir / f'{name}{suffix}'
                                                  for suffix in ('-request.json', '.json', '.log'))
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npy'))))
            print(f'[decoder-start] session={index + 1} mode={mode} log={log_path}', flush=True)
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
            print(f'[decoder-session] session={index + 1} mode={mode} '
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
            'Fresh sequential processes per layout/session; production fused QKV, cache, full last block, '
            'batched CPU compositing and immediate finite checks; cuDNN benchmark=false.',
            'Only decoder Conv2d weights and decoder input use channels_last in the candidate; decoder output '
            'returns to contiguous NCHW before finite checks, tanh, normalization and production download.',
            'Bilinear interpolation with align_corners=True, masks, frame/window order, precision and recursive '
            '0.5 mixing are unchanged. CUDA uses production device FP32 quantization and uint8 download; CPU uses FP32.',
            'Input/output layout conversions and identical lightweight decoder counters are inside timed bands; '
            'decoder_calls/frame_visits include first, warmup and measured bands, not GPU kernel counts.',
            'Synchronized layout_setup_seconds includes one-time weight conversion; first band excludes loading/setup '
            'and includes remaining lazy initialization. Weight layout restoration occurs after band timing.',
            'Setup and per-band GPU peaks include model/workspace/allocator state; no extra retained weight copy, '
            'no empty_cache between bands and no profiler/CUDA Events.',
            'All first/warmup/measured outputs compare with the first independent contiguous reference without '
            'extra uint8 truncation; pixel differences suppress speedups and fail after saving report.',
            'Precision fallback/non-finite/execution errors fail the worker; CPU runs validate the tool only.',
            'Synthetic bands do not establish real-video throughput, quality or concurrency capacity.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[decoder-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
