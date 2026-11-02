"""独立进程比较 STTN 三次 Q/K/V 卷积与拼接权重后的一次卷积。

仅在基准内覆盖 project_qkv 实例方法；首层缓存与生产推理流程保持原样。
父进程只用标准库，输出目录必须不存在。
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
def qkv_execution(model, *, mode):
    """拼接发生在计时前，拆分视图在带内；退出时恢复实例覆盖。

    保留原参数，packed 权重属于额外常驻开销，必须计入显存。使用叶子
    Parameter 保持生产 autocast 的权重缓存条件，但仅支持无梯度顺序推理。
    """
    import torch
    from backend.inpaint.sttn.network_sttn import MultiHeadedAttention

    if mode not in ('separate', 'fused'):
        raise ValueError(f'未知 Q/K/V 模式: {mode}')
    if model.training:
        raise ValueError('Q/K/V 基准仅支持 eval 模型')
    state = dict(qkv_modules=0, qkv_calls=0, qkv_conv_calls=0, packed_weight_bytes=0)
    methods = []
    try:
        for module in list(model.modules()):
            if not isinstance(module, MultiHeadedAttention):
                continue
            previous = module.project_qkv
            if mode == 'fused':
                layers = [module.query_embedding, module.key_embedding, module.value_embedding]
                first = layers[0]
                if any(not isinstance(layer, torch.nn.Conv2d)
                       or layer.kernel_size != (1, 1) or layer.stride != (1, 1)
                       or layer.padding != (0, 0) or layer.dilation != (1, 1)
                       or layer.groups != 1 or layer.padding_mode != 'zeros'
                       or layer.in_channels != first.in_channels
                       or layer.out_channels != first.in_channels or layer.bias is None
                       for layer in layers):
                    raise ValueError('仅支持相同通道数、带 bias 的标准 1×1 Q/K/V 卷积')
                weight = torch.nn.Parameter(torch.cat([layer.weight.detach() for layer in layers]),
                                            requires_grad=first.weight.requires_grad)
                bias = torch.nn.Parameter(torch.cat([layer.bias.detach() for layer in layers]),
                                          requires_grad=first.bias.requires_grad)
                state['packed_weight_bytes'] += sum(t.numel() * t.element_size() for t in (weight, bias))

                def target(x, weight=weight, bias=bias, channels=first.out_channels):
                    return torch.nn.functional.conv2d(x, weight, bias).split(channels, dim=1)

            else:
                # 生产推理已默认合并，基线须显式执行原来的三个卷积。
                def target(x, module=module):
                    return (module.query_embedding(x), module.key_embedding(x),
                            module.value_embedding(x))

            def project(x, target=target):
                result = target(x)
                state['qkv_calls'] += 1
                state['qkv_conv_calls'] += 1 if mode == 'fused' else 3
                return result

            methods.append((module, 'project_qkv' in module.__dict__, previous))
            module.project_qkv = project
        state['qkv_modules'] = len(methods)
        if not methods:
            raise ValueError('模型没有 Q/K/V 投影')
        yield state
    finally:
        for module, had_override, previous in reversed(methods):
            if had_override:
                module.project_qkv = previous
            else:
                del module.project_qkv


def measure_qkv(engine, frames, masks, *, mode, warmup, repeats, reference=None):
    """复用同步整带计时和严格浮点比较，两种模式都记录调用次数。"""
    import torch

    if mode == 'fused' and reference is None:
        raise ValueError('fused 模式需要独立 separate 参考输出')
    cuda = engine.device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(engine.device)
        torch.cuda.reset_peak_memory_stats(engine.device)
    started = time.perf_counter()
    with qkv_execution(engine.model, mode=mode) as state:
        if cuda:
            torch.cuda.synchronize(engine.device)
        setup_seconds = time.perf_counter() - started
        setup_allocated = torch.cuda.max_memory_allocated(engine.device) / 1024**3 if cuda else None
        setup_reserved = torch.cuda.max_memory_reserved(engine.device) / 1024**3 if cuda else None
        report, reference = measure_mode(engine, frames, masks, enabled=False,
                                         warmup=warmup, repeats=repeats, reference=reference)
        if not state['qkv_calls']:
            raise RuntimeError('没有执行 Q/K/V 投影，基准无效')
        report.update(mode=mode, **state, setup_seconds=setup_seconds,
                      setup_peak_allocated_gib=setup_allocated,
                      setup_peak_reserved_gib=setup_reserved)
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
                           cache_first_qkv=options['cache'] == 'on', profile=False, fuse_qkv=False)
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
    report, reference = measure_qkv(engine, frames, masks, mode=mode,
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
    for mode in ('separate', 'fused'):
        rows = [row for row in sessions if row['mode'] == mode]
        warm = [run for row in rows for run in row['runs'] if run['phase'] == 'measured']
        summary[mode] = dict(
            first_median_seconds=statistics.median(row['first_seconds'] for row in rows),
            warm_median_seconds=statistics.median(run['wall_seconds'] for run in warm),
            setup_median_seconds=statistics.median(row['setup_seconds'] for row in rows),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in rows) if gpu_comparison else None,
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows) if gpu_comparison else None)
    valid = gpu_comparison and all(row['outputs_match'] for row in sessions)
    old, new = (summary[mode]['warm_median_seconds'] for mode in ('separate', 'fused'))
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
        for mode in (('separate', 'fused') if index % 2 == 0 else ('fused', 'separate')):
            name = f'{index + 1:02d}-{mode}'
            request_path, report_path, log_path = (output_dir / f'{name}{suffix}'
                                                  for suffix in ('-request.json', '.json', '.log'))
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npy'))))
            print(f'[qkv-start] session={index + 1} mode={mode} log={log_path}', flush=True)
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
            print(f'[qkv-session] session={index + 1} mode={mode} '
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
            'Fresh processes per mode/session; cuDNN benchmark is false in both modes.',
            'Only project_qkv instance methods change: Q/K/V weights and biases concatenate in that order.',
            'Packed parameters preserve the original dtype and autocast weight-cache eligibility; no explicit layout changes.',
            'Original parameters remain allocated; packed_weight_bytes reports the extra persistent weights.',
            'Packing is in setup_seconds; split views and downstream handling are inside timed bands.',
            'First band excludes model loading and setup; includes remaining lazy initialization.',
            'First-layer QKV cache, windows, precision checks and compositing use the production path.',
            'Synchronized band wall includes production transfers/compositing and lightweight counters; no profiler/CUDA Events.',
            'qkv_conv_calls counts Conv2d operations in the selected projection path, not GPU kernels.',
            'All phases compare float outputs with the first independent separate reference, without extra uint8 truncation.',
            'Per-band memory includes original/packed weights, workspace and allocator state; no empty_cache between bands.',
            'Pixel differences suppress speedups and fail after saving report; precision fallback/non-finite values fail the worker.',
            'CPU validates the tool only; synthetic bands do not establish real-video throughput or concurrency capacity.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[qkv-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
