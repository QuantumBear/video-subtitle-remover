"""独立进程比较 STTN 末层完整输出与仅目标帧输出卷积/FFN。

生产保持完整计算。实验保留前七层和末层完整 QKV/Attention，仅裁剪末层
输出卷积、残差和 FFN 的帧维度。父进程只用标准库，不加载模型。
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
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.sttn_cudnn_benchmark import measure_mode


@contextmanager
def last_block_execution(engine, *, mode):
    """仅用于单线程整带基准；按生产窗口记录目标数，退出时恢复实例方法。"""
    if mode not in ('full', 'targets'):
        raise ValueError(f'未知末层模式: {mode}')
    if engine.model.training or len(engine.model.transformer) != 8:
        raise ValueError('末层实验需要八层 STTN eval 模型')
    last = engine.model.transformer[-1]
    state = dict(last_block_calls=0, last_attention_frame_visits=0, target_frame_visits=0,
                 last_output_frame_visits=0, last_ffn_frame_visits=0, pruned_frame_visits=0)
    overrides = []
    pending_window = None
    active_targets = None
    previous_refs = engine.get_ref_index
    previous_infer = engine.model.infer
    previous_block = last.forward
    previous_output = last.attention.output_linear.forward
    previous_ffn = last.feed_forward.forward

    def replace(owner, name, function):
        overrides.append((owner, name, name in owner.__dict__, owner.__dict__.get(name)))
        setattr(owner, name, MethodType(function, owner))

    def references(_self, neighbor_ids, length):
        nonlocal pending_window
        refs = previous_refs(neighbor_ids, length)
        pending_window = (len(neighbor_ids), len(neighbor_ids) + len(refs))
        return refs

    def infer(_self, feat, masks, qkv_cache=None):
        nonlocal pending_window, active_targets
        if active_targets is not None or pending_window is None:
            raise RuntimeError('末层实验须由生产窗口顺序调用，不支持嵌套或并发')
        targets, total = pending_window
        pending_window = None
        if not 0 < targets <= total or feat.shape[0] != total or masks.shape[0] != total:
            raise RuntimeError('目标帧数与生产窗口形状不一致')
        active_targets = targets
        try:
            return previous_infer(feat, masks, qkv_cache=qkv_cache)
        finally:
            active_targets = None

    def block(_self, data):
        import torch

        if active_targets is None or data['b'] != 1 or torch.is_grad_enabled():
            raise RuntimeError('末层实验仅支持生产单带无梯度推理')
        x = data['x']
        total = x.shape[0]
        state['last_block_calls'] += 1
        state['last_attention_frame_visits'] += total
        state['target_frame_visits'] += active_targets
        if mode == 'full':
            return previous_block(data)
        # Attention 中的 Q/K/V、四路 patch 和矩阵乘保持完整；output_linear
        # 的临时包装在其输入处裁剪，之后仅对相同的目标帧执行残差和 FFN。
        x = x[:active_targets] + last.attention(
            x, data['m'], data['b'], data['c'], projected_qkv=data.get('qkv_cache'))
        x = x + last.feed_forward(x)
        state['pruned_frame_visits'] += total - active_targets
        return dict(x=x, m=data['m'], b=data['b'], c=data['c'])

    def output(_self, x):
        if active_targets is None:
            raise RuntimeError('缺少末层目标帧上下文')
        if mode == 'targets':
            x = x[:active_targets]
        state['last_output_frame_visits'] += x.shape[0]
        return previous_output(x)

    def ffn(_self, x):
        state['last_ffn_frame_visits'] += x.shape[0]
        return previous_ffn(x)

    try:
        replace(engine, 'get_ref_index', references)
        replace(engine.model, 'infer', infer)
        replace(last, 'forward', block)
        replace(last.attention.output_linear, 'forward', output)
        replace(last.feed_forward, 'forward', ffn)
        yield state
    finally:
        for owner, name, existed, previous in reversed(overrides):
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)


def measure_last_block(engine, frames, masks, *, mode, warmup, repeats, reference=None):
    if mode == 'targets' and reference is None:
        raise ValueError('targets 模式需要独立 full 参考输出')
    fused = sum(block.attention._qkv_weight is not None for block in engine.model.transformer)
    if engine.model.training or fused != len(engine.model.transformer):
        raise ValueError('末层 A/B 须保持各层生产 QKV 合并推理')
    started = time.perf_counter()
    with last_block_execution(engine, mode=mode) as state:
        setup_seconds = time.perf_counter() - started
        report, reference = measure_mode(engine, frames, masks, enabled=False,
                                         warmup=warmup, repeats=repeats, reference=reference)
        expected = (state['target_frame_visits'] if mode == 'targets'
                    else state['last_attention_frame_visits'])
        if (not state['last_block_calls'] or state['last_output_frame_visits'] != expected
                or state['last_ffn_frame_visits'] != expected):
            raise RuntimeError('末层输出卷积/FFN 帧次不一致，基准无效')
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
    report, reference = measure_last_block(engine, frames, masks, mode=mode,
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
    for mode in ('full', 'targets'):
        rows = [row for row in sessions if row['mode'] == mode]
        warm = [run for row in rows for run in row['runs'] if run['phase'] == 'measured']
        summary[mode] = dict(
            first_median_seconds=statistics.median(row['first_seconds'] for row in rows),
            warm_median_seconds=statistics.median(run['wall_seconds'] for run in warm),
            setup_median_seconds=statistics.median(row['setup_seconds'] for row in rows),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in rows) if gpu_comparison else None,
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows) if gpu_comparison else None)
    pruning = all(row['pruned_frame_visits'] > 0 for row in sessions if row['mode'] == 'targets')
    valid = gpu_comparison and pruning and all(row['outputs_match'] for row in sessions)
    old, new = (summary[mode]['warm_median_seconds'] for mode in ('full', 'targets'))
    return dict(summary=summary, pruning_effective=pruning,
                warm_speedup=old / new if valid else None,
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
        for mode in (('full', 'targets') if index % 2 == 0 else ('targets', 'full')):
            name = f'{index + 1:02d}-{mode}'
            request_path, report_path, log_path = (output_dir / f'{name}{suffix}'
                                                  for suffix in ('-request.json', '.json', '.log'))
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npy'))))
            print(f'[last-block-start] session={index + 1} mode={mode} log={log_path}', flush=True)
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
            print(f'[last-block-session] session={index + 1} mode={mode} '
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
            'Fresh sequential processes per mode/session; production fused QKV, cache, batched postprocess and immediate finite checks.',
            'Both modes use cuDNN benchmark=false and the same backend settings.',
            'Only final-block output convolution, residuals and FFN select target frames; all Q/K/V and attention remain full.',
            'Target frames are the production neighbor prefix; reference frames still participate in every attention layer.',
            'Frame-visit counters describe tensor batches, not GPU kernel counts or measured FLOPs.',
            'Slicing and lightweight counters in both modes are inside synchronized band wall; setup only installs instance wrappers.',
            'First band excludes model loading/setup and includes remaining lazy initialization.',
            'No profiler/CUDA Events; per-band GPU peaks include model/workspace/allocator state; no empty_cache between bands.',
            'All first/warmup/measured outputs compare with the first independent full reference without extra uint8 truncation.',
            'Pixel differences suppress speedups and fail after saving report; fallback/non-finite/execution errors fail the worker.',
            'No-pruning cases and CPU runs suppress speedups; synthetic bands do not establish real-video throughput or concurrency capacity.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[last-block-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
