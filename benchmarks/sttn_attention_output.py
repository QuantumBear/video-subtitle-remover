"""独立进程比较 STTN Attention 输出 cat 与直接写入，分配和复制计入整带。

仅替换基准实例 forward，不改生产网络；保留当前 uint8 CUDA 下载。
父进程仅用标准库，输出目录须不存在。
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

MODES = ('cat', 'direct')


def direct_attention_output(module, x, m, b, c, projected_qkv=None):
    """保持计算与 QKV 重排，仅将四路最终重排直接复制到连续 NCHW 目标。"""
    import torch

    bt, _, h, w = x.size()
    t = bt // b
    d_k = c // len(module.patchsize)
    if projected_qkv is None:
        _query, _key, _value = module.project_qkv(x)
    else:
        _query, _key, _value = projected_qkv
        if any(projected.shape != x.shape for projected in projected_qkv):
            raise ValueError('cached STTN Q/K/V shape must match feature shape')
    output = None
    for index, ((width, height), query, key, value) in enumerate(zip(
            module.patchsize, torch.chunk(_query, len(module.patchsize), dim=1),
            torch.chunk(_key, len(module.patchsize), dim=1),
            torch.chunk(_value, len(module.patchsize), dim=1))):
        out_w, out_h = w // width, h // height
        query = query.view(b, t, d_k, out_h, height, out_w, width)
        query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            b, t * out_h * out_w, d_k * height * width)
        key = key.view(b, t, d_k, out_h, height, out_w, width)
        key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            b, t * out_h * out_w, d_k * height * width)
        value = value.view(b, t, d_k, out_h, height, out_w, width)
        value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            b, t * out_h * out_w, d_k * height * width)
        y, _ = module.attention(query, key, value, None)
        if output is None:
            # autocast 下 x 可能仍为 FP32；目标必须跟实际 attention 输出 dtype。
            output = y.new_empty((bt, c, h, w))
        source = y.view(b, t, out_h, out_w, d_k, height, width).permute(0, 1, 4, 2, 5, 3, 6)
        target = output.view(b, t, c, h, w)[:, :, index * d_k:(index + 1) * d_k]
        target.view(b, t, d_k, out_h, height, out_w, width).copy_(source)
        del source, target, y
    return module.output_linear(output)


@contextmanager
def attention_output_execution(model, *, mode):
    """只覆盖 Attention 实例方法；正常和异常均恢复既有覆盖。"""
    from backend.inpaint.sttn.network_sttn import MultiHeadedAttention

    if mode not in MODES:
        raise ValueError(f'未知 Attention 输出模式: {mode}')
    forwards = []
    state = dict(multihead_calls=0, multihead_frame_visits=0, attention_branch_calls=0,
                 output_cat_calls=0, direct_branch_copy_calls=0)
    try:
        for module in model.modules():
            if not isinstance(module, MultiHeadedAttention):
                continue
            previous = module.forward
            forwards.append((module, 'forward' in module.__dict__, previous))

            def forward(x, m, b, c, projected_qkv=None, layer=module, original=previous):
                state['multihead_calls'] += 1
                state['multihead_frame_visits'] += x.shape[0]
                if mode == 'direct':
                    result = direct_attention_output(layer, x, m, b, c, projected_qkv)
                else:
                    result = original(x, m, b, c, projected_qkv=projected_qkv)
                state['attention_branch_calls'] += len(layer.patchsize)
                state['output_cat_calls'] += int(mode == 'cat')
                state['direct_branch_copy_calls'] += len(layer.patchsize) if mode == 'direct' else 0
                return result

            module.forward = forward
        if not forwards:
            raise ValueError('模型没有 MultiHeadedAttention')
        yield state
    finally:
        for module, had_override, previous in reversed(forwards):
            if had_override:
                module.forward = previous
            else:
                del module.forward


def measure_output(engine, frames, masks, *, mode, warmup, repeats, reference=None):
    """正式测量无 profiler；所有输出与独立 cat 基线严格比较。"""
    if mode not in MODES:
        raise ValueError(f'未知 Attention 输出模式: {mode}')
    if mode == 'direct' and reference is None:
        raise ValueError('direct 模式需要独立 cat reference')
    started = time.perf_counter()
    with attention_output_execution(engine.model, mode=mode) as state:
        setup_seconds = time.perf_counter() - started
        report, reference = measure_mode(engine, frames, masks, enabled=False,
                                         warmup=warmup, repeats=repeats, reference=reference)
    from backend.inpaint.sttn.network_sttn import MultiHeadedAttention

    report.update(mode=mode, **state, setup_seconds=setup_seconds,
                  attention_modules=sum(isinstance(layer, MultiHeadedAttention)
                                        for layer in engine.model.modules()),
                  qkv_fused_modules=sum(getattr(layer, '_qkv_weight', None) is not None
                                        for layer in engine.model.modules()))
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
    report, reference = measure_output(engine, frames, masks, mode=mode,
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
            setup_median_seconds=statistics.median(row['setup_seconds'] for row in rows),
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
        for mode in (MODES if index % 2 == 0 else tuple(reversed(MODES))):
            name = f'{index + 1:02d}-{mode}'
            request_path, report_path, log_path = (output_dir / f'{name}{suffix}'
                                                  for suffix in ('-request.json', '.json', '.log'))
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npy'))))
            print(f'[attention-output-start] session={index + 1} mode={mode} log={log_path}', flush=True)
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
            print(f'[attention-output-session] session={index + 1} mode={mode} '
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
            'Fresh sequential processes per mode/session; cuDNN benchmark=false, production fused QKV, '
            'first-layer cache, full last block, original decoder layout and immediate finite checks.',
            'Cat mode calls the original MultiHeadedAttention forward; direct mode writes each permuted branch '
            'to a slice of one newly allocated contiguous NCHW buffer before the same output convolution.',
            'Q/K/V projection and input reshape, matmuls, scaling, softmax and output convolution are unchanged. '
            'The output buffer uses the actual attention result dtype/device, including autocast.',
            'Output allocation, strided branch copies, slicing and matched lightweight counters are inside '
            'synchronized band wall. Setup only wraps instance methods; no cross-window buffer reuse.',
            'Branch/cat/copy counters describe selected successful Python paths, not measured CUDA kernels, '
            'allocator calls, FLOPs or memory traffic; all first/warmup/measured bands count.',
            'CUDA keeps production FP32 quantization and uint8 download; CPU keeps FP32 download. Masks, '
            'quantization position, window/frame order and recursive 0.5 mixing are unchanged.',
            'First band excludes imports/model loading/setup and includes remaining lazy initialization; '
            'per-band allocated/reserved peaks include model/workspace/allocator state; no empty_cache.',
            'All phases compare with the first independent cat reference without extra uint8 truncation; '
            'pixel differences suppress speedups and fail after saving report.',
            'Precision fallback/non-finite/execution errors fail the worker; no profiler/CUDA Events in timed bands.',
            'CPU runs validate the tool only; synthetic bands do not establish real-video throughput or concurrency capacity.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[attention-output-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
