"""独立进程比较 STTN FP32 回传与设备端 FP32 量化后 uint8 回传。

仅替换下载/量化边界；共用生产 CPU mask 合成，父进程只用标准库。
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


def fp32_download(prediction):
    """冻结接入前的下载路径，不随生产默认 dtype 改变。"""
    return prediction.float().cpu()


def quantized_download(prediction):
    """使用生产量化实现，CPU 模式也可验证工具与数值。"""
    from backend.inpaint.sttn_det_inpaint import _quantize_sttn_prediction

    return _quantize_sttn_prediction(prediction)


@contextmanager
def download_execution(*, mode):
    """仅供单线程顺序基准使用；正常/异常退出均恢复生产入口。"""
    from backend.inpaint import sttn_det_inpaint

    if mode not in ('fp32', 'uint8'):
        raise ValueError(f'未知下载模式: {mode}')
    previous_download = sttn_det_inpaint._download_sttn_prediction
    previous_process = sttn_det_inpaint._postprocess_sttn
    target = fp32_download if mode == 'fp32' else quantized_download
    state = dict(download_calls=0, download_frame_visits=0, download_payload_bytes=0)

    def download(prediction):
        result = target(prediction)
        state['download_calls'] += 1
        state['download_frame_visits'] += result.shape[0]
        state['download_payload_bytes'] += result.numel() * result.element_size()
        return result

    sttn_det_inpaint._download_sttn_prediction = download
    try:
        yield state
    finally:
        sttn_det_inpaint._download_sttn_prediction = previous_download
        sttn_det_inpaint._postprocess_sttn = previous_process


def measure_download(engine, frames, masks, *, mode, warmup, repeats, reference=None):
    if mode == 'uint8' and reference is None:
        raise ValueError('uint8 模式需要独立 fp32 回传参考输出')
    fused = sum(block.attention._qkv_weight is not None for block in engine.model.transformer)
    if engine.model.training or fused != len(engine.model.transformer):
        raise ValueError('下载 A/B 须保持各层生产 QKV 合并推理')
    started = time.perf_counter()
    with download_execution(mode=mode) as state:
        setup_seconds = time.perf_counter() - started
        report, reference = measure_mode(engine, frames, masks, enabled=False,
                                         warmup=warmup, repeats=repeats, reference=reference)
        if not state['download_calls']:
            raise RuntimeError('未执行预测下载，基准无效')
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
    report, reference = measure_download(engine, frames, masks, mode=mode,
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
    for mode in ('fp32', 'uint8'):
        rows = [row for row in sessions if row['mode'] == mode]
        warm = [run for row in rows for run in row['runs'] if run['phase'] == 'measured']
        summary[mode] = dict(
            first_median_seconds=statistics.median(row['first_seconds'] for row in rows),
            warm_median_seconds=statistics.median(run['wall_seconds'] for run in warm),
            setup_median_seconds=statistics.median(row['setup_seconds'] for row in rows),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in rows) if gpu_comparison else None,
            peak_reserved_gib=max(row['peak_reserved_gib'] for row in rows) if gpu_comparison else None)
    valid = gpu_comparison and all(row['outputs_match'] for row in sessions)
    old, new = (summary[mode]['warm_median_seconds'] for mode in ('fp32', 'uint8'))
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
        for mode in (('fp32', 'uint8') if index % 2 == 0 else ('uint8', 'fp32')):
            name = f'{index + 1:02d}-{mode}'
            request_path, report_path, log_path = (output_dir / f'{name}{suffix}'
                                                  for suffix in ('-request.json', '.json', '.log'))
            request_path.write_text(json.dumps(dict(options=options, mode=mode,
                                                    reference_path=str(output_dir / 'reference.npy'))))
            print(f'[download-start] session={index + 1} mode={mode} log={log_path}', flush=True)
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
            print(f'[download-session] session={index + 1} mode={mode} '
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
            'Fresh sequential processes per mode/session; production fused QKV, cache, full last block, batched CPU '
            'compositing and immediate finite checks; cuDNN benchmark=false.',
            'Modes name download dtype, not inference precision: fp32 downloads prediction.float() before NumPy '
            'multiply/cast; uint8 multiplies in device FP32 then truncates to uint8 before download.',
            'No clamp/round, normalization, finite-check, mask or recursive 0.5 mixing changes; both keep NCHW downloads.',
            'download_calls/frame_visits/payload_bytes count all first/warmup/measured windows; bytes are tensor '
            'payload, not measured PCIe traffic or total band transfers; CPU runs do not perform D2H transfers.',
            'Quantization reduces prediction payload to one quarter; this is not a claim about band time or total traffic.',
            'FP32 conversion, multiply, uint8 cast, temporary allocations/layout, download and CPU compositing are '
            'inside synchronized band wall, with the same lightweight download counters in both modes.',
            'Setup only replaces helpers; first band excludes loading/setup and includes remaining lazy initialization.',
            'All first/warmup/measured outputs compare with first independent fp32 reference without extra uint8 truncation.',
            'Per-band GPU peaks include model/workspace/allocator state; no empty_cache; no profiler/CUDA Events.',
            'Pixel differences suppress speedups and fail after saving report; precision fallback/non-finite/execution '
            'errors fail the worker.',
            'CPU runs validate the tool only; synthetic bands do not establish real-video throughput or concurrency capacity.'])
    (output_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'sessions'},
                     ensure_ascii=False, indent=2))
    print(f'[download-report] {output_dir / "report.json"}', flush=True)
    if not match:
        raise SystemExit('输出像素不一致，不能只比较速度；差异已保存在 report.json')


if __name__ == '__main__':
    main()
