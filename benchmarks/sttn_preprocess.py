"""固定输入、权重、窗口和 QKV 设置，对比旧/新 STTN CPU 预处理。

python benchmarks/sttn_preprocess.py --device cuda --frames 50 --warmup 3 --repeats 5
python benchmarks/sttn_preprocess.py --device cpu --preprocess-only
整带计时包含 CPU 前后处理和 GPU 推理，不含模型加载、OCR、ROI 缩放或视频编解码。
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.inpaint import sttn_det_inpaint
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor


def legacy_preprocess(frames, masks):
    """冻结改动前的完整预处理，仅供 A/B 使用。"""
    def to_tensors(images):
        return ToTorchFormatTensor()(Stack()(images))

    feats = to_tensors([np.asarray(f).copy() for f in frames]).unsqueeze(0) * 2 - 1
    binary_masks = [np.expand_dims((np.array(m) > 0.5).astype(np.uint8), 2)
                    for m in masks]
    masks_tensor = (to_tensors([np.asarray(m).copy() for m in masks])
                    .unsqueeze(0) > 0.5).float()
    return feats, masks_tensor, binary_masks


@contextmanager
def use_preprocess(implementation):
    """仅在此单线程基准内替换转换函数，不向生产引擎增加旧路径开关。"""
    previous = sttn_det_inpaint._preprocess_sttn
    sttn_det_inpaint._preprocess_sttn = implementation
    try:
        yield
    finally:
        sttn_det_inpaint._preprocess_sttn = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--model', type=Path,
                        default=ROOT / 'backend/models/sttn-det/sttn.pth')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1, help='PyTorch CPU 线程数')
    parser.add_argument('--cache', choices=('on', 'off'), default='on',
                        help='两条路径固定使用同一 QKV 设置；默认沿用引擎设置')
    parser.add_argument('--profile', action='store_true', help='整带分阶段计时；预热不输出')
    parser.add_argument('--preprocess-only', action='store_true',
                        help='只测 CPU 转换，不加载模型或执行 CUDA')
    args = parser.parse_args()
    if min(args.frames, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error('frames/repeats/threads 必须为正，warmup 必须非负')
    if args.preprocess_only and args.profile:
        parser.error('--profile 用于整带测量，不能与 --preprocess-only 同用')
    if not args.preprocess_only and args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('当前环境没有 CUDA；CPU 验证请指定 --device cpu')

    torch.set_num_threads(args.threads)
    device = torch.device('cpu' if args.preprocess_only else args.device)
    engine = None
    width, height = 432, 240
    if not args.preprocess_only:
        engine = sttn_det_inpaint.STTNDetInpaint(
            device, str(args.model), cache_first_qkv=args.cache == 'on')
        width, height = engine.model_input_width, engine.model_input_height
        expected_precision = engine.precision
        expected_fallbacks = engine.precision_fallbacks

    # 与 sttn_qkv_cache.py 相同的合成修复带。
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
              for _ in range(args.frames)]
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in frames]
    for i, mask in enumerate(masks):
        if i % 10:
            x = width // 3 + i % 8
            mask[height // 2:height // 2 + 32, x:x + 128] = 255

    implementations = {'legacy': legacy_preprocess,
                       'batched': sttn_det_inpaint._preprocess_sttn}
    reference = legacy_preprocess(frames, masks)
    actual = implementations['batched'](frames, masks)
    if (not all(torch.equal(a, b) for a, b in zip(reference[:2], actual[:2]))
            or not all(np.array_equal(a, b) for a, b in zip(reference[2], actual[2]))):
        raise RuntimeError('新旧预处理不等价，停止性能对比')
    del reference, actual

    def run(name):
        with use_preprocess(implementations[name]):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            result = (implementations[name](frames, masks) if args.preprocess_only
                      else engine.inpaint(frames, masks))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            peak = (torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == 'cuda' else None)
        if engine is not None and (engine.precision != expected_precision
                                   or engine.precision_fallbacks != expected_fallbacks):
            raise RuntimeError('测量中发生精度回退，A/B 无效；请先检查日志')
        return result, elapsed, peak

    for _ in range(args.warmup):
        for name in implementations:
            run(name)

    if engine is not None:
        engine.profile = args.profile
    timings = {name: [] for name in implementations}
    peaks = {name: [] for name in implementations}
    max_pixel_diff, total_pixel_diff, pixel_count = 0.0, 0.0, 0
    for repeat in range(args.repeats):
        outputs = {}
        order = ('legacy', 'batched') if repeat % 2 == 0 else ('batched', 'legacy')
        for name in order:
            result, elapsed, peak = run(name)
            if not args.preprocess_only:
                outputs[name] = result
            del result
            timings[name].append(elapsed)
            if peak is not None:
                peaks[name].append(peak)
            print(f'preprocess={name} repeat={repeat + 1} seconds={elapsed:.6f}', flush=True)
        if not args.preprocess_only:
            for old, new in zip(outputs['legacy'], outputs['batched']):
                diff = np.abs(old.astype(np.float32) - new.astype(np.float32))
                if not np.isfinite(diff).all():
                    raise RuntimeError('输出存在非有限值')
                max_pixel_diff = max(max_pixel_diff, float(diff.max()))
                total_pixel_diff += float(diff.sum(dtype=np.float64))
                pixel_count += diff.size

    legacy_s, batched_s = (statistics.median(timings[name]) for name in implementations)
    print(json.dumps({
        'scope': 'preprocess_only' if args.preprocess_only else 'inpaint_band',
        'profile': args.profile,
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__,
        'threads': torch.get_num_threads(), 'opencv_threads': cv2.getNumThreads(),
        'frames': args.frames, 'warmup': args.warmup, 'repeats': args.repeats,
        'precision': engine.precision if engine is not None else 'fp32',
        'precision_fallbacks': engine.precision_fallbacks if engine is not None else 0,
        'cache_active': (engine.cache_first_qkv and args.frames > engine.neighbor_stride
                         if engine is not None else False),
        'preprocess_exact_match': True,
        'legacy_seconds': legacy_s, 'batched_seconds': batched_s,
        'speedup': legacy_s / batched_s,
        'time_saved_percent': 100 * (1 - batched_s / legacy_s),
        'max_pixel_difference': max_pixel_diff if pixel_count else None,
        'mean_pixel_difference': total_pixel_diff / pixel_count if pixel_count else None,
        'legacy_peak_allocated_gib': max(peaks['legacy'], default=None),
        'batched_peak_allocated_gib': max(peaks['batched'], default=None),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
