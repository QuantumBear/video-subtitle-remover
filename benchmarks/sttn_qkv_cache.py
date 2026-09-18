"""使用真实 STTN 权重对比首层 Q/K/V 缓存；输入是固定随机种子的合成修复带。

python benchmarks/sttn_qkv_cache.py --device cuda --frames 50
计时包含缓存构建、窗口切片、编码和解码，不包含模型加载、OCR 或 ProPainter。
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.inpaint.sttn_det_inpaint import STTNDetInpaint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "backend/models/sttn-det/sttn.pth")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1, help="CPU 线程数")
    parser.add_argument("--profile", action="store_true",
                        help="对比时输出 STTN 分阶段耗时；预热阶段不输出")
    args = parser.parse_args()
    if min(args.frames, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error("frames/repeats/threads 必须为正，warmup 必须非负")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境没有 CUDA；CPU 功能验证可指定 --device cpu")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    engine = STTNDetInpaint(device, str(args.model))
    width, height = engine.model_input_width, engine.model_input_height
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
              for _ in range(args.frames)]
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in frames]
    for i, mask in enumerate(masks):
        if i % 10:
            x = width // 3 + i % 8
            mask[height // 2:height // 2 + 32, x:x + 128] = 255

    def run(cached):
        engine.cache_first_qkv = cached
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        # Stack 会修改输入列表；每次都用新列表，帧本身保持相同。
        result = engine.inpaint(frames.copy(), masks.copy())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak = (torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda" else None)
        return result, elapsed, peak

    for _ in range(args.warmup):
        for cached in (False, True):
            run(cached)

    engine.profile = args.profile
    timings = {False: [], True: []}
    peaks = {False: [], True: []}
    max_pixel_diff = 0.0
    total_pixel_diff, pixel_count = 0.0, 0
    for repeat in range(args.repeats):
        outputs = {}
        # 交替顺序，减小先后运行顺序对结果的影响。
        for cached in ((False, True) if repeat % 2 == 0 else (True, False)):
            outputs[cached], elapsed, peak = run(cached)
            timings[cached].append(elapsed)
            if peak is not None:
                peaks[cached].append(peak)
            print(f"cache={cached} repeat={repeat + 1} seconds={elapsed:.3f}", flush=True)
        for direct, cached in zip(outputs[False], outputs[True]):
            diff = np.abs(direct.astype(np.float32) - cached.astype(np.float32))
            if not np.isfinite(diff).all():
                raise RuntimeError("输出存在非有限值")
            max_pixel_diff = max(max_pixel_diff, float(diff.max()))
            total_pixel_diff += float(diff.sum(dtype=np.float64))
            pixel_count += diff.size

    direct_s, cached_s = (statistics.median(timings[key]) for key in (False, True))
    print(json.dumps({
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "frames": args.frames,
        "cache_active": args.frames > engine.neighbor_stride,
        "uncached_seconds": direct_s,
        "cached_seconds": cached_s,
        "speedup": direct_s / cached_s,
        "time_saved_percent": 100 * (1 - cached_s / direct_s),
        "max_pixel_difference": max_pixel_diff,
        "mean_pixel_difference": total_pixel_diff / pixel_count,
        "uncached_peak_allocated_gib": max(peaks[False], default=None),
        "cached_peak_allocated_gib": max(peaks[True], default=None),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
