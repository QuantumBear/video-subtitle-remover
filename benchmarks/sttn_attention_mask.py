"""固定权重、输入及窗口，对比保留/移除无用注意力 mask 的整带推理。

python benchmarks/sttn_attention_mask.py --device cuda --frames 50 --warmup 3 --repeats 5
旧路径仅在此顺序 A/B 基准内恢复，不向生产引擎增加开关。
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.inpaint.sttn.network_sttn import MultiHeadedAttention
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint


def legacy_attention_forward(self, x, m, b, c, projected_qkv=None):
    """冻结改动前的四路 patch attention，包含未使用的 mm 构造。"""
    bt, _, h, w = x.size()
    t = bt // b
    d_k = c // len(self.patchsize)
    output = []
    if projected_qkv is None:
        _query, _key, _value = self.project_qkv(x)
    else:
        _query, _key, _value = projected_qkv
        if any(projected.shape != x.shape for projected in projected_qkv):
            raise ValueError('cached STTN Q/K/V shape must match feature shape')
    for (width, height), query, key, value in zip(
            self.patchsize, torch.chunk(_query, len(self.patchsize), dim=1),
            torch.chunk(_key, len(self.patchsize), dim=1),
            torch.chunk(_value, len(self.patchsize), dim=1)):
        out_w, out_h = w // width, h // height
        mm = m.view(b, t, 1, out_h, height, out_w, width)
        mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            b, t*out_h*out_w, height*width)
        mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
        query = query.view(b, t, d_k, out_h, height, out_w, width)
        query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            b, t*out_h*out_w, d_k*height*width)
        key = key.view(b, t, d_k, out_h, height, out_w, width)
        key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            b, t*out_h*out_w, d_k*height*width)
        value = value.view(b, t, d_k, out_h, height, out_w, width)
        value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
            b, t*out_h*out_w, d_k*height*width)
        y, _ = self.attention(query, key, value, mm)
        y = y.view(b, t, out_h, out_w, d_k, height, width)
        y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
        output.append(y)
    return self.output_linear(torch.cat(output, 1))


@contextmanager
def use_attention(forward):
    """只用于单进程顺序基准；任务返回或异常时恢复原实现。"""
    previous = MultiHeadedAttention.forward
    MultiHeadedAttention.forward = forward
    try:
        yield
    finally:
        MultiHeadedAttention.forward = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--model', type=Path,
                        default=ROOT / 'backend/models/sttn-det/sttn.pth')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--cache', choices=('on', 'off'), default='on')
    parser.add_argument('--profile', action='store_true', help='整带分阶段计时；预热不输出')
    args = parser.parse_args()
    if min(args.frames, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error('frames/repeats/threads 必须为正，warmup 必须非负')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('当前环境没有 CUDA；CPU 验证请指定 --device cpu --precision fp32')

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    engine = STTNDetInpaint(device, str(args.model), precision=args.precision,
                           cache_first_qkv=args.cache == 'on')
    expected_precision, expected_fallbacks = engine.precision, engine.precision_fallbacks
    width, height = engine.model_input_width, engine.model_input_height
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
              for _ in range(args.frames)]
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in frames]
    for index, mask in enumerate(masks):
        if index % 10:
            x = width // 3 + index % 8
            mask[height // 2:height // 2 + 32, x:x + 128] = 255
    implementations = dict(legacy=legacy_attention_forward,
                           optimized=MultiHeadedAttention.forward)

    def run(name):
        with use_attention(implementations[name]):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            output = engine.inpaint(frames, masks)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            peak = (torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == 'cuda' else None)
        if engine.precision != expected_precision or engine.precision_fallbacks != expected_fallbacks:
            raise RuntimeError('测量中发生精度回退，A/B 无效；请先检查日志')
        return output, elapsed, peak

    for _ in range(args.warmup):
        for name in implementations:
            run(name)
    engine.profile = args.profile
    timings = {name: [] for name in implementations}
    peaks = {name: [] for name in implementations}
    max_diff, total_diff, count = 0., 0., 0
    for repeat in range(args.repeats):
        outputs = {}
        order = ('legacy', 'optimized') if repeat % 2 == 0 else ('optimized', 'legacy')
        for name in order:
            outputs[name], elapsed, peak = run(name)
            timings[name].append(elapsed)
            if peak is not None:
                peaks[name].append(peak)
            print(f'attention_mask={name} repeat={repeat + 1} seconds={elapsed:.6f}', flush=True)
        for old, new in zip(outputs['legacy'], outputs['optimized']):
            diff = np.abs(old.astype(np.float32) - new.astype(np.float32))
            if not np.isfinite(diff).all():
                raise RuntimeError('输出存在非有限值')
            max_diff = max(max_diff, float(diff.max()))
            total_diff += float(diff.sum(dtype=np.float64))
            count += diff.size

    legacy_s, optimized_s = (statistics.median(timings[name]) for name in implementations)
    print(json.dumps(dict(
        scope='inpaint_band', profile=args.profile, device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(),
        frames=args.frames, warmup=args.warmup, repeats=args.repeats,
        requested_precision=args.precision, precision=engine.precision,
        precision_fallbacks=engine.precision_fallbacks,
        cache_active=engine.cache_first_qkv and args.frames > engine.neighbor_stride,
        legacy_seconds=legacy_s, optimized_seconds=optimized_s,
        speedup=legacy_s / optimized_s, time_saved_percent=100 * (1 - optimized_s / legacy_s),
        max_pixel_difference=max_diff, mean_pixel_difference=total_diff / count,
        outputs_match=max_diff == 0,
        legacy_peak_allocated_gib=max(peaks['legacy'], default=None),
        optimized_peak_allocated_gib=max(peaks['optimized'], default=None)),
        ensure_ascii=False, indent=2))
    if max_diff:
        raise SystemExit('输出像素不一致，不能只比较速度')


if __name__ == '__main__':
    main()
