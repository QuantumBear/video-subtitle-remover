"""只用于顺序基准的 STTN 算子分析；不修改生产模型计算。

python benchmarks/sttn_operator_profile.py --device cuda --frames 50 \
    --warmup 3 --repeats 5 --output-dir benchmark_sttn_operators_01
"""

import argparse
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.inpaint.sttn_det_inpaint import STTNDetInpaint


@contextmanager
def model_ranges(model):
    """仅在采集期间为当前实例加范围；正常/异常退出均恢复实例属性。"""
    targets = [(model.encoder, 'forward', 'encoder'),
               (model.decoder, 'forward', 'decoder'),
               (model.transformer, 'forward', 'transformer'),
               (model, 'project_first_qkv', 'qkv_cache')]
    for block in model.transformer:
        attention = block.attention
        targets.extend((module, 'forward', label) for module, label in (
            (attention, 'multihead'), (attention.attention, 'attention'),
            (attention.query_embedding, 'qkv_projection'),
            (attention.key_embedding, 'qkv_projection'),
            (attention.value_embedding, 'qkv_projection'),
            (attention.output_linear, 'attention_output'),
            (block.feed_forward, 'feed_forward')))
    saved = []

    def wrapped(original, label):
        @wraps(original)
        def forward(*args, **kwargs):
            with record_function(f'sttn::{label}'):
                return original(*args, **kwargs)
        return forward

    try:
        for module, attribute, label in targets:
            saved.append((module, attribute, attribute in module.__dict__,
                          module.__dict__.get(attribute)))
            setattr(module, attribute, wrapped(getattr(module, attribute), label))
        yield
    finally:
        for module, attribute, existed, original in reversed(saved):
            if existed:
                setattr(module, attribute, original)
            else:
                delattr(module, attribute)


def device_time_us(event, *, exclusive):
    """Torch 新版使用 device 字段，2.2 使用 cuda 字段；不将两者相加。"""
    names = (('self_device_time_total', 'self_cuda_time_total') if exclusive else
             ('device_time_total', 'cuda_time_total'))
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.


def operator_category(name):
    """互斥的诊断分组；原始算子名同时保留，避免分组掩盖细节。"""
    lower = name.lower()
    if 'synchronize' in lower or name in ('aten::item', 'aten::_local_scalar_dense'):
        return 'host_wait_or_scalar'
    if name in ('aten::isfinite', 'aten::isinf', 'aten::isnan', 'aten::all',
                'aten::abs', 'aten::ne', 'aten::eq'):
        return 'finite_check'
    if 'conv' in lower:
        return 'convolution'
    if name in ('aten::matmul', 'aten::bmm', 'aten::mm', 'aten::addmm'):
        return 'matmul'
    if 'softmax' in lower:
        return 'softmax'
    if name in ('aten::contiguous', 'aten::clone', 'aten::copy_', 'aten::to',
                'aten::_to_copy', 'aten::cat', 'aten::index', 'aten::index_select',
                'aten::permute', 'aten::transpose', 'aten::view', 'aten::reshape',
                'aten::_unsafe_view', 'aten::as_strided') or 'memcpy' in lower:
        return 'layout_copy_index'
    if any(word in lower for word in ('empty', 'resize_', 'alloc', 'free')):
        return 'allocation'
    if name in ('aten::add', 'aten::sub', 'aten::mul', 'aten::div', 'aten::tanh',
                'aten::leaky_relu_', 'aten::leaky_relu'):
        return 'elementwise'
    return 'other'


def summarize_events(events, *, cuda):
    """只累计 CPU 事件的 self 时间；独立列出原始设备事件，禁止重复相加。"""
    operators, regions, kernels = {}, {}, {}
    for event in events:
        if event.device_type == torch.autograd.DeviceType.CUDA:
            row = kernels.setdefault(event.name, dict(name=event.name, calls=0, device_ms=0.))
            row['calls'] += 1
            row['device_ms'] += device_time_us(event, exclusive=False) / 1000
            continue
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        if event.name.startswith('sttn::'):
            row = regions.setdefault(event.name, dict(
                name=event.name, calls=0, cpu_total_ms=0., device_total_ms=0. if cuda else None))
            row['calls'] += 1
            row['cpu_total_ms'] += event.cpu_time_total / 1000
            if cuda:
                row['device_total_ms'] += device_time_us(event, exclusive=False) / 1000
            continue
        parent, scope, region = event.cpu_parent, 'outside_model', 'outside_model'
        while parent is not None:
            if parent.name.startswith('sttn::'):
                label = parent.name.removeprefix('sttn::')
                if region == 'outside_model':
                    region = label
                if label in ('encoder', 'decoder', 'transformer', 'qkv_cache'):
                    scope = label
            parent = parent.cpu_parent
        category = operator_category(event.name)
        shapes = str(event.input_shapes) if event.input_shapes else ''
        key = scope, region, event.name, shapes
        row = operators.setdefault(key, dict(
            scope=scope, region=region, name=event.name, input_shapes=shapes,
            category=category, calls=0, self_cpu_ms=0., self_device_ms=0. if cuda else None))
        row['calls'] += 1
        row['self_cpu_ms'] += event.self_cpu_time_total / 1000
        if cuda:
            row['self_device_ms'] += device_time_us(event, exclusive=True) / 1000

    device_total = sum(row['device_ms'] for row in kernels.values())
    attributed = sum(row['self_device_ms'] or 0. for row in operators.values())
    status = ('cpu_only' if not cuda else 'attributed' if attributed > 0 else
              'unattributed' if device_total > 0 else 'unavailable')
    if status in ('unattributed', 'unavailable'):
        for row in operators.values():
            row['self_device_ms'] = None
        for row in regions.values():
            row['device_total_ms'] = None

    categories = {}
    for row in operators.values():
        key = row['scope'], row['category']
        group = categories.setdefault(key, dict(
            scope=row['scope'], category=row['category'], calls=0, self_cpu_ms=0.,
            self_device_ms=0. if status == 'attributed' else None))
        group['calls'] += row['calls']
        group['self_cpu_ms'] += row['self_cpu_ms']
        if group['self_device_ms'] is not None:
            group['self_device_ms'] += row['self_device_ms']
    sort_key = 'self_device_ms' if status == 'attributed' else 'self_cpu_ms'
    return dict(
        device_timing_status=status,
        operator_self_cpu_total_ms=sum(row['self_cpu_ms'] for row in operators.values()),
        attributed_self_device_total_ms=attributed if status == 'attributed' else None,
        device_event_total_ms=device_total if device_total > 0 else None,
        operators=sorted(operators.values(), key=lambda row: row[sort_key], reverse=True),
        categories=sorted(categories.values(), key=lambda row: row[sort_key], reverse=True),
        regions=list(regions.values()),
        device_events=sorted(kernels.values(), key=lambda row: row['device_ms'], reverse=True))


def run_profile(engine, frames, masks, output_dir, *, warmup=3, repeats=5, top=20,
                record_shapes=False, model_path=None):
    """无 profiler 预热/计时后仅采集一次整带；返回完整 JSON 并输出热点。"""
    if engine.profile:
        raise ValueError('算子基准须关闭引擎阶段 profile，避免混入额外 CUDA Event/同步')
    device = engine.device
    cuda = device.type == 'cuda'
    precision = engine.precision, engine.precision_fallbacks

    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)

    def check_precision():
        if (engine.precision, engine.precision_fallbacks) != precision:
            raise RuntimeError('测量中发生精度回退，profile 无效；请检查日志')

    for _ in range(warmup):
        engine.inpaint(frames, masks)
        check_precision()
    timings, reference = [], None
    max_diff, total_diff, count = 0., 0., 0

    def compare(output):
        nonlocal max_diff, total_diff, count
        if len(output) != len(reference):
            raise RuntimeError('采集前后输出帧数不同')
        for expected, actual in zip(reference, output):
            if expected.shape != actual.shape:
                raise RuntimeError('采集前后输出尺寸不同')
            diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
            if not np.isfinite(diff).all():
                raise RuntimeError('输出存在非有限值')
            max_diff = max(max_diff, float(diff.max()))
            total_diff += float(diff.sum())
            count += diff.size

    for repeat in range(repeats):
        synchronize()
        start = time.perf_counter()
        output = engine.inpaint(frames, masks)
        synchronize()
        timings.append(time.perf_counter() - start)
        check_precision()
        if reference is None:
            reference = output
        compare(output)
        print(f'[operator-baseline] repeat={repeat + 1} seconds={timings[-1]:.6f}', flush=True)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    synchronize()
    start = time.perf_counter()
    with profile(activities=activities, record_shapes=record_shapes) as capture:
        with model_ranges(engine.model):
            with record_function('sttn::band'):
                output = engine.inpaint(frames, masks)
        synchronize()
    capture_wall = time.perf_counter() - start
    check_precision()
    compare(output)
    capture.export_chrome_trace(str(output_dir / 'trace.json'))
    report = summarize_events(capture.events(), cuda=cuda)
    report.update(
        scope='inpaint_band', device=str(device),
        model=str(Path(model_path).resolve()) if model_path is not None else None,
        device_name=torch.cuda.get_device_name(device) if cuda else 'CPU',
        torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads(),
        frames=len(frames), frame_shape=list(frames[0].shape),
        neighbor_stride=engine.neighbor_stride, ref_length=engine.ref_length,
        requested_precision=engine.requested_precision, precision=engine.precision,
        precision_fallbacks=engine.precision_fallbacks,
        cache_active=engine.cache_first_qkv and len(frames) > engine.neighbor_stride,
        warmup=warmup, repeats=repeats, record_shapes=record_shapes,
        baseline_seconds=timings, baseline_median_seconds=statistics.median(timings),
        profiled_runs=1, capture_wall_seconds=capture_wall,
        outputs_match=max_diff == 0, max_pixel_difference=max_diff,
        mean_pixel_difference=total_diff / count,
        trace=str(output_dir / 'trace.json'),
        timing_notes=[
            'baseline_seconds: no profiler, no ranges; synchronized band wall time',
            'capture_wall_seconds includes profiler overhead and export preparation; not a speed baseline',
            'operators/categories: exclusive self CPU and attributed device times; CPU and GPU must not be added',
            'regions: inclusive nested ranges; do not sum them or add to operator times',
            'device_events: separate raw kernel/copy total; do not add to attributed device time',
            'host_wait_or_scalar includes CPU waiting, not pure GPU compute; inspect trace for gaps',
            'outside_model includes preprocessing, gather, finite checks, transfers and CPU composition; '
            'NumPy/Python work is visible as band gaps, not ATen operator time'])
    (output_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'[operator-profile] baseline_median={report["baseline_median_seconds"]:.6f}s '
          f'capture_wall={capture_wall:.6f}s device_timing={report["device_timing_status"]} '
          f'outputs_match={report["outputs_match"]}', flush=True)
    for row in report['categories']:
        print(f'[operator-category] scope={row["scope"]} category={row["category"]} '
              f'self_cpu_ms={row["self_cpu_ms"]:.3f} self_device_ms={row["self_device_ms"]}')
    for row in report['regions']:
        print(f'[operator-region] name={row["name"]} calls={row["calls"]} '
              f'cpu_total_ms={row["cpu_total_ms"]:.3f} device_total_ms={row["device_total_ms"]}')
    clocks = ['cpu'] + (['device'] if report['device_timing_status'] == 'attributed' else [])
    for clock in clocks:
        rows = sorted(report['operators'], key=lambda row: row[f'self_{clock}_ms'], reverse=True)
        for row in rows[:top]:
            print(f'[operator-hotspot] sort={clock} scope={row["scope"]} region={row["region"]} '
                  f'name={row["name"]} calls={row["calls"]} '
                  f'self_cpu_ms={row["self_cpu_ms"]:.3f} self_device_ms={row["self_device_ms"]}')
    print(f'[operator-report] {output_dir / "report.json"}', flush=True)
    if max_diff:
        raise RuntimeError('输出像素不一致，profile 结果不能作为等价推理依据')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--model', type=Path, default=ROOT / 'backend/models/sttn-det/sttn.pth')
    parser.add_argument('--frames', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--cache', choices=('on', 'off'), default='on')
    parser.add_argument('--top', type=int, default=20)
    parser.add_argument('--record-shapes', action='store_true', help='额外记录算子输入形状，采集开销更高')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark_sttn_operators_01'))
    args = parser.parse_args()
    if min(args.frames, args.repeats, args.threads, args.top) < 1 or args.warmup < 0:
        parser.error('frames/repeats/threads/top 必须为正，warmup 必须非负')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('当前环境没有 CUDA；CPU 验证请指定 --device cpu --precision fp32')
    if args.device == 'cpu' and args.precision != 'fp32':
        parser.error('CPU 验证请指定 --precision fp32')
    torch.set_num_threads(args.threads)
    engine = STTNDetInpaint(args.device, str(args.model), precision=args.precision,
                           cache_first_qkv=args.cache == 'on')
    width, height = engine.model_input_width, engine.model_input_height
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
              for _ in range(args.frames)]
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in frames]
    for index, mask in enumerate(masks):
        if index % 10:
            x = width // 3 + index % 8
            mask[height // 2:height // 2 + 32, x:x + 128] = 255
    run_profile(engine, frames, masks, args.output_dir, warmup=args.warmup,
                repeats=args.repeats, top=args.top, record_shapes=args.record_shapes,
                model_path=args.model)


if __name__ == '__main__':
    main()
