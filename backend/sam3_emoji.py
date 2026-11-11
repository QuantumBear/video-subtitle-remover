"""SAM3 emoji discovery in fixed text neighborhoods with official video confirmation."""
import gc
import os
from pathlib import Path
import sys
import tempfile
import time

import av
import cv2
import numpy as np
import torch


def plan_segments(store, cuts, max_frames=120):
    """Bound video state, never propagate across a scene cut, fix each crop before tracking."""
    if not isinstance(max_frames, int) or max_frames < 1:
        raise ValueError('SAM3 emoji max_frames 必须为正整数')
    boundaries = sorted({0, store.count} | {int(n) for n in cuts if 0 < n < store.count})
    h, w = store.shape
    for start, end in zip(boundaries, boundaries[1:]):
        for lo in range(start, end, max_frames):
            hi = min(end, lo + max_frames)
            union = np.zeros(store.shape, np.uint8)
            for n in range(lo, hi):
                union |= store[n]
            ys, xs = np.nonzero(union)
            if len(ys):
                pad = 2 * int(ys.max() - ys.min() + 1)
                region = (max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1),
                          max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1))
            else:
                region = (0, h, 0, w)
            yield lo, hi, region


def add_emoji_timeline(input_path, store, cuts, region, directory, *, checkpoint,
                       code_dir=None, device='auto', score=.3, confirm=True,
                       max_frames=120, dilate=0, progress=None):
    """Merge confirmed emoji pixels into text masks, releasing SAM3 before inpainting."""
    if not 0 < score < 1:
        raise ValueError('SAM3 emoji score 必须介于0和1')
    if not isinstance(dilate, int) or dilate < 0:
        raise ValueError('SAM3 emoji dilate 必须为非负整数')
    target = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if device == 'auto' else device)
    if target.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('SAM3 emoji 视频发现与确认需要 CUDA；CPU 请设置 --sam3-emoji-score 0')
    segments = list(plan_segments(store, cuts, max_frames))
    if code_dir:
        sys.path.insert(0, str(Path(code_dir).expanduser().resolve()))
    from sam3.model_builder import build_sam3_video_predictor

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stats = {'sam3_emoji_frames': 0, 'sam3_emoji_nonempty_frames': 0,
             'sam3_emoji_added_pixels': 0, 'sam3_emoji_instances': 0,
             'sam3_emoji_segments': len(segments), 'sam3_emoji_confirm': bool(confirm)}
    started = time.perf_counter()
    predictor = None
    print(f'[sam3-emoji] prompt=emoji score={score} confirm={confirm} '
          f'consecutive=3 max_frames={max_frames} segments={len(segments)}')
    try:
        gpu = target.index if target.index is not None else torch.cuda.current_device()
        predictor = build_sam3_video_predictor(checkpoint_path=os.fspath(Path(checkpoint).expanduser()),
                                               strict_state_dict_loading=True, compile=False,
                                               gpus_to_use=[gpu])
        model = predictor.model
        model.score_threshold_detection = score
        model.new_det_thresh = score
        model.masklet_confirmation_enable = bool(confirm)
        model.masklet_confirmation_consecutive_det_thresh = 3
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with av.open(os.fspath(input_path)) as src:
            frames = iter(src.decode(video=0))
            for lo, hi, (y0, y1, x0, x1) in segments:
                with tempfile.TemporaryDirectory(prefix='emoji-', dir=directory) as crops:
                    for n in range(lo, hi):
                        frame = next(frames, None)
                        if frame is None:
                            raise ValueError(f'emoji 第二遍解码提前结束: {n}/{store.count}')
                        image = frame.to_ndarray(format='rgb24')
                        if image.shape[:2] != store.shape:
                            raise ValueError('emoji 视频尺寸中途变化')
                        path = Path(crops) / f'{n-lo:05d}.png'
                        if not cv2.imwrite(str(path), cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)):
                            raise OSError(f'无法写入 SAM3 emoji 输入: {path}')
                    sid = None
                    try:
                        with torch.inference_mode(), torch.autocast('cuda', dtype=amp_dtype):
                            sid = predictor.handle_request({'type': 'start_session', 'resource_path': crops,
                                                             'offload_video_to_cpu': True})['session_id']
                            predictor.handle_request({'type': 'add_prompt', 'session_id': sid,
                                                      'frame_index': 0, 'text': 'emoji', 'output_prob_thresh': .1})
                            seen = set()
                            for response in predictor.handle_stream_request({
                                    'type': 'propagate_in_video', 'session_id': sid,
                                    'propagation_direction': 'forward', 'start_frame_index': 0,
                                    'output_prob_thresh': .1}):
                                i, result = response['frame_index'], response['outputs']
                                if not isinstance(i, int) or not 0 <= i < hi-lo or i in seen:
                                    raise ValueError(f'SAM3 emoji 返回重复或越界帧: {i}')
                                seen.add(i)
                                masks = np.asarray(result['out_binary_masks'])
                                scores = np.asarray(result['out_probs'])
                                if (masks.dtype != bool or masks.shape != (len(scores), y1-y0, x1-x0)
                                        or not np.isfinite(scores).all()):
                                    raise ValueError('SAM3 emoji 输出尺寸、类型或置信度异常')
                                mask = np.zeros(store.shape, np.uint8)
                                if len(masks):
                                    mask[y0:y1, x0:x1] = masks.any(0).astype(np.uint8)*255
                                if dilate:
                                    mask = cv2.dilate(mask, np.ones((2*dilate+1, 2*dilate+1), np.uint8))
                                ry0, ry1, rx0, rx1 = region
                                clipped = np.zeros_like(mask)
                                clipped[ry0:ry1, rx0:rx1] = mask[ry0:ry1, rx0:rx1]
                                original = store[lo+i]
                                merged = cv2.bitwise_or(original, clipped)
                                store.replace(lo+i, merged)
                                stats['sam3_emoji_frames'] += 1
                                stats['sam3_emoji_nonempty_frames'] += int(clipped.any())
                                stats['sam3_emoji_added_pixels'] += int(np.count_nonzero(merged)-np.count_nonzero(original))
                                stats['sam3_emoji_instances'] += len(scores)
                            if len(seen) != hi-lo:
                                raise ValueError(f'SAM3 emoji 输出不完整: {len(seen)}/{hi-lo}')
                    finally:
                        if sid is not None:
                            predictor.handle_request({'type': 'close_session', 'session_id': sid})
                    # Session state and cropped PNGs are bounded to this segment.
                print(f'[sam3-emoji] frames={hi}/{store.count} nonempty={stats["sam3_emoji_nonempty_frames"]}')
                if progress:
                    progress(hi, store.count, 'SAM3 emoji 视频确认')
            if next(frames, None) is not None:
                raise ValueError('emoji 第二遍解码帧数多于文字检测')
    finally:
        try:
            if predictor is not None:
                predictor.shutdown()
        finally:
            predictor = model = None
            gc.collect()
            with torch.cuda.device(target):
                torch.cuda.empty_cache()
    stats['sam3_emoji_seconds'] = time.perf_counter()-started
    print(f'[sam3-emoji] done {stats}')
    return stats
