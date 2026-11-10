"""SAM3 masks must remain masks through scheduling and optional repair."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import av
import cv2
import numpy as np
import pytest

from vsr_pipeline import Pipeline


def make_video(path, count=3):
    with av.open(str(path), 'w') as dst:
        stream = dst.add_stream('libx264rgb', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'rgb24'
        stream.options = {'crf': '0', 'bf': '0'}
        for i in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 80, np.uint8), format='rgb24')
            frame.pts = i
            for packet in stream.encode(frame):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)


def mask_fixture():
    mask = np.zeros((48, 64), np.uint8)
    # A hole inside the component catches accidental bounding-box conversion.
    mask[12:24, 12:30] = 255
    mask[16:20, 18:24] = 0
    return mask


def test_sam3_constructor_never_loads_ocr(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail('SAM3 must not initialize PaddleOCR')
    monkeypatch.setitem(sys.modules, 'paddleocr', SimpleNamespace(TextDetection=forbidden))
    pipe = Pipeline(mask_backend='sam3', device='cpu', inpaint_mode='sttn')
    assert pipe.ocr is None


@pytest.mark.parametrize('repair', [False, True])
def test_sam3_masks_reach_sttn_and_deferred_repair_without_box_fill(tmp_path, monkeypatch, repair):
    import backend.sam3_mask as sam3_mask

    source, output = tmp_path/'input.mp4', tmp_path/'output.mp4'
    make_video(source)
    mask = mask_fixture()
    calls = []
    class Detector:
        def __init__(self, **options):
            pass
        def predict(self, image):
            calls.append('sam3')
            # Last frame has no overlay: it must not inherit the previous mask.
            return mask.copy() if len(calls) < 3 else np.zeros_like(mask)
        def close(self):
            calls.append('released')
    monkeypatch.setattr(sam3_mask, 'Sam3MaskDetector', Detector)
    monkeypatch.setattr('vsr_pipeline.cuda_memory_snapshot', lambda *a, **kw: None)
    pipe = Pipeline(mask_backend='sam3', device='cpu', inpaint_mode='sttn')
    def forbidden(*args, **kwargs):
        pytest.fail('legacy mask code reached')
    pipe.detect = pipe.auto_region = pipe._detect_timeline = pipe._ensure_sticker_detector = forbidden
    seen_sttn, seen_pp = [], []
    def sttn(frames, masks, x_bounds=None):
        assert 'released' in calls
        seen_sttn.extend(m.copy() for m in masks)
        # Deliberately change outside the mask: composition must protect it.
        return [np.full_like(f, 200) for f in frames]
    pipe.inpainter = sttn
    pipe._ensure_sttn = lambda: None
    class Repair:
        def inpaint(self, frames, masks):
            seen_pp.extend(m.copy() for m in masks)
            return [np.full_like(f, 220) for f in frames]
    pipe._propainter_inpainter = Repair()
    pipe._ensure_propainter = lambda: None
    pipe._residual_mask = lambda fixed, original, boxes: mask.copy() if boxes else np.zeros_like(mask)
    stats = pipe.process_video(source, output, sttn_residual_propainter=repair,
                               sttn_residual_propainter_max_windows=1,
                               sttn_residual_propainter_max_frames=3)
    assert stats['frames'] == 3 and stats['ocr_calls'] == 0
    assert stats['mask_backend'] == 'sam3' and stats['sam3_calls'] == 3
    assert len(seen_sttn) == 3
    for actual in seen_sttn[:2]:
        np.testing.assert_array_equal(actual, mask)
    assert not seen_sttn[2].any()
    if repair:
        assert stats['sttn_propainter_calls'] == 1
        for actual in seen_pp[:2]:
            np.testing.assert_array_equal(actual, mask)
        assert not seen_pp[2].any()
    else:
        assert not seen_pp
    with av.open(str(output)) as result:
        frames = [f.to_ndarray(format='rgb24') for f in result.decode(video=0)]
    assert len(frames) == 3
    assert abs(int(frames[0][18, 20, 0]) - 80) < 15  # hole must remain original
    assert int(frames[0][13, 13, 0]) > 170
    assert np.abs(frames[2].astype(int) - 80).max() < 15


def test_mask_workspace_cleans_up_after_detection_error(tmp_path, monkeypatch):
    import backend.sam3_mask as sam3_mask
    source = tmp_path/'input.mp4'
    make_video(source, 1)
    workspaces = []
    original = sam3_mask.MaskTimeline.__init__
    def record(self, directory, *args, **kwargs):
        workspaces.append(Path(directory))
        return original(self, directory, *args, **kwargs)
    monkeypatch.setattr(sam3_mask.MaskTimeline, '__init__', record)
    class Broken:
        def __init__(self, **kwargs):
            pass
        def predict(self, image):
            raise RuntimeError('detector failure')
        def close(self):
            pass
    monkeypatch.setattr(sam3_mask, 'Sam3MaskDetector', Broken)
    pipe = Pipeline(mask_backend='sam3', device='cpu', inpaint_mode='sttn')
    with pytest.raises(RuntimeError, match='detector failure'):
        pipe.process_video(source, tmp_path/'out.mp4')
    assert workspaces and all(not p.exists() for p in workspaces)
    assert not (tmp_path/'out.mp4').exists()


def test_mask_timeline_fails_on_missing_cache_frame(tmp_path):
    from backend.sam3_mask import MaskTimeline
    store = MaskTimeline(tmp_path, (48, 64))
    store.append(mask_fixture())
    np.testing.assert_array_equal(store[0], mask_fixture())
    next(tmp_path.glob('*.png')).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        store[0]


@pytest.mark.parametrize('script,repair', [('run_sttn.sh', False), ('run_sttn_propainter.sh', True)])
def test_launchers_use_sam3_and_forward_extra_arguments(tmp_path, script, repair):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path/'python'
    fake.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    env = dict(os.environ, SAM3_PYTHON=str(fake), SAM3_SCORE='0.3')
    result = subprocess.run(['bash', str(root/script), 'my input.mp4', 'my output.mp4',
                             '--sttn-profile'], env=env, text=True, capture_output=True, check=True)
    args = json.loads(result.stdout.strip().splitlines()[-1])
    assert args[args.index('--mask-backend')+1] == 'sam3'
    assert args[args.index('--sam3-score')+1] == '0.3'
    assert args[args.index('-i')+1] == 'my input.mp4'
    assert args[-1] == '--sttn-profile'
    assert '--sticker-backend' not in args
    assert ('--sttn-residual-propaint' in args) is repair
