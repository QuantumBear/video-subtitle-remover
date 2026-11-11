"""Emoji video masks must preserve text, ROI boundaries, and frame identity."""
from pathlib import Path
from types import SimpleNamespace
import sys

import av
import cv2
import numpy as np
import pytest
import torch

from backend.sam3_mask import MaskTimeline


def video(path, count=4):
    with av.open(str(path), 'w') as dst:
        stream = dst.add_stream('libx264rgb', rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, 'rgb24'
        stream.options = {'crf': '0', 'bf': '0'}
        for n in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), 60+n, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)


def text_store(tmp_path, count=4):
    store = MaskTimeline(tmp_path/'text', (24, 32))
    for n in range(count):
        mask = np.zeros((24, 32), np.uint8)
        if n < 2:
            mask[10:12, 12:15] = 255
        store.append(mask)
    return store


def test_segments_use_fixed_text_neighborhood_and_break_at_cuts_and_limit(tmp_path):
    from backend.sam3_emoji import plan_segments
    store = text_store(tmp_path, 5)
    segments = list(plan_segments(store, [2], max_frames=2))
    assert segments == [(0, 2, (6, 16, 8, 19)),
                        (2, 4, (0, 24, 0, 32)), (4, 5, (0, 24, 0, 32))]


def fake_predictor(monkeypatch, *, fail=False, duplicate=False):
    calls = []
    class Predictor:
        def __init__(self):
            self.model = SimpleNamespace()
            self.closed = []
            self.shutdown_called = False
        def handle_request(self, request):
            calls.append(request.copy())
            if request['type'] == 'start_session':
                self.paths = sorted(Path(request['resource_path']).glob('*.png'))
                self.shape = cv2.imread(str(self.paths[0])).shape[:2]
                return {'session_id': 'session'}
            if request['type'] == 'add_prompt':
                assert request['text'] == 'emoji' and request['frame_index'] == 0
            if request['type'] == 'close_session':
                self.closed.append(request['session_id'])
            return {}
        def handle_stream_request(self, request):
            if fail:
                raise RuntimeError('video inference failed')
            h, w = self.shape
            for i in range(len(self.paths)):
                mask = np.zeros((0 if i else 1, h, w), bool)
                if len(mask):
                    mask[0, 1, 1] = True
                    mask[0, 1, 2] = True
                yield {'frame_index': 0 if duplicate else i, 'outputs': {
                    'out_binary_masks': mask, 'out_probs': np.ones(len(mask), np.float32)*.4,
                    'out_obj_ids': np.arange(len(mask))}}
        def shutdown(self):
            self.shutdown_called = True
    instance = Predictor()
    def build(**kwargs):
        calls.append(kwargs)
        return instance
    monkeypatch.setitem(sys.modules, 'sam3.model_builder', SimpleNamespace(build_sam3_video_predictor=build))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: False)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: __import__('contextlib').nullcontext())
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: __import__('contextlib').nullcontext())
    return instance, calls


def test_video_emoji_merge_preserves_text_and_roi_without_copying_neighbor_masks(tmp_path, monkeypatch):
    from backend.sam3_emoji import add_emoji_timeline
    store = text_store(tmp_path)
    source = tmp_path/'in.mp4'
    video(source)
    predictor, calls = fake_predictor(monkeypatch)
    original = [store[i].copy() for i in range(store.count)]
    stats = add_emoji_timeline(source, store, [2], (0, 24, 0, 32), tmp_path/'work',
                               checkpoint='local.pt', device='cuda', score=.3, dilate=0)
    assert predictor.model.score_threshold_detection == predictor.model.new_det_thresh == .3
    assert predictor.model.masklet_confirmation_enable is True
    assert predictor.model.masklet_confirmation_consecutive_det_thresh == 3
    assert predictor.closed == ['session', 'session'] and predictor.shutdown_called
    assert stats['sam3_emoji_frames'] == 4 and stats['sam3_emoji_nonempty_frames'] == 2
    assert np.all(store[0][original[0]>0] == 255)
    assert store[0][7, 9] == 255 and store[0][7, 10] == 255  # crop offset
    np.testing.assert_array_equal(store[1], original[1])
    assert store[2][1, 1] == 255
    assert not store[3].any()  # no temporal mask copying
    assert not list((tmp_path/'work').glob('emoji-*'))
    assert all(not k.startswith('bounding') for c in calls for k in c)


def test_roi_clips_emoji_after_dilation(tmp_path, monkeypatch):
    from backend.sam3_emoji import add_emoji_timeline
    store = text_store(tmp_path, 1)
    source = tmp_path/'in.mp4'
    video(source, 1)
    fake_predictor(monkeypatch)
    add_emoji_timeline(source, store, [], (7, 24, 10, 32), tmp_path/'work',
                       checkpoint='local.pt', device='cuda', score=.3, dilate=1)
    assert not store[0][:7].any() and not store[0][:, :10].any()
    assert store[0][7, 10] == 255


@pytest.mark.parametrize('duplicate', [False, True])
def test_video_failures_close_session_shutdown_and_clean_crops(tmp_path, monkeypatch, duplicate):
    from backend.sam3_emoji import add_emoji_timeline
    store = text_store(tmp_path)
    source = tmp_path/'in.mp4'
    video(source)
    predictor, _ = fake_predictor(monkeypatch, fail=not duplicate, duplicate=duplicate)
    with pytest.raises((RuntimeError, ValueError)):
        add_emoji_timeline(source, store, [], (0, 24, 0, 32), tmp_path/'work',
                           checkpoint='local.pt', device='cuda', score=.3)
    assert predictor.closed == ['session'] and predictor.shutdown_called
    assert not list((tmp_path/'work').glob('emoji-*'))


def test_video_emoji_requires_cuda_instead_of_silently_skipping(tmp_path):
    from backend.sam3_emoji import add_emoji_timeline
    store = text_store(tmp_path, 1)
    with pytest.raises(ValueError, match='CUDA'):
        add_emoji_timeline('unused.mp4', store, [], (0, 24, 0, 32), tmp_path,
                           checkpoint='unused.pt', device='cpu', score=.3)
