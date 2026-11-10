"""Empty-frame fallback must use current-image evidence without extending nonempty masks."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.sam3_mask import Sam3MaskDetector


def make_detector(tmp_path, monkeypatch, predictions, **options):
    """Replace the expensive external model, keeping production selection and morphology real."""
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class Processor:
        def __init__(self, model, device, confidence_threshold):
            self.confidence_threshold = confidence_threshold
            self.images = self.prompts = 0

        def set_image(self, image):
            self.images += 1
            return {}

        def reset_all_prompts(self, state):
            state.clear()

        def set_text_prompt(self, prompt, state):
            self.prompts += 1
            entries = predictions.pop(0)
            scores = torch.tensor([score for score, _ in entries], dtype=torch.float32)
            masks = torch.zeros((len(entries), 1, 9, 11), dtype=torch.bool)
            for i, (_, pixels) in enumerate(entries):
                for y, x in pixels:
                    masks[i, 0, y, x] = True
            keep = scores > self.confidence_threshold
            state.update(scores=scores[keep], masks=masks[keep],
                         boxes=torch.zeros((int(keep.sum()), 4)),
                         masks_logits=masks[keep].float())
            return state

    monkeypatch.setitem(sys.modules, 'sam3.model_builder',
                        SimpleNamespace(build_sam3_image_model=lambda **kwargs: Model()))
    monkeypatch.setitem(sys.modules, 'sam3.model.sam3_image_processor',
                        SimpleNamespace(Sam3Processor=Processor))
    checkpoint = tmp_path/'sam3.pt'
    torch.save({'detector.weight': torch.zeros(1)}, checkpoint)
    return Sam3MaskDetector(checkpoint=checkpoint, device='cpu', **options)


def image():
    return np.zeros((9, 11, 3), np.uint8)


def pixels(mask):
    assert mask.dtype == np.uint8 and set(np.unique(mask)).issubset({0, 255})
    return set(map(tuple, np.argwhere(mask)))


def test_empty_frame_uses_low_candidates_with_strict_score_boundaries(tmp_path, monkeypatch):
    detector = make_detector(tmp_path, monkeypatch, [[
        (.49, [(2, 3)]), (.5, [(3, 4)]), (.3, [(4, 5)]), (.29, [(5, 6)]),
    ]], empty_score=.3)
    assert pixels(detector.predict(image())) == {(2, 3), (3, 4)}
    assert (detector.empty_frames, detector.fallback_frames, detector.recovered_frames) == (1, 1, 1)
    assert detector.processor.images == detector.processor.prompts == 1


def test_primary_mask_does_not_append_low_score_instances(tmp_path, monkeypatch):
    detector = make_detector(tmp_path, monkeypatch, [[
        (.8, [(2, 3)]), (.4, [(6, 7)]),
    ]], empty_score=.3)
    assert pixels(detector.predict(image())) == {(2, 3)}
    assert (detector.empty_frames, detector.fallback_frames, detector.recovered_frames) == (0, 0, 0)


def test_fallback_waits_for_all_prompts_before_deciding_frame_is_empty(tmp_path, monkeypatch):
    detector = make_detector(tmp_path, monkeypatch, [
        [(.4, [(6, 7)])], [(.8, [(2, 3)])],
    ], prompts=['caption', 'overlay'], empty_score=.3)
    assert pixels(detector.predict(image())) == {(2, 3)}
    assert detector.fallback_frames == 0
    assert detector.processor.images == 1 and detector.processor.prompts == 2


def test_empty_high_score_instance_does_not_block_nonempty_fallback(tmp_path, monkeypatch):
    detector = make_detector(tmp_path, monkeypatch, [[
        (.8, []), (.4, [(2, 3)]),
    ]], empty_score=.3)
    assert pixels(detector.predict(image())) == {(2, 3)}
    assert detector.recovered_frames == 1


def test_new_content_and_absence_never_inherit_previous_mask(tmp_path, monkeypatch):
    detector = make_detector(tmp_path, monkeypatch, [
        [(.4, [(2, 3)])], [(.4, [(6, 7)])], [],
    ], empty_score=.3)
    assert pixels(detector.predict(image())) == {(2, 3)}
    assert pixels(detector.predict(image())) == {(6, 7)}
    assert pixels(detector.predict(image())) == set()
    assert (detector.empty_frames, detector.fallback_frames, detector.recovered_frames) == (3, 3, 2)


def test_fallback_is_disabled_by_default(tmp_path, monkeypatch):
    detector = make_detector(tmp_path, monkeypatch, [[(.4, [(2, 3)])]])
    assert pixels(detector.predict(image())) == set()
    assert (detector.empty_frames, detector.fallback_frames, detector.recovered_frames) == (1, 0, 0)


def test_dilate_applies_to_selected_fallback_in_original_pixels(tmp_path, monkeypatch):
    detector = make_detector(tmp_path, monkeypatch, [[(.4, [(4, 5)])]], empty_score=.3, dilate=1)
    assert pixels(detector.predict(image())) == {
        (3, 4), (3, 5), (3, 6), (4, 4), (4, 5), (4, 6), (5, 4), (5, 5), (5, 6),
    }


def test_roi_clipping_happens_after_whole_frame_fallback_selection(tmp_path, monkeypatch):
    import av
    from backend import sam3_mask

    detector = make_detector(tmp_path, monkeypatch, [
        [(.8, [(1, 1)]), (.4, [(6, 7)])], [(.4, [(6, 7)])],
    ], empty_score=.3)
    source = tmp_path/'source.mp4'
    with av.open(str(source), 'w') as dst:
        stream = dst.add_stream('libx264rgb', rate=30)
        stream.width, stream.height, stream.pix_fmt = 11, 9, 'rgb24'
        stream.options = {'crf': '0', 'bf': '0'}
        for n in range(2):
            frame = av.VideoFrame.from_ndarray(image(), format='rgb24')
            frame.pts = n
            for packet in stream.encode(frame):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    monkeypatch.setattr(sam3_mask, 'Sam3MaskDetector', lambda **options: detector)
    _, stats, masks = sam3_mask.detect_timeline(source, (5, 9, 5, 11), tmp_path/'masks', empty_score=.3)
    assert pixels(masks[0]) == set()  # High-score target outside ROI still blocks fallback.
    assert pixels(masks[1]) == {(6, 7)}
    assert (stats['sam3_empty_frames'], stats['sam3_fallback_frames'], stats['sam3_recovered_frames']) == (1, 1, 1)


@pytest.mark.parametrize('empty_score', [-.1, .6, float('nan'), float('inf')])
def test_invalid_fallback_score_fails_before_loading_model(tmp_path, empty_score):
    with pytest.raises(ValueError, match='empty_score'):
        Sam3MaskDetector(checkpoint=tmp_path/'missing.pt', empty_score=empty_score)
