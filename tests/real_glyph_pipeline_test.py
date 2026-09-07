"""Optional real-frame regression; OCR and GPU inference stay offline."""

import hashlib
import os
from itertools import islice
from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from glyph_pipeline_test import write_frames
from vsr_pipeline import Pipeline


@pytest.mark.skipif(not os.environ.get("VSR_TEMPLATE_REGRESSION_VIDEO"),
                    reason="set VSR_TEMPLATE_REGRESSION_VIDEO to the diagnostic source video")
def test_real_opening_recovers_white_clothing_glyphs_across_actual_cuts(tmp_path):
    with av.open(os.environ["VSR_TEMPLATE_REGRESSION_VIDEO"]) as container:
        frames = [frame.to_ndarray(format="rgb24")
                  for frame in islice(container.decode(video=0), 90)]
    assert len(frames) == 90 and frames[0].shape == (1280, 720, 3)
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, frames)
    lookup = {hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).tobytes()).digest(): n
              for n, frame in enumerate(frames)}
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    # Fixed OCR localization of this phrase; scene detection and masks are real.
    box, region = (465, 504, 166, 537), (450, 1010, 0, 720)
    pipe.detect = lambda *args: [box]
    pipe.inpainter = SimpleNamespace(inpaint=lambda frames, masks: [np.zeros_like(f) for f in frames])
    real_repair = pipe._repair_propainter_segment
    windows, captured = [], {}

    def audit_segment(images, masks, boxes, white_glyph_check=True):
        numbers = [lookup[hashlib.sha256(image.tobytes()).digest()] for image in images]
        windows.append((numbers[0], numbers[-1]))
        result, repairs = real_repair(images, masks, boxes, white_glyph_check)
        for n, original, mask, fixed in zip(numbers, images, masks, result):
            np.testing.assert_array_equal(fixed[mask == 0], original[mask == 0])
            assert not np.any(mask[:region[0]]) and not np.any(mask[region[1]:])
            if n in (60, 75):
                captured[n] = mask.copy(), fixed.copy()
        return result, repairs

    pipe._repair_propainter_segment = audit_segment
    stats = pipe.process_video(source, output, region=region,
                               locate_stickers=False, white_glyph_check=False)
    assert windows == [(0, 28), (29, 58), (59, 89)]
    assert stats["frames"] == 90
    suffix = (slice(473, 500), slice(340, 533))
    for n in (60, 75):
        base = pipe.propainter_boxes_to_mask([box], frames[n], region)
        mask, result = captured[n]
        added = (mask > 0) & (base == 0)
        assert np.count_nonzero(added[suffix]) > 1000, f"frame {n}: suffix still outside model mask"
        assert not np.any(result[added])
