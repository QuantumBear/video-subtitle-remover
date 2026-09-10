"""Optional original-pixel regression, with frozen OCR and no model/network calls."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip('av')
cv2 = pytest.importorskip('cv2')

from backend.temporal_glyphs import TemporalGlyphs
from vsr_pipeline import Pipeline


@pytest.fixture(scope='module')
def real_glyph_window():
    video = os.environ.get('VSR_TEMPORAL_GLYPH_VIDEO')
    if not video:
        pytest.skip('set VSR_TEMPORAL_GLYPH_VIDEO to the diagnostic original video')
    fixture = json.loads((Path(__file__).parent / 'fixtures/temporal_glyph_ocr.json').read_text())
    numbers = fixture['numbers']
    frames = []
    with av.open(video) as container:
        for n, frame in enumerate(container.decode(video=0)):
            if n > numbers[-1]:
                break
            if n >= numbers[0]:
                frames.append(frame.to_ndarray(format='bgr24'))
    assert len(frames) == len(numbers) == 60
    assert all(list(f.shape) == fixture['shape'] for f in frames)
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.tobytes())
    assert digest.hexdigest() == fixture['bgr_sha256'], 'wrong video or decoder pixels'
    pipe = Pipeline.__new__(Pipeline)
    masks = [pipe.propainter_boxes_to_mask(boxes, cv2.cvtColor(f, cv2.COLOR_BGR2RGB), fixture['roi'])
             for f, boxes in zip(frames, fixture['boxes'])]
    refined, boxes = TemporalGlyphs(fixture['roi']).refine(frames, masks, fixture['boxes'], numbers)
    return fixture, frames, masks, refined, boxes


@pytest.mark.parametrize('label', ['slip', 'sturdy_suffix', 'leading_bullet'])
def test_real_truncated_suffix_core_is_covered(real_glyph_window, label):
    fixture, _, original, refined, _ = real_glyph_window
    index = fixture['numbers'].index(fixture['target_frame'])
    ys, xs = np.array(fixture['core_labels'][label]).T
    assert len(ys) > 40
    assert np.any(original[index][ys, xs] == 0), 'fixture must reproduce original missing mask'
    assert np.all(refined[index][ys, xs] > 0), f'{label}: readable strokes still outside mask'


def test_real_glyphs_preserve_original_mask_and_background_gaps(real_glyph_window):
    fixture, _, original, refined, _ = real_glyph_window
    y1, y2, x1, x2 = fixture['roi']
    for old, new in zip(original, refined):
        assert np.all(new[old > 0] > 0)
        assert not new[:y1].any() and not new[y2:].any()
        assert not new[:, :x1].any() and not new[:, x2:].any()
        # Known spaces between text rows and beyond the last word.
        assert not np.any((new[866:869, 260:300] > 0) & (old[866:869, 260:300] == 0))
        assert not np.any((new[824:844, 240:300] > 0) & (old[824:844, 240:300] == 0))
