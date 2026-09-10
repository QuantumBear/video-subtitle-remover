"""Optional original-pixel replay using cached GroundingDINO candidates."""

import json
import os
from itertools import islice
from pathlib import Path

import pytest

from backend.sticker_detect import filter_candidates
from backend.sticker_tracking import StickerTracker
from backend.subtitle_tracking import associate_sticker_hits, sticker_match_score


pytestmark = pytest.mark.skipif(
    not os.environ.get('VSR_STICKER_REGRESSION_VIDEO'),
    reason='set VSR_STICKER_REGRESSION_VIDEO to the diagnostic source video')


@pytest.fixture(scope='module')
def opening():
    import av

    evidence = json.loads((Path(__file__).parent / 'fixtures' /
                           'sticker_opening_candidates.json').read_text())
    with av.open(os.environ['VSR_STICKER_REGRESSION_VIDEO']) as src:
        frames = [f.to_ndarray(format='rgb24') for f in
                  islice(src.decode(video=0), evidence['total_frames'])]
    assert len(frames) == 100 and frames[0].shape == (1280, 720, 3)
    samples = {int(n): filter_candidates(raw, evidence['roi'],
                                        evidence['score_threshold'], evidence['max_area_px'])
               for n, raw in evidence['observations'].items()}
    tracked = StickerTracker(evidence['text_timeline'], len(frames),
                             score_threshold=evidence['score_threshold'],
                             scene_change_frames=evidence['scene_change_frames'])
    for n, candidates in samples.items():
        tracked.add_sample(n, frames[n], candidates)
    for n, rgb in enumerate(frames):
        tracked.observe(n, rgb, samples.get(n, []))
    return evidence, frames, samples, tracked.finish(), tracked.stats


def test_original_opening_repairs_all_44_missing_left_sticker_frames(opening):
    evidence, _, samples, result, _ = opening
    left = (496, 527, 313, 341)
    baseline = associate_sticker_hits(
        {n: [c.box for c in candidates if c.score >= 0.22] for n, candidates in samples.items()},
        evidence['text_timeline'], 100, scene_change_frames=evidence['scene_change_frames'])
    missing = [n for n in range(90)
               if not any(sticker_match_score(b, left) >= 0 for b in baseline.get(n, []))]
    expected = [n for start, end in evidence['old_left_gaps'] for n in range(start, end + 1)]
    assert missing == expected and len(missing) == 44
    for n in range(90):
        assert any(sticker_match_score(b, left) >= 0 for b in result.get(n, [])), f'left missing: {n}'


def test_original_opening_keeps_all_three_stickers_separate_and_stops_at_90(opening):
    _, _, _, result, stats = opening
    for n in range(90):
        assert len(result.get(n, [])) == 3, f'frame {n}: {result.get(n, [])}'
    assert not any(result.get(n) for n in range(90, 100))
    assert stats['weak_continuations'] > 0
    assert stats['appearance_recovered'] > 0
    assert stats['appearance_unknown'] == 0


@pytest.mark.skipif(not os.environ.get('VSR_STICKER_PROPAINTER'),
                    reason='set VSR_STICKER_PROPAINTER=1 for slow offline CPU model comparison')
def test_real_propainter_opening_reduces_left_emoji_residue(opening, tmp_path):
    import cv2
    import numpy as np
    import torch

    from backend.inpaint.propainter_inpaint import PropainterInpaint
    from vsr_pipeline import Pipeline, STICKER_MASK_PAD

    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    evidence, frames, samples, tracked, _ = opening
    baseline = associate_sticker_hits(
        {n: [c.box for c in candidates if c.score >= 0.22] for n, candidates in samples.items()},
        evidence['text_timeline'], 100, scene_change_frames=evidence['scene_change_frames'])
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpainter = PropainterInpaint(
        torch.device('cpu'), str(Path(__file__).resolve().parents[1] / 'backend/models/propainter'),
        sub_video_length=60, use_fp16=False)
    crops = [cv2.cvtColor(rgb[400:656, 256:512], cv2.COLOR_RGB2BGR) for rgb in frames[:29]]
    outputs = {}
    measurements = {}
    for label, timeline in [('baseline', baseline), ('tracked', tracked)]:
        masks = []
        for n, rgb in enumerate(frames[:29]):
            ry1, ry2, rx1, rx2 = evidence['roi']
            boxes = [(max(ry1, y1 - STICKER_MASK_PAD), min(ry2, y2 + STICKER_MASK_PAD),
                      max(rx1, x1 - STICKER_MASK_PAD), min(rx2, x2 + STICKER_MASK_PAD))
                     for y1, y2, x1, x2 in timeline.get(n, [])]
            mask = pipe.propainter_boxes_to_mask(
                evidence['text_timeline'][n], rgb, evidence['roi'], sticker_boxes=boxes)
            masks.append(mask[400:656, 256:512].copy())
        outputs[label], repairs = pipe._repair_propainter_segment(
            crops, masks, [[] for _ in crops], white_glyph_check=False)
        assert repairs == 0
        for original, fixed, mask in zip(crops, outputs[label], masks):
            np.testing.assert_array_equal(fixed[mask == 0], original[mask == 0])
        rows = []
        for n in (0, 7, 14, 15, 22, 28):
            before = cv2.cvtColor(crops[n][93:130, 54:87], cv2.COLOR_BGR2HSV)
            after = cv2.cvtColor(outputs[label][n][93:130, 54:87], cv2.COLOR_BGR2HSV)
            orange = lambda hsv: ((hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 35) &
                                  (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 90))
            rows.append(dict(frame=n, source_orange=int(orange(before).sum()),
                             remaining_orange=int((orange(before) & orange(after)).sum())))
        measurements[label] = rows
    print(json.dumps(measurements))
    before_count = sum(row['remaining_orange'] for row in measurements['baseline'])
    after_count = sum(row['remaining_orange'] for row in measurements['tracked'])
    assert before_count > 100
    assert after_count < before_count * 0.1
    for n in (0, 7, 14, 15, 22, 28):
        cv2.imwrite(str(tmp_path / f'opening-{n}.jpg'), np.hstack([
            crops[n], outputs['baseline'][n], outputs['tracked'][n]]))
