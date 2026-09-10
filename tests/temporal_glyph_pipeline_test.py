"""Decode, window, mask and composition contract for opt-in temporal glyphs."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
pytest.importorskip('av')

import vsr_pipeline
from backend.temporal_glyphs import TemporalGlyphs
from glyph_pipeline_test import text_frame, write_frames
from temporal_glyphs_test import REGION, observations
from vsr_pipeline import Pipeline


def pipeline_with_observations(boxes, cuts=()):
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = 'propainter'
    pipe._detect_timeline = lambda *args: (boxes, {
        'sampled': len(boxes), 'refined': 0, 'ocr_calls': len(boxes),
        'tracks': 1, 'discarded': 0, 'scene_change_frames': list(cuts),
    })
    return pipe


@pytest.mark.parametrize('enabled', [False, True])
def test_effective_masks_reach_model_and_exact_composition(tmp_path, enabled):
    frames, _, boxes, ink = observations()
    source, output = tmp_path / 'source.mp4', tmp_path / 'result.mp4'
    write_frames(source, frames)
    pipe = pipeline_with_observations(boxes)
    original_masks = [pipe.propainter_boxes_to_mask(b, f, REGION) for b, f in zip(boxes, frames)]
    model_masks, composites = [], []

    def inpaint(images, masks):
        model_masks.extend(m.copy() for m in masks)
        return [np.full_like(f, 90) for f in images]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    repair = pipe._repair_propainter_segment

    def record(images, masks, effective_boxes, check):
        result, repaired = repair(images, masks, effective_boxes, check)
        for image, mask, comp in zip(images, masks, result):
            np.testing.assert_array_equal(image[mask == 0], comp[mask == 0])
            assert np.all(comp[mask > 0] == 90)
        composites.extend(result)
        return result, repaired

    pipe._repair_propainter_segment = record
    options = {'temporal_glyphs': True} if enabled else {}
    stats = pipe.process_video(source, output, region=REGION, locate_stickers=False, **options)
    json.dumps(stats)
    added = [np.count_nonzero((new > 0) & (old == 0))
             for new, old in zip(model_masks, original_masks)]
    assert len(model_masks) == 6 and stats['frames'] == 6
    assert stats['temporal_glyphs_enabled'] is enabled
    assert stats['temporal_glyph_recovered'] == sum(n > 0 for n in added)
    assert stats['temporal_glyph_added_pixels'] == sum(added)
    assert stats['template_recovered'] == stats['repaired'] == 0
    assert stats['residual_check_enabled'] is False
    missing = (ink > 240) & (original_masks[-1] == 0)
    assert missing.sum() > 100
    if enabled:
        assert np.all(model_masks[-1][missing] > 0)
        assert np.all(composites[-1][missing] == 90)
    else:
        assert not any(added)


def test_overlap_counts_only_emitted_frames(tmp_path):
    frames, _, boxes, _ = observations()
    frames = [frames[i % 6] for i in range(70)]
    boxes = [boxes[i % 6] for i in range(70)]
    source, output = tmp_path / 'source.mp4', tmp_path / 'result.mp4'
    write_frames(source, frames)
    pipe = pipeline_with_observations(boxes)
    calls = []

    def inpaint(images, masks):
        calls.append([m.copy() for m in masks])
        return images

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, region=REGION, locate_stickers=False,
                               temporal_glyphs=True)
    assert [len(m) for m in calls] == [60, 30]
    emitted = calls[0][:40] + calls[1]
    original = [pipe.propainter_boxes_to_mask(b, f, REGION) for b, f in zip(boxes, frames)]
    added = [np.count_nonzero((new > 0) & (old == 0)) for new, old in zip(emitted, original)]
    assert sum(added) > 0
    assert stats['temporal_glyph_added_pixels'] == sum(added)
    assert stats['temporal_glyph_recovered'] == sum(n > 0 for n in added) <= 70


def test_scene_cut_does_not_supply_old_caption_evidence(tmp_path):
    clear, box = text_frame('Sturdy tile', thin_shadow=True)
    changed, _ = text_frame('Sturdy file', light_background=True, thin_shadow=True)
    frames = [clear] * 3 + [changed] * 3
    source, output = tmp_path / 'source.mp4', tmp_path / 'result.mp4'
    write_frames(source, frames)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = 'propainter'
    pipe.detect = lambda *args: [box]
    calls = []

    def inpaint(images, masks):
        calls.append(([f.copy() for f in images], [m.copy() for m in masks]))
        return images

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    pipe.process_video(source, output, region=REGION, locate_stickers=False, temporal_glyphs=True)
    assert [len(f) for f, _ in calls] == [3, 3]
    for (images, masks), expected in zip(calls, [clear, changed]):
        bgr = cv2.cvtColor(expected, cv2.COLOR_RGB2BGR)
        original = pipe.propainter_boxes_to_mask([box], expected, REGION)
        independent, _ = TemporalGlyphs(REGION).refine([bgr] * 3, [original] * 3, [[box]] * 3, [0, 1, 2])
        for image, actual, wanted in zip(images, masks, independent):
            np.testing.assert_array_equal(image, bgr)
            np.testing.assert_array_equal(actual, wanted)


def test_sticker_exclusions_reach_refiner_without_changing_sticker_masks(tmp_path, monkeypatch):
    frames, _, boxes, _ = observations()
    source, output = tmp_path / 'source.mp4', tmp_path / 'result.mp4'
    write_frames(source, frames)
    pipe = pipeline_with_observations(boxes)
    pipe.sticker_backend = 'gdino'
    pipe._ensure_sticker_detector = lambda: object()
    hits = {i: [(58, 80, 200 + 8 * i, 214 + 8 * i)] for i in range(6)}
    monkeypatch.setattr(vsr_pipeline, 'locate_stickers_gdino', lambda *args, **kw: hits)
    calls = []

    def inpaint(images, masks):
        calls.extend(m.copy() for m in masks)
        return images

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, region=REGION, temporal_glyphs=True)
    pad = vsr_pipeline.STICKER_MASK_PAD
    exclusions = [[(y1 - pad, y2 + pad, x1 - pad, x2 + pad) for y1, y2, x1, x2 in hits[i]]
                  for i in range(6)]
    originals = [pipe.propainter_boxes_to_mask(b, f, REGION, sticker_boxes=s)
                 for f, b, s in zip(frames, boxes, exclusions)]
    bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    expected, _ = TemporalGlyphs(REGION).refine(bgr, originals, boxes, list(range(6)), exclusions)
    unexcluded, _ = TemporalGlyphs(REGION).refine(bgr, originals, boxes, list(range(6)))
    assert any(not np.array_equal(a, b) for a, b in zip(expected, unexcluded))
    assert stats['temporal_glyphs_enabled'] is True and len(calls) == 6
    for actual, wanted, sticker_boxes in zip(calls, expected, exclusions):
        np.testing.assert_array_equal(actual, wanted)
        for y1, y2, x1, x2 in sticker_boxes:
            assert np.all(actual[y1:y2, x1:x2] == 255)


@pytest.mark.parametrize('args,expected', [([], False), (['--temporal-glyphs'], True)])
def test_cli_passes_temporal_glyphs(monkeypatch, args, expected):
    calls = []
    monkeypatch.setattr(vsr_pipeline, 'Pipeline', lambda **kw: SimpleNamespace(
        process_video=lambda *a, **kw: calls.append(kw) or {}))
    monkeypatch.setattr(sys, 'argv', ['vsr_pipeline.py', '-i', 'in.mp4', '-o', 'out.mp4',
                                    '--inpaint-mode', 'propainter', *args])
    vsr_pipeline.main()
    assert calls[0]['temporal_glyphs'] is expected


def test_conflicting_recovery_options_fail_before_opening_video(tmp_path):
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = 'propainter'
    with pytest.raises(ValueError, match='template_refine.*temporal_glyphs'):
        pipe.process_video(tmp_path / 'missing.mp4', tmp_path / 'output.mp4',
                           template_refine=True, temporal_glyphs=True)


def test_cli_conflict_fails_before_loading_models(monkeypatch):
    monkeypatch.setattr(vsr_pipeline, 'Pipeline', lambda **kw: pytest.fail('loaded model'))
    monkeypatch.setattr(sys, 'argv', ['vsr_pipeline.py', '-i', 'in.mp4', '-o', 'out.mp4',
                                    '--template-refine', '--temporal-glyphs'])
    with pytest.raises(SystemExit) as caught:
        vsr_pipeline.main()
    assert caught.value.code == 2


def test_lama_cannot_silently_ignore_temporal_glyphs(tmp_path):
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = 'lama'
    with pytest.raises(ValueError, match='temporal_glyphs.*propainter'):
        pipe.process_video(tmp_path / 'missing.mp4', tmp_path / 'out.mp4', temporal_glyphs=True)
