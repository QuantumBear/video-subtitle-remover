"""Behavioral tests for original-frame, window-local glyph consensus."""

import importlib

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')


@pytest.fixture
def glyphs_class():
    try:
        return importlib.import_module('backend.temporal_glyphs').TemporalGlyphs
    except ModuleNotFoundError:
        pytest.fail('TemporalGlyphs must recover original strokes across partial observations')


REGION = (0, 128, 0, 360)
BOX = (47, 89, 15, 310)


def caption(text='Non-slip Sturdy', bright=None, dx=0):
    frame = np.full((128, 360, 3), (70, 85, 105), dtype=np.uint8)
    if bright is not None:
        frame[:, bright[0]:bright[1]] = 248
    ink = np.zeros(frame.shape[:2], np.uint8)
    for destination, color, thickness, origin in [
            (frame, (60, 60, 60), 2, (32 + dx, 81)),
            (frame, (255, 255, 255), 2, (31 + dx, 80)),
            (ink, 255, 2, (31 + dx, 80))]:
        cv2.putText(destination, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (22 + dx, 70), 3, (255, 255, 255), -1)
    cv2.circle(ink, (22 + dx, 70), 3, 255, -1)
    return frame, ink


def observations():
    frames = [caption(bright=bright)[0] for bright in [(175, 320), (85, 210), (0, 145),
                                                     (0, 100), (160, 360), (0, 160)]]
    ink = caption()[1]
    complete = cv2.dilate((ink > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    masks = [complete.copy() for _ in frames]
    boxes = [[BOX] for _ in frames]
    for i, (lo, hi) in enumerate([(175, 360), (85, 210), (0, 145),
                                (0, 100), (160, 360), (160, 360)]):
        masks[i][:, lo:hi] = 0
    boxes[-1] = [(47, 89, 30, 160)]
    return frames, masks, boxes, ink


def test_complementary_observations_recover_suffix_and_bullet_without_rectangles(glyphs_class):
    frames, masks, boxes, ink = observations()
    originals = [f.copy() for f in frames + masks]
    refined, result_boxes = glyphs_class(REGION).refine(frames, masks, boxes, list(range(6)))
    missing = (ink > 240) & (masks[-1] == 0)
    assert missing.sum() > 100
    assert np.all(refined[-1][missing] > 0)
    assert refined[-1][70, 22] > 0
    allowed = cv2.dilate((ink > 0).astype(np.uint8), np.ones((11, 11), np.uint8))
    assert not np.any((refined[-1] > 0) & (allowed == 0))
    assert len(result_boxes[-1]) == 1 and result_boxes[-1][0][3] > 230
    for current, original in zip(frames + masks, originals):
        np.testing.assert_array_equal(current, original)


def test_white_objects_outside_text_cannot_become_strokes(glyphs_class):
    frames, masks, boxes, ink = observations()
    for f in frames:
        f[:, 315:340] = 255
    result, _ = glyphs_class(REGION).refine(frames, masks, boxes, list(range(6)))
    assert all(not mask[:, 315:].any() for mask in result)


@pytest.mark.parametrize('x1,x2,y1,y2', [(275, 278, 64, 76), (8, 11, 64, 76),
                                      (275, 281, 67, 73)])
def test_unobserved_white_objects_near_letters_are_not_punctuation(glyphs_class, x1, x2, y1, y2):
    frame, ink = caption()
    frames = [frame.copy() for _ in range(6)]
    mask = cv2.dilate(ink, np.ones((7, 7), np.uint8))
    mask[:, 264:] = 0
    for f in frames:
        f[y1:y2, x1:x2] = 255
    result, _ = glyphs_class(REGION).refine(frames, [mask] * 6,
                                           [[(47, 89, 15, 264)]] * 6, list(range(6)))
    assert all(not m[y1:y2, x1:x2].any() for m in result)


def test_loose_edges_cannot_reintroduce_a_rejected_neighboring_object(glyphs_class):
    frame, ink = caption()
    frame[:, 261:340] = 255
    mask = cv2.dilate(ink, np.ones((7, 7), np.uint8))
    mask[:, 260:] = 0
    allowed = cv2.dilate((ink > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
    result, _ = glyphs_class(REGION).refine([frame] * 6, [mask] * 6,
                                           [[(47, 89, 15, 260)]] * 6, list(range(6)))
    for new in result:
        assert not np.any((new[:, 261:340] > 0) & (allowed[:, 261:340] == 0))


def test_no_current_box_does_not_extend_lifetime(glyphs_class):
    frames, masks, boxes, _ = observations()
    frames.append(np.full_like(frames[0], 250))
    masks.append(np.zeros_like(masks[0]))
    boxes.append([])
    result, result_boxes = glyphs_class(REGION).refine(frames, masks, boxes, list(range(7)))
    assert not result[-1].any() and result_boxes[-1] == []


def test_stale_box_on_bright_blank_frame_cannot_restore_old_letters(glyphs_class):
    frames, masks, boxes, _ = observations()
    frames[3] = np.full_like(frames[3], 255)
    masks[3] = np.zeros_like(masks[3])
    result, _ = glyphs_class(REGION).refine(frames, masks, boxes, list(range(6)))
    assert not result[3].any()


def test_unsampled_disappearance_is_checked_against_target_pixels(glyphs_class):
    frames, masks, boxes, _ = observations()
    frames = [frames[i % 6].copy() for i in range(60)]
    masks = [masks[i % 6].copy() for i in range(60)]
    boxes = [[BOX] for _ in frames]
    # Frame 2 is not one of the 16 uniformly selected reference frames.
    frames[2][:] = 255
    masks[2][:] = 0
    result, _ = glyphs_class(REGION).refine(frames, masks, boxes, list(range(60)))
    assert not result[2].any()


@pytest.mark.parametrize('texture', ['stripes', 'white_block'])
def test_bright_nontext_structure_cannot_confirm_old_caption(glyphs_class, texture):
    frame, ink = caption()
    frames = [frame.copy() for _ in range(60)]
    masks = [cv2.dilate(ink, np.ones((7, 7), np.uint8)) for _ in frames]
    frames[2][:] = 225
    if texture == 'stripes':
        frames[2][:, ::4] = 255
    else:
        frames[2][47:89, 15:310] = 255
    masks[2][:] = 0
    result, _ = glyphs_class(REGION).refine(frames, masks, [[BOX]] * 60, list(range(60)))
    assert not result[2].any()


def test_moving_caption_does_not_leave_a_stationary_trail(glyphs_class):
    frames, masks, boxes, _ = observations()
    for i, dx in enumerate([0, 5, 10, 15, 20, 25]):
        frames[i], ink = caption(dx=dx)
        masks[i][:] = 0
        masks[i][:, :150] = ink[:, :150]
        boxes[i] = [(47, 89, 15 + dx, 310 + dx)]
    result, _ = glyphs_class(REGION).refine(frames, masks, boxes, list(range(6)))
    for old, new in zip(masks, result):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize('offsets', [[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
def test_small_motion_only_recovers_current_strokes(glyphs_class, offsets):
    frames, masks, boxes, allowed = [], [], [], []
    for dx in offsets:
        frame, ink = caption(dx=dx)
        complete = cv2.dilate((ink > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
        allowed.append(complete)
        partial = complete.copy() * 255
        partial[:, 150:] = 0
        frames.append(frame)
        masks.append(partial)
        boxes.append([(47, 89, 15 + dx, 310 + dx)])
    result, _ = glyphs_class(REGION).refine(frames, masks, boxes, list(range(6)))
    for new, current in zip(result, allowed):
        assert not np.any((new > 0) & (current == 0))


@pytest.mark.parametrize('old_text,new_text', [('Sturdy tile', 'Sturdy file'),
                                             ('Sturdy file', 'Sturdy tile')])
def test_similar_changed_letters_do_not_receive_old_strokes(glyphs_class, old_text, new_text):
    frames = [caption(old_text)[0] for _ in range(60)]
    ink = caption(old_text)[1]
    masks = [cv2.dilate(ink, np.ones((7, 7), np.uint8)) for _ in frames]
    frames[2], new_ink = caption(new_text)
    masks[2][:] = 0
    result, _ = glyphs_class(REGION).refine(frames, masks, [[BOX]] * 60, list(range(60)))
    allowed = cv2.dilate((new_ink > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
    assert not np.any((result[2] > 0) & (allowed == 0))


def test_duplicate_boxes_cannot_replace_distinct_observations(glyphs_class):
    frames, masks, _, _ = observations()
    result, _ = glyphs_class(REGION).refine(frames[:2], masks[:2], [[BOX] * 4] * 2, [0, 1])
    for old, new in zip(masks, result):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize('replacement', ['Non-slip Steady', 'Non-slip', 'New subtitle'])
def test_changed_caption_cannot_receive_disappeared_strokes(glyphs_class, replacement):
    frames, masks, boxes, _ = observations()
    changed, ink = caption(replacement, bright=(0, 160))
    frames[-1] = changed
    masks[-1][:] = 0
    result, _ = glyphs_class(REGION).refine(frames, masks, boxes, list(range(6)))
    allowed = cv2.dilate((ink > 0).astype(np.uint8), np.ones((11, 11), np.uint8))
    assert not np.any((result[-1] > 0) & (allowed == 0))


def test_roi_and_sticker_exclusions_bound_new_pixels(glyphs_class):
    frames, masks, boxes, _ = observations()
    region = (52, 88, 20, 250)
    exclusions = [[(60, 83, 150, 195)] for _ in frames]
    for mask in masks:
        mask[60:83, 150:195] = 255
    result, _ = glyphs_class(region).refine(frames, masks, boxes, list(range(6)),
                                           excluded_boxes=exclusions)
    for old, new in zip(masks, result):
        assert not new[:52].any() and not new[88:].any()
        assert not new[:, :20].any() and not new[:, 250:].any()
        np.testing.assert_array_equal(new[60:83, 150:195], old[60:83, 150:195])


def test_sparse_or_repeated_frame_numbers_cannot_invent_consensus(glyphs_class):
    frames, masks, boxes, _ = observations()
    for indices in ([0, 10, 20, 30, 40, 50], [0, 0, 0, 0, 0, 0]):
        result, _ = glyphs_class(REGION).refine(frames, masks, boxes, indices)
        for actual, expected in zip(result, masks):
            np.testing.assert_array_equal(actual, expected)


def test_refiner_does_not_remember_prior_windows(glyphs_class):
    frames, masks, boxes, _ = observations()
    refiner = glyphs_class(REGION)
    refiner.refine(frames, masks, boxes, list(range(6)))
    blank = np.full_like(frames[0], 255)
    result, _ = refiner.refine([blank] * 3, [np.zeros_like(masks[0])] * 3,
                              [[BOX]] * 3, [10, 11, 12])
    assert all(not m.any() for m in result)


def test_invalid_input_and_window_limit(glyphs_class):
    frames, masks, boxes, _ = observations()
    refiner = glyphs_class(REGION)
    assert refiner.refine([], [], [], []) == ([], [])
    with pytest.raises(ValueError):
        refiner.refine(frames, masks[:1], boxes, list(range(6)))
    with pytest.raises(ValueError):
        refiner.refine(frames, [m[:20] for m in masks], boxes, list(range(6)))
    with pytest.raises(ValueError):
        refiner.refine(frames * 11, masks * 11, boxes * 11, list(range(66)))
