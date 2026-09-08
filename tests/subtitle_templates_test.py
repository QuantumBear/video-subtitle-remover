import importlib
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def templates_class():
    try:
        module = importlib.import_module("backend.subtitle_templates")
    except ModuleNotFoundError:
        pytest.fail("SubtitleTemplates must provide verified cross-frame glyph recovery")
    return module.SubtitleTemplates


def subtitle(text="Sturdy steel", dx=0, dy=0, light_background=False):
    frame = np.full((128, 360, 3), (62, 81, 105), dtype=np.uint8)
    if light_background:
        frame[:, 95:290] = 242
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    origin = (25 + dx, 83 + dy)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (15, 15, 15), 6, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(mask, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                255, 6, cv2.LINE_AA)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    box = (int(ys.min()) - 3, int(ys.max()) + 4,
           int(xs.min()) - 3, int(xs.max()) + 4)
    return frame, mask, box


def partial(mask):
    result = mask.copy()
    result[:, 110:225] = 0
    return result


def refine_one(templates, frame, mask, box, number):
    masks, boxes = templates.refine([frame], [mask], [box], [number])
    return masks[0], boxes[0]


def test_observation_match_verifies_both_original_glyphs_without_changing_cache(templates_class):
    source, source_mask, source_box = subtitle()
    target, target_mask, target_box = subtitle(dx=3, dy=-2)
    target[np.all(target == (62, 81, 105), axis=2)] = (105, 125, 80)
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, source, source_mask, [source_box], 0)
    cache_stats = templates.stats.copy()
    originals = [item.copy() for item in (source, source_mask, target, target_mask)]

    pairs = templates.match_observations(source, source_mask, [source_box],
                                         target, target_mask, [target_box])

    assert pairs == [(0, 0)]
    assert templates.stats == cache_stats
    for actual, original in zip((source, source_mask, target, target_mask), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("source_text,target_text", [
    ("Sturdy steel", "Smooth stone"), ("Sturdy steel", "Sturdy wheel"),
    ("The tile", "The file"), ("The file", "The tile"),
    ("Sturdy tile", "Sturdy file"), ("Sturdy file", "Sturdy tile"),
])
def test_observation_match_rejects_changed_words_and_strokes(templates_class, source_text, target_text):
    source, source_mask, source_box = subtitle(source_text)
    target, target_mask, target_box = subtitle(target_text)
    templates = templates_class((0, 128, 0, 360))
    assert templates.match_observations(source, source_mask, [source_box],
                                        target, target_mask, [target_box]) == []


@pytest.mark.parametrize("narrow_side", [0, 1])
def test_observation_match_checks_changed_suffix_beyond_the_shorter_box(templates_class, narrow_side):
    observations = [subtitle("Sturdy steel"), subtitle("Sturdy wheel")]
    frame, mask, box = observations[narrow_side]
    mask[:, 160:] = 0
    observations[narrow_side] = frame, mask, (box[0], box[1], box[2], 160)
    source, source_mask, source_box = observations[0]
    target, target_mask, target_box = observations[1]
    templates = templates_class((0, 128, 0, 360))
    assert templates.match_observations(source, source_mask, [source_box],
                                        target, target_mask, [target_box]) == []


@pytest.mark.parametrize("damaged_side", [0, 1])
@pytest.mark.parametrize("mask_kind", ["partial", "rectangle", "empty"])
def test_observation_match_requires_two_reliable_raw_masks(templates_class, damaged_side, mask_kind):
    frame, complete, box = subtitle()
    damaged = partial(complete) if mask_kind == "partial" else np.zeros_like(complete)
    if mask_kind == "rectangle":
        damaged[box[0]:box[1], box[2]:box[3]] = 255
    masks = [complete, complete]
    masks[damaged_side] = damaged
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, frame, complete, [box], 0)
    assert templates.match_observations(frame, masks[0], [box],
                                        frame, masks[1], [box]) == []


def test_observation_match_reports_only_supported_line_indices(templates_class):
    source, source_mask, source_boxes = multiline_subtitle("Sturdy")
    target, target_mask, target_boxes = multiline_subtitle("Steady")
    templates = templates_class((0, 384, 0, 360))
    assert templates.match_observations(source, source_mask, source_boxes,
                                        target, target_mask, list(reversed(target_boxes))) == [(1, 1), (2, 0)]


def test_observation_match_ignores_boxes_outside_roi_without_renumbering(templates_class):
    frame, mask, boxes = multiline_subtitle()
    templates = templates_class((128, 256, 0, 360))
    assert templates.match_observations(frame, mask, boxes, frame, mask, boxes) == [(1, 1)]


def test_observation_match_does_not_reuse_a_current_box(templates_class):
    frame, mask, box = subtitle()
    templates = templates_class((0, 128, 0, 360))
    assert templates.match_observations(frame, mask, [box, box], frame, mask, [box]) == [(0, 0)]


def test_recovers_missing_glyphs_without_filling_interword_background(templates_class):
    frame, complete, box = subtitle()
    templates = templates_class((0, 128, 0, 360))
    refined, _ = templates.refine([frame, frame], [complete, partial(complete)],
                                  [[box], [box]], [0, 1])
    np.testing.assert_array_equal(refined[1], complete)
    assert np.count_nonzero(refined[1]) < 0.6 * (box[1] - box[0]) * (box[3] - box[2])


def test_narrow_ocr_box_recovers_prefix_and_expands_only_to_glyphs(templates_class):
    frame, complete, box = subtitle()
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, frame, complete, [box], 0)
    cropped = complete.copy()
    cropped[:, :58] = 0
    narrow = (box[0], box[1], 58, box[3])
    result, boxes = refine_one(templates, frame, cropped, [narrow], 1)
    np.testing.assert_array_equal(result, complete)
    assert len(boxes) == 1
    assert boxes[0][2] == int(np.nonzero(complete)[1].min())
    assert boxes[0][0:2] == narrow[0:2]


def test_aligns_small_translation_before_restoring_glyphs(templates_class):
    source, complete, source_box = subtitle()
    target, expected, target_box = subtitle(dx=5, dy=-3)
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, source, complete, [source_box], 0)
    result, _ = refine_one(templates, target, partial(expected), [target_box], 1)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("replacement", ["Smooth stone", "Sturdy wheel", "Sturdy steed",
                                         "Sturdy steal", "Sturdy Steel", "Sturdy steep"])
def test_same_position_changed_text_cannot_reuse_template(templates_class, replacement):
    source, complete, source_box = subtitle()
    target, target_mask, target_box = subtitle(replacement)
    damaged = partial(target_mask)
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, source, complete, [source_box], 0)
    result, boxes = refine_one(templates, target, damaged, [target_box], 1)
    np.testing.assert_array_equal(result, damaged)
    assert boxes == [target_box]


@pytest.mark.parametrize("source_text,target_text", [("The tile", "The file"),
                                                     ("The file", "The tile"),
                                                     ("Sturdy tile", "Sturdy file")])
def test_small_changed_stroke_rejects_high_contrast_template(templates_class, source_text, target_text):
    source, complete, source_box = subtitle(source_text)
    target, target_mask, target_box = subtitle(target_text)
    damaged = partial(target_mask)
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, source, complete, [source_box], 0)
    result, boxes = refine_one(templates, target, damaged, [target_box], 1)
    np.testing.assert_array_equal(result, damaged)
    assert boxes == [target_box]


def test_no_current_box_does_not_extend_subtitle_lifetime(templates_class):
    frame, complete, box = subtitle()
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, frame, complete, [box], 0)
    result, boxes = refine_one(templates, frame, np.zeros_like(complete), [], 1)
    assert not result.any()
    assert boxes == []


def test_blank_frame_with_stale_box_does_not_reuse_template(templates_class):
    frame, complete, box = subtitle()
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, frame, complete, [box], 0)
    blank = np.full_like(frame, 242)
    result, _ = refine_one(templates, blank, np.zeros_like(complete), [box], 1)
    assert not result.any()


def test_reset_separates_scenes(templates_class):
    frame, complete, box = subtitle()
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, frame, complete, [box], 0)
    templates.reset()
    result, _ = refine_one(templates, frame, partial(complete), [box], 1)
    np.testing.assert_array_equal(result, partial(complete))


def test_expiration_uses_real_frame_numbers(templates_class):
    frame, complete, box = subtitle()
    templates = templates_class((0, 128, 0, 360), max_age=5)
    refine_one(templates, frame, complete, [box], 10)
    result, _ = refine_one(templates, frame, partial(complete), [box], 16)
    np.testing.assert_array_equal(result, partial(complete))


def test_roi_clips_recovery_and_inputs_are_unchanged(templates_class):
    frame, complete, box = subtitle()
    region = (58, 85, 35, 250)
    damaged = partial(complete)
    frames, masks, boxes = [frame, frame.copy()], [complete, damaged], [[box], [box]]
    old_frames, old_masks = [f.copy() for f in frames], [m.copy() for m in masks]
    templates = templates_class(region)
    result, result_boxes = templates.refine(frames, masks, boxes, [0, 1])
    expected = np.zeros_like(complete)
    expected[58:85, 35:250] = complete[58:85, 35:250]
    np.testing.assert_array_equal(result[1], expected)
    for actual, original in zip(frames + masks, old_frames + old_masks):
        np.testing.assert_array_equal(actual, original)
    assert boxes == [[box], [box]]
    assert all(58 <= y1 < y2 <= 85 and 35 <= x1 < x2 <= 250
               for batch in result_boxes for y1, y2, x1, x2 in batch)


def test_changed_background_preserves_matching_text_structure(templates_class):
    source, complete, box = subtitle()
    target, expected, _ = subtitle(light_background=True)
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, source, complete, [box], 0)
    result, _ = refine_one(templates, target, partial(expected), [box], 1)
    np.testing.assert_array_equal(result, expected)


def test_damaged_observations_do_not_replace_clear_reference(templates_class):
    frame, complete, box = subtitle()
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, frame, complete, [box], 0)
    first = complete.copy()
    first[:, :120] = 0
    refine_one(templates, frame, first, [box], 1)
    last = complete.copy()
    last[:, 120:] = 0
    result, _ = refine_one(templates, frame, last, [box], 2)
    np.testing.assert_array_equal(result, complete)


def test_repeated_overlapping_windows_keep_a_bounded_patch_cache(templates_class):
    frame, complete, box = subtitle()
    templates = templates_class((0, 128, 0, 360), max_age=60)
    for start in range(0, 200, 20):
        numbers = list(range(start, start + 40))
        templates.refine([frame] * 40, [complete] * 40, [[box]] * 40, numbers)
    assert 0 < templates.stats["cached_templates"] <= 64
    assert templates.stats["cached_bytes"] < 64 * frame.nbytes
    result, _ = refine_one(templates, frame, partial(complete), [box], 220)
    np.testing.assert_array_equal(result, complete)


def multiline_subtitle(short_text="Sturdy"):
    rows = [subtitle(text) for text in [short_text, "Sturdy steel", "Smoothest stone"]]
    frame = np.concatenate([image for image, _, _ in rows])
    complete = np.concatenate([mask for _, mask, _ in rows])
    boxes = [(y1 + 128 * row, y2 + 128 * row, x1, x2)
             for row, (_, _, (y1, y2, x1, x2)) in enumerate(rows)]
    return frame, complete, boxes


@pytest.fixture(params=[((0, 60),), ((0, 40), (20, 60))],
                ids=["full-window", "overlapping-windows"])
def crowded_multiline_cache(templates_class, request):
    frame, complete, boxes = multiline_subtitle()
    damaged = complete.copy()
    damaged[:128, 110:225] = 0
    assert np.count_nonzero(damaged[:128]) < np.count_nonzero(complete[:128])
    templates = templates_class((0, 384, 0, 360))
    for start, stop in request.param:
        numbers = list(range(start, stop))
        observations = [complete if number == 0 else damaged for number in numbers]
        jittered_boxes = []
        for number in numbers:
            # Each coordinate moves at most one pixel without exactly repeating a box.
            offsets = [(number // 3 ** axis) % 3 - 1 for axis in range(4)]
            jittered_boxes.append([tuple(value + offset for value, offset in zip(box, offsets))
                                   for box in boxes])
        result, _ = templates.refine([frame] * len(numbers), observations, jittered_boxes, numbers)
        assert templates.stats["cached_templates"] == 64
    return templates, complete, result[-1]


def test_long_line_observations_cannot_evict_short_line_reference(crowded_multiline_cache):
    _, complete, result = crowded_multiline_cache
    np.testing.assert_array_equal(result, complete)


def test_retained_short_line_reference_cannot_restore_changed_text(crowded_multiline_cache):
    templates, _, _ = crowded_multiline_cache
    frame, complete, boxes = multiline_subtitle("Steady")
    damaged = complete.copy()
    damaged[:128, 110:225] = 0
    result, _ = refine_one(templates, frame, damaged, boxes, 60)
    np.testing.assert_array_equal(result, damaged)


def test_overlapping_window_keeps_reference_valid_for_its_earliest_frames(templates_class):
    frame, complete, box = subtitle()
    damaged = partial(complete)
    templates = templates_class((0, 128, 0, 360), max_age=60)
    first_masks = [complete if number == 30 else damaged for number in range(60)]
    first, _ = templates.refine([frame] * 60, first_masks, [[box]] * 60, list(range(60)))
    np.testing.assert_array_equal(first[40], complete)
    repeated, _ = templates.refine([frame] * 60, [damaged] * 60, [[box]] * 60,
                                   list(range(40, 100)))
    np.testing.assert_array_equal(repeated[0], complete)
    np.testing.assert_array_equal(repeated[-1], damaged)
    assert templates.stats["cached_templates"] <= 64


def test_later_clear_frame_can_repair_earlier_frame_in_same_window(templates_class):
    frame, complete, box = subtitle()
    templates = templates_class((0, 128, 0, 360))
    result, _ = templates.refine([frame, frame], [partial(complete), complete],
                                 [[box], [box]], [0, 1])
    np.testing.assert_array_equal(result[0], complete)


def test_thin_shadow_text_matches_when_white_background_hides_light_edges(templates_class):
    complete = np.zeros((128, 360), dtype=np.uint8)
    frames = []
    for background in [70, 242]:
        frame = np.full((128, 360, 3), background, dtype=np.uint8)
        cv2.putText(frame, "Sturdy steel", (26, 84), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (65, 65, 65), 2, cv2.LINE_AA)
        cv2.putText(frame, "Sturdy steel", (25, 83), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(frame)
    for origin in [(25, 83), (26, 84)]:
        cv2.putText(complete, "Sturdy steel", origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 255, 2, cv2.LINE_AA)
    complete = np.where(complete > 0, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(complete)
    box = (int(ys.min()) - 3, int(ys.max()) + 4, int(xs.min()) - 3, int(xs.max()) + 4)
    templates = templates_class((0, 128, 0, 360))
    result, _ = templates.refine(frames, [complete, partial(complete)],
                                 [[box], [box]], [0, 1])
    np.testing.assert_array_equal(result[1], complete)


def test_recent_matching_reference_is_not_hidden_by_larger_wrong_text(templates_class):
    templates = templates_class((0, 128, 0, 360))
    for number, text in enumerate(["Smoothest stone", "Brightest stone", "Silvery stones",
                                   "Little stones", "Better stones", "Simple stones",
                                   "Yellow stones", "Longer stones", "Hidden stones"]):
        frame, mask, box = subtitle(text)
        refine_one(templates, frame, mask, [box], number)
    frame, complete, box = subtitle()
    refine_one(templates, frame, complete, [box], 9)
    damaged = complete.copy()
    damaged[:, 120:] = 0
    result, _ = refine_one(templates, frame, damaged, [box], 10)
    np.testing.assert_array_equal(result, complete)


def test_incomplete_reference_does_not_authorize_a_changed_phrase(templates_class):
    source, complete, source_box = subtitle()
    reference = complete.copy()
    reference[:, 120:] = 0
    target, target_mask, target_box = subtitle("Sturdy wheel")
    damaged = target_mask.copy()
    damaged[:, :120] = 0
    templates = templates_class((0, 128, 0, 360))
    refine_one(templates, source, reference, [source_box], 0)
    result, _ = refine_one(templates, target, damaged, [target_box], 1)
    np.testing.assert_array_equal(result, damaged)


@pytest.mark.skipif(not os.environ.get("VSR_TEMPLATE_REGRESSION_VIDEO"),
                    reason="set VSR_TEMPLATE_REGRESSION_VIDEO to the diagnostic source video")
def test_diagnostic_frame_60_recovers_verified_glyphs_from_frame_30(templates_class):
    av = pytest.importorskip("av")
    from vsr_pipeline import Pipeline

    frames = []
    with av.open(os.environ["VSR_TEMPLATE_REGRESSION_VIDEO"]) as container:
        for number, frame in enumerate(container.decode(video=0)):
            if number in (30, 60):
                frames.append(frame.to_ndarray(format="bgr24"))
            if number == 60:
                break
    assert len(frames) == 2
    region, box = (0, 1280, 0, 720), (464, 505, 177, 540)
    pipe = Pipeline.__new__(Pipeline)
    masks = [pipe.propainter_boxes_to_mask([box], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), region)
             for frame in frames]
    templates = templates_class(region)
    result, _ = templates.refine(frames, masks, [[box], [box]], [30, 60])
    added = (result[1] > 0) & (masks[1] == 0)
    assert np.count_nonzero(added[:, 360:535]) > 500
    assert not np.any(added & (masks[0] == 0))


@pytest.mark.skipif(not os.environ.get("VSR_TEMPLATE_REGRESSION_VIDEO"),
                    reason="set VSR_TEMPLATE_REGRESSION_VIDEO to the diagnostic source video")
def test_diagnostic_frame_120_recovers_sturdy_prefix_from_frame_180(templates_class):
    av = pytest.importorskip("av")
    from vsr_pipeline import Pipeline

    frames = {}
    with av.open(os.environ["VSR_TEMPLATE_REGRESSION_VIDEO"]) as container:
        for number, frame in enumerate(container.decode(video=0)):
            if number in (120, 180):
                frames[number] = frame.to_ndarray(format="bgr24")
            if number == 180:
                break
    source_boxes = [(918, 963, 92, 293), (879, 927, 94, 376), (848, 888, 95, 242),
                    (810, 858, 94, 218), (774, 824, 95, 241)]
    target_boxes = [(917, 963, 94, 294), (880, 927, 92, 372), (848, 891, 94, 242),
                    (823, 849, 139, 204), (776, 820, 92, 240)]
    region = (0, 1280, 0, 720)
    pipe = Pipeline.__new__(Pipeline)
    source_mask = pipe.propainter_boxes_to_mask(
        source_boxes, cv2.cvtColor(frames[180], cv2.COLOR_BGR2RGB), region)
    target_mask = pipe.propainter_boxes_to_mask(
        target_boxes, cv2.cvtColor(frames[120], cv2.COLOR_BGR2RGB), region)
    templates = templates_class(region)
    refine_one(templates, frames[180], source_mask, source_boxes, 180)
    result, boxes = refine_one(templates, frames[120], target_mask, target_boxes, 120)
    assert np.count_nonzero(result[820:850, 100:139]) > 200
    assert boxes[3][2] <= 110
