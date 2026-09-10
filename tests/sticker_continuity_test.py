"""Model-free regression for appearance-confirmed sticker timelines."""

from types import SimpleNamespace

import numpy as np
import pytest

from backend import sticker_tracking


BOX = (30, 50, 30, 50)
TEXT = (55, 65, 20, 90)


def candidate(box=BOX, score=0.3):
    return SimpleNamespace(box=box, score=score)


def picture(boxes=(BOX,), variant=0):
    frame = np.full((100, 180, 3), 45, dtype=np.uint8)
    for y1, y2, x1, x2 in boxes:
        frame[y1:y2, x1:x2] = (235, 125, 25) if variant == 0 else (25, 125, 235)
        frame[y1 + 3:y1 + 8, x1 + 3:x1 + 8] = (80, 35, 15)
        frame[y2 - 6:y2 - 3, x1 + 5:x2 - 3] = (250, 230, 180)
    return frame


def tracker(total=40, **kwargs):
    text = kwargs.pop('text_timeline', [[TEXT] for _ in range(total)])
    return sticker_tracking.StickerTracker(text, total, score_threshold=0.22, **kwargs)


def replay(tracked, frames, samples):
    for n, observations in sorted(samples.items()):
        tracked.add_sample(n, frames[n], observations)
    for n, frame in enumerate(frames):
        tracked.observe(n, frame, samples.get(n, []))
    return tracked.finish()


def test_low_score_and_empty_detections_continue_only_visible_known_sticker():
    frames = [picture() for _ in range(40)]
    samples = {0: [candidate()], 10: [candidate(score=0.18)],
               20: [], 39: [candidate()]}
    result = replay(tracker(), frames, samples)
    assert set(result) == set(range(40))
    assert all(boxes == [BOX] for boxes in result.values())


def test_low_score_candidates_never_create_a_target():
    frames = [picture() for _ in range(20)]
    result = replay(tracker(20), frames,
                    {n: [candidate(score=0.21)] for n in (0, 10, 19)})
    assert result == {}


def test_weak_and_local_recoveries_do_not_qualify_a_distant_singleton():
    frames = [picture() for _ in range(40)]
    result = replay(tracker(text_timeline=[[(80, 95, 175, 180)]] * 40), frames,
                    {0: [candidate()], 10: [candidate(score=0.18)], 39: []})
    assert result == {}


def test_nearby_singleton_keeps_six_frame_limit_even_with_appearance():
    frames = [picture() for _ in range(40)]
    result = replay(tracker(), frames, {20: [candidate()]})
    assert set(result) == set(range(14, 27))


def test_local_checks_stop_at_disappearance_even_between_positive_samples():
    frames = [picture() if n < 10 or n >= 20 else picture(()) for n in range(40)]
    result = replay(tracker(), frames, {0: [candidate()], 39: [candidate()]})
    assert set(result) == set(range(10)) | set(range(20, 40))


def test_low_score_at_wrong_appearance_cannot_override_disappearance():
    frames = [picture() if n < 10 else picture(variant=1) for n in range(40)]
    result = replay(tracker(), frames,
                    {0: [candidate()], 9: [candidate()], 20: [candidate(score=0.18)]})
    assert set(result) == set(range(10))


def test_adjacent_similar_targets_are_not_merged_or_borrowed():
    right = (30, 50, 51, 71)
    frames = [picture((BOX, right)) if n < 10 else picture((right,)) for n in range(25)]
    result = replay(tracker(25), frames,
                    {0: [candidate(), candidate(right)],
                     9: [candidate(right), candidate()], 24: [candidate(right)]})
    assert all(set(result[n]) == {BOX, right} for n in range(10))
    assert all(result[n] == [right] for n in range(10, 25))


def test_scene_boundary_requires_fresh_strong_evidence():
    frames = [picture() for _ in range(40)]
    result = replay(tracker(scene_change_frames=[20]), frames,
                    {0: [candidate()], 10: [candidate()], 25: [candidate(score=0.18)]})
    assert set(result) == set(range(20))


def test_prediction_and_low_scores_do_not_refresh_reference_age():
    frames = [picture() for _ in range(40)]
    result = replay(tracker(max_gap=10), frames,
                    {0: [candidate()], 5: [candidate()], 20: [candidate(score=0.18)]})
    assert set(result) == set(range(16))


def test_uniform_reference_is_not_safe_for_propagation():
    frames = [np.full((100, 180, 3), 100, dtype=np.uint8) for _ in range(20)]
    tracked = tracker(20)
    result = replay(tracked, frames, {0: [candidate()], 19: [candidate()]})
    assert set(result) == {0, 19}
    assert tracked.stats['appearance_unknown'] > 0


def test_repeated_observation_in_same_frame_does_not_create_stability():
    tracked = tracker(text_timeline=[[(80, 95, 175, 180)]] * 40)
    tracked.add_sample(0, picture(), [candidate()])
    tracked.add_sample(0, picture(), [candidate()])
    for n in range(40):
        tracked.observe(n, picture(), [])
    assert tracked.finish() == {}


def test_different_high_score_appearances_cannot_qualify_a_distant_track():
    frames = [picture() if n < 10 else picture(variant=1) if n >= 30 else picture(())
              for n in range(40)]
    result = replay(tracker(text_timeline=[[(80, 95, 175, 180)]] * 40), frames,
                    {0: [candidate()], 39: [candidate()]})
    assert result == {}


def test_feedback_bridged_tracks_emit_only_one_box_per_target():
    tracked = tracker(161)
    for n in (0, 100, 50):
        tracked.add_sample(n, picture(), [candidate()])
    for n in range(161):
        tracked.observe(n, picture())
    result = tracked.finish()
    assert set(result) == set(range(161))
    assert all(boxes == [BOX] for boxes in result.values())


def test_feedback_bridge_keeps_a_visible_neighbor_and_a_disappearance_gap():
    right = (30, 50, 51, 71)
    frames = [picture((right,)) if 60 <= n < 70 else picture((BOX, right))
              for n in range(161)]
    tracked = tracker(161)
    for n in (0, 100, 50):
        tracked.add_sample(n, frames[n], [candidate(), candidate(right)])
    for n, frame in enumerate(frames):
        tracked.observe(n, frame)
    result = tracked.finish()
    for n in range(161):
        expected = {right} if 60 <= n < 70 else {BOX, right}
        assert set(result.get(n, [])) == expected, f'frame {n}'


def test_small_motion_tracks_original_reference_without_freezing_the_box():
    boxes = [(30, 50, 30 + n, 50 + n) for n in range(15)]
    frames = [picture((box,)) for box in boxes]
    result = replay(tracker(15), frames,
                    {0: [candidate(boxes[0])], 14: [candidate(boxes[14])]})
    assert all(result[n] == [box] for n, box in enumerate(boxes))


def test_small_changed_edge_pixels_do_not_hide_unchanged_sticker_core():
    frames = [picture() for _ in range(20)]
    for frame in frames[1:19]:
        frame[30:33, 30:50] = (20, 20, 240)
    result = replay(tracker(20), frames, {0: [candidate()], 19: [candidate()]})
    assert set(result) == set(range(20))


def test_large_appearance_change_is_not_treated_as_background_noise():
    frames = [picture() for _ in range(20)]
    for frame in frames[1:19]:
        frame[30:43, 30:50] = (20, 20, 240)
    result = replay(tracker(20), frames, {0: [candidate()], 19: [candidate()]})
    assert set(result) == {0, 19}


def test_appearance_exception_is_unknown_not_absence_or_layer_failure(monkeypatch):
    tracked = tracker(10)
    frames = [picture() for _ in range(10)]
    tracked.add_sample(0, frames[0], [candidate()])
    tracked.add_sample(9, frames[9], [candidate()])

    def unavailable(*args, **kwargs):
        raise sticker_tracking.cv2.error('appearance check failed')

    monkeypatch.setattr(sticker_tracking.cv2, 'matchTemplate', unavailable)
    for n, frame in enumerate(frames):
        tracked.observe(n, frame, [])
    assert set(tracked.finish()) == {0, 9}
    assert tracked.stats['appearance_unknown'] == 8
    assert tracked.stats['appearance_absent'] == 0


def test_reference_extraction_failure_does_not_discard_other_confirmed_stickers(monkeypatch):
    tracked = tracker(10)
    tracked.add_sample(0, picture(), [candidate()])
    tracked.add_sample(9, picture(), [candidate()])

    def unavailable(*args, **kwargs):
        raise sticker_tracking.cv2.error('reference extraction failed')

    monkeypatch.setattr(sticker_tracking.cv2, 'cvtColor', unavailable)
    right = (30, 50, 65, 85)
    tracked.add_sample(5, picture((BOX, right)), [candidate(right)])
    for n in range(10):
        tracked.observe(n, picture((BOX, right)))
    result = tracked.finish()
    assert all(BOX in result[n] for n in range(10))
    assert right in result[5]
    assert all(right not in result[n] for n in range(10) if n != 5)
    assert tracked.stats['appearance_unknown'] == 9


@pytest.mark.parametrize('failure', ['uniform', 'exception'])
def test_invalid_new_reference_does_not_hide_valid_same_target_reference(monkeypatch, failure):
    tracked = tracker()
    tracked.add_sample(0, picture(), [candidate()])
    if failure == 'uniform':
        tracked.add_sample(10, picture(), [candidate((36, 44, 38, 46))])
    else:
        def unavailable(*args, **kwargs):
            raise sticker_tracking.cv2.error('reference extraction failed')

        with monkeypatch.context() as scoped:
            scoped.setattr(sticker_tracking.cv2, 'cvtColor', unavailable)
            tracked.add_sample(10, picture(), [candidate()])
    tracked.add_sample(39, picture(), [candidate()])
    for n in range(40):
        tracked.observe(n, picture())
    assert set(tracked.finish()) == set(range(40))
    assert tracked.stats['appearance_unknown'] == 0


def test_invalid_reference_does_not_extend_expired_valid_appearance(monkeypatch):
    tracked = tracker(max_gap=10)
    tracked.add_sample(0, picture(), [candidate()])
    tracked.add_sample(5, picture(), [candidate()])

    def unavailable(*args, **kwargs):
        raise sticker_tracking.cv2.error('reference extraction failed')

    monkeypatch.setattr(sticker_tracking.cv2, 'cvtColor', unavailable)
    tracked.add_sample(10, picture(), [candidate()])
    for n in range(40):
        tracked.observe(n, picture())
    assert set(tracked.finish()) == set(range(16))
    assert tracked.stats['appearance_unknown'] > 0


def test_low_score_proposal_cannot_jump_to_a_neighbor_lookalike():
    nearby = (30, 50, 42, 62)
    frames = [picture() if n in (0, 19) else picture((nearby,)) for n in range(20)]
    samples = {0: [candidate()], 19: [candidate()], 10: [candidate(nearby, 0.18)]}
    result = replay(tracker(20), frames, samples)
    assert set(result) == {0, 19}


def test_feedback_reference_can_recheck_prior_locally_visible_frames(tmp_path):
    from glyph_pipeline_test import write_frames
    from backend.sticker_detect import StickerCandidate

    right = (30, 50, 65, 85)
    frames = [picture((BOX, right) if 1 <= n < 8 else (right,)) for n in range(30)]
    for n, frame in enumerate(frames):
        frame[0, 0] = n
    source = tmp_path / 'recovered.mp4'
    write_frames(source, frames)

    class Detector:
        def detect_candidates(self, crop, region, prompt, score_threshold, max_area_px):
            n = int(crop[0, 0, 0])
            result = [StickerCandidate(right, 0.3)]
            if 1 <= n < 8:
                result.append(StickerCandidate(BOX, 0.3 if n == 3 else 0.18))
            return result

    result, stats = sticker_tracking.locate_tracked_stickers(
        source, (0, 100, 0, 180), [0, 29], Detector(), [[TEXT]] * 30, 30,
        max_calls=8, score_threshold=0.22, base_step=8)
    assert [n for n, boxes in sorted(result.items()) if BOX in boxes] == list(range(1, 8))
    assert stats['strong_hits'] >= 3


def test_stable_strong_feedback_does_not_force_frame_by_frame_detection(tmp_path):
    from glyph_pipeline_test import write_frames
    from backend.sticker_detect import StickerCandidate

    source = tmp_path / 'stable.mp4'
    write_frames(source, [picture()] * 80)

    class Detector:
        def detect_candidates(self, *args):
            return [StickerCandidate(BOX, 0.3)]

    result, stats = sticker_tracking.locate_tracked_stickers(
        source, (0, 100, 0, 180), [0, 79], Detector(), [[TEXT]] * 80, 80,
        max_calls=80, score_threshold=0.22, base_step=30)
    assert set(result) == set(range(80))
    assert stats['calls'] < 15


def test_budget_reserves_all_priority_frames_and_counts_failed_attempts():
    schedule = sticker_tracking.FeedbackSchedule([0, 10, 20], max_calls=5, max_step=8)
    assert schedule.priority_frames == [0, 10, 20]
    for n in schedule.priority_frames:
        schedule.record_priority(n)
    assert schedule.should_check(1, changed=True, active=True)
    schedule.record_feedback(1, changed=True)
    assert schedule.should_check(2, changed=False, active=True)
    schedule.record_feedback(2, changed=False)
    assert not schedule.should_check(3, changed=True, active=True)
    assert schedule.calls == 5


def test_stable_feedback_backs_off_and_changes_reset_density():
    schedule = sticker_tracking.FeedbackSchedule([], max_calls=20, max_step=8)
    seen = []
    for n in range(25):
        changed = n in (0, 17)
        if schedule.should_check(n, changed=changed, active=True):
            seen.append(n)
            schedule.record_feedback(n, changed=changed)
    assert seen == [0, 1, 3, 7, 15, 17, 18, 20, 24]


@pytest.mark.parametrize('budget,expected', [(0, []), (1, [10]), (2, [10, 20])])
def test_small_budgets_preserve_supplied_priority_order(budget, expected):
    schedule = sticker_tracking.FeedbackSchedule([10, 20, 0, 10], budget)
    assert schedule.priority_frames == expected
    assert not schedule.should_check(1, changed=True, active=True)


def test_tracked_location_reserves_late_priority_and_uses_bounded_feedback(tmp_path):
    from glyph_pipeline_test import write_frames
    from backend.sticker_detect import StickerCandidate

    assert hasattr(sticker_tracking, 'locate_tracked_stickers'), 'tracked video detection is missing'
    frames = [picture() if n < 10 else picture(()) for n in range(30)]
    for n, frame in enumerate(frames):
        frame[0, 0] = n
    source = tmp_path / 'stickers.mp4'
    write_frames(source, frames)
    calls = []

    class Detector:
        def detect_candidates(self, crop, region, prompt, score_threshold, max_area_px):
            n = int(crop[0, 0, 0])
            calls.append(n)
            if n == 1:
                raise RuntimeError('failed model inference')
            return [StickerCandidate(BOX, 0.3)] if n < 10 else []

    result, stats = sticker_tracking.locate_tracked_stickers(
        source, (0, 100, 0, 180), [0, 9, 29], Detector(), [[TEXT]] * 30, 30,
        max_calls=7, score_threshold=0.22, base_step=8)
    assert calls[:3] == [0, 9, 29]
    assert len(calls) == len(set(calls)) <= 7
    assert set(result) == set(range(10))
    assert stats['calls'] == len(calls)
    assert stats['model_failed'] == 1
    assert stats['budget_skipped'] > 0


def test_zero_budget_does_not_open_video_or_run_models():
    assert hasattr(sticker_tracking, 'locate_tracked_stickers'), 'tracked video detection is missing'
    result, stats = sticker_tracking.locate_tracked_stickers(
        'missing.mp4', (0, 100, 0, 180), [0], None, [[TEXT]], 1, max_calls=0)
    assert result == {} and stats['calls'] == 0


def test_zero_budget_pipeline_does_not_load_sticker_model(tmp_path):
    import vsr_pipeline
    from glyph_pipeline_test import write_frames

    source, output = tmp_path / 'input.mp4', tmp_path / 'output.mp4'
    write_frames(source, [picture()] * 2)
    pipe = vsr_pipeline.Pipeline.__new__(vsr_pipeline.Pipeline)
    pipe.inpaint_mode = 'propainter'
    pipe.sticker_backend = 'gdino'
    pipe._detect_timeline = lambda *args: ([[], []], dict(
        sampled=2, refined=0, ocr_calls=2, tracks=0, discarded=0, scene_change_frames=[]))
    pipe._ensure_sticker_detector = lambda: pytest.fail('zero budget must not load a model')
    stats = pipe.process_video(source, output, region=(0, 100, 0, 180),
                               sticker_max_frames=0)
    assert stats['frames'] == 2 and stats['inpainted'] == 0


def test_confirmed_gdino_masks_reach_engine_without_absence_interpolation(tmp_path, monkeypatch, capsys):
    import vsr_pipeline
    from glyph_pipeline_test import write_frames
    from backend.sticker_detect import StickerCandidate

    frames = [picture() if n < 4 or n >= 8 else picture(()) for n in range(12)]
    source, output = tmp_path / 'input.mp4', tmp_path / 'output.mp4'
    write_frames(source, frames)
    pipe = vsr_pipeline.Pipeline.__new__(vsr_pipeline.Pipeline)
    pipe.inpaint_mode = 'propainter'
    pipe.sticker_backend = 'gdino'
    text = (75, 85, 120, 170)
    pipe._detect_timeline = lambda *args: ([[text]] * 12, dict(
        sampled=2, refined=0, ocr_calls=2, tracks=1, discarded=0, scene_change_frames=[]))
    monkeypatch.setattr(vsr_pipeline, 'associate_sticker_hits',
                        lambda *a, **kw: pytest.fail('confirmed masks must not be re-interpolated'))

    class Detector:
        def locate(self, *args, **kwargs):
            pytest.fail('GDINO pipeline still discards scored candidates')

        def detect_candidates(self, crop, region, prompt, score_threshold, max_area_px):
            return [StickerCandidate(BOX, 0.3)] if crop[32, 32, 0] > 200 else []

    pipe._ensure_sticker_detector = lambda: Detector()
    masks_seen = []
    real_repair = pipe._repair_propainter_segment
    pipe.inpainter = SimpleNamespace(inpaint=lambda images, masks: [np.zeros_like(f) for f in images])

    def repair(images, masks, boxes, white_glyph_check=True):
        masks_seen.extend(mask.copy() for mask in masks)
        fixed, stats = real_repair(images, masks, boxes, white_glyph_check)
        for rgb, mask, output in zip(images, masks, fixed):
            np.testing.assert_array_equal(output[mask == 0], rgb[mask == 0])
        return fixed, stats

    pipe._repair_propainter_segment = repair
    stats = pipe.process_video(source, output, region=(0, 100, 0, 180),
                               sticker_max_frames=12, sticker_score=0.22)
    assert len(masks_seen) == 12
    assert [bool(mask[40, 40]) for mask in masks_seen] == [True] * 4 + [False] * 4 + [True] * 4
    assert stats.get('residual_check_enabled') is False
    assert '残留复核 未启用' in capsys.readouterr().out
