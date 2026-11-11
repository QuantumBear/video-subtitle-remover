"""SAM3 text masks plus sparse DINO emoji detection, without text-mask regression."""
import numpy as np


def test_time_schedule_is_bounded_and_keeps_scene_starts():
    from backend.sam3_stickers import plan_samples
    samples = plan_samples(687, 30, [29, 59, 90, 665], 200)
    assert samples == sorted(set(range(0, 687, 30)) | {0, 15, 30, 45, 60, 29, 59, 90, 665, 686})
    assert plan_samples(0, 30, [], 200) == []
    assert plan_samples(687, 30, [29, 665], 0) == []
    short = plan_samples(687, 30, [29, 665], 4)
    assert set(short) == {0, 29, 665, 686}


def test_text_neighborhood_uses_current_frame_and_clips_to_manual_roi():
    from backend.sam3_stickers import detection_regions
    roi = (5, 95, 10, 170)
    text = [[(40, 50, 60, 100)], [], [(75, 85, 140, 165)]]
    assert detection_regions(text, roi) == [(20, 70, 40, 120), roi, (55, 95, 120, 170)]


def test_mask_union_preserves_holes_and_reschedules_emoji_only_frame(tmp_path):
    from backend.sam3_stickers import merge_masks
    from backend.sam3_mask import MaskTimeline
    store = MaskTimeline(tmp_path, (48, 64))
    text = np.zeros(store.shape, np.uint8)
    text[10:20, 10:20] = 255
    text[12:16, 12:16] = 0
    store.append(text)
    store.append(np.zeros_like(text))
    timeline = [[(10, 20, 10, 20)], []]
    stats = merge_masks(store, timeline, {0: [(25, 28, 30, 33)], 1: [(1, 5, 1, 5)]},
                        (3, 45, 3, 60), pad=2)
    assert store[0][13, 13] == 0
    assert np.all(store[0][text != 0] == 255)
    assert np.all(store[0][23:30, 28:35] == 255)
    assert not store[1][:3].any() and not store[1][:, :3].any()
    assert timeline[1] == [(3, 7, 3, 7)]
    assert stats == {'gdino_emoji_frames': 2, 'gdino_emoji_added_pixels': 65}
