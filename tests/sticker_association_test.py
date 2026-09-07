from backend import subtitle_tracking as tracking


TEXT = (400, 430, 100, 300)
STICKER = (440, 475, 200, 235)


def text_timeline(total, start=0, end=None, box=TEXT):
    end = total - 1 if end is None else end
    return [[box] if start <= frame <= end else [] for frame in range(total)]


def test_vlm_sampling_preserves_dense_opening_and_call_limit():
    frames = tracking.plan_vlm_frames(687, [[] for _ in range(687)])
    assert {0, 15, 30, 45, 60}.issubset(frames)
    assert len(frames) == len(set(frames)) <= 32
    assert all(0 <= frame < 687 for frame in frames)


def test_vlm_sampling_adapts_opening_density_to_base_step():
    frames = tracking.plan_vlm_frames(150, [[] for _ in range(150)], base_step=24)
    assert frames[:5] == [0, 12, 24, 36, 48]


def test_vlm_sampling_prioritizes_each_spatial_track_before_base_samples():
    timeline = text_timeline(1000, 101, 109)
    for frame in range(107, 121):
        timeline[frame].append((700, 730, 100, 300))
    frames = tracking.plan_vlm_frames(1000, timeline, max_calls=9)
    assert frames == [0, 15, 30, 45, 60, 101, 109, 107, 120]


def test_vlm_sampling_prioritizes_late_short_text_window_before_base_samples():
    timeline = text_timeline(1000, 911, 919)
    frames = tracking.plan_vlm_frames(1000, timeline, max_calls=8)
    assert frames == [0, 15, 30, 45, 60, 911, 919, 915]


def test_vlm_sampling_keeps_track_edges_on_both_sides_of_scene_change():
    frames = tracking.plan_vlm_frames(
        1000, text_timeline(1000, 101, 999), max_calls=9, scene_change_frames=[700]
    )
    assert frames == [0, 15, 30, 45, 60, 101, 699, 700, 999]


def test_vlm_sampling_handles_short_and_empty_videos():
    assert tracking.plan_vlm_frames(0, []) == []
    assert tracking.plan_vlm_frames(100, [], max_calls=0) == []
    frames = tracking.plan_vlm_frames(4, text_timeline(4))
    assert set(frames) <= {0, 1, 2, 3}
    assert 0 in frames and 3 in frames


def test_sticker_match_keeps_touching_adjacent_boxes_separate():
    assert tracking.sticker_match_score((490, 525, 310, 350), (490, 525, 340, 380)) < 0


def test_sticker_grouping_matches_three_adjacent_objects_one_to_one():
    boxes = [(490, 525, x, x + 30) for x in (312, 343, 374)]
    hits = {0: boxes, 40: list(reversed(boxes)), 80: boxes}
    tracks = tracking.group_sticker_boxes(hits)
    assert len(tracks) == 3
    assert all(frames == [0, 40, 80] for _, frames in tracks)


def test_single_nearby_sticker_hit_has_only_short_propagation():
    result = tracking.associate_sticker_hits({20: [STICKER]}, text_timeline(50), 50)
    assert set(result) == set(range(14, 27))
    assert all(boxes == [STICKER] for boxes in result.values())


def test_single_distant_sticker_hit_is_discarded():
    far = (10, 45, 600, 635)
    assert tracking.associate_sticker_hits({20: [far]}, text_timeline(50), 50) == {}


def test_single_hit_propagation_stays_near_text_in_time():
    timeline = text_timeline(50, start=10, end=10)
    result = tracking.associate_sticker_hits({15: [STICKER]}, timeline, 50)
    assert set(result) == set(range(9, 17))


def test_stable_sticker_fills_short_gap_and_extends_to_absence_midpoints():
    hits = {0: [], 10: [STICKER], 20: [STICKER], 30: []}
    result = tracking.associate_sticker_hits(hits, text_timeline(40), 40)
    assert set(result) == set(range(5, 26))


def test_successful_empty_frame_prevents_sticker_interpolation():
    hits = {10: [STICKER], 20: [], 30: [STICKER]}
    result = tracking.associate_sticker_hits(hits, text_timeline(50), 50)
    assert set(result) == set(range(4, 17)) | set(range(24, 37))
    assert 20 not in result


def test_scene_change_prevents_matching_identical_stickers_across_scenes():
    result = tracking.associate_sticker_hits(
        {8: [STICKER], 22: [STICKER]},
        text_timeline(30),
        30,
        scene_change_frames=[15],
    )
    assert set(result) == set(range(2, 15)) | set(range(16, 29))


def test_single_hit_propagation_stops_before_next_scene():
    result = tracking.associate_sticker_hits(
        {14: [STICKER]}, text_timeline(30), 30, scene_change_frames=[15]
    )
    assert set(result) == set(range(8, 15))


def test_stable_track_extension_stays_inside_its_scene():
    result = tracking.associate_sticker_hits(
        {0: [], 12: [STICKER], 14: [STICKER], 29: []},
        text_timeline(30),
        30,
        scene_change_frames=[10, 15],
    )
    assert set(result) == set(range(10, 15))


def test_single_hit_cannot_borrow_subtitles_from_another_scene():
    result = tracking.associate_sticker_hits(
        {14: [STICKER]}, text_timeline(30, start=15), 30, scene_change_frames=[15]
    )
    assert result == {}


def test_stable_hits_cannot_borrow_subtitles_from_another_scene():
    result = tracking.associate_sticker_hits(
        {12: [STICKER], 14: [STICKER]},
        text_timeline(30, start=15),
        30,
        scene_change_frames=[15],
    )
    assert result == {}


def test_sticker_gap_larger_than_limit_is_not_filled():
    result = tracking.associate_sticker_hits(
        {10: [STICKER], 80: [STICKER]}, text_timeline(100), 100, max_gap=60
    )
    assert set(result) == set(range(4, 17)) | set(range(74, 87))


def test_separate_stable_tracks_do_not_bridge_a_gap_larger_than_limit():
    result = tracking.associate_sticker_hits(
        {0: [STICKER], 10: [STICKER], 80: [STICKER], 90: [STICKER]},
        text_timeline(100),
        100,
    )
    assert set(result) == set(range(0, 17)) | set(range(74, 97))


def test_single_hit_cannot_propagate_across_nearby_successful_empty_sample():
    result = tracking.associate_sticker_hits(
        {18: [], 20: [STICKER], 22: []}, text_timeline(50), 50
    )
    assert set(result) == {19, 20, 21}


def test_stable_track_extension_is_capped_even_when_absence_samples_are_far_away():
    result = tracking.associate_sticker_hits(
        {0: [], 200: [STICKER], 210: [STICKER], 450: []},
        text_timeline(500),
        500,
    )
    assert set(result) == set(range(140, 271))


def test_sticker_grouping_matches_gradual_movement_against_latest_box():
    hits = {
        frame: [(440, 475, 200 + frame, 230 + frame)]
        for frame in range(0, 101, 10)
    }
    groups = tracking.group_sticker_boxes(hits)
    assert len(groups) == 1
    assert groups[0][1] == list(range(0, 101, 10))


def test_stable_sticker_also_requires_nearby_subtitles():
    far = (10, 45, 600, 635)
    assert tracking.associate_sticker_hits(
        {10: [far], 20: [far]}, text_timeline(40), 40
    ) == {}


def test_stable_sticker_propagation_respects_short_text_window():
    hits = {10: [STICKER], 40: [STICKER]}
    result = tracking.associate_sticker_hits(hits, text_timeline(60, 20, 25), 60)
    assert set(result) == set(range(14, 32))


def test_moving_sticker_interpolates_boxes_without_spatial_union():
    moved = (442, 477, 210, 245)
    result = tracking.associate_sticker_hits(
        {10: [STICKER], 20: [moved]}, text_timeline(40), 40
    )
    assert result[10] == [STICKER]
    assert result[15] == [(441, 476, 205, 240)]
    assert result[20] == [moved]


def test_three_adjacent_stickers_keep_separate_interpolated_boxes():
    boxes = [(440, 475, x, x + 30) for x in (200, 231, 262)]
    result = tracking.associate_sticker_hits(
        {10: boxes, 20: list(reversed(boxes))}, text_timeline(40), 40
    )
    assert set(result[15]) == set(boxes)
    assert len(result[15]) == 3


def test_empty_video_or_empty_detections_produce_no_stickers():
    assert tracking.associate_sticker_hits({0: [STICKER]}, [], 0) == {}
    assert tracking.associate_sticker_hits({0: [], 10: []}, text_timeline(20), 20) == {}
