import pytest

from backend.subtitle_tracking import (
    BoxTrack,
    materialize_tracks,
    merge_closed_ranges,
    merge_residual_runs,
    plan_ocr_frames,
    refine_ranges,
    track_text_boxes,
)


def test_plan_ocr_frames_uses_five_frame_stride_and_last_frame():
    assert plan_ocr_frames(19, stride=5) == [0, 5, 10, 15, 18]


def test_track_text_boxes_keeps_two_non_adjacent_regions_separate():
    sampled = {
        0: [(100, 130, 80, 280), (900, 930, 90, 300)],
        5: [(102, 132, 82, 282), (902, 932, 92, 302)],
        10: [(104, 134, 84, 284), (904, 934, 94, 304)],
    }
    tracks = track_text_boxes(sampled, total_frames=11, max_gap=10)
    assert len(tracks) == 2
    assert all(track.frames == [0, 5, 10] for track in tracks)


def test_materialize_tracks_fills_short_gap_without_global_union():
    sampled = {0: [(100, 130, 80, 280)], 10: [(110, 140, 100, 300)]}
    tracks = track_text_boxes(sampled, total_frames=11, max_gap=10)
    timeline = materialize_tracks(tracks, total_frames=11, max_interpolation_gap=10)
    assert timeline[5]
    assert timeline[5][0][0] == 105
    assert timeline[5][0][2] == 90


def test_isolated_box_is_not_materialized():
    tracks = track_text_boxes(
        {20: [(400, 430, 100, 300)]},
        total_frames=40,
        min_hits=2,
        max_gap=10,
    )
    assert tracks == []
    assert materialize_tracks(tracks, total_frames=40) == [[] for _ in range(40)]


def test_refine_ranges_covers_track_edges_and_scene_change():
    ranges = refine_ranges(
        total_frames=100,
        track_frames=[20, 25, 30],
        scene_change_frames=[60],
        radius=15,
    )
    assert ranges == [(5, 75)]


def test_refine_ranges_merges_overlapping_windows():
    assert refine_ranges(100, [20, 30], [35], radius=10) == [(10, 45)]


def test_merge_residual_runs_adds_context_and_caps_one_run():
    assert merge_residual_runs([10, 11, 13, 40], total=60, context=5, max_runs=1) == [(5, 18)]


def test_merge_closed_ranges_merges_adjacent_closed_intervals():
    assert merge_closed_ranges([(8, 10), (11, 15), (20, 21), (20, 22)]) == [(8, 15), (20, 22)]


def test_box_track_rejects_mismatched_frame_and_box_counts():
    with pytest.raises(ValueError, match="frames.*boxes"):
        BoxTrack(0, [0, 5], [(100, 130, 80, 280)])


def test_text_tracks_do_not_jump_between_adjacent_lines():
    sampled = {
        0: [(100, 130, 80, 280)],
        5: [(190, 220, 80, 280)],
        10: [(190, 220, 80, 280)],
    }
    tracks = track_text_boxes(sampled, total_frames=11)
    assert [track.frames for track in tracks] == [[5, 10]]


def test_two_adjacent_text_lines_keep_independent_tracks():
    top = (100, 130, 80, 280)
    bottom = (135, 165, 80, 280)
    tracks = track_text_boxes({0: [top, bottom], 5: [bottom, top]}, 6)
    assert len(tracks) == 2
    assert {tuple(track.boxes) for track in tracks} == {(top, top), (bottom, bottom)}


def test_text_tracks_require_horizontal_overlap():
    tracks = track_text_boxes(
        {0: [(100, 130, 20, 50)], 5: [(100, 130, 70, 100)]}, 6
    )
    assert tracks == []


def test_text_tracks_keep_gradual_movement():
    sampled = {
        frame: [(100 + frame, 130 + frame, 80 + frame, 280 + frame)]
        for frame in range(0, 26, 5)
    }
    tracks = track_text_boxes(sampled, 26)
    assert len(tracks) == 1
    assert tracks[0].frames == [0, 5, 10, 15, 20, 25]


def test_scene_change_starts_new_track_without_cross_scene_interpolation():
    box = (100, 130, 80, 280)
    tracks = track_text_boxes(
        {0: [box], 4: [box], 6: [box], 10: [box]},
        11,
        scene_change_frames=[5],
    )
    assert [track.frames for track in tracks] == [[0, 4], [6, 10]]
    assert materialize_tracks(tracks, 11)[5] == []


def test_scene_change_frame_belongs_to_new_scene():
    box = (100, 130, 80, 280)
    tracks = track_text_boxes(
        {0: [box], 5: [box], 10: [box]}, 11, scene_change_frames=[5]
    )
    assert [track.frames for track in tracks] == [[5, 10]]


def test_text_tracks_discard_invalid_boxes():
    invalid = [(100, 100, 20, 50), (130, 100, 20, 50), (100, 130, 50, 20)]
    assert track_text_boxes({0: invalid, 5: invalid}, 6) == []


def test_short_missing_observation_can_be_interpolated():
    box = (100, 130, 80, 280)
    tracks = track_text_boxes({0: [box], 5: [], 10: [box]}, 11)
    assert materialize_tracks(tracks, 11)[5] == [box]


def test_long_gap_does_not_join_tracks():
    box = (100, 130, 80, 280)
    tracks = track_text_boxes({0: [box], 11: [box]}, 12)
    assert tracks == []


def test_empty_video_has_no_sampling_or_tracks():
    assert plan_ocr_frames(0) == []
    assert track_text_boxes({0: [(1, 2, 3, 4)]}, 0) == []
    assert materialize_tracks([], 0) == []
