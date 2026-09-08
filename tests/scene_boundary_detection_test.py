from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from vsr_pipeline import Pipeline


REGION = (30, 165, 12, 348)


def scene_frame(texts=("Sturdy steel", "Second line"), background=60, number=0):
    image = np.full((192, 360, 3), background, dtype=np.uint8)
    image[0, 0] = number
    boxes = []
    for row, text in enumerate(texts):
        if not text:
            continue
        glyph = np.zeros(image.shape[:2], dtype=np.uint8)
        origin = (25, 75 + 65 * row)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (15, 15, 15), 6, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(glyph, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, 255, 6, cv2.LINE_AA)
        ys, xs = np.nonzero(glyph)
        boxes.append((int(ys.min()) - 3, int(ys.max()) + 4,
                      int(xs.min()) - 3, int(xs.max()) + 4))
    return image, boxes


def write_scenes(path, frames):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=30)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "bf": "0"}
        for number, image in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = number
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def detecting_pipeline(observations):
    pipe = Pipeline.__new__(Pipeline)
    calls = []

    def detect(image, region):
        number = int(image[0, 0, 0])
        calls.append(number)
        return observations[number]

    pipe.detect = detect
    return pipe, calls


def test_consecutive_single_frame_cuts_preserve_both_observed_subtitle_lines(tmp_path):
    scenes = [scene_frame(background=value, number=number)
              for number, value in enumerate([60, 130, 190, 90, 90, 90])]
    path = tmp_path / "consecutive-cuts.mp4"
    write_scenes(path, [frame for frame, _ in scenes])
    observations = [boxes for _, boxes in scenes]
    pipe, calls = detecting_pipeline(observations)

    timeline, stats = pipe._detect_timeline(path, REGION)

    assert stats["scene_change_frames"] == [1, 2, 3]
    assert timeline == observations
    assert stats["ocr_calls"] == len(calls) == 6
    assert stats["tracks"] == 8
    assert stats["detected"] == stats["accepted"] == 12
    assert stats["discarded"] == 0


@pytest.mark.parametrize("source_text,target_text", [
    ("Sturdy steel", "Sturdy wheel"), ("Sturdy tile", "Sturdy file"),
    ("Sturdy file", "Sturdy tile"), ("Sturdy steel", ""),
])
def test_cut_does_not_confirm_changed_or_disappeared_text(tmp_path, source_text, target_text):
    scenes = [scene_frame((text,), background, number)
              for number, (text, background) in enumerate([(source_text, 60), (target_text, 140)])]
    path = tmp_path / "changed.mp4"
    write_scenes(path, [frame for frame, _ in scenes])
    # The disappearance case includes a stale OCR box and must reject its rectangle mask.
    observations = [scenes[0][1], scenes[1][1] or scenes[0][1]]
    pipe, _ = detecting_pipeline(observations)

    timeline, stats = pipe._detect_timeline(path, REGION)

    assert stats["scene_change_frames"] == [1]
    assert timeline == [[], []]
    assert stats["detected"] == stats["discarded"] == 2
    assert stats["accepted"] == 0


@pytest.mark.parametrize("narrow_side", [0, 1])
def test_cut_checks_changed_suffix_when_one_ocr_box_only_covers_prefix(tmp_path, narrow_side):
    scenes = [scene_frame((text,), background, number)
              for number, (text, background) in enumerate([
                  ("Sturdy steel", 60), ("Sturdy wheel", 140)])]
    observations = [boxes for _, boxes in scenes]
    y1, y2, x1, _ = observations[narrow_side][0]
    observations[narrow_side] = [(y1, y2, x1, 160)]
    path = tmp_path / "narrow-ocr.mp4"
    write_scenes(path, [frame for frame, _ in scenes])
    pipe, _ = detecting_pipeline(observations)

    timeline, stats = pipe._detect_timeline(path, REGION)

    assert stats["scene_change_frames"] == [1]
    assert timeline == [[], []]


def test_cut_only_confirms_the_matching_subtitle_line(tmp_path):
    scenes = [scene_frame(("Sturdy steel", second), background, number)
              for number, (second, background) in enumerate([
                  ("The tile", 60), ("The file", 140)])]
    path = tmp_path / "multiline.mp4"
    write_scenes(path, [frame for frame, _ in scenes])
    pipe, _ = detecting_pipeline([boxes for _, boxes in scenes])

    timeline, stats = pipe._detect_timeline(path, REGION)

    assert timeline == [[boxes[0]] for _, boxes in scenes]
    assert stats["detected"] == 4
    assert stats["accepted"] == stats["discarded"] == 2


@pytest.mark.parametrize("middle", ["empty-ocr", "changed-text"])
def test_consecutive_cuts_cannot_skip_an_intervening_observation(tmp_path, middle):
    texts = ["Sturdy steel", "Sturdy wheel" if middle == "changed-text" else "Sturdy steel",
             "Sturdy steel"]
    scenes = [scene_frame((text,), background, number)
              for number, (text, background) in enumerate(zip(texts, [60, 140, 80]))]
    observations = [boxes for _, boxes in scenes]
    if middle == "empty-ocr":
        observations[1] = []
    path = tmp_path / "intervening.mp4"
    write_scenes(path, [frame for frame, _ in scenes])
    pipe, _ = detecting_pipeline(observations)

    timeline, stats = pipe._detect_timeline(path, REGION)

    assert stats["scene_change_frames"] == [1, 2]
    assert timeline == [[], [], []]
    assert stats["accepted"] == 0


@pytest.mark.parametrize("middle", ["empty-ocr", "changed-text", "matching-text"])
def test_scene_cut_consumes_backfilled_observation_before_current_one(tmp_path, middle):
    earlier = "Sturdy wheel" if middle == "matching-text" else "Sturdy steel"
    intervening = "Sturdy wheel" if middle == "changed-text" else "Sturdy steel"
    texts = [earlier] * 7 + [intervening, "Sturdy steel", ""]
    scenes = [scene_frame((text,), 140 if number == 8 else 60, number)
              for number, text in enumerate(texts)]
    observations = [boxes for _, boxes in scenes]
    if middle == "empty-ocr":
        observations[7] = []
    path = tmp_path / "backfill.mp4"
    write_scenes(path, [frame for frame, _ in scenes])
    pipe, calls = detecting_pipeline(observations)

    timeline, stats = pipe._detect_timeline(path, REGION)

    assert stats["scene_change_frames"] == [8, 9]
    assert calls == [0, 1, 2, 4, 6, 8, 7, 9]
    assert stats["sampled_frames"] == [0, 1, 2, 4, 6, 8, 9]
    assert stats["refined"] == 1
    assert stats["ocr_calls"] == 8
    assert timeline[8] == (observations[8] if middle == "matching-text" else [])
    assert timeline[9] == []


@pytest.mark.parametrize("cut", [False, True])
def test_content_confirmation_is_only_available_across_scene_boundaries(tmp_path, cut):
    scenes = [scene_frame(("Sturdy steel",), 140 if cut and number else 60, number)
              for number in range(2)]
    observations = [boxes for _, boxes in scenes]
    y1, y2, x1, _ = observations[0][0]
    # This size change cannot form a normal geometry-only track.
    observations[0] = [(y1, y2, x1, 130)]
    path = tmp_path / "boundary-required.mp4"
    write_scenes(path, [frame for frame, _ in scenes])
    pipe, _ = detecting_pipeline(observations)

    timeline, stats = pipe._detect_timeline(path, REGION)

    assert stats["scene_change_frames"] == ([1] if cut else [])
    assert timeline == (observations if cut else [[], []])


def test_distant_single_detections_do_not_support_each_other(tmp_path):
    scenes = [scene_frame(("Sturdy steel",), 140 if number == 11 else 60, number)
              for number in range(12)]
    observations = [boxes if number in (0, 11) else []
                    for number, (_, boxes) in enumerate(scenes)]
    path = tmp_path / "distant.mp4"
    write_scenes(path, [frame for frame, _ in scenes])
    pipe, _ = detecting_pipeline(observations)

    timeline, stats = pipe._detect_timeline(path, REGION, ocr_stride=5)

    assert stats["scene_change_frames"] == [11]
    assert timeline == [[] for _ in scenes]


def test_single_frame_cut_masks_reach_model_without_cross_scene_or_outside_mask_changes(tmp_path):
    scenes = [scene_frame(background=value, number=number)
              for number, value in enumerate([60, 130, 190, 90, 90, 90])]
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_scenes(source, [frame for frame, _ in scenes])
    pipe, _ = detecting_pipeline([boxes for _, boxes in scenes])
    pipe.inpaint_mode = "propainter"
    windows, composed = [], set()

    def inpaint(frames, masks):
        numbers = [int(frame[0, 0, 0]) for frame in frames]
        windows.append(numbers)
        assert len({int(frame[30, 12, 0]) for frame in frames}) == 1
        for number, mask in zip(numbers, masks):
            for y1, y2, x1, x2 in scenes[number][1]:
                assert np.count_nonzero(mask[y1:y2, x1:x2]) > 100
        return [np.zeros_like(frame) for frame in frames]

    repair_segment = pipe._repair_propainter_segment

    def verify_composition(frames, masks, boxes, white_glyph_check):
        result, repairs = repair_segment(frames, masks, boxes, white_glyph_check)
        for original, mask, actual in zip(frames, masks, result):
            np.testing.assert_array_equal(actual[mask == 0], original[mask == 0])
            outside_roi = np.ones(mask.shape, dtype=bool)
            y1, y2, x1, x2 = REGION
            outside_roi[y1:y2, x1:x2] = False
            assert not np.any(mask[outside_roi])
            np.testing.assert_array_equal(actual[outside_roi], original[outside_roi])
            composed.add(int(original[0, 0, 0]))
        return result, repairs

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    pipe._repair_propainter_segment = verify_composition

    stats = pipe.process_video(source, output, region=REGION,
                               locate_stickers=False, white_glyph_check=False)

    assert windows == [[0, 0], [1, 1], [2, 2], [3, 4, 5]]
    assert composed == set(range(6))
    assert stats["frames"] == stats["inpainted"] == 6
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 6
    assert [round(float(frame.pts * frame.time_base) * 30) for frame in frames] == list(range(6))
