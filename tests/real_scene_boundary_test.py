"""Optional real tail replay; cached OCR and a fake GPU keep it offline."""

import hashlib
import json
import os
from bisect import bisect_right
from fractions import Fraction
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from glyph_pipeline_test import write_frames
from vsr_pipeline import Pipeline


pytestmark = pytest.mark.skipif(
    not os.environ.get("VSR_TEMPLATE_REGRESSION_VIDEO"),
    reason="set VSR_TEMPLATE_REGRESSION_VIDEO to the diagnostic source video",
)


def frame_key(image):
    return hashlib.sha256(image.tobytes()).digest()


@pytest.fixture(scope="module")
def tail_replay(tmp_path_factory):
    fixture_path = Path(__file__).parent / "fixtures" / "scene_boundary_tail_ocr.json"
    evidence = json.loads(fixture_path.read_text())
    start, end = evidence["start_frame"], evidence["end_frame"]
    with av.open(os.environ["VSR_TEMPLATE_REGRESSION_VIDEO"]) as container:
        frames = [frame.to_ndarray(format="rgb24")
                  for frame in islice(container.decode(video=0), start, end + 1)]
    assert len(frames) == end - start + 1
    assert frames[0].shape == (1280, 720, 3)
    source = tmp_path_factory.mktemp("scene-tail") / "source.mp4"
    write_frames(source, frames)
    return SimpleNamespace(
        source=source, frames=frames, start=start, region=tuple(evidence["region"]),
        cuts=evidence["scene_change_frames"],
        observations={int(n): [tuple(box) for box in boxes]
                      for n, boxes in evidence["observations"].items()},
        rgb_numbers={frame_key(frame): start + n for n, frame in enumerate(frames)},
        bgr_numbers={frame_key(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)): start + n
                     for n, frame in enumerate(frames)},
    )


def replay_pipeline(replay):
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    calls = []

    def detect(image, region):
        assert tuple(region) == replay.region
        number = replay.rgb_numbers[frame_key(image)]
        assert number in replay.observations, f"unexpected extra OCR call at {number}"
        calls.append(number)
        return list(replay.observations[number])

    pipe.detect = detect
    return pipe, calls


def test_real_tail_keeps_raw_ocr_hits_at_each_consecutive_cut(tail_replay):
    replay = tail_replay
    pipe, calls = replay_pipeline(replay)
    timeline, stats = pipe._detect_timeline(replay.source, replay.region)
    assert [n + replay.start for n in stats["scene_change_frames"]] == replay.cuts
    assert len(calls) == len(set(calls)) == stats["ocr_calls"] == 33
    assert stats["sampled"] == 26 and stats["refined"] == 7
    for number in (665, 666, 667):
        expected = replay.observations[number]
        assert len(expected) == 2
        assert timeline[number - replay.start] == expected, f"frame {number}: OCR hits lost"
        mask = pipe.propainter_boxes_to_mask(
            timeline[number - replay.start], replay.frames[number - replay.start], replay.region)
        assert np.count_nonzero(mask) > 10000, f"frame {number}: subtitle mask missing"
    assert stats["discarded"] == 0
    json.dumps(stats)


def test_real_tail_restored_masks_reach_engine_and_preserve_scene_isolation(tail_replay, tmp_path):
    replay = tail_replay
    pipe, calls = replay_pipeline(replay)
    windows, model_windows, model_pixels, audited = [], [], {}, set()

    def inpaint(images, masks):
        numbers = [replay.bgr_numbers[frame_key(image)] for image in images]
        assert len({bisect_right(replay.cuts, n) for n in numbers}) == 1
        assert len(numbers) >= 2, "single-frame scenes must repeat themselves for RAFT"
        model_windows.append(numbers)
        for number, mask in zip(numbers, masks):
            model_pixels[number] = int(np.count_nonzero(mask))
        return [np.zeros_like(image) for image in images]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    real_repair = pipe._repair_propainter_segment

    def audit_segment(images, masks, boxes, white_glyph_check=True):
        numbers = [replay.bgr_numbers[frame_key(image)] for image in images]
        windows.append(numbers)
        # Temporarily use the real method for its single-frame recursive call.
        pipe._repair_propainter_segment = real_repair
        try:
            result, repairs = real_repair(images, masks, boxes, white_glyph_check)
        finally:
            pipe._repair_propainter_segment = audit_segment
        for number, original, mask, current_boxes, fixed in zip(numbers, images, masks, boxes, result):
            np.testing.assert_array_equal(fixed[mask == 0], original[mask == 0])
            assert not np.any(mask[:replay.region[0]])
            assert not np.any(mask[replay.region[1]:])
            if number in (665, 666, 667):
                assert len(current_boxes) == 2
                assert np.count_nonzero(mask) > 10000
                assert np.all(fixed[mask > 0] == 0)
                audited.add(number)
        return result, repairs

    pipe._repair_propainter_segment = audit_segment
    output = tmp_path / "result.mp4"
    stats = pipe.process_video(replay.source, output, region=replay.region,
                               locate_stickers=False, white_glyph_check=False)
    assert audited == {665, 666, 667}, "restored subtitle masks did not reach repair"
    assert [(numbers[0], numbers[-1]) for numbers in windows] == [
        (650, 664), (665, 665), (666, 666), (667, 667), (668, 686)]
    for number in (665, 666, 667):
        assert [number, number] in model_windows
        assert model_pixels[number] > 10000
    assert stats["frames"] == stats["inpainted"] == 37
    assert len(calls) == len(set(calls)) == stats["ocr_calls"] == 33
    json.dumps(stats)
    with av.open(str(output)) as container:
        assert container.streams.video[0].average_rate == 30
        decoded = list(container.decode(video=0))
        assert len(decoded) == 37
        assert [frame.pts * frame.time_base for frame in decoded] == [
            Fraction(n, 30) for n in range(37)]
