from types import SimpleNamespace
from fractions import Fraction
import subprocess

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("cv2")

from vsr_pipeline import Pipeline


def make_video(path, values, rate=30):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "rgb24"
        stream.options = {"crf": "0", "bf": "0"}
        for i, value in enumerate(values):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), value, dtype=np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_stable_detection_gradually_skips_frames(tmp_path):
    path = tmp_path / "stable.mp4"
    make_video(path, [0] * 687)
    pipe = Pipeline.__new__(Pipeline)
    calls = []
    box = (20, 30, 5, 55)
    pipe.detect = lambda img, roi: calls.append(roi) or [box]
    timeline, stats = pipe._detect_timeline(str(path), (0, 48, 0, 64))
    assert timeline == [[box]] * 687
    assert 140 <= len(calls) <= 170
    assert stats["ocr_calls"] == len(calls)
    assert stats["sampled_frames"][:7] == [0, 1, 2, 4, 6, 10, 14]


def test_changed_detection_backfills_skipped_frames(tmp_path):
    path = tmp_path / "changing.mp4"
    make_video(path, list(range(60)))
    pipe = Pipeline.__new__(Pipeline)
    box = (20, 30, 5, 55)
    pipe.detect = lambda img, roi: [box] if 21 <= int(img[0, 0, 0]) <= 36 else []
    timeline, stats = pipe._detect_timeline(str(path), (0, 48, 0, 64))
    assert all(timeline[i] == [box] for i in range(21, 37))
    assert all(not timeline[i] for i in list(range(21)) + list(range(37, 60)))
    assert stats["refined"] > 0
    assert 25 in stats["sampled_frames"]  # change at 24 forces next frame


def test_scene_change_breaks_same_position_track(tmp_path):
    path = tmp_path / "cut.mp4"
    make_video(path, [0] * 20 + [200] * 20)
    pipe = Pipeline.__new__(Pipeline)
    pipe.detect = lambda img, roi: [(20, 30, 5, 55)]
    timeline, stats = pipe._detect_timeline(str(path), (0, 48, 0, 64))
    assert all(timeline)
    assert stats["tracks"] == 2
    assert 20 in stats["sampled_frames"] and 21 in stats["sampled_frames"]


def test_detect_clips_padding_to_manual_roi():
    pipe = Pipeline.__new__(Pipeline)
    pipe.ocr = SimpleNamespace(predict=lambda img: [{
        "dt_polys": [np.array([[0, 0], [40, 0], [40, 10], [0, 10]])]}])
    boxes = pipe.detect(np.zeros((80, 100, 3), dtype=np.uint8), (20, 30, 30, 70))
    assert boxes == [(20, 30, 30, 70)]


def test_vlm_caps_attempts_and_keeps_successful_empty_samples(tmp_path, monkeypatch, capsys):
    import requests
    import vsr_pipeline

    path = tmp_path / "vlm.mp4"
    make_video(path, [0] * 12)
    monkeypatch.setattr(vsr_pipeline, "_dashscope_key", lambda: "test-key")
    attempts = []

    def post(*args, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise requests.Timeout("secret-must-not-be-logged")
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "choices": [{"message": {"content": "[]"}}]})

    monkeypatch.setattr(requests, "post", post)
    hits = vsr_pipeline.locate_stickers_vlm(str(path), (0, 48, 0, 64),
                                          sample_frames=[0, 5, 10], max_calls=2)
    assert len(attempts) == 2
    assert hits == {5: []}
    log = capsys.readouterr().out
    assert "calls=2/2" in log and "Timeout" in log
    assert "secret-must-not-be-logged" not in log


def test_vlm_explicit_empty_schedule_makes_no_requests(tmp_path, monkeypatch):
    import requests
    import vsr_pipeline

    path = tmp_path / "empty-vlm.mp4"
    make_video(path, [0] * 12)
    monkeypatch.setattr(vsr_pipeline, "_dashscope_key", lambda: "test-key")
    monkeypatch.setattr(requests, "post", lambda *a, **kw: pytest.fail("unexpected request"))
    assert vsr_pipeline.locate_stickers_vlm(str(path), (0, 48, 0, 64), sample_frames=[]) == {}


@pytest.mark.parametrize("roi", [None, (10, 40, 5, 60)])
@pytest.mark.parametrize("mode", ["lama", "propainter"])
def test_process_video_keeps_frame_count_pts_and_roi(tmp_path, roi, mode):
    path, out = tmp_path / "input.mp4", tmp_path / "output.mp4"
    make_video(path, [0] * 83)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = mode
    regions, windows = [], []
    pipe.auto_region = lambda *a: pytest.fail("obsolete auto_region was called")
    pipe.detect = lambda img, region: regions.append(region) or [(20, 30, 10, 50)]

    def inpaint(frames, masks):
        if mode == "lama":
            return frames.copy()
        windows.append(len(frames))
        return [f.copy() for f in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stat = pipe.process_video(str(path), str(out), region=roi, locate_stickers=False)
    assert stat["frames"] == 83
    assert set(regions) == {roi or (0, 48, 0, 64)}
    if mode == "propainter":
        assert windows and max(windows) <= 60
    with av.open(str(out)) as result:
        frames = list(result.decode(video=0))
        assert len(frames) == 83
        assert [round(float(f.pts * f.time_base) * 30) for f in frames] == list(range(83))


def test_no_detections_preserves_fractional_rate_and_audio(tmp_path):
    from vsr_pipeline import FFMPEG

    raw, source, output = (tmp_path / name for name in ("silent.mp4", "audio.mp4", "out.mp4"))
    rate = Fraction(30000, 1001)
    make_video(raw, [0] * 31, rate=rate)
    subprocess.run([FFMPEG, "-v", "error", "-i", str(raw), "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=44100:duration=1.0344",
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    "-shortest", str(source)], check=True)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: []
    pipe._ensure_propainter = lambda: pytest.fail("no-text video loaded model")
    stats = pipe.process_video(source, output, locate_stickers=False)
    assert stats["frames"] == 31 and stats["inpainted"] == 0
    with av.open(str(output)) as result:
        assert len(result.streams.audio) == 1
        stream = result.streams.video[0]
        assert stream.average_rate == rate
        frames = list(result.decode(video=0))
        assert len(frames) == 31
        assert frames[-1].pts * frames[-1].time_base == Fraction(30, 1) / rate


def test_propainter_windows_do_not_cross_scene_changes(tmp_path):
    source, output = tmp_path / "scenes.mp4", tmp_path / "out.mp4"
    make_video(source, [0] * 23 + [120] * 20)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [(20, 30, 10, 50)]
    windows = []

    def inpaint(frames, masks):
        values = {int(f[0, 0, 0]) for f in frames}
        assert len(values) == 1, "window crosses scene boundary"
        windows.append(len(frames))
        return [f.copy() for f in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, locate_stickers=False)
    assert windows == [23, 20] and stats["frames"] == 43
