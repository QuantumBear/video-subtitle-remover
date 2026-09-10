import sys
from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("cv2")

import vsr_pipeline
from glyph_pipeline_test import write_frames
from mask_layers_test import white_text_frame
from vsr_pipeline import Pipeline


@pytest.mark.parametrize("options,edge_value", [
    ({}, 255), ({"subtitle_strength": "light"}, 255),
    ({"subtitle_strength": "conservative"}, 0),
])
def test_strength_reaches_model_and_composition(tmp_path, options, edge_value):
    image = white_text_frame()
    box, region = (24, 48, 15, 135), (0, 80, 0, 240)
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 3)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    calls = []

    def inpaint(frames, masks):
        calls.append([mask.copy() for mask in masks])
        return [np.full_like(frame, 90) for frame in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, region=region, locate_stickers=False, **options)
    assert len(calls) == 1
    assert all(mask[35, 16] == edge_value for mask in calls[0])
    assert stats["template_recovered"] == stats["repaired"] == 0
    assert stats["residual_check_enabled"] is False
    with av.open(str(output)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(frames) == 3
    expected = 90 if edge_value else 0
    assert all(abs(float(frame[35, 16].mean()) - expected) < 10 for frame in frames)
    assert all(float(frame[35, 75].mean()) < 10 for frame in frames)


@pytest.mark.parametrize("args,expected", [
    ([], "light"), (["--subtitle-strength", "light"], "light"),
    (["--subtitle-strength", "conservative"], "conservative"),
])
def test_cli_passes_subtitle_strength(monkeypatch, args, expected):
    calls = []

    def process(*paths, **options):
        calls.append(options)
        return {}

    monkeypatch.setattr(vsr_pipeline, "Pipeline", lambda **kwargs: SimpleNamespace(process_video=process))
    monkeypatch.setattr(sys, "argv", ["vsr_pipeline.py", "-i", "source.mp4", "-o", "result.mp4",
                                      "--inpaint-mode", "propainter", *args])
    vsr_pipeline.main()
    assert calls[0]["subtitle_strength"] == expected


def test_invalid_strength_is_rejected_before_opening_video(tmp_path):
    pipe = Pipeline.__new__(Pipeline)
    with pytest.raises(ValueError, match="subtitle_strength"):
        pipe.process_video(tmp_path / "missing.mp4", tmp_path / "result.mp4", subtitle_strength="strong")
