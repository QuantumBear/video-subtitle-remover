import json
from types import SimpleNamespace

import numpy as np
import pytest

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")

from vsr_pipeline import Pipeline


def write_frames(path, frames):
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


def text_frame(text="Sturdy steel", *, light_background=False, thin_shadow=False):
    frame = np.full((128, 360, 3), 80, dtype=np.uint8)
    if light_background:
        frame[:, 120:310] = 242
    scale = 0.9 if thin_shadow else 1.2
    origin = (26, 84) if thin_shadow else (25, 83)
    color = (65, 65, 65) if thin_shadow else (15, 15, 15)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2 if thin_shadow else 6, cv2.LINE_AA)
    cv2.putText(frame, text, (25, 83), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return frame, (48, 94, 18, 260)


def test_completed_glyphs_reach_model_and_composition_even_from_empty_masks(tmp_path):
    image, box = text_frame()
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 4)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    complete = pipe.propainter_boxes_to_mask([box], image, (0, 128, 0, 360))
    provided = iter([complete] + [np.zeros_like(complete)] * 3)
    pipe.propainter_boxes_to_mask = lambda *args, **kwargs: next(provided).copy()
    model_masks = []

    def inpaint(frames, masks):
        model_masks.extend(m.copy() for m in masks)
        return [np.full_like(frame, 80) for frame in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, locate_stickers=False,
                               template_refine=True, white_glyph_check=False)
    json.dumps(stats)
    assert len(model_masks) == 4
    assert all(np.array_equal(mask, complete) for mask in model_masks)
    assert stats["template_recovered"] == 3
    with av.open(str(output)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    assert len(frames) == 4
    assert all(np.abs(f[complete > 0].astype(float) - 80).mean() < 3 for f in frames)


def test_template_refine_disabled_by_default_for_diagnosis(tmp_path):
    """诊断开关:模板补全默认关闭,原始 OCR mask 原样进模型。

    服务器验证 0-3 秒残留根因时需默认禁用 templates.refine 与白字自检;
    本测试锁定默认禁用状态,后续若恢复默认开启需同步更新。
    """
    image, box = text_frame()
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 4)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    complete = pipe.propainter_boxes_to_mask([box], image, (0, 128, 0, 360))
    provided = iter([complete] + [np.zeros_like(complete)] * 3)
    pipe.propainter_boxes_to_mask = lambda *args, **kwargs: next(provided).copy()
    model_masks = []

    def inpaint(frames, masks):
        model_masks.extend(m.copy() for m in masks)
        return [np.full_like(frame, 80) for frame in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, locate_stickers=False,
                               white_glyph_check=False)
    assert stats["template_recovered"] == 0
    assert np.array_equal(model_masks[0], complete)
    assert all(not mask.any() for mask in model_masks[1:])


def test_unresolved_after_bounded_retry_is_reported(tmp_path, capsys):
    image, box = text_frame()
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 4)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    calls = []

    def inpaint(frames, masks):
        calls.append(len(frames))
        return [frame.copy() for frame in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, locate_stickers=False,
                               template_refine=True, white_glyph_check=True)
    assert len(calls) == 2
    assert stats["unresolved"] == 4
    assert "疑似残留 4" in capsys.readouterr().out


def test_unrecoverable_empty_masks_do_not_load_model(tmp_path):
    image, box = text_frame()
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 4)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    pipe.propainter_boxes_to_mask = lambda *args, **kwargs: np.zeros(image.shape[:2], np.uint8)
    pipe._ensure_propainter = lambda: pytest.fail("empty masks loaded model")
    stats = pipe.process_video(source, output, locate_stickers=False,
                               white_glyph_check=True)
    assert stats["frames"] == 4 and stats["inpainted"] == 0
    assert stats["unresolved"] == 4


def test_final_audit_failure_keeps_first_pass_and_reports_unchecked_frames(tmp_path, capsys):
    image, box = text_frame()
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [image] * 4)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    pipe.inpainter = SimpleNamespace(inpaint=lambda frames, masks: [f.copy() for f in frames])

    def fail_check(*args):
        raise cv2.error("test check failure")

    pipe._residual_mask = fail_check
    stats = pipe.process_video(source, output, locate_stickers=False,
                               white_glyph_check=True)
    assert stats["residual_check_failed"] == 4
    assert "复核未完成 4" in capsys.readouterr().out
    with av.open(str(output)) as container:
        assert len(list(container.decode(video=0))) == 4


def test_matching_subtitle_survives_scene_cut_without_mixing_model_frames(tmp_path):
    clear, box = text_frame(thin_shadow=True)
    bright, _ = text_frame(light_background=True, thin_shadow=True)
    region = (0, 128, 0, 360)
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [clear] * 3 + [bright] * 3)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    complete = pipe.propainter_boxes_to_mask([box], clear, region)
    damaged = pipe.propainter_boxes_to_mask([box], bright, region)
    raw = pipe.white_glyph(bright, region)
    filtered = pipe.filter_glyph_by_height(raw)
    hidden_white = (complete > 0) & (raw > 0) & (filtered == 0)
    assert np.count_nonzero(hidden_white[:, 120:260]) > 200
    assert not np.any(damaged[hidden_white])
    _, detection = pipe._detect_timeline(source, region)
    assert detection["scene_change_frames"] == [3]
    calls = []

    def inpaint(frames, masks):
        calls.append(([frame.copy() for frame in frames], [mask.copy() for mask in masks]))
        return [np.full_like(frame, 80) for frame in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, region=region,
                               locate_stickers=False, template_refine=True,
                               white_glyph_check=False)
    assert [len(frames) for frames, _ in calls] == [3, 3]
    for (frames, _), expected in zip(calls, [clear, bright]):
        assert all(np.array_equal(frame, expected) for frame in frames)
    for mask in calls[1][1]:
        assert np.all(mask[hidden_white] > 0)
        np.testing.assert_array_equal(mask, np.maximum(complete, damaged))
    assert stats["frames"] == 6 and stats["template_recovered"] == 3


@pytest.mark.parametrize("source_text,target_text", [("Sturdy steel", "Sturdy wheel"),
                                                     ("Sturdy tile", "Sturdy file"),
                                                     ("Sturdy file", "Sturdy tile"),
                                                     ("Sturdy steel", "")])
def test_scene_cut_cannot_copy_changed_or_disappeared_subtitles(tmp_path, source_text, target_text):
    clear, box = text_frame(source_text, thin_shadow=True)
    changed, _ = text_frame(target_text, light_background=True, thin_shadow=True)
    region = (0, 128, 0, 360)
    source, output = tmp_path / "source.mp4", tmp_path / "result.mp4"
    write_frames(source, [clear] * 3 + [changed] * 3)
    pipe = Pipeline.__new__(Pipeline)
    pipe.inpaint_mode = "propainter"
    pipe.detect = lambda *args: [box]
    damaged = pipe.propainter_boxes_to_mask([box], changed, region)
    _, detection = pipe._detect_timeline(source, region)
    assert detection["scene_change_frames"] == [3]
    calls = []

    def inpaint(frames, masks):
        calls.append([mask.copy() for mask in masks])
        return [np.full_like(frame, 80) for frame in frames]

    pipe.inpainter = SimpleNamespace(inpaint=inpaint)
    stats = pipe.process_video(source, output, region=region,
                               locate_stickers=False, template_refine=True,
                               white_glyph_check=False)
    assert stats["frames"] == 6 and stats["template_recovered"] == 0
    if target_text:
        assert [len(masks) for masks in calls] == [3, 3]
        for mask in calls[1]:
            np.testing.assert_array_equal(mask, damaged)
    else:
        assert not damaged.any()
        assert [len(masks) for masks in calls] == [3]
        assert stats["inpainted"] == 3
