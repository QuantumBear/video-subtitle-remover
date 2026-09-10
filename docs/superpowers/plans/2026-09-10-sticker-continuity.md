# Sticker Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Repair missing emoji masks without lowering the threshold for new targets or erasing disappeared stickers.

**Architecture:** Preserve scored GroundingDINO candidates. Build trusted references from high-score detections, then check local appearance on every decoded frame with OpenCV. Reserve the existing priority samples before spending the remaining model budget on feedback-driven checks. Only GDINO uses the new confirmed timeline; VLM association stays unchanged.

**Tech Stack:** Python, NumPy, OpenCV, PyAV, pytest, existing offline GroundingDINO and ProPainter weights.

Work in the user-selected workspace. Preserve untracked weights. No commit, push, downloads, OCR changes, default residual-check changes, or model-window changes in this task.

## Task 1: Scored Candidates

Files: `backend/sticker_detect.py`, `tests/sticker_detect_test.py`.

- [x] Add failing tests for retained scores, weak candidates, non-finite detections, same-target duplicates, and adjacent targets.
- [x] Run `/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest tests/sticker_detect_test.py -q` and verify new assertions fail.
- [x] Add `StickerCandidate(box, score)` with full-frame yxyx coordinates, and `filter_candidates(detections, region, score_threshold, max_area_px)`. Its floor is `min(0.15, score_threshold)`. Apply existing absolute area filtering before coordinate conversion and class-independent IoU deduplication at 0.6, highest score first.
- [x] Add `GroundingDinoStickerDetector.detect_candidates(crop, region, prompt, score_threshold, max_area_px)`. Keep `detect_crop`, `filter_detections`, and `locate` compatible.
- [x] Re-run candidate tests and existing backend-forwarding tests.

```python
result = filter_candidates([
    {"score": 0.21, "box": [10, 20, 30, 40]},
    {"score": 0.30, "box": [11, 20, 31, 40]},
], (443, 1052, 0, 624), 0.22, 1000)
assert len(result) == 1
assert result[0].score == 0.30
assert result[0].box == (463, 483, 11, 31)
```

## Task 2: Appearance-Confirmed Tracking

Files: new `backend/sticker_tracking.py`, `tests/sticker_continuity_test.py`.

- [x] Add failing synthetic-frame tests for weak continuation, no-detection continuation, weak-only rejection, distant singleton rejection, duplicate/adjacent targets, appearance mismatch, disappearance/reappearance, cuts, invalid references, and reference expiry.
- [x] Verify failures with `/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest tests/sticker_continuity_test.py -q`.
- [x] Implement `StickerTracker`: add trusted high-score observations, keep original RGB reference patches, associate one-to-one using existing `sticker_match_score`, and check local original pixels per frame. Reference age is at most two seconds and references never cross scenes. Predictions do not create trusted evidence or refresh the reference.
- [x] Preserve distant singleton protection: only independent high-score frame observations qualify a stable track. A singleton requires subtitle proximity and the existing six-frame radius.
- [x] Use a bounded local OpenCV template comparison with color error and nonuniform-reference checks. Low-score candidates can propose a nearby location but cannot create a target or override an appearance mismatch. Missing/invalid evidence produces no new mask.
- [x] Re-run synthetic and existing association tests.

```python
tracker = StickerTracker(text_timeline, total_frames=90,
                         score_threshold=0.22, max_gap=60,
                         scene_change_frames=[29, 59])
tracker.add_sample(0, original_rgb, candidates)
tracker.observe(1, next_original_rgb, [])
timeline = tracker.finish()
```

## Task 3: Reserved Budget and Pipeline Integration

Files: `backend/sticker_tracking.py`, `vsr_pipeline.py`, `tests/sticker_continuity_test.py`, `tests/sticker_detect_test.py`.

- [x] Add failing tests for zero/small budgets, preserving future priority frames, feedback backoff, failed calls consuming budget, and GDINO confirmed masks reaching the repair engine without VLM interpolation.
- [x] Implement `locate_tracked_stickers`: first decode only to perform priority detections and store small trusted patches; second decode checks every relevant frame locally. Extra GDINO calls use the remaining budget only. Changed/lost targets trigger short intervals; stable checks double the interval up to the base sample step. Explicit statistics distinguish attempted calls, recovered observations, unknown checks, and budget limits.
- [x] Extend the wrapper with optional tracking context, preserving legacy callers that only ask for sampled boxes. The pipeline passes text timeline, total frame count, scene cuts, two-second age limit, and budget. VLM keeps `associate_sticker_hits`; GDINO does not re-interpolate confirmed masks.
- [x] Run all lightweight tests, including OCR, VLM, glyph, and mask-layer regressions. Do not enable template refinement or white-glyph repair by default.

## Task 4: Real Replay and Local Model Verification

Files: new `tests/fixtures/sticker_opening_candidates.json`, `tests/real_sticker_continuity_test.py`; scoped updates to the usage documentation and this checklist.

- [x] Store cached real opening candidates at frames 0,14,15,28,29,30,43,45,58,59,60,74,89,90,91,92,95,96. Include source video identity, ROI, cuts, score/area settings, and old missing-frame ranges.
- [x] Add an optional replay using `VSR_STICKER_REGRESSION_VIDEO`. Assert all 44 formerly missing frames receive a left sticker mask, adjacent targets remain distinct, and frames 90 onward do not extend the disappeared location. Reuse original video pixels and cached model outputs without OCR/model calls.
- [x] Run `/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest tests -q` and the real replay with the source-video environment variable set. Report skips accurately.
- [x] Run the actual CPU ProPainter on the first 29 frames at native-resolution 256 x 256 crop using final new masks, compare original-color residue against the already-recorded baseline, and verify pixels outside the effective mask are unchanged. Keep this separate from full CUDA validation.
- [x] Request independent spec review, then code-quality review; resolve actionable findings and re-run affected tests.
- [x] Document verified results, limitations, unchanged CLI settings, and residual checking remaining off by default.

## Baseline

2026-09-10: existing sticker detection, association, and tracking tests: **46 passed**. Production files untouched at baseline. Existing untracked GroundingDINO weights and approved design document preserved.

## Verification Results

- Final lightweight suite: **235 passed, 10 skipped**. The skips require opt-in real-video/model inputs.
- Original-pixel sticker replay: **2 passed, 1 deselected** (slow model comparison excluded).
- Offline CPU ProPainter comparison: **1 passed, 2 deselected**, 358.07 seconds. Six sampled left-emoji orange-residue counts changed from `[70, 101, 101, 72, 101, 101]` to six zeros; effective-mask-exterior pixels were unchanged. This is a 29-frame native-resolution crop, not full CUDA video validation.
- Offline real GroundingDINO first-frame candidate smoke: three candidates, scores `0.3537`, `0.2583`, `0.2491`.
- Independent specification and code-quality reviews passed after regression fixes for track bridging, invalid reference fallback, and zero-budget model loading.
- An additional opt-in template suite found one pre-existing failure in `tests/real_glyph_pipeline_test.py`: it expects template recovery without enabling `template_refine`. Running the original `HEAD` pipeline (`deb955e`) reproduced the same failure. No production default or unrelated test was changed.
- `git diff --check` passed. No commits, pushes, downloads, or localhost proxy use.
