# Temporal Glyph Consensus Implementation Plan

> **For agentic workers:** Execute this approved continuation with TDD and verify each checkpoint. Work in the user-selected workspace; do not commit or push.

**Goal:** Recover missing white subtitle strokes using multiple original frames, without filling whole text rectangles or copying stale captions.

**Architecture:** A bounded, window-local `TemporalGlyphs` module receives original BGR frames, original masks, text boxes, frame numbers and sticker exclusions. It groups nearby lines, constructs persistent stroke candidates from several distinct frames, and checks current-frame stroke evidence before adding pixels. ProPainter receives the refined mask and composition uses that identical mask. Existing template recovery and residual repair remain separate and off by default.

**Tech Stack:** NumPy, OpenCV, PyAV, pytest; existing local CPU ProPainter weights only.

## Approved Scope And Constraints

The preceding A/B/C/D experiment found that complete temporal glyph coverage removed the readable suffixes in frames 130-135 while keeping more structure than filled rows. The new code must derive candidates from OCR-localized original observations, not hard-coded coordinates or manual caption identity. Initial support is stationary white captions within a contiguous same-scene window. Missing evidence and ambiguous bright objects fall back to the current mask. Animated captions are not fully supported: only stable, currently verified stroke subsets can be added. No full-video local inference, network requests, new models, scene-window enlargement or changes to emoji tracking.

Expose `--temporal-glyphs` as an opt-in trial. Reject simultaneous `--template-refine` to keep experimental attribution clear. Retain current default strength, ROI and strict composition. Report recovered frame and added pixel counts separately from residual quality. These counts cannot assert that a video is clean.

## Task 1: Failing Behavioral Tests

- [x] Baseline: `/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest tests -q`: 247 passed, 10 skipped.
- [x] Create `tests/temporal_glyphs_test.py`. Synthesize fixed white captions over complementary moving bright backgrounds with shortened OCR boxes. Verify suffix/bullet coverage, preservation of interword gaps, no input mutation, ROI, no-box frames, disappeared and changed glyphs, large white objects, movement, sticker exclusions and bounded distinct observations.
- [x] Run the new tests before implementing the module and record the expected failure. Initial missing-module checks produced 11 setup failures; substantive red/green coverage followed in pipeline tests and the real `slip` core regression (39 uncovered pixels).

```python
refiner = TemporalGlyphs(region)
refined, boxes = refiner.refine(frames, masks, text_boxes, frame_numbers)
assert np.all(refined[target][missing_core] > 0)
assert not np.any(refined[target][protected_background])
```

## Task 2: Window-Local Consensus

- [x] Create `backend/temporal_glyphs.py` with `TemporalGlyphs(region, outline_radius=3)` and `refine(frames_bgr, masks, boxes, frame_numbers, excluded_boxes=None)`.
- [x] Validate aligned inputs; return independent ROI-clipped masks and boxes. Cap input at the current 60-frame window and keep no full-video cache.
- [x] Group geometrically compatible OCR lines and split runs at frame gaps or missing line observations. Distinct frame numbers, not duplicate boxes, determine evidence count.
- [x] Within bounded line neighborhoods, combine original channel-minimum images to reject transient bright background. Filter connected components using stroke shape, OCR-mask support and local contrast; never use a filled rectangle as a glyph template.
- [x] Confirm candidate strokes in each target original frame before adding them, including non-sampled targets. Exclude sticker regions from evidence; never promote inferred pixels to source evidence.
- [x] Re-run behavioral tests and inspect failure cases before changing thresholds. The background ring caused rejection of unchanged thin strokes; compare stroke-core structure and polarity instead. Eighteen synthetic tests, eight initial integration cases and three original-pixel regressions pass together.

## Task 3: Pipeline Integration

- [x] Add `tests/temporal_glyph_pipeline_test.py` using the real decode/window/compose path, substituting only OCR/model inference. Verify refined masks reach both the model and composition, sticker masks are preserved, windows do not cross scene cuts, option defaults and CLI forwarding.
- [x] In `vsr_pipeline.py`, add `temporal_glyphs=False` to the end of `process_video`, instantiate the refiner only when enabled, retain sticker boxes alongside window metadata and pass exclusions to the refiner. Never modify the buffered source masks in place.
- [x] Add `--temporal-glyphs`; reject simultaneous legacy template recovery before opening the video. Keep LAMA behavior unchanged.
- [x] Add explicit statistics for enabled state, recovered frames and added pixels, counted only for frames actually emitted from an overlapping window.
- [x] Run new integration tests and the full lightweight suite. Final lightweight suite: 282 passed, 14 skipped; includes ten temporal integration cases.

## Task 4: Real-Pixel Verification And Handoff

- [x] Create optional `tests/real_temporal_glyphs_test.py` keyed by `VSR_TEMPORAL_GLYPH_VIDEO`. Use cached/local OCR observations from the actual 96-155 same-scene window; no future frames outside that window and no manual full-row boxes.
- [x] Verify readable suffix and bullet cores at frame 135 are covered while known background gaps and exclusions remain unchanged. Store the OCR fixture provenance and source dimensions. The 193 slip, 74 Sturdy-suffix and 41 leading-bullet core pixels are all covered.
- [x] Run the real refiner on all original window frames, then run native 320x256 CPU ProPainter only on frames 130-135 with the resulting masks. Compare with the saved A baseline and manually verified D result. Inspect all key-frame images; do not substitute mask area or test counts for quality. Final CPU run: 39.12s; readable suffix remnants removed, bullets remain at frames 130/132 and small trouser seams remain. Original pixels outside the effective mask are byte-identical in all six results.
- [x] Update the usage guide with the trial flag, same-scene/stationary-white-caption limits and server full-video verification command.
- [x] Run full tests, optional real-frame tests, syntax compilation and `git diff --check`; record remaining limitations. Final results: 282 passed, 14 skipped (22.23s); optional original-pixel suite: 4 passed (23.67s); syntax compilation, CLI help and diff whitespace checks pass. Changes remain uncommitted.

## Review And Evidence

- Independent review reproduced two new white-object overmasking cases. Tightened unsupported candidates to compact leading bullets, and restricted loose edges to eligible connected components. Each fix followed a failing regression. Final independent recheck found no outstanding issues.
- Slow-movement probes (0-1px per frame and 1px oscillation) can recover stable stroke subsets, but added pixels stayed within current glyphs and 3px outlines. Three tests now record this narrower claim; complete animated-caption recovery is not claimed.
- Persistent artifacts: `out/temporal-glyph-trial.N9P460/`; complete local scripts and A/B/C/D/T PNGs: `/private/tmp/vsr-mask-ab.SwJH7A/`.
- No full-video model run, network request, model download, commit or push. Trial default remains off; emoji and legacy recovery behavior are unchanged.
