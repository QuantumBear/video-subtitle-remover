# STTN Model and Composite Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the wide STTN inference mask from the narrower final write-back mask so STTN keeps context while fewer non-subtitle pixels are replaced.

**Architecture:** `Pipeline` will build the existing per-frame rectangle masks for STTN inference and a second per-frame mask using `propainter_boxes_to_mask` with rectangle fallback. `STTNDetInpaint` will accept both lists, crop both through the horizontal ROI, use the first for model inference, and use the second for final `copyto`; the second mask will always be intersected with the first.

**Tech Stack:** Python, NumPy, OpenCV, PyTorch/STTN, pytest, PyAV.

---

### Task 1: Add failing contract tests for separate masks

**Files:**
- Modify: `tests/sttn_det_composite_test.py`
- Modify: `tests/sttn_pipeline_test.py`

- [ ] **Step 1: Add a fake-engine test that distinguishes inference and composite masks.**

Add a test using a wide `model_mask`, a narrower `composite_mask`, and the existing fake `inpaint` result. Assert the fake model receives the wide mask, while only the narrow mask contains `COMP_VALUE` in the returned frame.

- [ ] **Step 2: Add validation and context tests.**

Cover mismatched composite-mask length/shape with `ValueError`, ensure composite pixels outside the model mask are ignored, and ensure an all-black context frame remains unchanged even when a model mask is present in another frame.

- [ ] **Step 3: Add a Pipeline forwarding test.**

Replace the STTN engine with a recorder that accepts `composite_masks`, inject a white text-like frame and OCR box, and assert the recorder receives both the rectangle model masks and a composite mask with fewer pixels for the text frame and zero pixels for clean context frames.

- [ ] **Step 4: Run the focused tests and verify they fail before implementation.**

Run:

```bash
/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest -q tests/sttn_det_composite_test.py tests/sttn_pipeline_test.py
```

Expected: failures reporting the unsupported composite-mask argument or missing forwarding behavior.

### Task 2: Extend `STTNDetInpaint` with a separate composite mask

**Files:**
- Modify: `backend/inpaint/sttn_det_inpaint.py`
- Test: `tests/sttn_det_composite_test.py`

- [ ] **Step 1: Extend the call signature compatibly.**

Change the signature to accept `composite_mask=None` after `x_bounds`. Keep the current `input_mask` behavior when it is an ndarray. If `composite_mask is None`, reuse the normalized model masks so old callers produce identical output.

- [ ] **Step 2: Normalize and validate both mask lists.**

Normalize every mask to `uint8` values of `0/255`, require one mask per input frame and the same `(H, W)` shape, and compute `composite_masks[j] = composite_masks[j] & model_masks[j]`. Store each as `H x W x 1` before ROI handling.

- [ ] **Step 3: Pass both masks through horizontal ROI recursion.**

When `x_bounds` is narrower than the frame, crop frames, model masks, and composite masks using the same `x0:x1` slice, call the recursive full-width implementation with both mask lists, then paste the returned frames back into the original ROI.

- [ ] **Step 4: Use only the composite mask for final `copyto`.**

Keep `union_mask` and `get_inpaint_area_by_mask` based on model masks. During result write-back, replace pixels using the per-frame composite mask slice rather than the model mask slice. Preserve unchanged behavior when no composite mask is supplied.

- [ ] **Step 5: Run focused engine tests.**

Run the two test files from Task 1 and expect all mask-contract tests to pass.

### Task 3: Build conservative narrow masks in the STTN Pipeline

**Files:**
- Modify: `vsr_pipeline.py`
- Test: `tests/sttn_pipeline_test.py`

- [ ] **Step 1: Create per-frame composite masks alongside model masks.**

In `flush_sttn`, keep the current `masks` list as model masks. Build `composite_masks` with all-black masks for `seg_core == False`; for core frames call `propainter_boxes_to_mask` with the frame converted from BGR to RGB, the frame’s boxes, the configured subtitle `region`, and the current subtitle strength. If that result is empty for a frame with boxes, fall back to the frame’s rectangle mask.

- [ ] **Step 2: Forward both masks to the inpainter.**

Call:

```python
comps = self.inpainter(
    seg_frames,
    masks,
    x_bounds=(x_min, x_max),
    composite_mask=composite_masks,
)
```

Update STTN test doubles and compatibility lambdas to accept the optional keyword.

- [ ] **Step 3: Preserve final ROI guarding and residual probing.**

Leave `roi_mask` around the returned composition and continue residual probing against the original frame and original OCR boxes. The new composite mask only controls pixels that can be written by the STTN engine.

- [ ] **Step 4: Run Pipeline-focused tests.**

Run:

```bash
/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest -q tests/sttn_pipeline_test.py
```

Expected: all Pipeline tests pass, including ROI, context, per-frame, and forwarding tests.

### Task 4: Full verification and commit

**Files:**
- Modify: `backend/inpaint/sttn_det_inpaint.py`
- Modify: `vsr_pipeline.py`
- Modify: `tests/sttn_det_composite_test.py`
- Modify: `tests/sttn_pipeline_test.py`

- [ ] **Step 1: Run syntax and whitespace checks.**

```bash
/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m py_compile vsr_pipeline.py backend/inpaint/sttn_det_inpaint.py
git diff --check
```

- [ ] **Step 2: Run the complete test suite.**

```bash
/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest -q
```

Expected: all existing tests pass with the repository’s current skip count.

- [ ] **Step 3: Review the diff and commit.**

```bash
git diff --stat
git status --short
git add backend/inpaint/sttn_det_inpaint.py vsr_pipeline.py tests/sttn_det_composite_test.py tests/sttn_pipeline_test.py
git commit -m "feat: separate STTN model and composite masks"
```

- [ ] **Step 4: Push after user requests it.**

```bash
git push origin main
```
