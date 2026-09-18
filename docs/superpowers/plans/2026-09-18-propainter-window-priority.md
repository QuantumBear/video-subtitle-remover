# ProPainter 残留窗口优先调度 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在设置 `sttn_residual_propainter_max_windows` 时，从全片候选中选择残留核心帧最多、残留像素最多的 Top-K 窗口，再按时间顺序交给 ProPainter。

**Architecture:** STTN 第一遍继续按当前分段推理并写入临时视频；正数窗口预算下只收集候选元数据，不立即调用 ProPainter。第一遍结束后按严重度选出 Top-K，第二遍顺序读取临时视频，对选中窗口调用 ProPainter 并写出替换后的临时视频。不限预算 `max_windows=0` 保持原有流式调用。

**Tech Stack:** Python、PyAV、NumPy、PyTorch、pytest。

---

### Task 1: 固化跨段 Top-K 选择行为

**Files:**
- Modify: `tests/sttn_pipeline_test.py`
- Test: `tests/sttn_pipeline_test.py::test_sttn_residual_max_windows_prioritizes_most_severe_candidate`

- [x] **Step 1: Write the failing test**

  已增加跨 STTN 段测试：前段只有 1 个残留帧，后段有 10 个残留帧，预算为 1 时预期只调用后段 `50-64` 窗口。

- [x] **Step 2: Run test to verify it fails**

  运行 `...pytest -q tests/sttn_pipeline_test.py::test_sttn_residual_max_windows_prioritizes_most_severe_candidate -vv`，当前失败并显示调用了 `0-8`。

### Task 2: 收集并排序有预算的候选

**Files:**
- Modify: `vsr_pipeline.py:STTN process_video branch`
- Modify: `backend/subtitle_tracking.py` only if a pure ranking helper is needed

- [ ] **Step 1: Record candidate metadata**

  为每个通过最小核心帧过滤的候选保存原候选时间范围、截短后时间范围、逐帧框、残留核心帧集合、核心帧数和残留像素总量。正数预算时跳过即时 ProPainter 调用。

- [ ] **Step 2: Select global Top-K**

  第一遍结束后按 `(-core_frames, -residual_pixels, start_frame)` 排序，取 `max_windows` 个候选，再恢复时间顺序用于输出。

### Task 3: 对选中窗口做第二遍 ProPainter 合成

**Files:**
- Modify: `vsr_pipeline.py:STTN branch after first-pass close`
- Test: `tests/sttn_pipeline_test.py`

- [ ] **Step 1: Read the first-pass temporary video**

  对选中窗口按时间顺序收集 BGR 帧；上下文帧 mask 为零，残留核心帧使用保存的框生成 mask。

- [ ] **Step 2: Run ProPainter and write a replacement temporary video**

  仅将选中窗口核心帧的修复结果写回，其他帧保留第一遍输出；替换原临时视频后继续现有音频合并流程。

- [ ] **Step 3: Preserve statistics and logs**

  只为实际选中的窗口累计调用次数、输入帧数、核心帧数、截短和耗时统计；日志增加候选严重度排序结果，便于验证调度。

### Task 4: 文档与回归验证

**Files:**
- Modify: `docs/02-use/05-STTN残留转交预算.md`
- Modify: `docs/superpowers/specs/2026-09-18-propainter-window-priority-design.md`

- [ ] **Step 1: Document Top-K ordering**

  明确正数 `max_windows` 按核心残留帧数、残留像素总量、时间顺序选择；`0` 保持流式不限预算。

- [ ] **Step 2: Run focused and full verification**

  运行 STTN 流水线测试、相关追踪测试、`git diff --check`；确认模型目录不被暂存。

- [ ] **Step 3: Commit**

  使用提交信息 `feat: prioritize severe propainter windows`。

## Self-review

- 规格中的 Top-K 目标对应 Task 2，时间顺序输出对应 Task 3。
- `max_windows=0` 的兼容路径对应 Task 2 和 Task 4。
- 测试先于生产代码并已观察到预期失败。
