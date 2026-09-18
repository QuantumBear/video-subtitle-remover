# STTN 首层 Q/K/V 投影缓存 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 STTN 每个修复段内缓存首个 Transformer block 的 Q/K/V 投影，并验证缓存路径与原路径结果一致。

**Architecture:** Encoder 仍对整段帧只执行一次。新增模型级首层投影接口，`infer` 按窗口帧索引读取缓存；后续 Transformer blocks 和解码保持原逻辑。缓存仅是可选参数，缺省时保留原始计算。

**Tech Stack:** Python、PyTorch、pytest。

---

### Task 1: 首层缓存等价性测试

**Files:**
- Create: `tests/sttn_qkv_cache_test.py`

- [x] **Step 1: Write the failing test**

测试创建 `InpaintGenerator`，比较 `infer(features, masks)` 与 `infer(features, masks, qkv_cache=project_first_qkv(features))` 的输出。

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/sttn_qkv_cache_test.py`

系统 Python 缺少依赖；改用已有 Conda `vsr` 环境。将 HEAD 中的原始模型载入独立进程运行重叠窗口测试，确认因缺少 `project_first_qkv` 而失败。单窗口测试在加入短段判断前也确认失败。

### Task 2: 增加首层投影接口

**Files:**
- Modify: `backend/inpaint/sttn/network_sttn.py`

- [x] **Step 1: Expose the first attention projection**

Add a projection helper on `MultiHeadedAttention`, add a cached projection argument to its forward method, and expose `InpaintGenerator.project_first_qkv`.

- [x] **Step 2: Consume the cache only in the first Transformer block**

Pass the cache through the transformer dictionary for block zero and remove it before block one. Keep the no-cache branch unchanged.

- [x] **Step 3: Run the equivalence test**

Run: `python -m pytest -q tests/sttn_qkv_cache_test.py`

验证同 batch 严格相等、乱序窗口特征在 FP32 容差内相等，缓存不被修改。

### Task 3: Connect segment inference to the cache

**Files:**
- Modify: `backend/inpaint/sttn_det_inpaint.py`

- [x] **Step 1: Compute one cache per STTN segment**

After Encoder, call `project_first_qkv(feats)` once. For each window, index the cached tensors with `neighbor_ids + ref_ids` and pass them to `infer`. 增加默认开启的 `cache_first_qkv` 构造参数；单窗口跳过缓存；及时释放窗口切片。

- [x] **Step 2: Run STTN regression tests**

Run: `python -m pytest -q tests/sttn_qkv_cache_test.py tests/sttn_det_composite_test.py tests/sttn_pipeline_test.py`

结果：68 passed。新增测试还检查后续七层投影次数、连续两段缓存隔离、单窗口跳过缓存。

### Task 4: Static and syntax verification

**Files:**
- Modify: `backend/inpaint/sttn/network_sttn.py`
- Modify: `backend/inpaint/sttn_det_inpaint.py`

- [x] **Step 1: Compile modified modules**

Run: `python -m py_compile backend/inpaint/sttn/network_sttn.py backend/inpaint/sttn_det_inpaint.py`

- [x] **Step 2: Review the diff**

Run: `git diff --check` and inspect that no model files or unrelated changes are staged.

### Task 5: 可复现基准

**Files:**
- Create: `benchmarks/sttn_qkv_cache.py`

- [x] 使用真实权重、同一组合成输入、开关缓存两条路径；计时包含缓存构建和切片。
- [x] CUDA 预热、同步计时、交替测量顺序，输出耗时中位数、加速比、像素差和峰值 allocated。
- [x] 运行真实权重 CPU 冒烟验证；5090 性能数据需要在目标 GPU 上测量。

GPU 命令：`python benchmarks/sttn_qkv_cache.py --device cuda --frames 50 --warmup 1 --repeats 3`。

CPU 验证命令：`python benchmarks/sttn_qkv_cache.py --device cpu --frames 6 --warmup 0 --repeats 1 --threads 1`。缓存实际启用，最大和平均像素差均为 0；未缓存 39.019 秒，缓存 39.722 秒。该次无预热的 CPU 短段检查未观察到加速，不能用来推断 5090 性能。
