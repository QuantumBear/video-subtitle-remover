# 字幕字形补全与残留复核实施计划

> **For agentic workers:** Use superpowers:subagent-driven-development for bounded module work and review. Do not commit or push without the user's request.

**Goal:** 修复 test_pro_16 中的暗描边、白衣服上的半句残留和 OCR 局部漏框，保留人物及背景。

**Architecture:** 保持反馈式 OCR 和 ProPainter 引擎不变。字形层扩展有效擦除边界；短期模板只在文字结构匹配时补全局部遮罩和框；复核比较原文字结构与修复结果，局部二轮后显式报告未解决帧。

**Tech Stack:** Python 3.12、NumPy、OpenCV、PyAV、pytest；GPU 完整画质在服务器验收。

## 约束与批准范围

用户已批准先试用上述方案。隔离工作树基于 06b2ede；不改默认检测范围、不换模型、不增加云端调用、不使用整行矩形作为白字兜底。保留 ROI、切景、60 帧窗口和二轮失败降级。

## Task 1：字形边界和残留结构

状态：实现、需求复审和质量复审通过；补了引擎内部合成负例，
确认背景横线不会因最终裁切触发二轮。已纳入全量 120 项回归。

**Files:** `vsr_pipeline.py`、`tests/mask_layers_test.py`、`tests/glyph_residual_test.py`。

- [x] 写失败测试：深色描边必须覆盖，字间背景不变；原字形位置的暗笔画必须触发复核，新出现的背景边缘不能触发。
- [x] 运行 `python -m pytest -q tests/glyph_residual_test.py` 确認旧实现失败。
- [x] 高置信字形和局部亮边取并集后扩展 2 像素，限制在文字框和 ROI；保留全局大白物体排除。
- [x] `_residual_mask` 在允许字形邻域比较原/结果的亮暗局部对比结构，不再只接受亮度大于 180 的像素；保持背景白块负例。
- [x] 重放第 60、686 帧，并跑原有遮罩测试。

```python
mask = pipe.propainter_boxes_to_mask(boxes, original, region)
assert mask[dark_stroke_y, dark_stroke_x] == 255
assert mask[gap_y, gap_x] == 0
residual = pipe._residual_mask(result_bgr, original_bgr, boxes)
assert np.count_nonzero(residual[text_slice]) >= 50
```

## Task 2：有界跨帧字形模板

状态：实现、最终需求与质量复审通过，27 项模板回归通过（包含真实素材）；
近似字母误匹配和重叠窗口提前淘汰参考均已修复。

**Files:** 创建 `backend/subtitle_templates.py`、`tests/subtitle_templates_test.py`。

- [x] 写测试覆盖部分字形缺失、局部 OCR 框缩水、平移、同位置换字、文字消失、reset、时间过期、ROI 和输入不变性。
- [x] 运行测试观察缺少模块/接口失败。
- [x] 实现 `SubtitleTemplates(region, max_age=60)`，提供 `reset()` 和 `refine(frames_bgr, masks, boxes, frame_numbers)`，返回新的 `(masks, boxes)`。
- [x] 只缓存有限的局部图像/字形，不缓存全片；通过 OpenCV 文字边缘及局部对比度核实内容、估计小幅平移；仅几何重叠不允许复用。
- [x] 匹配成功时只传播字形形状，修正局部缩短的框；无当前文字框/结构不匹配时不传播。切景由调用方 reset，时间差由帧号限制。
- [x] 遮罩始终限制 ROI，支持 40+20 窗口重叠的重复帧号。

```python
templates = SubtitleTemplates(region)
refined_masks, refined_boxes = templates.refine(frames, masks, boxes, frame_numbers)
templates.reset()
```

## Task 3：接入、验收与日志

状态：主集成需求与质量复审通过；最终复核异常不再阻断写出，另计
`residual_check_failed`。全量 120 项通过，整方案最终需求与质量审查通过。

**Files:** `vsr_pipeline.py`、`tests/adaptive_detection_test.py`、使用文档。

- [x] 写集成测试固定模板在二轮前生效、切景重置、未解决残留日志以及非遮罩区域不变。
- [x] 每次 process_video 创建独立模板实例；ProPainter 分段送入前补全字形。合成和二轮使用补全后的有效遮罩/文字框。
- [x] 二轮结束后重查残留，打印未解决帧数量和补全统计；不做无上限循环修复。
- [x] 生成真实 2 秒、4 秒、结尾的遮罩覆盖对照；真实素材路径通过环境变量指定，常规测试不依赖本机视频。
- [x] 全量 pytest、py_compile、git diff --check；独立需求审查后做代码质量审查。
- [x] 更新试跑命令及验证限制，保留 GPU 全流程待验收状态，不声称画质已修复。

## 实际验证结果

- 全量：120 passed；设置 `VSR_DIAGNOSTIC_DIR` 与 `VSR_TEMPLATE_REGRESSION_VIDEO`，包含四项真实采样回归。
- 遮罩对照：`/private/tmp/vsr-result-analysis.aSCqKp/glyph-fix-{0060,0120,0686}.png`。
- 第 60 帧：本次扩边后 4118 像素，使用第 30 帧参考补为 8517 像素。
- 第 120 帧：本次扩边后 15962 像素，使用第 180 帧参考补为 16738 像素，恢复圆点和 S。该回放验证可用参考下的能力，不保证默认窗口能访问第 180 帧。
- 第 686 帧：旧遮罩 10060 像素，本次扩边后 16000 像素；新复核在旧成片字幕带检测到 447 个疑似残留像素。
- 未运行 CUDA ProPainter 全片推理，GPU 画质、显存及耗时验收仍待服务器试跑。

## 验证命令

```bash
PYTHONPATH=/private/tmp/vsr-test-deps.Lsr98F /Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest -q
/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m py_compile vsr_pipeline.py backend/subtitle_templates.py
git diff --check
```

## 18 成片反馈后的修正

用户批准继续修改，以 18 的手动 ROI 配置为基线。原先的“切景重置模板”
混淆了两个生命周期：ProPainter 图像窗口必须断开，但同一句叠加字幕
可以跨镜头持续。短期字形参考仍须逐帧核实文字内容，不能复制修复图像。
本次不提交或推送；使用现有隔离工作树验证后交付修改。

### Task 4：切景保留有界字形参考

**Files:** `vsr_pipeline.py`、`backend/subtitle_templates.py`、
`tests/glyph_pipeline_test.py`。

- [x] 核对干净基线：116 passed，4 skipped（可选真实素材未启用）。
- [x] 用真实 `process_video` 路径生成两个不同背景的连续字幕场景，
  下一场景白底使部分字形被大白块过滤；断言缺失字形补入模型遮罩，
  两个场景的模型调用仍分开。仅用测试替身替代 OCR 推理和 GPU 引擎。
- [x] 参数化换字、相似字母变化、字幕消失场景，断言不复制旧字形。
- [x] 运行新测试，确认旧实现因第二场景缺少遮罩失败，再作最小修改：

```python
if n - 1 in scene_changes:
    flush_segment(len(seg_frames))
    # Keep bounded, content-verified glyph references across camera cuts.
```

- [x] 更新模板类的生命周期说明；保留 `reset()`、60 帧时效、64 条缓存、
  当前文字框与逐字形核验，不放宽内容匹配阈值。
- [x] 运行 `python -m pytest -q tests/glyph_pipeline_test.py tests/subtitle_templates_test.py`。
  红灯 1 failed、33 passed、2 skipped；绿灯 34 passed、2 skipped。
  需求与质量分别独立复审通过，复核相关五组测试为 67 passed、4 skipped。

### Task 5：真实时间序列和背景保护验收

**Files:** `tests/real_glyph_pipeline_test.py`；诊断输出留在忽略目录 `out/`。

- [x] 按原片解码帧生成无损前缀，使用手动 ROI `(450, 1010, 0, 720)`
  和本地 OCR 建立可重复离线输入；禁用网络与 VLM，不加载修复模型。
- [x] 经真实检测、分段、模板及合成调用链重放，比较第 60、75、135 帧
  有效遮罩对 18 成片残留结构的覆盖。不能用尚未进入窗口的未来参考。
- [x] 进一步定位 135 帧；只有独立失败样例证实的同范围问题才改代码。
- [x] 将真实场景回归纳入可选测试；检查 ROI 和有效遮罩外像素严格不变。
- [x] 运行全量测试、`py_compile`、`git diff --check`，进行需求和质量复审。
- [x] 记录像素覆盖证据和 CUDA 未验收限制，给出同 ROI 的服务器试跑命令。

前 205 帧使用本地真实 OCR 建立时间线（164 次调用），再用缓存时间线
和替代 GPU 引擎重放分段、模板与合成。实际 ROI 切景为 29、59、96、165；
135 帧属于 96-155 窗口，不能使用 180 帧的参考。两项修正后的固定字幕带核验：

| 帧 | 18 残留结构像素 | 旧有效遮罩覆盖 | 新有效遮罩覆盖 |
| --- | --- | --- | --- |
| 60 | 732 | 14 | 732 |
| 75 | 796 | 0 | 794 |
| 89 | 832 | 0 | 832 |
| 135 | 730 | 93 | 93 |

证据图已检查：`out/vsr17-diagnostics/replay18/balanced-evidence-*.png`。
这些是启发式结构像素，不是字符计数，也不是 CUDA 成片擦净证明。
新增可选测试使用原片前 90 帧、固定 OCR 框和替代 GPU 引擎；旧实现第 60 帧
新增缺字遮罩为零，回归失败；修正后 1 passed（57.72 秒），实际模型窗口为
0-28、29-58、59-89，逐帧验证有效遮罩外及 ROI 保护。

### Task 6：多行字幕缓存预算

**Files:** `backend/subtitle_templates.py`、`tests/subtitle_templates_test.py`。

真实 ROI 窗口 96-155 的 64 个参考全部集中在最长字幕行，短行参考在
注册阶段就被全局按像素面积排序淘汰。诊断暂将上限改为 512 后可看到
短行参考和少量恢复，但完整短词仍被严格局部对比度检查拒绝；本次不放宽该检查。

- [x] 新增多行长短不同、OCR 框微抖动的回归；短行仅早帧完整、其后缺字，
  验证长行重复观测不能让短行失去补全能力。先确认原缓存策略失败。
- [x] 保留总量 64；按已有 `_nearby` 几何关系分配缓存预算，从占用最多的
  位置组淘汰面积最小、时间较旧的参考。几何分组不授权复用，`_align` 原样执行。
- [x] 验证换字、重叠窗口、时效和上限；重放实际 ROI 时间线并审查新增遮罩。
- [x] 独立需求审查后再作代码质量审查，记录 4.5 秒仍未补全的限制。

新增回归红灯为 2 failed、36 passed、2 skipped；完整与重叠窗口均缺少短行
863 个字形像素。修正后绿灯为 38 passed、2 skipped。真实 135 帧仍受局部
对比度核验限制，本次不放宽匹配阈值、不以整行矩形兜底；首帧 emoji 细边未修改。

18 后续修正最终验证：启用全部真实样例后，129 passed（68.63 秒）；
`py_compile` 和 `git diff --check` 通过，独立需求及最终代码质量审查通过。
未运行 CUDA ProPainter 成片验收，未提交或推送。
