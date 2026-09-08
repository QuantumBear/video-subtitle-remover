# 连续切景字幕命中保留实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for implementation and two-stage review. Steps use checkbox syntax. 本轮不提交或推送。

**Goal:** 恢复真实片尾 665-667 帧被单帧轨迹过滤的两行字幕命中。

**Architecture:** 检测阶段按时间顺序处理已有采样和回查观测，仅在相邻观测跨切景且间隔不超过 OCR 步长时复用现有字形核验。确认依据精确到帧号和当前 OCR 框，只影响低于 min_hits 的轨迹是否保留，不连接跨景轨迹。长期字形缓存与修复阶段不变。

**Tech Stack:** Python、OpenCV、NumPy、PyAV、pytest；OCR 及 GPU 推理以替身或已有离线结果隔离。

---

## 文件职责

- `backend/subtitle_templates.py`：从原始观测构建可靠局部字形参考，提供双向观测匹配；与 `_remember` 共享参考构建及 `_align`，不修改阈值和缓存规则。
- `backend/subtitle_tracking.py`：可选接收 `(frame_number, box)` 确认依据，默认旧行为。
- `vsr_pipeline.py`：有界保留上一条观测，顺序消费回查，再消费当前采样；向轨迹层传递确认依据。
- `tests/subtitle_templates_test.py`、`tests/subtitle_tracking_test.py`、`tests/adaptive_detection_test.py`、`tests/glyph_pipeline_test.py`：合成回归及真实调用链集成。
- `tests/scene_boundary_detection_test.py`：连续切景、逐行确认、回查顺序及单帧修复集成的聚焦回归。
- `tests/real_scene_boundary_test.py`、`tests/fixtures/scene_boundary_tail_ocr.json`：可选真实片尾回放；只纳入已缓存 OCR 框，不提交视频。
- `docs/02-use/04-实战去字幕流程-本机验证版.md`：记录原因、验证边界和同 ROI 的 20 试跑命令。

## Task 1：字形确认、轨迹保留及检测接入

这是紧密关联的一项实现，由同一个实现者负责生产代码与合成测试。

- [x] 创建隔离工作树，运行基线：124 passed、5 skipped。
- [x] 先写连续单帧切景的失败测试。以不同背景上的同一句两行文字生成无损短片，真实 `_detect_timeline` 应返回每帧两行，不得修改切景列表。

```python
timeline, stats = pipe._detect_timeline(source, region)
assert stats["scene_change_frames"] == [1, 2, 3]
assert timeline == [boxes] * 4
assert stats["ocr_calls"] == 4
assert stats["discarded"] == 0
```

- [ ] 先写拒绝测试：换字、tile/file 双向、同前缀不同后缀及缩水框、字幕消失、空 OCR、不同内容的中间观测、间隔超限、不跨切景和孤立误检。逐行确认，不得凭一行匹配保留另一行。
- [x] 跑新测试，记录因单帧轨迹被丢弃而失败的红灯结果；现有纯追踪测试默认调用行为保持不变。
- [x] 共享可靠原始参考的构建逻辑。新增观测匹配入口只返回具体框的双向匹配，不调用 `refine()`，不向长期缓存写入推断字形。不得降低 `_align`、`_core_consistent`、白物体过滤或参考完整性阈值。
- [x] 追踪器的可选确认集合按帧和精确原始框核对。普通轨迹仍须两次命中；带确认的单帧轨迹不跨镜头插值，不能给无 OCR 帧复制框。
- [x] 检测接入沿用原反馈调度。每次回查按时间顺序记录，最后记录当前帧；相邻观测为空或不匹配就不能跨过。只保留最近一条所需图像或局部参考，间隔上限为 stride，切景依据使用原列表。
- [ ] 跑聚焦测试，核对 OCR 调用数和检测/接受/丢弃统计、ROI、64 条与 60 帧长期缓存限制不变。

```bash
env PYTHONPATH=/private/tmp/vsr-test-deps.Lsr98F /Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest -q tests/subtitle_tracking_test.py tests/subtitle_templates_test.py tests/adaptive_detection_test.py tests/glyph_pipeline_test.py
```

## Task 2：真实片尾及修复链路验证

由主代理负责，只修改新的真实片尾测试、OCR fixture 和本文档，避免与实现者改同一文件。

- [x] 将诊断缓存中的 650-686 帧 OCR 框保存为结构化 JSON fixture，不复制视频，不重新推理或联网。
- [x] 使用 `VSR_TEMPLATE_REGRESSION_VIDEO` 指定原片，读取 650-686 帧生成无损测试片段；OCR 替身按图像哈希返回真实缓存框。
- [x] 在旧实现上运行可选测试，确认第 665-667 帧原始 OCR 两行但时间线为空。

真实片尾红灯：2 failed（21.92 秒）。时间线在 665 帧断言失败；真实
process_video 仅修复 34/37 帧，665-667 均未到达修复引擎。
切景保持 665/666/667/668，33 次 OCR（26 sampled、7 refined）。
- [x] 使用真实 `process_video`，仅替代 OCR 与 GPU 推理；记录分段窗口、有效遮罩和合成结果。665-667 每帧两行框、遮罩非零且进入引擎；窗口不跨原有切景，单帧时间上下文只能重复自身。

```python
for frame_number in (665, 666, 667):
    assert len(timeline[frame_number - 650]) == 2
    assert np.count_nonzero(model_masks[frame_number]) > 10000
    np.testing.assert_array_equal(composed[frame_number][mask == 0], original[mask == 0])
```

- [x] 核对输出 37 帧、30 fps、连续 PTS；ROI 外及有效遮罩外的编码前像素不变。生成局部遮罩证据图，人工查看。

真实片尾绿灯：2 passed（26.97 秒）。665-667 两行框均保留，原始字形遮罩
分别从 0 恢复为 14352、14221、14440 像素；37 帧全部进入修复模型。
检测统计为 72 命中、72 接受、0 丢弃；仍为 33 次 OCR 和原四个切景点。
证据图与结构化数据位于 `/private/tmp/vsr19-tail.vqgWoF/scene-boundary-coverage.*`，
已人工查看。666 的句末点号仍有局部范围外像素，不属于本次整句漏帧修复承诺。
- [ ] 依次进行独立需求审查、代码质量审查，发现问题先修再复审。
- [ ] 运行全量、启用真实素材的回归、语法编译和差异检查。

```bash
env PYTHONPATH=/private/tmp/vsr-test-deps.Lsr98F /Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest -q
env PYTHONPATH=/private/tmp/vsr-test-deps.Lsr98F VSR_TEMPLATE_REGRESSION_VIDEO=/Users/liusili/play/video-subtitle-remover/TikSave.io_7635080993354878239.mp4 /Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest -q tests/real_scene_boundary_test.py
/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m py_compile vsr_pipeline.py backend/subtitle_templates.py backend/subtitle_tracking.py
git diff --check
```

- [ ] 核对用户主工作区没有冲突后，以补丁交付并在主工作区复验，不提交或推送；不清理已有工作树。
- [ ] 记录 CUDA 画质未验收限制，服务器使用同 ROI 生成 `test_pro_20.mp4`，不得覆盖 19。
