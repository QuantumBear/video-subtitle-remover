# 操作日志

## 诊断开关:默认禁用 templates.refine 与白字自检 - 2026-09-09 14:10

背景:实验 A 证明本地基础流程能擦净 f90-94 的 "Non-slip",残留嫌疑收敛到服务器路径
独有的 templates.refine(mask 模板补全)与 white_glyph_check(残留复核+复修)两环节。
改为服务器端二分:默认禁用两环节,拉取即验证,省去本地 2 小时实验 C。

改动:
- vsr_pipeline.py:process_video 新增 template_refine=False;white_glyph_check 默认值
  True→False;flush_segment 条件化 templates.refine(禁用时原始 mask 直通);
  CLI 翻转 --no-white-glyph-check → --white-glyph-check,新增 --template-refine
- service/server.py:适配参数翻转(子进程命令),Form 默认 False,与 CLI 一致
- tests/glyph_pipeline_test.py:5 个依赖默认开启行为的测试显式传参以继续锁定功能;
  新增 test_template_refine_disabled_by_default_for_diagnosis 锁定默认禁用

验证:pytest 全套 175 通过 7 跳过;CLI --help 确认新参数注册。
验证结论:若服务器重跑后 0-3 秒干净 → 根因在两环节之一,再用 --template-refine /
--white-glyph-check 逐个恢复定位;若仍残留 → 嫌疑转回环境差异(region/模型版本)。

## 0-3 秒残留修复:自适应阈值重试 + 单帧空洞填补
时间:2026-09-10

### 标定结论(th_probe)
- 白色阈值 [228,200,215,225,235,245] 对 f52(暗背景)/f70(亮背景)/f114(白毛衣)标定:
  f70 主框 th228 时字形 7123px 被高度过滤剔剩 756px(保留率 10.6%),th235 仍 15%,
  **th245 时 1853px 全部保留且最大块高 25/40,与亮背景完全分离**;
  f52/f114 在各阈值下均 100% 保留,无回归。
- 方案 A(per-box 连通域)经 rule_probe 否定:f70 字幕在框内即与亮背景连通
  (组件 40×195 贴住框上下边),"贴边即弃"判据连字幕一起丢。
- "过滤后/raw 保留率"是干净分离信号:正常场景 100%,误杀 ≤15%。

### 实施内容
1. `propainter_boxes_to_mask`:框内字形保留率 <50%(GLYPH_KEEP_RATIO)时按
   WHITE_RETRY_DELTAS=(0,7,17) 逐级提高阈值重算,直到字形与背景分离;
   所有阈值都无法分离的(真大白物体)维持不擦;最低阈值部分字形作兜底(旧行为)。
   删除分支 B 写入空 local_glyph 的死语句(原 527-529,27 帧空 mask bug)。
2. `backend/subtitle_tracking.fill_single_frame_gaps`:轨迹物化后,对前后帧均有
   水平重叠框的单帧空洞用前帧框延拓(修 f58 场景切换前 OCR 漏检直通问题)。
   `_detect_timeline` 接入并统计 gap_filled。
3. 删除死代码:expand_timeline / filter_boxes_by_continuity / _boxes_overlap
   (06b2ede 引入轨迹机制后无调用点)。

### 测试
- 新增:亮背景字形恢复重试(亮带 242/白字 255 用 245 分离)、单帧空洞填补 ×3。
- 契约更新:test_matching_subtitle_survives_scene_cut... 中 hidden_white 断言——
  旧契约"合并白不直接擦"改为"真文字 255 必须直接擦除、亮带 242 不被整条吞掉"
  (旧断言混入了字形外扩圈压到的亮带像素)。
- 全量:179 passed, 7 skipped。
