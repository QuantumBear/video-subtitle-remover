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
