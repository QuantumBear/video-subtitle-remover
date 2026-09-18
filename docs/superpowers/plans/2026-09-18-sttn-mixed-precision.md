# STTN 可选 FP16 实施计划

**目标：** 增加可试验的 STTN FP16 参数，默认保持 FP32。

**结构：** 精度调度及整带重试放在 `backend/inpaint/sttn_det_inpaint.py`，CLI 与惰性加载参数放在 `vsr_pipeline.py`。移除 `backend/inpaint/sttn/network_sttn.py` 中不生效且不兼容 FP16 的运算。

**技术：** PyTorch CUDA autocast、NumPy、pytest。

- [ ] 在 `tests/sttn_precision_test.py` 增加 CPU 实网、数值异常、算子异常、OOM、CLI 和 CUDA 精度测试，运行并确认新增行为尚未实现。
- [ ] STTNDetInpaint 增加 `precision='fp32'`，以 `inpaint()` 包装单次 `_inpaint_once()`；用局部 autocast 执行模型，输出转 FP32，数值异常重算整带。
- [ ] 删除 Attention 的无效 masked_fill；通过半精度小张量验证与原 FP32 数学语义一致。
- [ ] Pipeline 增加 `sttn_precision='fp32'`，CLI 增加 `--sttn-precision` 并传到惰性 STTN 引擎。
- [ ] 更新 `docs/07-dev/03-STTN性能分析.md` 的用法、精度日志及回退计时口径。
- [ ] 用 `/Users/liusili/opt/anaconda3/envs/vsr/bin/python -m pytest -q tests/sttn_precision_test.py tests/sttn_qkv_cache_test.py tests/sttn_det_composite_test.py tests/sttn_pipeline_test.py tests/sttn_profile_test.py` 验证 STTN，再执行 DINO 与流水线相关回归及 `git diff --check`。

本轮沿用工作区中尚未提交的 DINO 默认 FP16 修改，不包含本地模型文件。
