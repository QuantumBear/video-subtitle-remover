# STTN 可选 FP16 推理

用户选择先提供可选参数，默认保持 FP32。

- `vsr_pipeline.py --sttn-precision fp32|fp16` 经 Pipeline 传到惰性加载的 STTNDetInpaint；默认 `fp16`（GPU 实测后设为默认）。
- CUDA 的 encoder、首层 QKV 缓存、transformer、decoder 使用局部 autocast；权重保持 FP32，不更换模型文件。输出转 FP32 后做 tanh、像素换算与合成。
- 非 CUDA 请求 FP16 时记录原因并使用 FP32。FP16 算子异常或非有限 decoder 输出触发整个当前修复带的 FP32 重算，并让该引擎后续调用保持 FP32。失败带的部分结果和 QKV 缓存不能复用。
- OOM 继续抛给现有处理，不尝试显存需求更高的 FP32。FP32 重试失败继续抛出。
- 移除 Attention 中丢弃返回值的 `scores.masked_fill(m, -1e9)`：其结果原本未生效，且常量超出 FP16 范围。本次保留已有 attention 数学语义，不改变 mask 算法。
- 初始化及 profile 展示请求精度、实际精度和累计回退次数。重试时带内 profile 只统计成功尝试，segment/total 包含失败尝试时间。
- CPU 回归覆盖默认行为、完整重试、有限性、OOM、CLI 传递和 mask 合成；CUDA 测试在无 CUDA 时跳过。真实速度和画质由用户 GPU 机器对比验证。
