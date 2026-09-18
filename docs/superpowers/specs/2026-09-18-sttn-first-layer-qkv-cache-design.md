# STTN 首层 Q/K/V 投影缓存设计

## 目标

在每个 STTN 修复段内复用首个 Transformer block 的 Q/K/V 投影，减少同一段滑动窗口反复执行的首层 `1x1` 卷积，同时保持现有窗口、mask、权重和输出合成语义不变。

## 方案

`STTNDetInpaint.inpaint` 在 Encoder 完成后，对整段特征计算一次首层投影缓存。每个滑动窗口仍按原顺序选择 `neighbor_ids + ref_ids`，同时按相同索引切出缓存的 Q/K/V，传给 `InpaintGenerator.infer`。首个 Transformer block 使用传入的投影，后续 block 正常从上一层输出重新计算投影。未提供缓存时，模型走原有路径。

缓存只存在于一次修复带的 `inpaint` 调用内，不跨视频段、不同 mask 或不同 ROI 复用。这样避免图像内容、mask 和窗口上下文改变时产生错误复用。默认开启；单窗口无需缓存，自动走原路径。

## 接口与正确性

- `InpaintGenerator.project_first_qkv(features)` 返回首层三个投影张量。
- `InpaintGenerator.infer(features, masks, qkv_cache=None)` 接受可选缓存。
- 两条路径执行相同的数学计算；整段与窗口投影的 batch 大小不同，底层卷积可能引入浮点舍入差异，不承诺 CUDA 上逐位一致。测试覆盖同 batch 严格相等，以及跨窗口特征 FP32 容差 `rtol=1e-5, atol=1e-6`、最终 0–255 像素差不超过 1。
- 缓存张量按帧维度索引，窗口排序仍由调用方控制。
- `STTNDetInpaint(..., cache_first_qkv=False)` 保留原推理路径，便于对比；已有调用默认启用。
- 不改权重参数及 state_dict 键名。首层消费缓存后不传递给后续层。

## 显存与收益

50 帧、256 通道、60×108 特征、FP32 下，整段三个投影约占 `3×50×256×60×108×4 / 1024³ = 0.927 GiB`；这是缓存张量体积，并非精确的峰值增量。窗口切片在 `infer` 后释放，整个缓存随 `inpaint` 返回释放，不常驻模型，也不会与随后 ProPainter 调用同时保持存活。

只减少首层投影，注意力、后续层和解码计算量不变。缓存切片和显存访问也有开销，实际 GPU 加速幅度需要测量。

## 验证

单测覆盖标准分辨率等价性、乱序且重叠的窗口、逐帧不同 mask、首层每段只投影一次、后续七层仍逐窗口投影、连续两段缓存隔离以及单窗口跳过缓存。现有 STTN 合成和流水线回归检查逐帧 mask、ROI 和帧输出契约。

真实权重基准（固定合成输入，包含缓存构建，不含 OCR、ProPainter、模型加载）:

```bash
python benchmarks/sttn_qkv_cache.py --device cuda --frames 50 --warmup 1 --repeats 3
```

输出两条路径的耗时中位数、加速比、像素差和 CUDA 峰值 allocated。该基准只衡量 STTN 修复带推理；视频端到端收益仍需用原视频测量。

本机验证：68 项相关测试通过；真实权重、432×240、6 帧 CPU 冒烟检查输出零差异。该次未预热单次计时为 39.019 秒（关闭）与 39.722 秒（开启），未观察到加速。本机无 CUDA，尚无 5090 性能和显存峰值数据。
