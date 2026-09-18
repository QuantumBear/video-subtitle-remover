# GroundingDINO 批量推理、混合精度与性能分析

`vsr_pipeline.py` 的 GDINO 后端默认把已确定的优先采样帧按 4 张一批推理。
反馈补查需要前面帧的结果来决定下一次检测时间，因此继续逐帧执行。
批量返回的候选按帧分别过滤，再按原来的帧序进入跟踪器。

同一检测器复用当前 prompt 的 tokenization 结果，换 prompt 时重新生成。
图像特征、文本模型前向和图文融合仍然正常计算；这个缓存不复用跨帧的
检测结果。仅缓存一个 prompt 的 CPU 张量，避免常驻服务的缓存不断增长。

默认使用 CUDA FP16 混合精度，可用 FP32 或 BF16 覆盖；CPU 或不兼容设备会自动回退 FP32。
图像预处理尺寸、
检测阈值、帧预算和跟踪判据不变。
不同 batch 的浮点计算可能存在微小差异，需要关注接近阈值的候选。

## 参数与对照方式

- `--sticker-batch-size 4`：默认值，仅对 GDINO 优先采样生效。
- `--sticker-batch-size 1`：关闭批量，仍复用 prompt token，可用于对照。
- `--sticker-precision fp16`：默认精度；在 CUDA 的模型 forward 中启用 autocast。
- `--sticker-precision fp32`：关闭混合精度，可用于基准和结果对照。
- `--sticker-precision bf16`：CUDA 可选精度，在模型 forward 中启用 autocast。
  同时作用于优先帧批次和单张反馈补查，后处理使用 FP32。
- `--sticker-profile`：输出模型初始化、检测分阶段和跟踪各遍耗时。

例如，先测单张，再把 batch 改成 4、输出文件名改为 `test_gdino_b4.mp4`：

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_gdino_b1.mp4 \
  --inpaint-mode sttn \
  --sticker-backend gdino \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-score 0.22 --sticker-max-area-px 1000 \
  --sticker-batch-size 1 --sticker-profile
```

固定素材、参数与 GPU 负载，对照多次运行。profile 会在 CUDA 阶段边界同步，
最终吞吐量应再关闭 profile 测一次。此示例关闭了 ProPainter 残留转交，
与此前 DINO profile 的设置一致。

### 混合精度对照

直接沿用 `backend/models/grounding-dino-tiny` 中的 FP32 权重，无需下载、
转换模型文件或重新训练。权重仍以 FP32 加载，autocast 在推理时选择适用算子的
计算精度。BF16 的数值范围更接近 FP32，可以先在支持 BF16 的 CUDA 设备上尝试：

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_sttn_gdino_bf16.mp4 \
  --inpaint-mode sttn \
  --sticker-backend gdino \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-score 0.22 --sticker-max-area-px 1000 \
  --sticker-batch-size 4 --sticker-precision bf16 \
  --sttn-profile --sticker-profile
```

保持其它参数相同，把精度改为 `fp32` / `fp16`，并使用不同输出文件名作对照。
确认最终 `[gdino-profile]` 的 `precision` 与请求一致、`precision_fallbacks=0`，
再比较 `inference`、跟踪 `wall` 和总耗时。模型内部也可能因低精度算子兼容问题
采用较慢的实现，因此启用 AMP 不保证提速。

有限数值的分数和框也可能有偏移，自动回退无法判断这种检测质量变化。
需要比较 `strong/weak/local/associated_frames` 等统计和成片中的贴纸残留、
误擦，尤其关注接近 `--sticker-score 0.22` 的检测；统计相同也不保证逐帧框相同。

## 日志口径

`[gdino-profile]` 中：

- `calls`：成功完成推理的帧数。
- `batches`：成功完成推理的批次数，包含反馈补查的单张调用。
- `text_tokenizations`：生成 prompt token 的次数；固定 prompt 通常为 1。
- `requested_precision` / `precision`：请求精度 / 当前实际精度。
- `precision_fallbacks`：设备不支持或 forward 异常引发的 FP32 回退次数。
- `preprocess/upload/inference/postprocess/filter`：各阶段累计秒数。

开启混合精度或 profile 时，初始化输出 `[gdino-precision]`；发生精度回退时，
即使没有开启 profile，也会输出 `[gdino-precision-fallback]` 及原因。

这些值在检测器实例内累计。`[gdino-profile-tracking]` 是单条视频的统计，
`priority_pass` 包含优先帧解码、推理与参考建立；`scan_pass` 包含外观核验和
反馈推理；`recheck_pass` 是用新增参考重新核验。外层耗时已经包含对应的
模型耗时，不能把它们相加当作总耗时。

如果仍检测 79 张优先帧和 59 张反馈帧，且没有失败，batch=4 时应看到
`calls=138 batches=79`，其中优先帧为 20 批，反馈帧为 59 批。
forward 次数减少不代表等比例提速：每批的计算量更大，收益需在目标 GPU 实测。
优先对照 `priority_pass`、`inference` 和跟踪 `wall`，同时检查候选与成片。

## 失败回退

非 CUDA 设备请求 FP16/BF16、或所选 CUDA 设备不支持 BF16 时，明确记录原因，
使用 FP32。混合精度 forward 的算子不兼容或输出 NaN、无效无穷值时，
当前批次以 FP32 重试一次，该检测器后续也使用 FP32。
GroundingDINO 合法的 `-inf` 文本填充不会触发回退。
FP32 重试仍异常时继续交由原有批量/帧失败处理，不静默返回空检测。
精度重试成功后，`calls/batches` 只计成功的帧/批，`inference` 包含失败尝试与重试耗时。

OOM 不触发精度回退，保持请求精度并交给下面的缩批逻辑处理。

批量异常或返回数量不符时，输出 `[sticker-gdino-batch-fallback]`，
当前批次逐帧重试，当前视频后续也改为单张。CUDA OOM 时先退出异常栈、
回收临时张量并释放空闲显存缓存，再重试。

`[sticker-gdino]` 的 `calls` 仍按尝试过的唯一帧数计入预算，
`batch_fallbacks` 单列批量失败次数。成功回退不算帧检测失败；
逐帧重试仍失败的帧才计入 `failed`。失败批次不计入检测器的成功 `batches`
或分阶段统计，但其耗时包含在跟踪 `wall` 中。

## 本地验证（2026-09-18）

混合精度回归覆盖 CLI 到惰性加载的参数传递、默认 FP16、CPU/不支持 BF16 的降级、
autocast 作用域、整数 token、后处理 FP32、合法文本填充、数值/算子异常与 OOM 分流。
本次相关回归及原片贴纸回放共 198 项通过、3 项跳过（2 项 CUDA 测试，
1 项可选慢速 ProPainter 对照）。FP16/BF16 的 CUDA 小模型测试在本地无 CUDA 时跳过；
它们只能验证封装行为，真实 GroundingDINO 的 CUDA 兼容性与速度仍需目标机器实测。

本地真实权重的 CPU 回退检查中，请求 BF16 后实际使用 FP32，原片第 0 帧
`(443,1052,0,624)` ROI 的 3 个候选与直接 FP32 forward 的整数框、分数完全一致。

此前批量推理相关回归及原片贴纸回放共 179 项通过，1 项可选的慢速 ProPainter 对照跳过。
覆盖尾批、帧顺序、反馈采样、预算、换 prompt、异常回退和 CLI 参数。

使用本地真实 GroundingDINO Tiny 权重（Transformers 4.44.2、PyTorch 2.2.2、
CPU FP32），对原片第 0、29、90、200 帧的 `(443,1052,0,624)` ROI 做对照：
原单张路径与优化后的 batch=1/4 都分别得到 3、3、0、0 个候选。
batch=4 的整数框坐标完全一致，置信度最大绝对差约 `1.60e-5`；
batch=1 的候选与分数完全一致。这是小样本正确性验证，CUDA 加速幅度和
全片最终遮罩仍需在目标服务器测量。
