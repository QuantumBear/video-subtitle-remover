# GroundingDINO 批量推理与性能分析

`vsr_pipeline.py` 的 GDINO 后端默认把已确定的优先采样帧按 4 张一批推理。
反馈补查需要前面帧的结果来决定下一次检测时间，因此继续逐帧执行。
批量返回的候选按帧分别过滤，再按原来的帧序进入跟踪器。

同一检测器复用当前 prompt 的 tokenization 结果，换 prompt 时重新生成。
图像特征、文本模型前向和图文融合仍然正常计算；这个缓存不复用跨帧的
检测结果。仅缓存一个 prompt 的 CPU 张量，避免常驻服务的缓存不断增长。

本轮保留 FP32、图像预处理尺寸、检测阈值、帧预算和跟踪判据。
不同 batch 的浮点计算可能存在微小差异，需要关注接近阈值的候选。

## 参数与对照方式

- `--sticker-batch-size 4`：默认值，仅对 GDINO 优先采样生效。
- `--sticker-batch-size 1`：关闭批量，仍复用 prompt token，可用于对照。
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

## 日志口径

`[gdino-profile]` 中：

- `calls`：成功完成推理的帧数。
- `batches`：成功完成推理的批次数，包含反馈补查的单张调用。
- `text_tokenizations`：生成 prompt token 的次数；固定 prompt 通常为 1。
- `preprocess/upload/inference/postprocess/filter`：各阶段累计秒数。

这些值在检测器实例内累计。`[gdino-profile-tracking]` 是单条视频的统计，
`priority_pass` 包含优先帧解码、推理与参考建立；`scan_pass` 包含外观核验和
反馈推理；`recheck_pass` 是用新增参考重新核验。外层耗时已经包含对应的
模型耗时，不能把它们相加当作总耗时。

如果仍检测 79 张优先帧和 59 张反馈帧，且没有失败，batch=4 时应看到
`calls=138 batches=79`，其中优先帧为 20 批，反馈帧为 59 批。
forward 次数减少不代表等比例提速：每批的计算量更大，收益需在目标 GPU 实测。
优先对照 `priority_pass`、`inference` 和跟踪 `wall`，同时检查候选与成片。

## 失败回退

批量异常或返回数量不符时，输出 `[sticker-gdino-batch-fallback]`，
当前批次逐帧重试，当前视频后续也改为单张。CUDA OOM 时先退出异常栈、
回收临时张量并释放空闲显存缓存，再重试。

`[sticker-gdino]` 的 `calls` 仍按尝试过的唯一帧数计入预算，
`batch_fallbacks` 单列批量失败次数。成功回退不算帧检测失败；
逐帧重试仍失败的帧才计入 `failed`。失败批次不计入检测器的成功 `batches`
或分阶段统计，但其耗时包含在跟踪 `wall` 中。

## 本地验证（2026-09-18）

相关回归及原片贴纸回放共 179 项通过，1 项可选的慢速 ProPainter 对照跳过。
覆盖尾批、帧顺序、反馈采样、预算、换 prompt、异常回退和 CLI 参数。

使用本地真实 GroundingDINO Tiny 权重（Transformers 4.44.2、PyTorch 2.2.2、
CPU FP32），对原片第 0、29、90、200 帧的 `(443,1052,0,624)` ROI 做对照：
原单张路径与优化后的 batch=1/4 都分别得到 3、3、0、0 个候选。
batch=4 的整数框坐标完全一致，置信度最大绝对差约 `1.60e-5`；
batch=1 的候选与分数完全一致。这是小样本正确性验证，CUDA 加速幅度和
全片最终遮罩仍需在目标服务器测量。
