# STTN 性能分析

在原有 `vsr_pipeline.py` 命令后追加 `--sttn-profile` 即可；默认关闭。
例如：

```bash
python vsr_pipeline.py -i input.mp4 -o output.mp4 \
  --inpaint-mode sttn --sttn-precision fp16 --sttn-profile \
  --sttn-residual-propaint --sttn-residual-propaint-max-windows 8
```

Python 调用可使用 `Pipeline(inpaint_mode='sttn', sttn_precision='fp16', sttn_profile=True)`，
或直接构造 `STTNDetInpaint(device, model_path, precision='fp16', profile=True)`。
该选项仅在 STTN 模式生效。

`--sttn-precision` 默认是 `fp16`。设置为 `fp16` 时，CUDA 上仅对
encoder、首层 Q/K/V、transformer 和 decoder 使用局部 autocast，模型权重仍以
FP32 加载，输出和 mask 合成回到 FP32。CPU 会自动回退到 FP32。若 CUDA
算子不支持 Half，或 decoder 产生非有限值，会丢弃当前修复带并完整用 FP32
重算；OOM 不触发回退，以便沿用上层显存处理策略。

需要对照旧路径时显式传入 `--sttn-precision fp32`。

## 日志范围

- `[sttn-profile]`：一次缩放后修复带的 `inpaint()` 调用。一个视频段可能包含多个修复带，因此可能输出多行；各阶段累计该修复带所有窗口的耗时。日志中的 `requested_precision`、`precision` 和 `precision_fallbacks` 分别表示请求精度、当前实际精度和累计回退次数。
- `[sttn-profile-band]`：一次视频段中实际生成的垂直修复带，包含 `band=当前/总数` 和原图坐标 `ymin,ymax,xmin,xmax`。同一视频段出现多条 `[sttn-profile]` 时，用这个日志判断它们对应的是哪些区域。
- `[sttn-profile-segment]`：一次完整 STTN 引擎调用，包含 ROI 裁剪、修复带缩放、推理、放大及 mask 合成。`segment` 是原视频帧号范围。不含模型加载、OCR、残留检测、ProPainter 和视频编码。
- `[sttn-profile-total]`：整条视频的 STTN 引擎调用 wall time 之和，以及已有的 ProPainter fallback wall time。每条视频重新统计，不等于流水线总耗时；完整耗时仍看 `[done]`。
- `[pipeline-profile-stage]`：启用 `--sttn-profile` 后增加的流水线主机计时，逐项打印 `stage`、累计 `wall` 和调用次数 `calls`，字段见下表。
- `[pipeline-profile-total]`：本次流水线的主机总时间 `wall`、已分配时间 `accounted`、未覆盖时间 `other`。各个新阶段使用排除子阶段后的时间，因此 `accounted + other = wall`；不要再将已有的 STTN、GDINO 内部计时加到这组总数上。

所有耗时单位为秒。引擎的 `last_profile` 保存最后一次修复带的数字统计，不累加到下一次调用。

## 流水线阶段

沿用现有开关即可，不需要增加命令参数。纯 STTN + GDINO 示例：

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_sttn_pipeline_gdino_15.mp4 \
  --inpaint-mode sttn \
  --sticker-backend gdino \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-score 0.22 --sticker-max-area-px 1000 \
  --sttn-profile --sticker-profile
```

| `stage` | 范围 |
| --- | --- |
| `ocr_timeline` | 整个 OCR 时间线构建，包括该遍视频解码、采样、OCR 调用和轨迹关联，不能视为纯 OCR 推理 |
| `sticker_init` | 检测器获取与首次加载，包括加载函数中的初始化日志 |
| `sticker_detect` | 整个贴纸检测/跟踪流程；GDINO 的细分时间包含在其中 |
| `sticker_associate` | VLM 路径的贴纸框关联，GDINO 关联已包含在 `sticker_detect` |
| `sticker_release` | 贴纸模型释放及该函数中的 GC、CUDA 缓存清理 |
| `video_open` | 修复阶段打开输入/输出视频容器 |
| `video_decode` | 修复阶段逐帧从解码迭代器取帧，不含取出帧后的转换、推理和写出 |
| `input_frame_convert` | 修复输入帧转 NumPy RGB，再转 BGR |
| `sttn_init` | 每段获取 STTN 引擎，包含首次加载；后续调用一般只做已加载检查 |
| `sttn_mask_prepare` | 时序邻域矩形 mask 和横向裁剪边界准备 |
| `sttn_infer` | 完整 STTN 引擎调用，含 ROI 与修复带往返；不含调用后的 band/segment 日志 |
| `roi_composite` | 输出时将原始 ROI 之外的像素贴回 |
| `residual_probe` | `_residual_mask()` 调用，包括补修候选与最终疑似残留诊断；不含调用外的像素计数和候选排序 |
| `output_frame_convert` | 输出 BGR 转 RGB、创建视频帧及设置时间戳 |
| `video_encode` | 逐帧编码和容器 mux；补修第二遍的写出也累计到这里 |
| `video_encode_flush` | STTN 第一遍末尾编码器 flush、mux 和关闭输出容器 |
| `cleanup_gc` | STTN 每段结束的 `gc.collect()` |
| `cleanup_cuda_cache` | STTN 每段结束的 `torch.cuda.empty_cache()`，仅 CUDA 可用时出现 |
| `propainter_init` | 补修引擎获取与首次加载，仅启用补修且实际执行时出现 |
| `propainter_infer` | 补修窗口处理，不含模型加载 |
| `propainter_second_pass` | Top-K 补修第二遍剩余工作：扣除已单列的补修加载/处理、残留诊断、帧转换与编码，包含重读视频、mask 准备、末尾 flush 等 |
| `audio_finalize` | 检查音轨、关闭输入容器、调用 ffmpeg 合音频及最终文件替换；无音频时只关闭和替换 |

只有执行过的阶段才打印；解码 `calls` 是取帧尝试数，含末尾 EOF 尝试。所有新阶段均用 `perf_counter` 测主机经过时间，不创建 CUDA Event 或额外同步。异步 GPU 等待可能落在随后阻塞的调用内，因此这些字段用于划分流水线时延，不能当作纯 GPU 算子时间。

新总计从自动 ROI 推断之后、OCR 时间线之前开始，到音频和输出文件处理完成后、阶段汇总打印之前结束。它与 `[done]` 的起点基本一致，后者还包含末尾日志等少量开销；两者均不包含 `Pipeline` 初始化及自动 ROI 推断，不能当作 CLI 冷启动全耗时。`other` 保留未覆盖的调度、框整理、日志及计时本身等开销，不会强行归为 OCR。

Python 返回统计在启用时增加 `pipeline_profile`，包含 `wall_seconds`、`stage_seconds`、`stage_calls`、`accounted_seconds` 和 `other_seconds`。每条视频重新累计；关闭时不添加该字段、不输出新日志，也不读取新增计时时钟。日志不改变检测、mask、模型推理、残留统计或视频编码参数。

## 修复带内的阶段

| 字段 | 范围 |
| --- | --- |
| `preprocess` | CPU 上图片/mask 转 tensor、归一化及合成 mask 准备 |
| `upload` | 帧和 mask tensor 转到推理设备 |
| `encoder` | 输入遮罩运算与 Encoder |
| `qkv` | 整个修复带的首层 Q/K/V 缓存构建；未启用时为 0 |
| `gather` | 各窗口收集帧特征、mask 和缓存 Q/K/V |
| `transformer` | 各窗口完整 Transformer 推理；关闭缓存时也包含首层投影 |
| `decoder` | 邻近帧解码、tanh 及输出归一化 |
| `download` | 结果 tensor 回传 CPU |
| `postprocess` | CPU 上转 NumPy、还原像素范围、mask 合成和重叠窗口混合 |
| `wall` | 此次 `inpaint()` 的总实际经过时间，包括计时开销 |

`frames` 为修复带帧数，`windows` 为滑动窗口数。
`input_frame_visits` 累计所有窗口输入帧数，含参考帧；
`decoded_frame_visits` 累计所有窗口解码的邻近帧数。
这两个值除以 `frames`，分别表示每帧平均参与 Transformer 和 Decoder 的次数。
`cache=on/off` 表示本次是否实际建立首层缓存，单窗口自动关闭。

## CUDA 计时口径

CUDA 上输出 `clock=cuda_event`。除 `preprocess`、`postprocess` 使用 CPU wall time 外，
其余阶段使用当前设备当前 stream 上的 CUDA Event。各阶段只记录事件，修复带结束时
新增一次 stream 同步读取耗时，不额外逐阶段同步；原有阻塞上传、回传保持不变。
事件区间可能包含 CPU 提交 kernel 的间隙，并非逐 kernel 执行时间的简单总和。

`upload_wait_wall`、`download_wait_wall` 是主机执行对应 `.to()` / `.cpu()` 的经过时间，
**可能包含等待前序 GPU 计算完成的时间，不能与 GPU 各阶段时间相加**。
尤其不能凭 `download_wait_wall` 较高就认定 PCIe 拷贝慢。
CPU/GPU 工作可能重叠，阶段之和也不保证等于 `wall`；总延迟以 wall time 为准。

CPU 上输出 `clock=wall`，各阶段均为主机计时。其他非 CUDA 设备也只记录主机时间，
不能据此当作准确的异步设备执行时间。关闭分析时不创建 CUDA Event、不读取计时时钟，
也不增加同步。

## 分析首层缓存收益

使用已有的固定输入基准，同时打印缓存关闭和开启时的阶段统计：

```bash
python benchmarks/sttn_qkv_cache.py --device cuda --frames 50 \
  --warmup 1 --repeats 3 --profile
```

重点比较 `qkv + gather + transformer` 和完整 `wall`。
缓存构建及索引搬运也有成本，不能只看开启缓存后 `transformer` 的下降。
若 `decoder` 占比高，首层投影缓存能影响的比例就有限；若整条视频的
`propainter_wall` 占比高，STTN 的局部加速也会被稀释。

性能分析本身有开销。定位后用相同输入、预热和重复次数，不带 `--profile` 再测最终加速比。
不要用本机 CPU 的耗时推算 5090 上的收益。
