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

## 按需执行 STTN 残留诊断

STTN 默认逐个核心帧进行最终疑似残留诊断，只影响统计，不修改输出像素。
已接受当前修复画质、希望省去这部分耗时时，可追加 `--sttn-skip-residual-check`：

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_sttn_pipeline_gdino_16.mp4 \
  --inpaint-mode sttn \
  --sticker-backend gdino \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-score 0.22 --sticker-max-area-px 1000 \
  --sttn-profile --sticker-profile \
  --sttn-skip-residual-check
```

不传该开关即保留原诊断行为；它不依赖 profile 开关。Python 对应
`pipe.process_video(..., sttn_residual_check=False)`，参数仅支持 STTN。
跳过后日志显示 `疑似残留 未检测`，返回 `unresolved=None`
（序列化成 JSON 为 `null`）、`sttn_residual_check_enabled=False`；
默认 STTN 返回 `sttn_residual_check_enabled=True`，`unresolved` 仍为整数。
原 `residual_check_enabled` 继续表示 `white_glyph_check` 白字复修开关，含义不变。

与 `--sttn-residual-propaint` 同时使用时，仍执行补修候选检测及窗口选择，
只跳过最终诊断和补修后的统计复查，`unresolved` 同样为 `None`；
候选及实际补修窗口统计继续返回。未启用补修时，跳过诊断后不应出现
`residual_probe` 阶段；启用补修后该阶段仍包含必要的候选检测。

本地编解码回归验证开关不改变输出像素、帧数及时间戳。目标 GPU 单次对照中，
该样片从 `120.724 s` 降到 `107.064 s`，少 `13.660 s`（`11.32%`）；
原诊断阶段 `11.345 s` 已消失，另外约 `2.315 s` 来自其余阶段及未分配时间的
净变化，不能全部计为开关收益。详细对照见
[首次 GPU 基线分析](06-STTN首次GPU基线分析.md)第 13 节。
稳定收益仍需相同环境多次 A/B 复测，正式吞吐测试应统一关闭两个 profile。

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

## 删除未使用的注意力 mask

`network_sttn.py` 的四路 patch attention 原先每层都从 mask 构造 `mm`，
执行重排、均值、阈值和 `repeat`，得到 token 数量平方大小的临时 bool 张量。
`Attention.forward()` 实际未读取该参数，当前实现只删除这部分无用计算，
仍按原来的 QK、softmax 和 V 完成注意力。函数参数及权重键保持原样。
输入帧上的空洞遮罩、逐帧合成 mask、窗口融合和精度回退继续使用原路径。
本次没有修改 `auto_sttn.py` 的独立实现，也不合并 GPU 数值检查。

固定随机输入、同一真实权重和窗口，可在目标 GPU 直接 A/B：

```bash
python benchmarks/sttn_attention_mask.py \
  --device cuda --frames 50 --warmup 3 --repeats 5
```

工具在同一引擎上顺序恢复冻结的旧 attention forward，与新路径交替运行；
生产引擎无需切换开关。两边默认 FP16、首层 QKV 缓存开启，模型加载不计时，
整带耗时包含预处理、上传、编码、窗口推理、解码、下载和 CPU 合成，不包含
外层修复带缩放、OCR、GDINO 或编解码。可用 `--precision fp32` 或
`--cache off` 分别验证另一精度和无缓存路径，两边始终使用相同设置。

先看 `outputs_match=true`、`max_pixel_difference=0`，再看 `legacy_seconds`、
`optimized_seconds` 的中位数和两边 `peak_allocated_gib`；峰值是 PyTorch
allocated，不含 reserved 或其它进程显存。工具不清空分配器缓存，预热后
交替顺序减少先后影响；精度回退或输出像素不一致时以失败退出。
需要定位 `transformer` 内的变化时另跑一次追加 `--profile`，正式速度比较
使用上面的无 profile 命令，避免将计时开销混入收益。

本地验证包含缓存开关、单帧/重叠窗口、batch>1、不同 mask 的特征及整带
输出等价，算子记录确认推理不再执行 `repeat`。真实 `sttn.pth` 权重的
3 帧、CPU FP32 单次冒烟中，新旧整带输出逐像素一致，无精度回退；无预热且
仅一次，不作为性能结论。本机没有 CUDA，目标 GPU 实测由用户执行，见下节。
本轮全量回归为 `490 passed, 22 skipped`，其中新增 9 项测试通过、3 项 CUDA
等价性测试因本机无 CUDA 跳过；独立只读审查未发现需修复的问题。

### 首轮 GPU A/B：输出一致，耗时差异很小

用户按上面的无 profile 命令测量：`NVIDIA RTX5880-Ada-24Q`，
PyTorch `2.11.0+cu128`、NumPy `2.2.5`，CPU 线程 1，50 帧，预热 3 轮、
测量 5 轮；两边 FP16、首层 QKV 缓存开启，无精度回退。

| 指标 | 旧路径 | 删除无用 mask |
| --- | ---: | ---: |
| 整带中位耗时 | 1.102735 s | 1.096091 s |
| 五轮耗时范围 | 1.094368–1.121409 s | 1.091761–1.111887 s |
| 峰值 allocated | 1.647917 GiB | 1.647479 GiB |

本轮中位数少 `6.644 ms`、`0.603%`，比值为 `1.0061×`；两边测量范围重叠，
差额小于本轮耗时波动，尚不能认定为稳定加速。峰值 allocated 仅少
`0.449 MiB`（471,040 字节），对当前显存预算没有实质影响。
静态累计分配/写入量不等于同时存活的张量峰值，不能据原累计约 `0.33 GiB`
推断应节省同量峰值显存。

`outputs_match=true`、最大和平均像素差均为 0。本轮支持保留这项无用计算
清理，但没有显示明显的性能收益；尚无改动后真实视频的完整流水线 A/B，
不能将本次合成单修复带比值直接外推到整片或并发吞吐。原始终端测量值：

- 旧路径：`1.104269 / 1.121409 / 1.102004 / 1.102735 / 1.094368 s`。
- 新路径：`1.099747 / 1.091761 / 1.111887 / 1.093564 / 1.096091 s`。

## Transformer 算子级分析

无用 mask 的首轮 GPU 收益很小，下一步先定位卷积、矩阵乘、布局搬运和同步
各自的成本。新增工具仅在基准采集时为当前模型实例加 `record_function`
范围，退出时恢复，不改生产计算、窗口、精度或清理策略：

```bash
python benchmarks/sttn_operator_profile.py \
  --device cuda --frames 50 --warmup 3 --repeats 5 \
  --output-dir benchmark_sttn_operators_01
```

使用真实 `sttn.pth`、固定随机种子 42、FP16 和首层 QKV 缓存；输入与
`sttn_attention_mask.py` 相同。先在没有 profiler、没有临时范围的情况下预热
和重复测量，再独立采集一次完整修复带。`baseline_median_seconds` 才是本轮
无采集开销的整带耗时，`capture_wall_seconds` 包含采集与收尾开销，不能用于
计算加速比。这是合成单带基准，不含 OCR、GDINO、外层带缩放或视频编解码。

终端打印分类、范围及分别按 CPU/GPU 排序的热点；输出目录生成：

- `report.json`：环境、窗口/缓存设置、每轮基线、输出差异、完整算子和分类表。
- `trace.json`：Chrome trace，可在支持该格式的 trace viewer 中查看 GPU 执行、
  CPU 提交间隙和同步等待。

重点先确认 `outputs_match=true`、`max_pixel_difference=0` 和精度未回退，
再比较 `scope=transformer` 下 `convolution`、`matmul`、`softmax`、
`layout_copy_index`。`region` 进一步区分 `qkv_projection`、`attention`、
`attention_output` 和 `feed_forward`；`qkv_cache` 是窗口外首层缓存构建。
`finite_check` 包含引擎中的非有限值检查，`host_wait_or_scalar` 包含 `.item()`
及同步；两者仍按原方式执行，本工具没有合并数值检查。

时间单位为毫秒，口径如下：

- `operators/categories` 使用 **self** CPU 时间和归属于该算子的 self GPU 时间，
  避免将 `conv2d → convolution → cudnn_convolution` 等父子重复累计。
  每个事件归入一个分类，分类是诊断提示，具体算子名保留在完整表中。
- `regions` 是包含子范围的时间，`band → transformer → multihead → attention`
  相互嵌套，**不能相加**，也不能再加到算子表。
- CPU 时间可能包含等待 GPU，**不能与 GPU 时间相加**。原始 `device_events`
  是单独的 kernel/copy 表，也不能再加到算子归属的 GPU 时间上。
- `outside_model` 包括预处理、窗口 gather、数值检查、传输和 CPU 合成；
  NumPy/Python 部分不进入 ATen 算子表，其间隙可在 `sttn::band` 的 trace 中查看。
- `device_timing_status=attributed` 才能根据算子 GPU 时间定位；若为
  `unattributed`，只能看到原始设备事件；`unavailable` 表示未采到设备时间，
  需要检查服务器 profiler/CUPTI 环境，不能将 `null` 解释为 GPU 耗时为零。

需要看输入形状时另跑一次加 `--record-shapes`（采集开销更高）；还可用
`--cache off`、`--precision fp32` 比较其它设置，勿混合这些设置的计时结果。
本机 CPU 冒烟使用真实权重、3 帧、预热 1 轮、测量 2 轮，带形状采集，
输出一致、最大像素差为 0；GPU 数据仍待采集，不据 CPU 热点预测 GPU 收益。
本轮全量回归 `498 passed, 23 skipped`；新增算子分析测试为 8 项通过，
1 项 CUDA 实采测试因本机无 CUDA 跳过。

### CUPTI 不可用时，用 CUDA Event 测范围

首轮服务器算子采集的无 profiler 中位数为 `1.095010 s`，五轮范围
`1.091726–1.103121 s`，输出一致；但 CUPTI 返回
`CUPTI_ERROR_NOT_INITIALIZED (15)`，`device_timing_status=unavailable`，
没有 GPU 算子时间。采集 wall `2.081678 s` 含 profiler 开销，不能作为基线。
其中 20 次 `_local_scalar_dense` 的 CPU self 时间共 `713.624 ms`，对应
10 窗口各两次 `.item()`。该时间包含等待前序 GPU 运算完成，不能视为
删除检查即可节省的时间，也不能用卷积的 CPU 提交耗时判断 GPU 卷积成本。

此环境先用不依赖 CUPTI 的模式采集同一完整修复带：

```bash
python benchmarks/sttn_operator_profile.py \
  --device cuda --frames 50 --warmup 3 --repeats 5 \
  --timing cuda-event --output-dir benchmark_sttn_events_01
```

该模式不调用 `torch.profiler`，不生成 Chrome trace，`report.json` 中
`trace=null`、`device_timing_status=stream_intervals`。
先按原设置完成无包装基线，再校准一带初始化可复用事件，最后单独重复采集
`--repeats` 带。校准输出也参与一致性检查；所有阶段均须保持精度，发生
精度回退或输出变化会以失败退出。现有两次数值检查、传输、窗口融合不改。

终端的 `[event-region]` 和报告的 `regions` 给出每带累计值的逐字段中位数：

| 范围 | 观察内容 |
| --- | --- |
| `qkv_cache` / `qkv_projection` | 窗口外缓存构建 / 每个 Q、K、V 卷积；后者也含缓存构建内的投影 |
| `attention` | 四路注意力的矩阵乘、缩放和 softmax，合计 320 次/50 帧带 |
| `attention_output` | 拼接后的输出卷积和激活 |
| `feed_forward` | 每层两次前馈卷积和激活 |
| `multihead` 的 self | 扣除投影、attention、输出卷积后的重排、拼接及提交间隙 |
| `transformer` 的 self | 扣除 multihead/FFN 后的残差运算及范围外间隙 |
| `band` 的 self | 模型模块之外的预处理、gather、数值检查、传输、CPU 合成及间隙 |

`inclusive_stream_ms` 是范围开始/结束事件之间的 stream 经过时间，含子范围；
`self_stream_ms` 扣除直接子范围，避免父子重复累计。GPU 事件只记录，
每带结束时统一同步一次读取，不新增逐模块同步；原有 `.item()` 和 `.cpu()`
仍会按生产路径等待。事件复用只用于同一进程中的顺序基准，当前只支持
STTN 的单 stream 路径，不代表多 stream 或并发容量。

事件区间包含 CPU 提交 kernel 的间隙、事件和包装开销，并非纯 kernel
执行时间；尤其 `band` 会包含 GPU 等待 CPU 的间隙。`inclusive_host_ms` /
`self_host_ms` 是对应主机经过时间，可能等待 GPU，不能再与 stream 时间相加。
只在同一带内按 self 汇总，`regions` 各字段中位数不保证可加；
需要核对时使用 `region_runs` 中各次原始范围。正式速度仍看无包装的
`baseline_median_seconds`，采集 wall 仅用于衡量工具开销。

CPU 可用 `--device cpu --precision fp32 --timing cuda-event` 冒烟；此时
`compute_clock=wall`，stream 字段均为 `null`，不能当 GPU 数据。
`--record-shapes` 只支持原 `--timing profiler` 模式。

本地真实权重 3 帧、预热 1 轮、基线及范围各测量 2 轮，输出像素差为 0，
精度回退为 0；该数据仅验证 CPU 冒烟。全量回归 `503 passed, 24 skipped`，
算子/范围基准测试 `13 passed, 2 skipped`，两项实采测试因本机无 CUDA 跳过。

### 首轮 CUDA Event 结果与 cuDNN A/B

`benchmark_sttn_events_01` 在 RTX5880-Ada-24Q、Torch `2.11.0+cu128`、
NumPy `2.2.5` 上测得以下结果。50 帧、FP16、首层 QKV 缓存开启、单线程，
预热 3 轮、正式 5 轮；输出一致，最大/平均像素差为 0，精度回退为 0。
无包装基线中位数 `1.123503 s`，事件采集中位数 `1.099923 s`；二者执行
同一公式，该差异反映波动与先后顺序，不能作为优化收益。

| 范围 | inclusive stream 中位数（ms） | self stream 中位数（ms） |
| --- | ---: | ---: |
| band | 1097.805 | 247.009 |
| encoder | 15.608 | 15.608 |
| qkv_projection | 89.252 | 89.252 |
| transformer | 681.123 | 26.038 |
| multihead | 410.429 | 64.340 |
| attention | 131.610 | 131.610 |
| attention_output | 129.767 | 129.767 |
| feed_forward | 244.171 | 244.171 |
| decoder | 150.771 | 150.771 |

按每轮原始数据，encoder、QKV 投影、attention 输出、FFN 和 decoder
这些不重叠的卷积相关范围合计占 band 的 `54.93–60.28%`，占比中位数
`56.50%`；attention 占比中位数 `11.90%`。这支持先试卷积算法选择。
这些范围包含激活、插值和提交间隙，不是纯卷积 kernel 时间；也不能
把表中父子范围或逐字段中位数直接相加。`band` self 还含预处理、
gather、数值检查、传输、CPU 合成及间隙，不能全部归因于同步。

新增独立基准只切换 `torch.backends.cudnn.benchmark`，生产设置不改：

```bash
python benchmarks/sttn_cudnn_benchmark.py \
  --device cuda --frames 50 --warmup 3 --repeats 5 --sessions 2 \
  --output-dir benchmark_sttn_cudnn_01
```

使用新输出目录；默认真实权重 `backend/models/sttn-det/sttn.pth`、FP16、
QKV 缓存开启、线程数 1。默认两组按 `off/on`、`on/off` 执行，共四个
顺序子进程；每个进程重新加载模型，避免同进程 cuDNN 算法缓存污染。
父进程只用标准库，子进程日志、逐次记录与汇总均保存在输出目录。

- `first_seconds` 是模型加载后的首次完整带 wall 时间。它仍含剩余惰性
  初始化和算法选择；不能把开关之间的差值全部当作纯算法搜索成本。
  模型加载、模块导入、子进程总 wall 另列，不混入带耗时。
- `runs` 保留首次、预热和正式每次耗时及 allocated/reserved 峰值；
  `summary` 比较各模式首次中位数、正式热态中位数和所有阶段最大峰值。
  每带前后同步、重置峰值，不在带间清空 allocator 缓存。首次搜索可能
  增加 workspace，需结合热态收益评估，不能只看单带速度。
- 所有阶段均与第一个 off 带输出逐像素比较，保留融合后的浮点值，
  不先转 uint8。像素不一致仍写完整报告，再以失败退出，速度收益字段
  为 `null`；精度回退、非有限值或执行失败直接使子进程失败，查看对应日志。
- `backend` 记录 enabled、deterministic、TF32、benchmark_limit、全局
  deterministic 等设置，除 benchmark 外须跨进程相同。工具不放宽
  精度约束，不设置无限搜索，也不修改窗口、mask、融合或清理策略。

本机可用 `--device cpu --precision fp32` 冒烟；CPU 的 `warm_speedup` 和
显存字段为 `null`，不能据此判断 cuDNN 收益。固定合成帧和单一带长度
只能用于初筛，GPU 收益、真实视频质量及三并发容量仍需分别实测。

本地真实权重 CPU 冒烟（3 帧、预热 1 轮、正式 2 轮、每模式 1 个进程）
输出最大/平均像素差均为 0，精度回退为 0。新增测试 `7 passed, 1 skipped`，
全量回归 `510 passed, 25 skipped`；新增 CUDA 显存采集测试因无 GPU 跳过。
