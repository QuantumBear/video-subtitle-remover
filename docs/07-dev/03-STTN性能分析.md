# STTN 性能分析

在原有 `vsr_pipeline.py` 命令后追加 `--sttn-profile` 即可；默认关闭。
例如：

```bash
python vsr_pipeline.py -i input.mp4 -o output.mp4 \
  --inpaint-mode sttn --sttn-profile \
  --sttn-residual-propaint --sttn-residual-propaint-max-windows 8
```

Python 调用可使用 `Pipeline(inpaint_mode='sttn', sttn_profile=True)`，
或直接构造 `STTNDetInpaint(device, model_path, profile=True)`。
该选项仅在 STTN 模式生效。

## 日志范围

- `[sttn-profile]`：一次缩放后修复带的 `inpaint()` 调用。一个视频段可能包含多个修复带，因此可能输出多行；各阶段累计该修复带所有窗口的耗时。
- `[sttn-profile-segment]`：一次完整 STTN 引擎调用，包含 ROI 裁剪、修复带缩放、推理、放大及 mask 合成。`segment` 是原视频帧号范围。不含模型加载、OCR、残留检测、ProPainter 和视频编码。
- `[sttn-profile-total]`：整条视频的 STTN 引擎调用 wall time 之和，以及已有的 ProPainter fallback wall time。每条视频重新统计，不等于流水线总耗时；完整耗时仍看 `[done]`。

所有耗时单位为秒。引擎的 `last_profile` 保存最后一次修复带的数字统计，不累加到下一次调用。

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
