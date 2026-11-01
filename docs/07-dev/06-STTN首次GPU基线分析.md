# STTN 首次 GPU 基线分析

数据来源：用户在当前测试机器执行命令后提供的输出；本地代理未远程执行。对应[自建 Serving 评估与优化路线](05-STTN自建Serving评估与优化路线.md)的首次性能实验。

## 1. 环境与测试范围

- GPU 名称：`NVIDIA RTX5880-Ada-24Q`。
- PyTorch：`2.11.0+cu128`。
- 用户确认目标就是本次测试机器，先前“5090”记忆不准确，后续不按 5090 推断性能。
- 设备名称提示可能是 vGPU 配置，但物理卡型号、共享或调度配额、驱动和 CPU 资源尚未核实，不能仅凭名称确定。
- 标准模型输入 `432×240`，50 帧固定合成修复带，10 个窗口，Transformer 输入 140 帧次、Decoder 输出 104 帧次。
- 两次运行均为预热 3 轮、测量 5 轮，分别开启和关闭 profile，交替对比首层 Q/K/V 缓存。
- 脚本默认 `--threads 1`，只固定了 PyTorch CPU 线程数，不能由此推断所有图像处理库均为单线程。
- 开启 profile 的全部测量均为 FP16，`precision_fallbacks=0`。

计时不含模型初始化、OCR、贴纸定位、视频解码编码或 ProPainter 补修，也不含完整修复带 ROI 往返缩放。不能将本结果直接作为业务端到端吞吐。

## 2. 命令与结果

```bash
python benchmarks/sttn_qkv_cache.py --device cuda --frames 50 --warmup 3 --repeats 5 --profile
python benchmarks/sttn_qkv_cache.py --device cuda --frames 50 --warmup 3 --repeats 5
```

关闭 profile 的五轮耗时，单位秒：

| 轮次 | cache off | cache on |
| --- | ---: | ---: |
| 1 | 1.187 | 1.465 |
| 2 | 1.253 | 1.201 |
| 3 | 1.170 | 1.208 |
| 4 | 1.211 | 1.184 |
| 5 | 1.214 | 1.223 |

脚本根据未舍入时间计算的汇总：

| 指标 | profile off | profile on |
| --- | ---: | ---: |
| cache off 耗时中位数 | 1.210894 s | 1.217810 s |
| cache on 耗时中位数 | 1.208386 s | 1.212188 s |
| 开缓存加速比 | 1.002075× | 1.004638× |
| 时间节省 | 0.2071% | 0.4617% |
| cache off 峰值 allocated | 1.184193 GiB | 1.184193 GiB |
| cache on 峰值 allocated | 1.649239 GiB | 1.649239 GiB |
| 最大像素差 | 0 | 0 |
| 平均像素差 | 0 | 0 |

最终速度判断采用 profile off 数据。两种模式像素零差只适用于本次合成输入，不代表所有视频或其他后端配置均零差。

## 3. Q/K/V 缓存结论

关闭 profile 时，缓存的耗时中位数仅减少约 2.51 ms，低于本次运行中几十至数百毫秒的波动，不能视为已经证实的提速。

缓存使峰值 allocated 增加约 `0.465 GiB`（`476 MiB`），相对增幅约 39.3%。这是当前 PyTorch 测量到的 allocated 峰值，不是整卡显存、reserved 或其他模型的显存预算。

开启 profile 时，按每轮相加后再取中位数：

| 范围 | cache off | cache on |
| --- | ---: | ---: |
| `qkv + gather + transformer` | 0.657 s | 0.666 s |

缓存构建和窗口切片开销基本抵消了省下的首层投影工作。只比较 `transformer` 字段会遗漏这部分成本。

当前不值得继续扩大首层缓存投入。可将关闭缓存作为后续实验对照，以保留显存余量；本次未修改其默认值。

## 4. 当前耗时分布

以下为 cache on 五轮 profile 字段各自的中位数，比例以 wall 中位数约 1.212 s 粗略计算：

| 阶段 | 中位数 | 占 wall 的粗略比例 |
| --- | ---: | ---: |
| Transformer | 0.643 s | 53% |
| CPU preprocess | 0.211 s | 17% |
| CPU postprocess | 0.157 s | 13% |
| Decoder | 0.146 s | 12% |
| Gather | 0.020 s | 约 2% |
| Encoder | 0.014 s | 约 1% |
| Upload | 0.010 s | 约 1% |
| Download | 0.007 s | 小于 1% |
| 首层 QKV 缓存构建 | 0.005 s | 小于 1% |

各阶段独立取中位数，不能要求总和等于某一轮 wall。CPU/GPU 工作也可能重叠；这些比例用于定位，不是严格可相加的算子执行份额。

GPU 字段采用 CUDA Event 区间，可能包含 CPU 提交 kernel 的间隙。Decoder 区间含有限值检查造成的等待，Transformer 字段尚未细分卷积、attention、mask 和布局转换，不能据此断言某个算子占主导。

当前 download 约 7 ms，不能声称 PCIe 下载带宽是主要瓶颈。窗口间同步是否存在额外空隙仍需时间线或对照实验确认。

按 cache on 稳态中位数计算，合成修复带约为 `41.4 帧/秒`；按 30 fps 算，相当于该孤立输入的 `1.38 倍实时`。这不是全分辨率视频去字幕的实时倍速，也不是当前机器的业务容量承诺。

## 5. 下一项实验建议

这组数据使优化重点收敛到 Transformer 执行及 CPU 前后处理，而非继续扩大 QKV 缓存。

建议第一项保持计算语义的优化是 **STTN CPU 预处理的批量张量化**：当前路径逐帧执行 BGR/RGB 转换、NumPy/PIL 包装、堆叠、连续化和浮点归一化，存在多次复制。探索直接批量构造张量，消除不必要的 PIL 往返；固定颜色、mask 阈值、数值类型、归一化顺序和输入不被修改的约定。

验证应包含：

1. 帧和 mask 张量与旧路径的等价性、输入保护和精度回退回归。
2. 同一真实权重、合成输入、线程数和缓存设置下的 A/B，对比 preprocess、总 wall 和显存。
3. 真实视频输出差异和完整流程耗时。

这项优化的收益不能直接按 CPU 前处理的 17% 宣称；仍需实测。之后分别验证删除无用 `mm`、子图编译和 GPU 结果合成，一次改变一个变量。

保持现有模型和窗口不变。不要基于本次缓存测试直接调整模型层数、参考帧、默认 batch 或启动多个竞争 GPU 的进程。

基础硬件信息可用 `nvidia-smi` 和 `nvidia-smi -q` 补齐；即使短期无法得知完整虚拟化配额，也可以在当前机器上做受控 A/B，只需固定负载和配置，不将结论外推为物理卡极限。

## 6. 已实现：CPU 预处理批量张量化

`STTNDetInpaint` 已改为直接将 uint8 BGR 帧写入连续 RGB/TCHW 缓冲区，移除逐帧 PIL 往返和冗余复制；浮点转换后在新建张量上按原顺序执行 `/255`、`*2`、`-1`。mask 也批量构造，仍分别保留模型 mask 归一化后 `>0.5`、合成 mask 原量级 `>0.5` 的语义。调用方输入不被修改，FP16 整带回退路径保持不变。

本次只改 STTN DET 预处理，未调整模型、窗口、参考帧、精度策略或 QKV 默认值，也未改 LAMA。

新增 `benchmarks/sttn_preprocess.py`，冻结旧转换实现，在同一进程、同一引擎和相同输入下交替对比旧/新转换。默认两条路径都开启 QKV 缓存，与当前引擎默认保持一致；可用 `--cache off` 固定关闭，但不要把不同缓存配置的数据混作本次 A/B。

在目标机器激活现有虚拟环境后运行：

```bash
# 先看分阶段耗时，确认没有精度回退。
python benchmarks/sttn_preprocess.py --device cuda --frames 50 --warmup 3 --repeats 5 --profile

# 用关闭 profile 的整带 wall 判断最终收益。
python benchmarks/sttn_preprocess.py --device cuda --frames 50 --warmup 3 --repeats 5

# 可选：只测 CPU 转换，不加载模型。
python benchmarks/sttn_preprocess.py --preprocess-only --frames 50 --warmup 3 --repeats 9
```

脚本先检查预处理逐值完全一致，若不一致则终止；整带测量会检查中途精度回退并拒绝将其混入结果，同时报告修复带输出像素差、峰值 GPU allocated、软件版本和线程配置。两条路径共用模型、窗口和 QKV 配置，交替测量，使用中位数。`scope=preprocess_only` 只表示转换耗时，此模式没有修复带输出像素差或 GPU 显存结果。

本地 macOS x86_64、PyTorch 2.2.2、NumPy 1.26.4、OpenCV 4.11.0，PyTorch 1 线程、OpenCV 8 线程，50 帧、预热 3 轮、测量 9 轮：旧/新 CPU 转换中位数为 `0.4415 s / 0.0522 s`，预处理张量逐值一致。这是本地 CPU 小基准，不能外推成目标 GPU 的整带加速比；后续目标机器 A/B 见下一节，真实视频输出仍待验证。

本地全量测试结果为 `456 passed, 19 skipped`。新增回归覆盖全部 uint8 值、RGB 通道顺序、两种 mask 阈值、单帧/多帧、非连续/反向视图、输入保护，以及真实八层小网络在 QKV 开/关时整带输出完全一致。另用实际权重完成 2 帧 CPU A/B 冒烟检查，修复带输出最大/平均像素差均为 0；CUDA 与需要外部视频素材的测试按环境跳过。

## 7. 目标 GPU 实测：批量预处理 A/B

数据来源：用户执行第 6 节两条整带命令后提供的输出。GPU 仍为 `NVIDIA RTX5880-Ada-24Q`，PyTorch `2.11.0+cu128`，NumPy `2.2.5`，OpenCV `4.10.0`。PyTorch 线程数为 1、OpenCV 配置线程数为 64；配置值不能直接当作实际活跃线程数或过度订阅的证据。

50 帧、预热 3 轮、测量 5 轮；旧/新路径均开启 QKV 缓存，均为 FP16 且 `precision_fallbacks=0`，`preprocess_exact_match=true`。

| 指标 | profile off | profile on |
| --- | ---: | ---: |
| 旧预处理整带耗时中位数 | 1.272387 s | 1.279378 s |
| 批量预处理整带耗时中位数 | 1.072662 s | 1.071274 s |
| 加速比 | 1.186196× | 1.194258× |
| 时间节省 | 15.6969% | 16.2660% |
| 最大/平均输出像素差 | 0 / 0 | 0 / 0 |
| 旧/新 GPU 峰值 allocated | 1.648881 / 1.648881 GiB | 1.648881 / 1.648881 GiB |

关闭 profile 的逐轮耗时，单位秒：

| 轮次 | 旧预处理 | 批量预处理 |
| --- | ---: | ---: |
| 1 | 1.447536 | 1.086259 |
| 2 | 1.386628 | 1.057157 |
| 3 | 1.209848 | 1.072662 |
| 4 | 1.272387 | 1.044269 |
| 5 | 1.212105 | 1.198129 |

以 profile off 中位数判断，本次合成修复带少用约 `199.7 ms`，耗时减少约 `15.7%`，对应吞吐约提高 `18.6%`（`39.30 → 46.61` 修复带帧/秒）。五个配对轮次方向均有利于批量路径，但单轮差距仍有明显波动；该结果支持保留这项优化，不是业务视频吞吐或整卡极限。

用户粘贴的 profile 文本有截断/拼接，批量路径第 1、2 轮阶段明细不完整，因此不计算该路径五轮阶段中位数。可见的完整行中，旧预处理为 `0.215–0.260 s`，批量路径为 `0.045–0.052 s`；Transformer 仍约 `0.64 s`，表明主要收益与预处理下降相符。profile 汇总与关闭 profile 的结果方向、量级接近。

下一步使用固定真实视频验证完整流水线：沿用同一视频及既有 ROI、OCR、贴纸和补修参数，只追加 `--sttn-profile` 并更换输出文件名，记录 `[sttn-profile-total]`、`[done]` 和最终统计；定位后关闭 profile 再测。`sttn_wall` 包含 ROI/修复带往返处理但不含其他流水线阶段，`[done]` 不包含 `Pipeline` 初始化及此前的自动 ROI 推断；需要冷启动总耗时则在 CLI 外层另计时。

真实视频中重点检查字幕残留、颜色、闪烁、边缘接缝、非遮罩区域及音画时长。若比较优化前后完整视频，必须固定同一输入和处理参数，并关注检测结果是否一致；不能用一条旧历史总耗时直接归因于本次修改。当前 GPU 合成 A/B 已足以验证预处理的局部收益，暂不需要重复 QKV 缓存实验。

## 8. 首次真实视频：STTN + GDINO + ProPainter 补修

用户运行的命令如下；本地未远程执行，成片尚未人工验收：

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_sttn_pipeline_gdino_13.mp4 \
  --inpaint-mode sttn \
  --sticker-backend gdino \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-score 0.22 --sticker-max-area-px 1000 \
  --sttn-residual-propaint \
  --sttn-residual-propaint-max-windows 8 \
  --sttn-profile
```

输入 687 帧，自动 ROI 为 `(443,1052,0,624)`。OCR 调用 452 次、文字轨迹 39 条；GDINO 共检测 138 帧（priority 79、feedback 59），关联到 212 帧，无失败或 batch 回退。STTN 各修复带均为 FP16、回退次数为 0。

### 整片计时

| 范围 | 耗时 | 占记录的总时间 |
| --- | ---: | ---: |
| STTN 引擎调用合计（含 ROI/修复带往返） | 33.367 s | 5.54% |
| ProPainter 补修窗口合计 | 340.908 s | 56.64% |
| 尚未细分的其余时间（相减所得） | 227.616 s | 37.82% |
| `[done]` / `seconds` | 601.892 s | 100% |

`seconds` 从自动 ROI 推断之后、OCR 时间线开始之前计时，包含后续检测与修复流程，不含 `Pipeline` 初始化和自动 ROI 推断。STTN 计时不包含模型初始化、残留检测和段末 `gc.collect()` / `empty_cache()`；补修窗口计时也不含 ProPainter 首次模型加载。其余时间还包含检测、残留分析、视频读写、Top-K 补修所需的第二遍编解码及音频合并，现有日志无法拆分，不能全部归为 OCR 或 GDINO。

在其他阶段工作量和耗时固定的简化条件下，即便将 STTN 引擎阶段提速 2 倍，整片也只少约 `16.7 s`，总耗时降幅约 `2.77%`；该阶段耗时占比约 `5.54%`，因此无法仅靠 STTN kernel 优化解释或解决这条混合命令的全部时延。

### STTN 内部

18 段共生成 28 条修复带，累计输入 1,145 修复带帧次、234 个窗口、Transformer 输入 3,159 帧次、Decoder 输出 2,381 帧次。帧次含不同修复带对同一视频帧的重复处理，不能当作独立视频帧计数。

对日志中保留三位小数的阶段数求和：

| 范围 | 累计耗时 |
| --- | ---: |
| 修复带内部 wall | 约 26.749 s |
| Transformer | 约 15.015 s |
| CPU postprocess | 约 5.098 s |
| Decoder | 约 3.513 s |
| CPU preprocess | 约 1.402 s |
| 引擎调用合计减修复带 wall（ROI、缩放、拷贝、贴回等外围工作） | 约 6.618 s |

GPU 字段仍为 CUDA Event 区间，不能视为纯 kernel 时间或与 CPU 阶段严格可加。最后 3 个 50 帧长段各生成 3 条修复带，单段引擎调用约 `4.1–4.3 s`；多条修复带是实际工作量，不能直接删带或扩大输入凑成单带而假定画质不变。

部分短段/新形状出现单阶段异常偏长（例如 4 帧段 Transformer `0.366 s`、2 帧段 Decoder `0.170 s`），仅凭区间日志无法区分首次形状执行、分配、主机提交和设备调度等因素，暂不归因。预处理也有波动，不能将孤立高值直接视为本次批量实现回退。

### 补修量与质量口径

检测到 333 个疑似残留帧、15 个候选窗口，按严重度选中 8 个窗口；ProPainter 实际输入共 332 帧、其中残留核心帧 286 帧，约占全片 41.6%。单窗 19–50 帧，单次耗时约 15.8–52.6 秒。

当前只设置了窗口数量上限，`sttn_residual_propainter_max_frames=0`，因此单窗帧数仍不限。332 帧不能理解为 332 个经人工确认必须补修的帧。ProPainter 峰值 allocated/reserved 为 `14.211 / 21.514 GiB`，明显高于 STTN 阶段的 `1.77 / 4.98 GiB`；本次没有 OOM 证据。

最终 `unresolved=184` 是启发式结构判据标记的疑似残留帧数，不是人工验收失败数，也不能直接用来计算擦净率。`residual_check_enabled=False` 对应独立的 `white_glyph_check` 开关，STTN 自带残留探测和 ProPainter 转交在本次确实已执行。`repaired=0` 对应另一套补擦计数，不代表 8 个 ProPainter 窗口没有运行。

### 下一项受控实验

先以同一素材和检测参数运行纯 STTN + GDINO，对比本次混合成片的真实质量和耗时，并追加已有 `--sticker-profile` 定位 GDINO 耗时。它用于判断补修带来的质量收益与时间成本，不是默认关闭生产补修的决定：

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_sttn_pipeline_gdino_14_sttn_only_profile.mp4 \
  --inpaint-mode sttn \
  --sticker-backend gdino \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-score 0.22 --sticker-max-area-px 1000 \
  --sttn-profile --sticker-profile
```

重点观察原补修区间 `29–58`、`96–145`、`146–164`、`219–257`、`345–394`、`395–438`、`563–612`、`613–662`。补修只写回其中残留核心帧，区间还可能含上下文。相较本次命令，去掉的是两项 ProPainter 参数，新增 GDINO 计时会引入测量开销；最终性能仍需在关闭 profile 后复测。

不能直接把 `601.892 - 340.908` 当作纯 STTN 的预期耗时，因为关闭补修同时改变残留分析次数及第二遍视频读写工作量。若人工对照确认补修有明确价值，再单独测试已有 `--sttn-residual-propaint-max-frames 30` 预算；该开关会裁短候选并可能改变 Top-K 排名，属于覆盖范围与质量的权衡，不能当作输出等价加速。

继续优化 STTN 时，优先分别验证 Transformer 无用计算、CPU 合成和外围 ROI 拷贝；现有 GPU 日志无法进一步确定卷积/attention 的实际份额。不要为此改变现有 LAMA 设备约定。

## 9. 纯 STTN + GDINO：基线与画质反馈

用户完成第 8 节受控实验，实际输出文件为 `test_sttn_pipeline_gdino_14.mp4`，命令为：

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_sttn_pipeline_gdino_14.mp4 \
  --inpaint-mode sttn \
  --sticker-backend gdino \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-score 0.22 --sticker-max-area-px 1000 \
  --sttn-profile --sticker-profile
```

用户反馈：“看起来不用 ProPainter 补修，效果也可以接受”。因此后续效率工作以这条纯 STTN + GDINO 路径为当前样片基线，暂不要求 ProPainter 常规补修。这是该样片的人工判断，不代表其他素材已经通过验收，也不代表已经迁移生产。

### 路径对比

| 指标 | STTN + GDINO + ProPainter | 纯 STTN + GDINO |
| --- | ---: | ---: |
| `[done]` / `seconds` | 601.891558 s | 115.608768 s |
| STTN 引擎调用合计 | 33.367 s | 31.035 s |
| ProPainter 窗口合计 | 340.908 s | 0 s |
| ProPainter 调用 / 输入帧次 | 8 / 332 | 0 / 0 |
| 修复帧数 | 651 | 651 |
| 启发式疑似残留帧数 | 184 | 333 |

本次计时约快 `5.206×`，下降 `80.79%`，少用约 `486.283 s`。两次自动 ROI、OCR 452 次、文字轨迹 39 条、GDINO 138 次检测及关联 212 帧均一致；STTN 均为 FP16、无精度回退，均执行 18 段、28 条修复带。

差额不能全部归因于 ProPainter 模型推理：关闭 Top-K 补修同时去掉无损中间视频、第二遍读取和编码、额外残留处理及模型加载等工作；两次还存在运行波动和 profile 配置差异。也不能把 `5.206×` 归为批量预处理收益，两次均已使用新预处理。

疑似残留数是启发式统计，与用户接受画质不矛盾；不能据此认定 333 帧均有可见缺陷，或认定补修真实修好了其中 149 帧。关闭补修时，`sttn_residual_frames=0` 表示没有进入补修候选统计，不能解读为没有疑似残留。

### 纯路径耗时分布

| 范围 | 耗时 | 占本次记录总时间 |
| --- | ---: | ---: |
| STTN 引擎调用合计（含 ROI/修复带往返） | 31.035 s | 26.84% |
| GDINO 跟踪流程 | 26.181 s | 22.65% |
| GDINO processor 与模型初始化 | 3.300 s | 2.85% |
| 尚未细分的其余时间（相减所得） | 55.093 s | 47.65% |
| `[done]` / `seconds` | 115.609 s | 100% |

GDINO 模型调用的 `wall=18.610 s` 已包含在跟踪流程 `26.181 s` 内，不能重复相加。其内部为预处理 `3.720 s`、上传 `0.321 s`、推理 `14.274 s`、后处理 `0.286 s`、过滤 `0.004 s`，138 帧共 79 个 batch，文本只分词 1 次，无精度或 batch 回退。跟踪流程包含优先采样、扫描和复核三遍工作，外观核验 `0.673 s` 同样是内部计时。

剩余约 55 秒不能全部算作 OCR；它还含 STTN 模型加载、残留诊断、清理、视频读写、音频合并等尚未单独测量的部分。总时间沿用第 8 节口径，不含 `Pipeline` 初始化及自动 ROI 推断，因此也不是 CLI 冷启动全耗时。按 687 帧相除约为 `5.94 fps`，仅表示本次所计流程的平均处理速率，不是模型吞吐或服务容量。

### 下一步优先级

1. 保留该输出作为当前样片的质量与速度基线，后续比较固定输入、检测参数及模型配置，不再默认开启 ProPainter 补修。
2. 先补齐流水线计时，单独测 OCR 时间线、残留诊断、段末清理及视频读写等开销，定位约 55 秒。当前关闭补修后，`flush_sttn()` 仍对核心帧调用 `_residual_mask()`，只计数、不修改成片；若占比显著，可另行设计按需诊断，但不能凭源码猜测节省幅度，也不能直接删除统计或清理保护。
3. STTN 内继续按输出语义不变的方向优化 Transformer 无用计算、CPU 后处理和 ROI 往返；先测算子占比，再决定编译或自定义 kernel。即使整个 STTN 引擎阶段提速 2 倍，其他部分固定时，本次计时也只少约 `15.52 s`、下降 `13.42%`，因此同时需要关注外围工作。

本次仅记录和分析实测，没有新增 CLI 开关、修改检测或模型参数，也没有修改 LAMA 固定 CPU 的约定。后续正式比较性能时关闭两个 profile，并在更广的代表性素材上记录人工通过率。

## 10. 补充流水线阶段日志

用户随后要求进一步定位第 9 节尚未细分的约 55 秒。已沿用 `--sttn-profile` 增加 `[pipeline-profile-stage]` 与 `[pipeline-profile-total]`，单列 OCR 时间线、贴纸加载/检测/释放、STTN 加载与 mask 准备、残留诊断、ROI 合成、修复阶段视频读写与帧转换、GC/CUDA 缓存清理及合音频。新计时不额外同步 CUDA，不改变检测或修复参数。

汇总包含已分配时间 `accounted` 与剩余 `other`，嵌套阶段扣除子阶段，避免补修与视频写出重复累计。统计每条视频重置；启用时 Python 返回值增加 `pipeline_profile`。命令和字段口径见 [STTN 性能分析](03-STTN性能分析.md#流水线阶段)。目标 GPU 尚待用新日志复测，本次没有给约 55 秒分配推测结果，也没有删除残留诊断或清理。

本地全量回归为 `463 passed, 19 skipped`。受控时钟测试覆盖嵌套阶段互斥计时、解码取帧排除消费时间和跨视频重置；小视频验证启用/关闭日志的输出像素一致，并覆盖带音轨、无检测及即时/Top-K 补修路径。CUDA 相关验证仍受本地无 CUDA 环境限制。
