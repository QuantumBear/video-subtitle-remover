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

本地 macOS x86_64、PyTorch 2.2.2、NumPy 1.26.4、OpenCV 4.11.0，PyTorch 1 线程、OpenCV 8 线程，50 帧、预热 3 轮、测量 9 轮：旧/新 CPU 转换中位数为 `0.4415 s / 0.0522 s`，预处理张量逐值一致。这是本地 CPU 小基准，不能外推成目标 GPU 的整带加速比；目标机器整带耗时、显存和真实视频输出仍待验证。

本地全量测试结果为 `456 passed, 19 skipped`。新增回归覆盖全部 uint8 值、RGB 通道顺序、两种 mask 阈值、单帧/多帧、非连续/反向视图、输入保护，以及真实八层小网络在 QKV 开/关时整带输出完全一致。另用实际权重完成 2 帧 CPU A/B 冒烟检查，修复带输出最大/平均像素差均为 0；CUDA 与需要外部视频素材的测试按环境跳过。
