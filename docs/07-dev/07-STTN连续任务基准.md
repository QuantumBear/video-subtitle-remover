# STTN 连续任务基准

用于回答当前 CLI 每片启动的成本中，有多少能通过复用进程及 Pipeline 摊薄。
固定纯 STTN + GDINO、跳过最终残留诊断，无 ProPainter；模型精度、检测参数、
遮罩、编码及清理行为沿用当前实现。工具不接入服务，也不改变模型生命周期。

## 运行

在仓库根目录及同一个 Python 环境执行：

```bash
python benchmarks/sttn_pipeline_reuse.py \
  -i TikSave.io_7635080993354878239.mp4 \
  --output-dir benchmark_sttn_reuse_01 \
  --repeats 3 \
  --device cuda \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-score 0.22 \
  --sticker-max-area-px 1000
```

`--output-dir` 必须是尚不存在的新目录，避免覆盖成片或已有测量。默认三种
模式均测，单个输入重复 3 次时共处理 9 条视频，全部顺序执行。按当前约
107 秒的样片处理时间，先为测量预留约 20 分钟及 9 份输出视频空间，实际
用时还包含 Pipeline 初始化、自动 ROI 推断和进程启动退出。

可用 `--modes fresh-process reuse-pipeline` 缩小比较范围。需要多片验证时，
`-i clip_a.mp4 clip_b.mp4` 按原列表顺序重复。线程默认沿用运行环境；明确设置
`--threads N` 时所有模式相同。工具固定开启两个 profile，以获得阶段明细，
这些数据用于定位成本，不作为关闭 profile 后的正式服务容量。

## 三种模式

| 模式 | 进程生命周期 | Pipeline 生命周期 | 能回答的问题 |
| --- | --- | --- | --- |
| `fresh-process` | 每条视频新进程 | 每条视频新建 | 当前每任务 CLI 模式的启动和首次导入成本 |
| `new-pipeline` | 所有视频共用进程 | 每条视频新建，随后释放并 GC/清理 CUDA 缓存 | 只复用 Python 进程的收益 |
| `reuse-pipeline` | 所有视频共用进程 | 同一个 Pipeline 依次处理 | 进一步复用 OCR/STTN 模型的收益和任务状态隔离 |

贴纸检测器目前在每片检测后释放，**三种模式都仍逐片加载 GDINO**。复用
Pipeline 不等于 GDINO 常驻；后续任务还会发生 GDINO 与常驻 STTN 的显存共存。
需要看实际显存记录，不能据单任务峰值直接推断三并发容量。此工具不实现并发。

## 输出与计时口径

每个会话有 `.log`、`.request.json`、`.json`，成片写到会话同名子目录。
独立进程每片一个会话，另外两种模式每种一个多任务会话。根目录的
`report.json` 汇总全部任务、环境和输出对照：

- `sessions[].process_wall_seconds`：父进程测量的子进程启动到退出时间，含首次
  模块导入、所有任务、任务间清理、进程退出及报告写出。
- `pipeline_module_import_seconds`：首次导入 `vsr_pipeline` 的时间。
- `jobs[].pipeline_init_seconds`：本次 Pipeline 构造时间，含 Paddle/OCR 初始化。
  复用时后续任务为 0。
- `jobs[].process_video_seconds`：整个 `process_video()` 的外层时间，含元数据
  读取、自动 ROI 推断、阶段日志与收尾。
- `jobs[].task_seconds`：该任务 Pipeline 初始化（若有）与 `process_video()`。
  不含进程启动、首次导入 `vsr_pipeline`、任务间清理和进程退出。
- `jobs[].stats.pipeline_profile`：原流水线分阶段计时，起点晚于自动 ROI 推断。
  它包含在 `process_video_seconds` 中，不能重复相加。
- `jobs[].gdino_import_seconds`：在原惰性加载位置显式执行 transformers 模块导入
  及 `AutoProcessor`、`AutoModelForZeroShotObjectDetection` 命名属性解析的时间。
  它包含在 `sticker_init` 内；未用贴纸检测器时为 0。此记录器只在基准生效，
  不提前预热。用于缩小原外层/内部计时差额的归因，不能当作全部可省时间。
- `jobs[].gdino_init`：原检测器内部 processor、模型加载及初始化时间和精度状态。
  与命名导入相互分开；其他外层工作仍可能存在。若初始化失败被流水线跳过，
  基准直接报错，避免把降级成无贴纸检测当成加速。
- `jobs[].gdino_tracking`：原跟踪器的检测、关联、逐帧失败及批量回退统计。
  检测异常被流水线捕获或出现 `model_failed > 0` 时，基准报错并保留会话日志，
  不将漏做检测算成有效测量；成功完成逐帧重试的 batch 回退仍保留记录。
- `jobs[].cuda_after`：该片结束后的 allocated/reserved/free/total，不是峰值。
  每段/检测阶段峰值仍看会话日志，STTN 的实际精度及回退累计值另见 `sttn_precision`。

`summary` 区分三种模式的任务中位数和完整进程摊销时间：
`task_median_seconds` 包含每个会话首条任务，`warm_task_median_seconds` 只取同一
进程的后续任务；每条都新进程时后者是 `null`。有多个输入时这些数是混合
素材的汇总，逐片对照应看 `jobs`。
`amortized_process_seconds_per_video` 将该模式全部进程时间除以处理视频数，
包括起止成本，能避免将首片初始化摊销漏掉。`median_seconds` 展示关键阶段的
中位数，首片/后续明细仍应单独比较。

所有测量完成后，工具流式读取成片，对同一输入比较解码 RGB 像素、帧数和呈现
时间的 SHA-256，记录在 `comparisons` 和 `outputs_match`。验收耗时不混入上面
的性能时间，比较不含音轨。若不一致，保存报告并以失败退出，应先定位差异
再判断复用是否可用。记录计数相同不能代替像素比较；像素通过也不能代替人工
画质验收。GPU 算子可能存在非确定性，不应为让基准通过擅改推理设置。

## 解释结果

先比较 `fresh-process` 与 `new-pipeline`，再看 `reuse-pipeline` 的首片和后续
`pipeline_init_seconds`、`gdino_import_seconds`、`sticker_init`、`sttn_init`。
模型逐片释放的加载成本应继续出现，首次模块导入通常只在每个进程首片出现。
不要把 `sticker_init` 的全部差额都视为模型加载，也不要将任何单次差额直接
视为稳定的吞吐收益。

默认顺序是先独立进程，再同进程新建，最后复用；OS 文件缓存、GPU 预热及负载
可能影响比较。首次运行后可用新输出目录和反向顺序复测：

```bash
python benchmarks/sttn_pipeline_reuse.py \
  -i TikSave.io_7635080993354878239.mp4 \
  --output-dir benchmark_sttn_reuse_02 \
  --repeats 3 --device cuda \
  --modes reuse-pipeline new-pipeline fresh-process
```

第二条命令省略的贴纸参数仍采用相同默认值：本地 tiny 权重、阈值 0.22、
面积 1000。真正冷启动（系统磁盘缓存清空）不在本基准范围内。多次结果用于
决定是否进入常驻 worker 及 GDINO 模型共存实验；本轮不修改现有 API 服务。

## 首轮目标 GPU 实测

用户在原 `RTX5880-Ada-24Q` 机器按本文首条命令完成测量，输出目录为
`benchmark_sttn_reuse_01`。同一 687 帧视频每模式重复 3 次，两个 profile 开启、
跳过残留诊断。以下数值来自用户提供的终端汇总，尚未读取完整 `report.json`
或会话日志；未据此核对逐任务显存、精度回退及全部检测计数。

| 指标 | 每片独立进程 | 同进程新建 Pipeline | 同进程复用 Pipeline |
| --- | ---: | ---: | ---: |
| 处理视频数 / 进程数 | 3 / 3 | 3 / 1 | 3 / 1 |
| 完整进程总耗时 | 393.560 s | 320.369 s | 316.877 s |
| 每片完整摊销 | 131.187 s | 106.790 s | 105.626 s |
| 任务中位数 | 123.300 s | 95.470 s | 93.896 s |
| 后续任务中位数（各 2 片） | 不适用 | 95.426 s | 93.567 s |

独立进程三次完整耗时分别为 `134.431 / 130.072 / 129.057 s`。同进程新建
Pipeline 相比独立进程每片少 `24.397 s`（`18.60%`）；进一步复用 Pipeline
本轮只再少 `1.164 s`（相对同进程新建少 `1.09%`）。两项合计每片少
`25.561 s`（`19.48%`），相同工作量按该摊销比值计算的顺序吞吐提高约
`24.20%`。这些是本轮测量比值，不是并发容量或长期稳定收益。

| 阶段中位数 | 每片独立进程 | 同进程新建 Pipeline | 同进程复用 Pipeline |
| --- | ---: | ---: | ---: |
| `pipeline_init_seconds` | 13.679 s | 0.123 s | 0 s |
| `process_video_seconds` | 109.621 s | 95.347 s | 93.896 s |
| `gdino_import_seconds` | 6.748 s | 0.000014 s | 0.000021 s |
| `sticker_init` | 10.077 s | 0.634 s | 0.642 s |
| `ocr_timeline` | 17.796 s | 14.956 s | 14.857 s |
| `sticker_detect` | 26.758 s | 25.652 s | 25.272 s |
| `sttn_init` | 0.903 s | 0.813 s | 0.000033 s |
| `sttn_infer` | 31.318 s | 31.360 s | 31.120 s |

主要收益已在只复用进程时出现：首次 Pipeline 构造包含 Paddle/OCR 的导入和
框架初始化，后续重新构造已很便宜；transformers 命名导入从约 `6.748 s`
降到微秒量级。GDINO 仍每片重新加载，后续 `sticker_init` 约 `0.64 s`，
不能解释为检测器常驻，也不能将它与独立进程的全部差额归为模型导入。
STTN 复用主要省去约 `0.8–0.9 s` 的引擎加载，实际推理仍约 `31 s`。
OCR/GDINO 执行阶段也有变化，受预热、系统缓存和测量顺序影响，不能把
全部差额归为生命周期变化；各阶段中位数也不能相加还原总耗时。

`outputs_match=true` 表示本轮 9 份输出的解码像素、帧数和呈现时间全部一致，
比较不含音轨。复用模式后续任务约 `93.567 s`，但完整三片会话摊销仍为
`105.626 s/片`，首次加载等成本尚在其中。此前 CLI 的 `107.064 s` 来自
自动 ROI 之后的流水线计时，未包含 Pipeline 初始化、进程启动和自动 ROI
推断，不能用它与本次 `131.187 s/片` 直接判断性能退化。

下一步优先实现常驻进程顺序接单、复用一个 Pipeline 的最小 worker，保留
GDINO 每片释放及原有清理策略，先在服务化入口获得本轮已测到的收益。
进一步让 GDINO 常驻的收益需另测，目前不能继续按冷启动的 `10 s` 估算。
原 API 服务尚未修改；不同素材连续任务、完整日志中的模型共存显存及三并发
吞吐仍待验证。必要时先按上文反向顺序复测，以降低本轮固定顺序的影响。

## 本地验证范围

本轮全量回归为 `481 passed, 19 skipped`，包含新增的 10 项基准测试，
覆盖流水线捕获检测异常和跟踪器逐帧失败时的拒绝行为。
另使用 3 帧、64×48 的无字幕小视频，在 CPU 上以每模式 2 条任务运行真实
子进程冒烟：三种模式全部完成，6 份输出的像素、帧数和呈现时间一致。
该样本只验证 OCR 初始化、编解码、进程组织和报告生成，没有执行真实
GDINO/STTN 推理，不能据此估算目标 GPU 的加速或显存收益。
