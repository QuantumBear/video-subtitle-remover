# STTN 残留转交预算

`--sttn-residual-propaint-max-frames` 限制每次转交 ProPainter 的总输入帧数，
包含残留核心帧和上下文帧。默认 `0` 表示不限；启用时必须是至少 `2` 的整数，
因为光流模型需要帧对。仅设置限长参数不会开启转交，仍需加
`--sttn-residual-propaint`。

建议先固定全片最多 8 次转交、单次最多 30 帧，关闭小窗口过滤进行对照：

```bash
python vsr_pipeline.py -i TikSave.io_7635080993354878239.mp4 \
  -o test_sttn_pipeline_gdino_13.mp4 --inpaint-mode sttn \
  --sticker-backend gdino \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-score 0.22 --sticker-max-area-px 1000 \
  --sttn-residual-propaint \
  --sttn-residual-propaint-max-windows 8 \
  --sttn-residual-propaint-min-core-frames 0 \
  --sttn-residual-propaint-max-frames 30
```

库调用对应 `process_video(..., sttn_residual_propainter=True,
sttn_residual_propainter_max_frames=30)`。

## 窗口选择

每个候选窗口保留原有前后最多 5 帧上下文。候选超过上限时，在候选范围内
选择残留核心帧最多的连续子窗口；同分选最早位置。上下文也受上限约束，
截短后可能减少或完全没有上下文。不会把长窗口拆成多次调用；被截掉的帧
保留 STTN 结果，因此可能牺牲长残留段的修复覆盖率。

`max_windows` 为正数时，处理顺序为：

1. 按 `max-frames` 截短候选窗口。
2. 用截短后的核心帧数检查 `min-core-frames`，不足则跳过整个候选。
3. 收集整条视频的合格候选，按截短后的残留核心帧数从多到少排序。
4. 核心帧数相同时，按窗口残留像素总量从多到少排序；仍相同则选时间更早的窗口。
5. 从排序结果取前 `max-windows` 个，再按时间顺序执行 ProPainter；过滤掉的候选不消耗名额。

正数预算会先完成 STTN 第一遍，再对选中的窗口执行第二遍 ProPainter，因此后出现但
残留更严重的窗口也能获得预算。`max-windows=0` 仍按时间流式处理所有合格候选。

只有选中窗口里的残留核心帧写回 ProPainter 结果，其余帧保留 STTN 结果。
单帧窗口会复制为两帧供模型计算，输出仍只有原来的一帧。

## 如何验证

逐次日志会显示 `candidate_window`（原候选）、`window`（实际选择）、
`input_frames` 和 `max_frames`。上例每次 `input_frames` 应不超过 30，
总调用不超过 8 次，总输入不超过 240 帧。

最终统计新增以下字段；截短统计只计入实际转交的窗口，过滤或预算跳过的窗口不计入：

| 字段 | 含义 |
| --- | --- |
| `sttn_residual_propainter_max_frames` | 配置的单次输入帧数上限，0 为不限 |
| `sttn_propainter_windows_trimmed` | 实际转交中发生截短的窗口数 |
| `sttn_propainter_input_frames_trimmed` | 这些窗口累计减少的输入帧数，含上下文 |
| `sttn_propainter_core_frames_trimmed` | 这些窗口被截掉、保留 STTN 结果的核心帧数 |

既有 `sttn_propainter_frames` 及逐次 `input_frames` 包含单帧窗口的复制补齐帧，
`sttn_propainter_core_frames` 按原视频核心帧计数，不重复计算补齐帧。
小窗口过滤帧数按被跳过的原候选核心帧数计，包含其截短范围外的核心帧。

服务器对照时，除输出名与 `max-frames` 外保持参数一致，用 `0` 和 `30`
分别运行，比较转交输入帧数、转交耗时、总耗时和画面残留。减少输入量不保证
耗时或显存按比例下降，GPU 收益与画质需要实际视频验证。
