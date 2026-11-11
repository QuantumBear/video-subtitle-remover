# SAM3 文字 mask 与 DINO emoji 补漏评估

日期：2026-11-11（DGX UTC）。GB10 / aarch64，DINO FP16，纯 STTN FP16。

**结论：值得保留为可选实验方向，但当前不能直接默认启用。** SAM3 负责文字，DINO 在由 SAM3 生成的字幕邻域内能重新发现漏掉的 emoji；采样优化后，22.9 秒样片的额外检测与跟踪为 20.20 秒，另有本次进程的模型加载 6.58 秒。完整实验也发现人物戒指被当成贴纸并擦掉，证明内容准入仍是明确阻碍。

本轮只增加实验脚本、产物和本报告，未修改生产源码、两个启动脚本或默认参数，未提交或推送。完整实验产物在本地 [output/sam3_dino_hybrid_20261111](../../output/sam3_dino_hybrid_20261111/)，DGX 对应 `quality_eval_20261109/sam3_dino_hybrid_20261111/`。

## 1. 固定已有文字 mask，拆开测试补漏和修复

输入为 `input1.mp4`，720×1280、30fps、687 帧、22.9 秒。输入 SHA256 为 `1ca7950176a859824c705bf1d6ab850efe8ee99872e4de7331991c9cd67f7170`。

复用此前完整逐帧 SAM3 候选：score=0.5，整帧为空时使用当前帧 score=0.3 候选，最后外扩 3px。DINO 不修改或替换这些文字像素，不重新引入 OCR。SAM3 缓存读取和调度框生成放在 DINO 计时之外；检测、跟踪和视频扫描计入增量耗时。

先统一检查 18 个固定帧，覆盖 tik、car、third、mop 四个来源，包括 emoji、门锁、通风口、设备标签、无字幕帧等。比较三条提示，不逐帧选优：

```text
legacy: emoji. sticker. cartoon icon. dog face icon.
generic: emoji. sticker. cartoon icon.
emoji: emoji.
```

DINO 使用已有本地 `grounding-dino-tiny` 权重，score=0.22、最大原图框面积 1200px²，保留原弱候选、外观核验和反馈补查。最大面积沿用旧实验配置，不能推广成任意分辨率和任意大小 emoji 的通用保证。

## 2. 全画幅直接恢复旧分支不可用

全画幅固定帧上，generic 在前三个 emoji 检查点 f0/f30/f50 均无强候选；legacy 只在 f50 找到两个。整片 generic 按原调度运行 128 次检测，耗时 31.70 秒，没有有效强命中或接受轨迹。

原始候选也出现了把一排三个 emoji 当成一个大框的情况，这种框被原面积上限过滤。降低阈值或增大面积上限会改变误检范围，不能当成没有副作用的修复。

这说明此前字幕 ROI 下的 DINO 召回与耗时都不能直接外推到当前 SAM3 全画幅流程。

## 3. SAM3 字幕邻域恢复小目标检测，但不是内容分类

独立裁剪探针按当前 SAM3 mask 外接矩形生成邻域，向四边扩展该矩形高度的一倍和两倍。所有样本用相同规则；没有样片坐标、颜色或狗表情模板。两倍版本为 emoji 边缘保留了更多空间，后续整片沿用它。

完整跟踪使用最终 SAM3 mask 的外接矩形，向四边扩展其高度的两倍，无 mask 时回退全屏。字幕文字仍保留原始 SAM3 像素轮廓；DINO 候选框才按旧配置外扩 12px，并加入逐帧实验 mask，重新生成调度框。

generic 在固定 f0/f30/f50 都能找到三个 emoji；这三张图证明输入尺度影响明显，不能据此宣称完整成片没有漏检。固定检查帧的 generic 没有新增门锁、通风口、设备标签等强候选，但整片随后发现了新的戒指误检。

此邻域本身会遗漏远离字幕的独立图形，不能作为全画面贴纸发现方案。字幕上下文也不能证明邻域内目标就是后期叠加内容。

## 4. 采样调度可减少开销

| 整片设置 | 检测帧次 | 优先 / 反馈 | 检测＋跟踪耗时 | 模型加载 |
|---|---:|---:|---:|---:|
| 全画幅 generic，旧组件调度 | 128 | 128 / 0 | 31.70s | 5.58s |
| 自动邻域 generic，旧组件调度 | 157 | 128 / 29 | 35.56s | 5.32s |
| 自动邻域 generic，时间与切镜调度 | 71 | 41 / 30 | 20.20s | 6.58s |

原调度用 SAM3 逐帧连通块框生成轨迹，细碎边界变化导致大量优先采样。实验调度采用每秒网格、切镜帧、开头两秒半秒网格及最后一帧，仍保留原反馈补查与最大 200 次预算。没有以手工 emoji 时间段限制采样。

三组均无推理错误、batch fallback 或精度回退。邻域整片运行峰值 PyTorch allocated 约 9.51GiB。它包含该进程的模型和临时张量，不是三模型同时常驻峰值，也不能用于推算服务 slots；异形裁剪混批的 padding 仍有优化空间。

这是单次顺序实验，包含视频多遍扫描和轻量记录；没有足够重复建立稳定时延分布。不能把 20.20 秒说成完整服务只慢这些时间：新 mask 还会影响后续修复带和计算。

## 5. 逐帧核对：补 emoji 有效，戒指误删使默认接入不合格

使用旧人工核验过的 9 条 emoji 轨迹、270 个框次作诊断参照，以 IoU≥0.6 匹配新框。它们实际来自同一排三个狗 emoji 跨三个镜头，不是 270 个独立目标或像素级真值。

- 旧组件调度匹配 269/270，f6 漏一个，f47 有一个重复框。
- 时间与切镜调度匹配 270/270，f0–89 每帧均恰有三个框。
- 两组均接受了 f665/667/668/669/670/671/675 的戒指候选，共 7 帧；其中有外观跟踪续接位置偏移。

人工放大原图确认，这些新候选是人物手上真实戒指，原 SAM3 mask 在候选框内为零。合并后新增约 1554–1763px/帧的 mask 支持，并实际把戒指擦掉。这里的面积只解释新增支持，不代表画质得分。

这一反例不能靠增加命中次数、保持外观一致、靠近文字或固定面积阈值自动解决。所有轨迹都应有内容证据入口；先前报告中的相对背景运动可以作为后续反证候选，但仍未证明足以保护静止物体、戒指等目标。本轮没有把未经验证的分类门槛写入生产。

## 6. 纯 STTN 成片验收与实际边界

冻结 SAM3 基线与优化采样 hybrid 均通过原生产纯 STTN 修复路径，ProPainter 加载被禁止。两段成片均完整解码 687 帧，音轨存在，ProPainter 调用为 0，精度回退为 0；所有 SAM3 mask 像素均保留。

| 冻结检测结果后的整片运行 | 耗时 | SHA256 |
|---|---:|---|
| SAM3 基线 | 49.02s | `481affc0398e825d297c3ced5561f90a3747246caab16e6dd4d4b8ce4b1c7881` |
| SAM3＋DINO，优化采样 | 48.43s | `38868ffb3462f29e4a732618c79cd35f767fff4ec9a54498f78c6459ff81c74b` |

基线 SHA 与用户反馈的 `output1.mp4` 一致。两组耗时包含冻结 mask 读取、调度与修复等，不含实际 SAM3 推理或 DINO；差值小且没有重复，不能称作修复提速。

关键帧原图／基线／hybrid 对照见 [final_compare.jpg](../../output/sam3_dino_hybrid_20261111/final_compare.jpg)。前三秒关键帧的 emoji 已被移除，但 f29/f30 等结构区域仍有重建涂抹，补漏不等于背景恢复完美。f668 对照清楚显示戒指被误擦。没有逐帧动态盲评，不能宣称整片稳定或已达到 Kaipai 画质。

DGX 实验视频：

```text
/home/vibbit/video-subtitle-remover/quality_eval_20261109/sam3_dino_hybrid_20261111/tik_baseline.mp4
/home/vibbit/video-subtitle-remover/quality_eval_20261109/sam3_dino_hybrid_20261111/tik_hybrid_uniform.mp4
```

## 7. 建议与复核入口

保留 SAM3 文字 mask；DINO 的补漏价值已经得到具体证据，但先不改两个 sh 的默认行为。接入前优先解决两点：所有轨迹的内容核验，以及少量帧的稳定采样。字幕邻域只是帮助检测小目标，不能兼任内容准入。

验证集需增加不同 emoji、远字幕图形、手部饰物、近字幕实体、实体印字和无字幕场景，并按原片划分。现有数据集中在少量来源，没有证据支持通用默认启用。

复跑使用 DGX 已有 SAM3 环境，不安装依赖，DINO 权重离线加载：

```bash
cd /home/vibbit/video-subtitle-remover
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
quality_eval_20261109/sam3_mask_compare/env/bin/python \
  quality_eval_20261109/sam3_dino_hybrid_20261111/probe.py \
  --phase tracking --prompt generic --nearby --crop-pad 2 --schedule uniform

quality_eval_20261109/sam3_mask_compare/env/bin/python \
  quality_eval_20261109/sam3_dino_hybrid_20261111/verify.py
```

`verification.json` 保存两片完整解码结果、音轨/零回退断言及脚本和视频 SHA。`tracking_nearby2_generic_uniform.json` 保存实际调度、候选、轨迹和计时；`probe.json` / `crop2.json` 保存固定帧原始检测；`coverage_check.json` 保存原组件调度的漏帧诊断。框和坐标只用于人工评估、可视化，不输入生产准入。
