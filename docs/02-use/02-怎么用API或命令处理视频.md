# 怎么用 API 或命令处理视频

这篇讲不用界面，直接用命令或服务调用来处理视频。

## 生产流水线：动态全屏检测

`vsr_pipeline.py` 不传 `-c` 时检测整个画面，适用于字幕在多个位置出现的视频：

```bash
python vsr_pipeline.py \
  -i TikSave.io_7635080993354878239.mp4 \
  -o test_auto_dynamic.mp4 \
  --inpaint-mode propainter
```

OCR 按反馈调整采样间隔：开始逐帧检测；连续两次稳定后，间隔按
`1 → 2 → 4 → 5` 帧增大。字幕框的数量、位置、尺寸或框内画面变化时，
立即回到逐帧检测，并补查刚跳过的帧；每帧还用低分辨率画面检查场景切换。
变化持续时会继续逐帧检测，稳定后再拉长间隔。

可调参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--ocr-stride` | `5` | 稳定时的最大采样间隔；`1` 为逐帧 |
| `--ocr-refine-radius` | `15` | 检测变化后向前补查的最大帧数 |
| `--vlm-max-calls` | `32` | 单视频贴纸定位请求上限，失败也计数 |
| `--no-locate-stickers` | 关闭时不请求 VLM | 跳过贴纸定位 |
| `--no-white-glyph-check` | 关闭时不补擦 | 跳过白字残留复核，适合彩色字幕 |

`--ocr-stride` 越大，稳定片段调用越少，但短暂出现又消失的字幕可能落在
两次检测之间；低分辨率场景检查也不能保证捕获这类变化。默认上限 5 帧。
实际 OCR 次数取决于变化程度，持续变化时可能接近总帧数。

已知固定区域时，可用手动 ROI 限定检测和遮罩范围：

```bash
python vsr_pipeline.py -i TikSave.io_7635080993354878239.mp4 \
  -o test_manual_roi.mp4 --inpaint-mode propainter -c 450 1010 0 720
```

贴纸定位使用 `DASHSCOPE_API_KEY`，未配置或请求失败时继续处理 OCR 字幕。
ProPainter 使用最多 60 帧的输入窗口；白字幕使用字形核心和窄灰边遮罩，
局部残留最多追加一次修复，失败保留首轮结果。完整 GPU 画质和显存需要
在目标服务器验证；CPU 可用 `--inpaint-mode lama`。

下文的 `backend/main.py` 是另一套后端入口，其模式和参数与生产流水线不同。

## 命令行入口

入口文件是：

```text
backend/main.py
```

基本命令：

```bash
python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4
```

意思是：

- `-i test/test.mp4`：输入视频。
- `-o test/test_no_sub.mp4`：输出视频。

当前实现建议显式指定 `-o`。虽然参数帮助把输出路径标为可选，但未传入时可能覆盖后端默认输出路径，导致路径处理异常。

## 指定字幕区域

可以用 `-c` 指定处理区域。这个区域格式是原视频中的像素坐标：

```bash
python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4 -c 600 700 100 1200
```

区域格式是：

```text
ymin ymax xmin xmax
```

也就是：

- 上边界
- 下边界
- 左边界
- 右边界

如果不传 `-c`，后端会使用全屏区域。对于检测模式，这意味着选区内的其他文字也可能被处理。

## 指定修复模式

可以用 `--inpaint-mode` 指定模式：

```bash
python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4 --inpaint-mode sttn-auto
```

可选值：

- `sttn-auto`
- `sttn-det`
- `lama`
- `propainter`
- `opencv`

`sttn-auto` 不做 OCR，而是直接修复指定区域；其他模式通常会先检测文字框，再执行修复。

## 参数从哪里解析

命令行参数由这个文件处理：

```text
backend/tools/args_handler.py
```

它负责把你输入的命令变成程序能用的数据。

## 命令最后流向哪里

简单路线：

```text
backend/tools/args_handler.py
        |
        v
backend/main.py
        |
        v
SubtitleRemover.run()
```

## 你现在只需要记住

命令行和 API 调用最后都会走到 `SubtitleRemover`。

命令行是手动输入参数，API 调用是发送结构化请求，但核心处理流程相同。
