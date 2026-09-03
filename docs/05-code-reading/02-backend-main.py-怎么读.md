# backend/main.py 怎么读

这篇讲项目最核心的后端文件。

## 这个文件为什么重要

`backend/main.py` 定义了 `SubtitleRemover`。

它不是某一个具体模型，而是负责准备输入、选择处理路线、写出视频和合并音频的总控对象。

## 先看 `__init__()`

初始化时，它会准备：

- 输入视频或图片路径。
- OpenCV 读取器、尺寸、帧数和 FPS。
- 临时视频文件和视频写入器。
- 模型路径、硬件设备和进度状态。
- 默认输出路径。

视频写入器优先使用 FFmpeg 的 `libx264`，失败时回退到 OpenCV `mp4v`。

## 再看 `run()`

`run()` 是主流程入口，大致做这些事：

1. 如果没有选区，就把整个画面作为处理区域。
2. 创建进度条并记录模型信息。
3. 图片走单帧 LAMA 路线，视频走选定的模式。
4. 释放读取器和写入器。
5. 视频处理完成后合并原音频。
6. 更新完成状态并清理临时文件。

## 模式选择

模式选择代码在 `run()` 中：

```text
PROPAINTER -> propainter_mode()
STTN_AUTO  -> sttn_auto_mode()
STTN_DET   -> video_inpaint(sttn_det_inpaint)
LAMA       -> video_inpaint(lama_inpaint)
OPENCV     -> video_inpaint(OpenCVInpaint())
```

这张分发关系是读懂后端的关键。

## `video_inpaint()` 是什么

这是“先检测，再修复”的通用视频流程，供 STTN_DET、LAMA 和 OpenCV 使用。

它会：

1. 创建 `SubtitleDetect`。
2. 按 FPS 抽样检测文字框。
3. 插值、统一文字框并合并字幕时间段。
4. 将时间段前后扩展少量帧。
5. 收集该段出现过的遮罩区域。
6. 分批读取帧并调用传入的模型。
7. 把修复结果写入视频。

没有进入字幕时间段的帧会直接写回，不经过修复模型。

## `sttn_auto_mode()` 是什么

它不调用 OCR，而是把指定选区直接变成遮罩。

之后 `STTNAutoInpaint` 按视频块读取帧，利用相邻帧和参考帧修复遮罩区域。

这里的 “auto” 指自动进行视频补全，不是自动识别字幕。

## `propainter_mode()` 是什么

ProPainter 有单独的处理流程。

它会先检测文字时间段，再用场景检测结果切开镜头切换，最后按批次调用 ProPainter。

ProPainter 内部还会计算光流、补全光流并进行视频传播，因此显存和时间开销都较高。

## `merge_audio_to_video()` 是什么

它使用 FFmpeg 从原视频提取音频，再将音频流复制到临时画面视频中。

画面会重新编码，音频通常不重新编码。

## 图片分支

图片没有视频时间轴。

`run()` 会先调用一次文本检测，然后创建遮罩并用 LAMA 修复。作为服务调用时，如果输入是单张图片，通常也会默认使用 LAMA 模式。

## 命令行入口

文件最后的 `if __name__ == '__main__':` 负责：

1. 解析 `-i`、`-o`、`-c` 和 `--inpaint-mode`。
2. 创建 `SubtitleRemover`。
3. 设置区域、输出路径和修复模式。
4. 调用 `run()`。

当前文档和代码需要注意：CLI 的 `-o` 虽然在参数解析中标为可选，但不传时会把默认输出路径覆盖成 `None`，实际使用时建议显式指定 `-o`。

## 读这个文件的顺序

建议按下面顺序阅读：

```text
__init__()
  -> run()
      -> sttn_auto_mode()
      -> video_inpaint()
      -> propainter_mode()
      -> merge_audio_to_video()
```

## 你现在只需要记住

`SubtitleRemover` 是流程总控，不等于某一个修复模型。

模型的具体细节在 `backend/inpaint/`，检测和遮罩细节在 `backend/tools/`。
