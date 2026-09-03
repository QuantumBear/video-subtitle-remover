# tools 工具代码怎么读

这篇讲 `backend/tools/`。

## tools 是什么

`backend/tools/` 是工具箱。

里面每个文件通常只负责一种辅助功能。

## args_handler.py

负责命令行参数。

例如：

```bash
python backend/main.py -i input.mp4 -o output.mp4
```

这里的 `-i` 和 `-o` 就由它解析。

## common_tools.py

通用工具。

它负责：

- 判断是不是视频文件。
- 判断是不是图片文件。
- 读取图片。
- 合并被拆分的大模型文件。
- 处理 Windows 路径兼容问题。

## subtitle_detect.py

字幕检测。

它负责：

- 调用 OCR。
- 找出文字框。
- 根据指定选区过滤文字框。
- 找连续出现字幕的帧区间。
- 合并相似字幕区域。

## ocr.py

OCR 结果处理。

它把 OCR 返回的多边形点，整理成项目使用的矩形坐标。

## inpaint_tools.py

修复辅助工具。

常见职责：

- 创建遮罩。
- 按批次切分帧。
- 处理 AB 区间。
- 扩展字幕帧范围。

## model_config.py

模型路径配置。

它告诉程序不同模型文件在哪里。

例如：

- LAMA 模型位置。
- STTN 模型位置。
- ProPainter 模型位置。
- OCR 检测模型位置。

## hardware_accelerator.py

硬件加速检测。

它判断当前机器能不能用：

- CUDA
- DirectML
- MPS
- ONNX Runtime Provider

## ffmpeg_cli.py

找到当前系统对应的 ffmpeg 文件。

Windows、Linux、macOS 路径不一样，所以这里统一处理。

## video_io.py

视频输入输出工具。

里面两个重要类：

- `FramePrefetcher`：后台提前读取视频帧。
- `FFmpegVideoWriter`：用 ffmpeg 写出视频。

## process_manager.py

管理正在运行的子进程。

服务收到取消请求或停止运行时，需要终止处理进程。

## version_service.py

检查版本更新。

它会访问项目发布信息。

## 你现在只需要记住

`backend/tools/` 不是主流程本身。

它是给主流程提供各种工具。
