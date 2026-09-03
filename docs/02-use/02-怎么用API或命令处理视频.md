# 怎么用 API 或命令处理视频

这篇讲不用界面，直接用命令或服务调用来处理视频。

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
