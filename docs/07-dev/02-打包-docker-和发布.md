# 打包、Docker 和发布

这篇讲项目怎么被部署成服务。

## Docker 是什么

Docker 可以理解成“把运行环境一起装进盒子”。

项目有 Dockerfile：

```text
docker/Dockerfile
```

它会：

- 使用 Python 3.12 镜像。
- 安装系统依赖。
- 复制项目代码。
- 根据参数安装 CUDA、DirectML 或 CPU 依赖。
- 设置默认启动命令。

## Docker 适合谁

适合：

- 想在服务器上跑的人。
- 想减少本机环境污染的人。
- 想把服务交给别的程序调用的人。

## 预构建包

README 里列了几种包：

- Windows CPU
- Windows DirectML
- Windows NVIDIA CUDA 11.8
- Windows NVIDIA CUDA 12.6
- Windows NVIDIA CUDA 12.8

区别主要是：

- Python 版本
- Paddle 版本
- Torch 版本
- 是否支持 GPU
- 支持哪类显卡

## 为什么有这么多版本

因为 AI 依赖和显卡环境强相关。

一个包不可能对所有电脑都最合适。

## 你现在只需要记住

源码运行、Docker、预构建包，最后都是运行同一套项目代码。

只是环境准备方式不同。服务部署优先考虑 Docker。
