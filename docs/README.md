# VSR 项目小白阅读入口

这套文档的目标只有一个：让完全没有编程基础的人，也能按顺序慢慢看懂这个项目。

项目名字是 `video-subtitle-remover`，简称 VSR。它做的事情是：把视频或图片里的硬字幕、文字水印找出来，然后用算法把文字所在的画面补上。

## 推荐阅读顺序

请不要从代码开始看。先按下面顺序看文档。

1. `01-start/01-这是什么项目.md`
2. `01-start/02-最小知识包.md`
3. `01-start/03-术语词典.md`
4. `02-use/02-怎么用API或命令处理视频.md`
5. `02-use/03-输入输出和任务.md`
6. `03-project-map/01-项目文件夹地图.md`
7. `03-project-map/02-重要文件一览.md`
8. `04-core-flow/01-字幕去除总流程.md`
9. `04-core-flow/02-视频帧-字幕框-遮罩-修复-音频.md`
10. `05-code-reading/02-backend-main.py-怎么读.md`
11. `05-code-reading/04-tools-工具代码怎么读.md`
12. `06-models/01-算法和模型白话解释.md`
13. `06-models/02-配置项怎么理解.md`
14. `07-dev/01-开发环境安装.md`
15. `07-dev/02-打包-docker-和发布.md`
16. `08-troubleshooting/01-常见问题排查.md`
17. `08-troubleshooting/02-继续学习路线.md`

## 你只想会用服务

只看这些：

1. `01-start/01-这是什么项目.md`
2. `02-use/03-输入输出和任务.md`
3. `02-use/04-实战去字幕流程-本机验证版.md`（实测跑通的完整流程：模式选择、参数、差分验收与补擦闭环）
4. `08-troubleshooting/01-常见问题排查.md`

## 你想看懂代码

先看这些：

1. `03-project-map/01-项目文件夹地图.md`
2. `04-core-flow/01-字幕去除总流程.md`
3. `05-code-reading/02-backend-main.py-怎么读.md`

## 你想改代码

至少看这些：

1. `01-start/03-术语词典.md`
2. `03-project-map/02-重要文件一览.md`
3. `04-core-flow/02-视频帧-字幕框-遮罩-修复-音频.md`
4. `06-models/02-配置项怎么理解.md`
5. `07-dev/01-开发环境安装.md`

## 不要一开始就看的内容

下面这些文件很难，不适合第一天读：

- `backend/inpaint/video/`
- `backend/scenedetect/`
- `backend/tools/train/`
- `backend/models/`

它们不是没用，而是太底层。先知道项目怎么跑，再回来看它们。
