#!/usr/bin/env bash
set -euo pipefail

# 在项目根目录、已激活 Python 环境后执行。
# 用法：./run_sttn_propainter.sh [输入视频] [输出视频]
# ProPainter 补修预算：全片最多 8 窗，每窗最多 30 帧（包含上下文）。
input_video="${1:-input.mp4}"
output_video="${2:-output_sttn_propainter.mp4}"

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
exec python -u -c 'import paddle; import runpy; runpy.run_path("vsr_pipeline.py", run_name="__main__")' \
  -i "$input_video" \
  -o "$output_video" \
  --device cuda \
  --inpaint-mode sttn \
  --sttn-precision fp16 \
  --sticker-backend gdino \
  --sticker-model-id backend/models/grounding-dino-tiny \
  --sticker-precision fp16 \
  --sticker-batch-size 4 \
  --sticker-score 0.22 \
  --sticker-max-area-px 1000 \
  --threads 1 \
  --sttn-residual-propaint \
  --sttn-residual-propaint-max-windows 8 \
  --sttn-residual-propaint-min-core-frames 0 \
  --sttn-residual-propaint-max-frames 30
