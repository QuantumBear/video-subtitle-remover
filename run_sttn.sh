#!/usr/bin/env bash
set -euo pipefail

# 在项目根目录、已激活 Python 环境后执行。
# 用法：./run_sttn.sh [输入视频] [输出视频]
input_video="${1:-input.mp4}"
output_video="${2:-output_sttn.mp4}"

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
  --threads 1
