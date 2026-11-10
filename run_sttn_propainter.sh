#!/usr/bin/env bash
set -euo pipefail

# SAM3 caption mask + STTN + ProPainter 残留补修。
# 用法：./run_sttn_propainter.sh [输入视频] [输出视频] [额外流水线参数...]
# ProPainter 补修预算：全片最多 8 窗，每窗最多 30 帧（包含上下文）。
input_video="${1:-input.mp4}"
output_video="${2:-output_sam3_sttn_propainter.mp4}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi

exec bash "$(dirname -- "${BASH_SOURCE[0]}")/scripts/run_sam3_pipeline.sh" \
  "$input_video" "$output_video" \
  --sttn-residual-propaint \
  --sttn-residual-propaint-max-windows 8 \
  --sttn-residual-propaint-min-core-frames 0 \
  --sttn-residual-propaint-max-frames 30 \
  "$@"
