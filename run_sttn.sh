#!/usr/bin/env bash
set -euo pipefail

# SAM3 caption mask + STTN。优先复用 DGX 的独立 SAM3 实验环境。
# 默认外扩 3px；主阈值整帧为空时采用当前帧 0.3 候选。
# 用法：./run_sttn.sh [输入视频] [输出视频] [额外流水线参数...]
input_video="${1:-input.mp4}"
output_video="${2:-output_sam3_sttn.mp4}"
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi

exec bash "$(dirname -- "${BASH_SOURCE[0]}")/scripts/run_sam3_pipeline.sh" \
  "$input_video" "$output_video" "$@"
