#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for the two user-facing scripts. No Paddle/OCR import here.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
experiment_dir="$project_dir/quality_eval_20261109/sam3_mask_compare"
sam3_python="${SAM3_PYTHON:-}"
if [[ -z "$sam3_python" ]]; then
  if [[ -x "$experiment_dir/env/bin/python" ]]; then
    sam3_python="$experiment_dir/env/bin/python"
  else
    sam3_python=python
  fi
fi
sam3_code="${SAM3_CODE_DIR:-$project_dir/backend/models/sam3/code}"
sam3_checkpoint="${SAM3_CHECKPOINT:-$project_dir/backend/models/sam3/sam3.pt}"
if [[ -z "${SAM3_CODE_DIR:-}" && -f "$experiment_dir/sam3_code/sam3/model_builder.py" ]]; then
  sam3_code="$experiment_dir/sam3_code"
fi
if [[ -z "${SAM3_CHECKPOINT:-}" && -f "$experiment_dir/sam3_model/sam3.pt" ]]; then
  sam3_checkpoint="$experiment_dir/sam3_model/sam3.pt"
fi
input_video="$1"
output_video="$2"
shift 2

# Official video tracking may JIT-build Triton helpers; reuse DGX's existing headers.
sam3_headers="$experiment_dir/build_headers/extracted/usr/include"
if [[ "$sam3_python" == "$experiment_dir/env/bin/python" && -f "$sam3_headers/python3.12/Python.h" ]]; then
  export CPATH="$sam3_headers/python3.12:$sam3_headers/aarch64-linux-gnu/python3.12${CPATH:+:$CPATH}"
fi

exec "$sam3_python" -u "$project_dir/vsr_pipeline.py" \
  -i "$input_video" -o "$output_video" \
  --device cuda --inpaint-mode sttn --sttn-precision fp16 \
  --mask-backend sam3 --sam3-checkpoint "$sam3_checkpoint" \
  --sam3-code-dir "$sam3_code" --sam3-score "${SAM3_SCORE:-0.5}" \
  --sam3-dilate "${SAM3_DILATE:-3}" --sam3-empty-score "${SAM3_EMPTY_SCORE:-0.3}" \
  --sam3-emoji-score "${SAM3_EMOJI_SCORE:-0}" \
  --sam3-emoji-max-frames "${SAM3_EMOJI_MAX_FRAMES:-120}" \
  --sticker-backend gdino \
  --sticker-model-id "${STICKER_MODEL_ID:-$project_dir/backend/models/grounding-dino-tiny}" \
  --sticker-prompt "${STICKER_PROMPT:-emoji. sticker. cartoon icon.}" \
  --sticker-score "${STICKER_SCORE:-0.22}" \
  --sticker-max-area-px "${STICKER_MAX_AREA_PX:-1200}" \
  --sticker-max-frames "${STICKER_MAX_FRAMES:-200}" \
  --sticker-batch-size "${STICKER_BATCH_SIZE:-1}" --sticker-precision fp16 --threads 1 \
  "$@"
