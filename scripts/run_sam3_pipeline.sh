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

exec "$sam3_python" -u "$project_dir/vsr_pipeline.py" \
  -i "$input_video" -o "$output_video" \
  --device cuda --inpaint-mode sttn --sttn-precision fp16 \
  --mask-backend sam3 --sam3-checkpoint "$sam3_checkpoint" \
  --sam3-code-dir "$sam3_code" --sam3-score "${SAM3_SCORE:-0.5}" \
  --sam3-dilate "${SAM3_DILATE:-3}" --sam3-empty-score "${SAM3_EMPTY_SCORE:-0.3}" \
  --no-locate-stickers --threads 1 \
  "$@"
