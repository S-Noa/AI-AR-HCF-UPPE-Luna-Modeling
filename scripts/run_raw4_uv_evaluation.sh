#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
data_root="/mnt/Luna.jl-master"
model_dir="$data_root/rl_inverse_design/raw4_banded_mlp_v1"
status_dir="$data_root/rl_inverse_design"

source "$repo/scripts/cloud_luna_env.sh"

if [[ -f "$model_dir/pid" ]]; then
  train_pid="$(cat "$model_dir/pid")"
  while kill -0 "$train_pid" 2>/dev/null; do
    sleep 15
  done
fi

checkpoint="$model_dir/best_val_r2_model.pth"
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }

cd "$LUNA_PROJECT/examples/simple_interface"
python3 evaluate_raw4_uv_ranking.py \
  --input-dir "$data_root/processed_t650_global_log_raw4" \
  --checkpoint "$checkpoint" \
  --output-dir "$model_dir/uv_ranking"

touch "$status_dir/RAW4_UV_EVALUATION_COMPLETED"
