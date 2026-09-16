#!/usr/bin/env bash
# Train a small constrained RL smoke run for the fixed-z=5 cm raw4 surrogate.
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
data_root="/mnt/Luna.jl-master"
input="$data_root/processed_t650_global_log_raw4_z5cm"
checkpoint="$data_root/rl_inverse_design/raw4_z5cm_banded_mlp_v1/best_val_r2_model.pth"
output="$data_root/rl_inverse_design/raw4_z5cm_constrained_rl_smoke_v1"

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$output"
echo "$$" > "$output/runner.pid"
trap 'rm -f "$output/runner.pid"' EXIT

cd "$repo/Luna.jl-master/examples/simple_interface"
python3 train_rl_inverse.py \
  --input-dir "$input" \
  --checkpoint "$checkpoint" \
  --output-dir "$output" \
  --objective constrained_uv_power \
  --total-power-floor-percentile 25 \
  --total-power-penalty 1.0 \
  --timesteps 10000 \
  --baseline-budget 10000 \
  --top-k 10 \
  --episode-steps 20 \
  --action-scale 0.10 \
  --seed 123 \
  > "$output/train.log" 2>&1

touch "$output/COMPLETED"
