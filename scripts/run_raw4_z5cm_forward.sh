#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
data_root="/mnt/Luna.jl-master"
input="$data_root/processed_t650_global_log_raw4"
view="$data_root/processed_t650_global_log_raw4_z5cm"
output="$data_root/rl_inverse_design/raw4_z5cm_banded_mlp_v1"

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$output"
echo "$$" > "$output/runner.pid"
trap 'rm -f "$output/runner.pid"' EXIT

python3 "$repo/Luna.jl-master/examples/simple_interface/create_raw4_zslice_view.py" \
  --input-dir "$input" --output-dir "$view" --z-cm 5.0 --batch-rows 128 \
  --log-file "$output/create_view.log"

cd "$repo/Luna.jl-master/examples/simple_interface"
python3 train_mlp.py \
  --input-dir "$view" --output-dir "$output" \
  --model banded_mlp --band-boundaries-nm 200 700 1200 1800 2500 \
  --output-activation identity --epochs 300 --batch-size 128 --learning-rate 1e-3 \
  --selection-metric val_r2 --visualize \
  > "$output/train.log" 2>&1

python3 evaluate_raw4_uv_ranking.py \
  --input-dir "$view" --checkpoint "$output/best_val_r2_model.pth" \
  --output-dir "$output/uv_ranking" \
  > "$output/evaluate.log" 2>&1
touch "$output/COMPLETED"
