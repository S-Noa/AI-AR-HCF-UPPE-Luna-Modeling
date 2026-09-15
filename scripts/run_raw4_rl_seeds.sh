#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
data_root="/mnt/Luna.jl-master"
output_root="$data_root/rl_inverse_design"
metrics="$output_root/raw4_banded_mlp_v1/uv_ranking/uv_ranking_metrics.json"

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$output_root/logs"
echo "$$" > "$output_root/rl_formal.pid"
trap 'rm -f "$output_root/rl_formal.pid"' EXIT

python3 - "$metrics" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metrics = json.load(handle)
if metrics["final_logpower_r2"] < 0.90 or metrics["uv_fraction_spearman"] < 0.90:
    raise SystemExit(f"Forward surrogate gate failed: {metrics}")
print("Forward surrogate gate passed:", metrics)
PY

for seed in 123 456 789; do
  output_dir="$output_root/rl_sac_seed${seed}_v1"
  mkdir -p "$output_dir"
  python3 "$repo/Luna.jl-master/examples/simple_interface/train_rl_inverse.py" \
    --input-dir "$data_root/processed_t650_global_log_raw4" \
    --checkpoint "$output_root/raw4_banded_mlp_v1/best_val_r2_model.pth" \
    --output-dir "$output_dir" \
    --seed "$seed" \
    --timesteps 200000 \
    --episode-steps 20 \
    --action-scale 0.10 \
    --top-k 20 \
    --baseline-budget 200000 \
    > "$output_dir/train.log" 2>&1
  touch "$output_dir/COMPLETED"
done

touch "$output_root/RL_FORMAL_COMPLETED"
