#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
root="/mnt/Luna.jl-master/rnn_complexity_benchmark"
output="$root/visualizations"

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$output"
echo "$$" > "$output/runner.pid"
trap 'rm -f "$output/runner.pid"' EXIT

cd "$repo/rnnnonlinear-master/rnnnonlinear-master"
python3 visualize_rnn_complexity_targets.py \
  --manifest "$root/processed/manifest.csv" \
  --output "$output/representative_simple_complex_targets.png"

result="$root/results/pytorch_simple_original_dbm_seed123"
if [ -f "$result/COMPLETED" ]; then
  python3 visualize_rnn_rollout_results.py \
    --stepwise-mat "$result/stepwise_predictions.mat" \
    --autoregressive-mat "$result/autoregressive_predictions.mat" \
    --output-dir "$output/pytorch_simple_original_dbm_seed123" \
    --experiment-name "pytorch_simple_original_dbm_seed123" \
    --test-evo 50 --steps 200 --wavelength-points 251 \
    --normalization-label "original dBm target space" \
    --sample-indices 0 1 2 3
fi

touch "$output/COMPLETED"
