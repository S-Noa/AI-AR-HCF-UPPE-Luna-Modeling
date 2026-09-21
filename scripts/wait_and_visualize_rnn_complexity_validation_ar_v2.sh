#!/usr/bin/env bash
# Export relative-dB figures only after the complete v2 benchmark matrix exists.
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
root="/mnt/Luna.jl-master/rnn_complexity_benchmark"
results="$root/results_validation_ar_v2"
output="$root/visualizations_validation_ar_v2"
logdir="$root/logs"
mkdir -p "$output" "$logdir"
echo "$$" > "$root/validation_ar_v2_visualization_watcher.pid"
trap 'rm -f "$root/validation_ar_v2_visualization_watcher.pid"' EXIT

while [ ! -f "$root/VALIDATION_AR_V2_TRAINING_COMPLETED" ]; do
  echo "[$(date -Is)] waiting for validation-AR v2 training completion"
  sleep 300
done

source "$repo/scripts/cloud_luna_env.sh"
cd "$repo"
BENCHMARK_RESULTS_ROOT="$results" \
BENCHMARK_VISUAL_OUTPUT="$output" \
bash scripts/run_rnn_complexity_visualizations.sh
touch "$root/VALIDATION_AR_V2_VISUALIZATION_COMPLETED"
