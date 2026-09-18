#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
root="/mnt/Luna.jl-master/rnn_complexity_benchmark"
output="$root/moderate_smoke_candidates"

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$output" "$root/logs"
echo "$$" > "$root/moderate_smoke.pid"
trap 'rm -f "$root/moderate_smoke.pid"' EXIT

cd "$repo/Luna.jl-master/examples/simple_interface"
julia --project="$LUNA_PROJECT" generate_rnn_complexity_benchmark.jl \
  --class moderate --output-dir "$output" --count 64 --seed 20260918 --max-attempts 32 \
  > "$root/logs/moderate_generation_smoke.log" 2>&1

completed=$(find "$output" -maxdepth 1 -type f -name 'candidate_*.h5.done' | wc -l)
if [ "$completed" -ne 64 ]; then
  echo "Expected 64 completed Moderate candidates; found $completed" >&2
  exit 1
fi
cd "$repo/rnnnonlinear-master/rnnnonlinear-master"
python3 visualize_rnn_simple_gallery.py \
  --candidate-dir "$output" --class-label moderate \
  --output-dir "$root/moderate_smoke_visualizations" --examples 12
touch "$root/MODERATE_SMOKE_COMPLETED"
