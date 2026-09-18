#!/usr/bin/env bash
# Generate the full Moderate candidate pool after the smoke gallery is approved.
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
root="/mnt/Luna.jl-master/rnn_complexity_benchmark"
output="$root/moderate_candidates"
count=1600

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$output" "$root/logs"
echo "$$" > "$root/moderate_generation.pid"
trap 'rm -f "$root/moderate_generation.pid"' EXIT

cd "$repo/Luna.jl-master/examples/simple_interface"
julia --project="$LUNA_PROJECT" generate_rnn_complexity_benchmark.jl \
  --class moderate --output-dir "$output" --count "$count" \
  --seed 20260919 --max-attempts 32 \
  > "$root/logs/moderate_generation.log" 2>&1

completed=$(find "$output" -maxdepth 1 -type f -name 'candidate_*.h5.done' | wc -l)
if [ "$completed" -ne "$count" ]; then
  echo "Expected $count completed Moderate candidates; found $completed" >&2
  exit 1
fi

cd "$repo/rnnnonlinear-master/rnnnonlinear-master"
python3 visualize_rnn_simple_gallery.py \
  --candidate-dir "$output" --class-label moderate \
  --output-dir "$root/moderate_visualizations" --examples 16
touch "$root/MODERATE_GENERATION_COMPLETED"
