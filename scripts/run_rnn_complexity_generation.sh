#!/usr/bin/env bash
set -euo pipefail

count="${1:-1600}"
root="/mnt/Luna.jl-master/rnn_complexity_benchmark"
repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$root/logs"
cd "$LUNA_PROJECT/examples/simple_interface"

for label in simple complex; do
  log="$root/logs/${label}_generation.log"
  julia --project="$LUNA_PROJECT" generate_rnn_complexity_benchmark.jl \
    --class "$label" \
    --output-dir "$root/${label}_candidates" \
    --count "$count" \
    >"$log" 2>&1
done

touch "$root/GENERATION_COMPLETED"
