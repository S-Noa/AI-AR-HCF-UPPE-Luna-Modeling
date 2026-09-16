#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
validation="/mnt/Luna.jl-master/rl_inverse_design/luna_validation_reexport_v3"
output="$validation/analysis"
source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$output"
echo "$$" > "$output/runner.pid"
trap 'rm -f "$output/runner.pid"' EXIT
python3 "$repo/Luna.jl-master/examples/simple_interface/evaluate_rl_luna_validation.py" \
  --validation-dir "$validation" --output-dir "$output"
touch "$output/COMPLETED"
