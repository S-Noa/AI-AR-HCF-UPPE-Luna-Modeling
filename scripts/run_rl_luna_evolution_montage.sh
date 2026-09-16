#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
validation="/mnt/Luna.jl-master/rl_inverse_design/luna_validation_reexport_v3"
analysis="$validation/analysis"
source "$repo/scripts/cloud_luna_env.sh"
python3 "$repo/Luna.jl-master/examples/simple_interface/plot_rl_luna_evolution_montage.py" \
  --validation-dir "$validation" --analysis-dir "$analysis" \
  --output "$analysis/successful_luna_spectral_evolution_montage.png"
