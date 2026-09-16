#!/usr/bin/env bash
# Validate constrained fixed-z candidates with direct 5 cm Luna propagation,
# then generate final-spectrum and full spectral-evolution diagnostics.
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
data_root="/mnt/Luna.jl-master"
rl_output="${1:-$data_root/rl_inverse_design/raw4_z5cm_constrained_rl_smoke_v1}"
validation="$rl_output/luna_validation"

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$validation/validation_h5" "$validation/analysis"
echo "$$" > "$validation/runner.pid"
trap 'rm -f "$validation/runner.pid"' EXIT

cp "$rl_output/inverse_candidates.csv" "$validation/luna_candidates.csv"
echo "rank,seed,uv_fraction_surrogate,reason" > "$validation/luna_validation_failures.csv"

cd "$LUNA_PROJECT/examples/simple_interface"
tail -n +2 "$validation/luna_candidates.csv" | while IFS=, read -r rank seed method uv_fraction objective log_uv log_total energy_j tau_s pressure_bar diameter_m energy_uj tau_fs diameter_um; do
  output_h5="$validation/validation_h5/candidate_rank$(printf '%03d' "$rank")_seed${seed}.h5"
  if [ -f "$output_h5" ]; then
    echo "skip existing $output_h5"
    continue
  fi
  echo "[$(date -Is)] Luna rank=$rank E=${energy_uj}uJ tau=${tau_fs}fs p=${pressure_bar}bar d=${diameter_um}um"
  if ! julia --project="$LUNA_PROJECT" anti_resonant_simulation.jl \
    -e "$energy_uj" --tau "$tau_fs" -p "$pressure_bar" -d "$diameter_um" -t 0.65 \
    --length-cm 5 -o "$output_h5"; then
    rm -f "$output_h5"
    printf '%s,%s,%s,%s\n' "$rank" "$seed" "$uv_fraction" "luna_simulation_failed" >> "$validation/luna_validation_failures.csv"
  fi
done

python3 "$repo/Luna.jl-master/examples/simple_interface/evaluate_rl_luna_validation.py" \
  --validation-dir "$validation" --output-dir "$validation/analysis"
python3 "$repo/Luna.jl-master/examples/simple_interface/plot_rl_luna_evolution_montage.py" \
  --validation-dir "$validation" --analysis-dir "$validation/analysis" \
  --output "$validation/analysis/luna_spectral_evolution_montage.png"
touch "$validation/LUNA_VALIDATION_COMPLETED"
