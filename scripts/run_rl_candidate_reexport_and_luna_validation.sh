#!/usr/bin/env bash
# Re-export SAC candidates in physical units and verify a deduplicated top set with Luna.
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
data_root="/mnt/Luna.jl-master"
rl_root="$data_root/rl_inverse_design"
output="$rl_root/luna_validation_reexport_v2"

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$output/validation_h5"
echo "$$" > "$output/runner.pid"
trap 'rm -f "$output/runner.pid"' EXIT

python3 "$repo/Luna.jl-master/examples/simple_interface/export_rl_candidates.py" \
  --input-dir "$data_root/processed_t650_global_log_raw4" \
  --checkpoint "$rl_root/raw4_banded_mlp_v1/best_val_r2_model.pth" \
  --rl-root "$rl_root" \
  --output-dir "$output" \
  --top-k-per-seed 20 \
  --episodes 200

# A policy can converge to the same boundary candidate from many restarts.
# Keep the first 20 unique physical settings after rounding to simulator units.
python3 - "$output/rl_sac_candidates_reexported.csv" "$output/luna_candidates.csv" <<'PY'
import csv
import sys

source, destination = sys.argv[1:]
seen, retained = set(), []
with open(source, newline='', encoding='utf-8') as handle:
    for row in csv.DictReader(handle):
        key = tuple(round(float(row[name]), digits) for name, digits in [
            ('energy_uj', 5), ('tau_fs', 4), ('pressure_bar', 4), ('diameter_um', 4)
        ])
        if key not in seen:
            seen.add(key)
            retained.append(row)
        if len(retained) == 20:
            break
if not retained:
    raise SystemExit('No RL candidates were exported')
with open(destination, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=retained[0].keys())
    writer.writeheader(); writer.writerows(retained)
print(f'Retained {len(retained)} unique candidates')
PY

cd "$LUNA_PROJECT/examples/simple_interface"
while IFS=, read -r rank seed score energy_j tau_s pressure diameter_m energy_uj tau_fs diameter_um; do
  if [ "$rank" = "rank" ]; then continue; fi
  output_h5="$output/validation_h5/candidate_rank$(printf '%03d' "$rank")_seed${seed}.h5"
  if [ -f "$output_h5" ]; then
    echo "skip existing $output_h5"
    continue
  fi
  echo "[$(date -Is)] validating rank=$rank seed=$seed score=$score E=${energy_uj}uJ tau=${tau_fs}fs p=${pressure}bar d=${diameter_um}um"
  julia --project="$LUNA_PROJECT" anti_resonant_simulation.jl \
    -e "$energy_uj" --tau "$tau_fs" -p "$pressure" -d "$diameter_um" -t 0.65 \
    -o "$output_h5"
done < "$output/luna_candidates.csv"

touch "$output/LUNA_VALIDATION_COMPLETED"
