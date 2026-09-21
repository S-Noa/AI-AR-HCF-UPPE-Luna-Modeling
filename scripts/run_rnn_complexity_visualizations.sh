#!/usr/bin/env bash
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
root="/mnt/Luna.jl-master/rnn_complexity_benchmark"
results_root="${BENCHMARK_RESULTS_ROOT:-$root/results}"
output="${BENCHMARK_VISUAL_OUTPUT:-$root/visualizations}"

source "$repo/scripts/cloud_luna_env.sh"
mkdir -p "$output"
echo "$$" > "$output/runner.pid"
trap 'rm -f "$output/runner.pid"' EXIT

cd "$repo/rnnnonlinear-master/rnnnonlinear-master"
# Set BENCHMARK_CLASSES="complex" (or another space-separated subset) for a
# staged export.  With no override, export every available class.
if [ -n "${BENCHMARK_CLASSES:-}" ]; then
  read -r -a classes <<< "$BENCHMARK_CLASSES"
else
  classes=(simple complex)
  if [ -f "$root/processed/moderate_original_dbm.mat" ]; then
    classes+=(moderate)
  fi
fi

for class in "${classes[@]}"; do
  if [ ! -f "$root/processed/${class}_original_dbm.mat" ]; then
    echo "Missing processed data for requested class: $class" >&2
    exit 1
  fi
done

python3 visualize_rnn_complexity_targets.py \
  --manifest "$root/processed/manifest.csv" \
  --output "$output/representative_complexity_targets.png" \
  --classes "${classes[@]}"

for class in "${classes[@]}"; do
  python3 visualize_rnn_simple_gallery.py \
    --manifest "$root/processed/manifest.csv" \
    --class-label "$class" --output-dir "$output/${class}_gallery" --examples 12
done

# Export the same target/teacher-forced/autoregressive/final-spectrum template
# for every completed Keras and PyTorch benchmark run.  The fixed target grid
# is 251 wavelengths by 200 propagation planes for all controlled classes.
for class in "${classes[@]}"; do
  for framework in keras pytorch; do
    for representation in original_dbm per_sample_minmax; do
      for seed in 123 456 789; do
        name="${framework}_${class}_${representation}_seed${seed}"
        result="$results_root/$name"
        [ -f "$result/COMPLETED" ] || continue
        label="per-sample min-max target space"
        [ "$representation" = "original_dbm" ] && label="original dBm target space"
        display_args=(--target-representation "$representation" --relative-db-floor -50)
        if [ "$representation" = "per_sample_minmax" ]; then
          display_args+=(--raw-power-mat "$root/processed/${class}_raw_power.mat" --test-offset 1250)
        fi
        python3 visualize_rnn_rollout_results.py \
          --stepwise-mat "$result/stepwise_predictions.mat" \
          --autoregressive-mat "$result/autoregressive_predictions.mat" \
          --output-dir "$output/$name" --experiment-name "$name" \
          --test-evo 50 --steps 200 --wavelength-points 251 \
          --normalization-label "$label" --sample-indices 0 1 2 3 \
          "${display_args[@]}"
      done
    done
  done
done

touch "$output/COMPLETED"
