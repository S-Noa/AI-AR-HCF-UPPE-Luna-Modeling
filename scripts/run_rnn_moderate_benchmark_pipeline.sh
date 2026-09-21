#!/usr/bin/env bash
# Export the Moderate bridge set, train the 12-run Keras/PyTorch matrix, then
# create the uniform frequency-domain evolution visualization package.
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
data_root="/mnt/Luna.jl-master"
root="$data_root/rnn_complexity_benchmark"
processed="$root/processed"
logdir="$root/logs"
mkdir -p "$logdir"
echo "$$" > "$root/moderate_benchmark_pipeline.pid"
trap 'rm -f "$root/moderate_benchmark_pipeline.pid"' EXIT

if [ ! -f "$root/MODERATE_GENERATION_COMPLETED" ]; then
  echo "[$(date -Is)] Moderate candidate generation has not completed" >&2
  exit 1
fi

source "$repo/scripts/cloud_luna_env.sh"
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

# The Simple/Complex selection order is intentionally preserved by the export
# script, so refreshing the shared manifest does not invalidate their models.
if [ ! -f "$processed/moderate_original_dbm.mat" ] || \
   [ ! -f "$processed/moderate_per_sample_minmax.mat" ]; then
  python3 prepare_rnn_complexity_benchmark.py \
    --simple-dir "$root/simple_candidates" \
    --complex-dir "$root/complex_candidates" \
    --moderate-dir "$root/moderate_candidates" \
    --output-dir "$processed" \
    --keep-per-class 1300 --train-evolutions 1250 --test-evolutions 50 \
    --z-points 200 --lambda-points 251 \
    --log-file "$logdir/prepare_moderate_benchmark.log"
fi

for representation in original_dbm per_sample_minmax; do
  mat="$processed/moderate_${representation}.mat"
  for seed in 123 456 789; do
    pytorch_out="$root/results/pytorch_moderate_${representation}_seed${seed}"
    if [ ! -f "$pytorch_out/COMPLETED" ]; then
      mkdir -p "$pytorch_out"
      python3 train_luna_rnn.py \
        --data "$mat" --output-dir "$pytorch_out" \
        --training-mode open_source_legacy --conditioning none \
        --window-size 10 --hidden 250 --lstm-implementation keras_compatible \
        --learning-rate 1e-4 --stage2-learning-rate 1e-5 --stage2-start-epoch 51 \
        --epochs 80 --batch-size 128 --train-evolutions 1250 --test-evolutions 50 \
        --seed "$seed" --eval-fixed-horizons 1 2 3 4 5 6 7 8 9 10 \
        --fixed-horizon-origins both --log-file "$pytorch_out/train.log"
      touch "$pytorch_out/COMPLETED"
    fi

    keras_out="$root/results/keras_moderate_${representation}_seed${seed}"
    if [ ! -f "$keras_out/COMPLETED" ]; then
      mkdir -p "$keras_out"
      env -u PYTHONHOME -u PYTHONPATH PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        "$data_root/rnn_original_code_env/miniconda3/envs/rnn_tf1/bin/python" \
        "$repo/rnnnonlinear-master/rnnnonlinear-master/run_original_keras_benchmark.py" \
        --data "$mat" --output-dir "$keras_out" --normalization none --seed "$seed" \
        --train-evolutions 1250 --test-evolutions 50 --steps 200 --window-size 10 \
        --epochs-stage1 50 --epochs-stage2 30 \
        --eval-fixed-horizons 1 2 3 4 5 6 7 8 9 10 \
        > "$keras_out/train.log" 2>&1
      touch "$keras_out/COMPLETED"
    fi
  done
done

touch "$root/MODERATE_BENCHMARK_TRAINING_COMPLETED"
nohup bash "$repo/scripts/run_rnn_complexity_visualizations.sh" \
  > "$root/logs/visualize_all_complexity_benchmarks.nohup.log" 2>&1 < /dev/null &
echo $! > "$root/visualization_launcher.pid"
