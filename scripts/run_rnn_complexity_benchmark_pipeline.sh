#!/usr/bin/env bash
# Wait for controlled Luna generation, export datasets, then run the benchmark matrix serially.
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
data_root="/mnt/Luna.jl-master"
root="$data_root/rnn_complexity_benchmark"
processed="$root/processed"
logdir="$root/logs"
mkdir -p "$logdir"
echo "$$" > "$root/pipeline.pid"
trap 'rm -f "$root/pipeline.pid"' EXIT

while [ ! -f "$root/GENERATION_COMPLETED" ]; do
  echo "[$(date -Is)] waiting for controlled data generation"
  sleep 300
done

source "$repo/scripts/cloud_luna_env.sh"
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"
if [ ! -f "$processed/simple_original_dbm.mat" ] || [ ! -f "$processed/complex_original_dbm.mat" ] || \
   [ ! -f "$processed/simple_per_sample_minmax.mat" ] || [ ! -f "$processed/complex_per_sample_minmax.mat" ]; then
  python3 prepare_rnn_complexity_benchmark.py \
    --simple-dir "$root/simple_candidates" \
    --complex-dir "$root/complex_candidates" \
    --output-dir "$processed" \
    --keep-per-class 1300 --train-evolutions 1250 --test-evolutions 50 \
    --z-points 200 --lambda-points 251 \
    --log-file "$logdir/prepare_benchmark.log"
else
  echo "[$(date -Is)] benchmark MAT files already exist; skipping export"
fi

# PyTorch follows the 50 + 30 epoch reference schedule.  Keras is kept in the
# isolated TensorFlow-1 environment created for the original-code comparison.
for class in simple complex; do
  for representation in original_dbm per_sample_minmax; do
    mat="$processed/${class}_${representation}.mat"
    for seed in 123 456 789; do
      pytorch_out="$root/results/pytorch_${class}_${representation}_seed${seed}"
      if [ ! -f "$pytorch_out/COMPLETED" ]; then
        mkdir -p "$pytorch_out"
        python3 train_luna_rnn.py \
          --data "$mat" --output-dir "$pytorch_out" \
          --training-mode open_source_legacy --conditioning none \
          --window-size 10 --hidden 250 --lstm-implementation keras_compatible \
          --learning-rate 1e-4 --stage2-learning-rate 1e-5 --stage2-start-epoch 51 \
          --epochs 80 --batch-size 128 --train-evolutions 1250 --test-evolutions 50 \
          --seed "$seed" --eval-fixed-horizons 1 2 3 4 --fixed-horizon-origins both \
          --log-file "$pytorch_out/train.log"
        touch "$pytorch_out/COMPLETED"
      fi

      keras_out="$root/results/keras_${class}_${representation}_seed${seed}"
      if [ ! -f "$keras_out/COMPLETED" ]; then
        mkdir -p "$keras_out"
        # The legacy TF1 environment must not inherit the project's Python
        # paths; the current working directory already exposes its modules.
        env -u PYTHONHOME -u PYTHONPATH PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
          "$data_root/rnn_original_code_env/miniconda3/envs/rnn_tf1/bin/python" \
          "$repo/rnnnonlinear-master/rnnnonlinear-master/run_original_keras_benchmark.py" \
          --data "$mat" --output-dir "$keras_out" \
          --normalization none --seed "$seed" --train-evolutions 1250 --test-evolutions 50 \
          --steps 200 --window-size 10 --epochs-stage1 50 --epochs-stage2 30 \
          > "$keras_out/train.log" 2>&1
        touch "$keras_out/COMPLETED"
      fi
    done
  done
done

touch "$root/BENCHMARK_TRAINING_COMPLETED"
