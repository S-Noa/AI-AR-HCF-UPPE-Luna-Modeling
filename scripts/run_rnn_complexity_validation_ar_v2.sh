#!/usr/bin/env bash
# Controlled RNN benchmark with validation-only autoregressive checkpoint selection.
set -euo pipefail

repo="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
data_root="/mnt/Luna.jl-master"
root="$data_root/rnn_complexity_benchmark"
processed="$root/processed"
results="$root/results_validation_ar_v2"
logdir="$root/logs"
mkdir -p "$results" "$logdir"
echo "$$" > "$root/validation_ar_v2_pipeline.pid"
trap 'rm -f "$root/validation_ar_v2_pipeline.pid"' EXIT

source "$repo/scripts/cloud_luna_env.sh"
cd "$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"

# The existing controlled MAT files contain 1,300 trajectories.  Each run uses
# the first 1,150 for fitting, the next 100 only for checkpoint selection, and
# the final 50 only for the final reported test metrics.
for class in simple complex moderate; do
  for representation in original_dbm per_sample_minmax; do
    mat="$processed/${class}_${representation}.mat"
    if [ ! -f "$mat" ]; then
      echo "Missing benchmark data: $mat" >&2
      exit 1
    fi
    for seed in 123 456 789; do
      pytorch_out="$results/pytorch_${class}_${representation}_seed${seed}"
      if [ ! -f "$pytorch_out/COMPLETED" ]; then
        mkdir -p "$pytorch_out"
        python3 train_luna_rnn.py \
          --data "$mat" --output-dir "$pytorch_out" \
          --training-mode open_source_legacy --conditioning none \
          --window-size 10 --hidden 250 --lstm-implementation keras_compatible \
          --learning-rate 1e-4 --stage2-learning-rate 1e-5 --stage2-start-epoch 51 \
          --epochs 80 --batch-size 128 \
          --train-evolutions 1250 --validation-evolutions 100 --test-evolutions 50 \
          --eval-autoregressive-samples 100 --eval-autoregressive-every 5 \
          --seed "$seed" --eval-fixed-horizons 1 2 3 4 5 6 7 8 9 10 --fixed-horizon-origins both \
          --log-file "$pytorch_out/train.log"
        touch "$pytorch_out/COMPLETED"
      fi

      keras_out="$results/keras_${class}_${representation}_seed${seed}"
      if [ ! -f "$keras_out/COMPLETED" ]; then
        mkdir -p "$keras_out"
        env -u PYTHONHOME -u PYTHONPATH PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
          "$data_root/rnn_original_code_env/miniconda3/envs/rnn_tf1/bin/python" \
          "$repo/rnnnonlinear-master/rnnnonlinear-master/run_original_keras_benchmark.py" \
          --data "$mat" --output-dir "$keras_out" --normalization none --seed "$seed" \
          --train-evolutions 1250 --validation-evolutions 100 --test-evolutions 50 \
          --checkpoint-every 5 --steps 200 --window-size 10 \
          --epochs-stage1 50 --epochs-stage2 30 \
          --eval-fixed-horizons 1 2 3 4 5 6 7 8 9 10 \
          > "$keras_out/train.log" 2>&1
        touch "$keras_out/COMPLETED"
      fi
    done
  done
done

touch "$root/VALIDATION_AR_V2_TRAINING_COMPLETED"
