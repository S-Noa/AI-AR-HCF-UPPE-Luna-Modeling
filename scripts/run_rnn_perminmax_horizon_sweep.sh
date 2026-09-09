#!/usr/bin/env bash
set -euo pipefail

# Run the two remaining short-horizon RNN diagnostics serially.  The 51-point
# companion has already completed; these runs quantify stability as the number
# of autoregressive propagation steps increases.
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh

RNN_DIR="$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"
RESULT_ROOT="$LUNA_LEGACY_DATA_ROOT/rnn_earlydense"
LOG_DIR="$RESULT_ROOT/logs"
mkdir -p "$LOG_DIR"
cd "$RNN_DIR"

run_case() {
    local label="$1"
    local data_path="$2"
    local output_dir="$RESULT_ROOT/$label"

    if [[ -e "$output_dir" ]]; then
        echo "[$(date '+%F %T')] refusing to overwrite existing $output_dir" >&2
        return 1
    fi

    echo "[$(date '+%F %T')] start $label data=$data_path"
    python3 train_luna_rnn.py \
        --data "$data_path" \
        --output-dir "$output_dir" \
        --training-mode scheduled_sampling \
        --conditioning features_z \
        --prediction-target direct \
        --output-activation sigmoid \
        --rollout-steps-start 10 \
        --rollout-steps-end 41 \
        --scheduled-sampling-start 0.0 \
        --scheduled-sampling-end 0.03 \
        --rollout-start-mode mixed \
        --zero-start-prob 0.9 \
        --no-detach-feedback \
        --window-size 10 \
        --hidden 250 \
        --learning-rate 3e-5 \
        --grad-clip 1.0 \
        --epochs 30 \
        --batch-size 16 \
        --eval-autoregressive-every 1 \
        --eval-autoregressive-samples 256 \
        --autoregressive-eval-batch-size 8 \
        --early-stop-on-autoreg \
        --autoreg-patience 8 \
        --checkpoint-every 1 \
        --log-file "$output_dir/train.log"
    echo "[$(date '+%F %T')] done $label"
}

run_case \
    results_features_z_nodetach_z10cm_101_lambda1000_perminmax_v1 \
    "$RESULT_ROOT/simulations/luna_t0p6_earlydense_z10cm_101_conditional.mat"

run_case \
    results_features_z_nodetach_z10cm_201_lambda1000_perminmax_v1 \
    "$RESULT_ROOT/simulations/luna_t0p6_earlydense_z10cm_201_conditional.mat"

echo "[$(date '+%F %T')] all perminmax horizon-sweep experiments done"
