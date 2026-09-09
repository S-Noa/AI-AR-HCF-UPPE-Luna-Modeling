#!/usr/bin/env bash
set -euo pipefail

source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh

RESULT_ROOT="$LUNA_LEGACY_DATA_ROOT/rnn_earlydense"
HORIZON_PID_FILE="$RESULT_ROOT/run_perminmax_horizon_sweep.pid"
OUTPUT_DIR="$LUNA_LEGACY_DATA_ROOT/rnn_visual_diagnostics/data_dynamics_comparison"
SCRIPT="$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master/compare_rnn_data_dynamics.py"

if [[ -f "$HORIZON_PID_FILE" ]]; then
    horizon_pid=$(cat "$HORIZON_PID_FILE")
    while kill -0 "$horizon_pid" 2>/dev/null; do
        echo "[$(date '+%F %T')] waiting for RNN horizon sweep pid=$horizon_pid"
        sleep 300
    done
fi

mkdir -p "$OUTPUT_DIR"
python3 "$SCRIPT" \
    --original-data-h5 "$LUNA_LEGACY_DATA_ROOT/rnn_original_data/converted/SC_spec_251_dBm.h5" \
    --luna-raw-mat "$LUNA_LEGACY_DATA_ROOT/rnn_original_code_env/luna_t0p6_earlydense_z10cm_51_lambda251_rawpower_originalcode.mat" \
    --original-autoreg-mat "$LUNA_LEGACY_DATA_ROOT/rnn_original_data/results_pytorch_SC_spec_251_dBm_v1/autoregressive_predictions.mat" \
    --luna-autoreg-mat "$LUNA_LEGACY_DATA_ROOT/rnn_original_code_env/raw_power_dBm_luna_lambda251/dBm_3e/results/autoregressive.mat" \
    --output-dir "$OUTPUT_DIR" \
    --max-windows 10000 \
    --seed 20260909
touch "$OUTPUT_DIR/COMPLETED"
echo "[$(date '+%F %T')] data-dynamics comparison completed"
