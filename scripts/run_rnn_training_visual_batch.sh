#!/usr/bin/env bash
set -euo pipefail

# Process one result at a time: individual saved prediction pairs can be large.
source /mnt/AI-AR-HCF-UPPE-Luna-Modeling/scripts/cloud_luna_env.sh

RNN_DIR="$AI_AR_HCF_REPO/rnnnonlinear-master/rnnnonlinear-master"
RESULT_ROOT="$LUNA_LEGACY_DATA_ROOT/rnn_earlydense"
OUTPUT_ROOT="$LUNA_LEGACY_DATA_ROOT/rnn_visual_diagnostics/training_batch"

mkdir -p "$OUTPUT_ROOT"
cd "$RNN_DIR"

echo "[$(date '+%F %T')] Starting uniform RNN visualization batch"
python3 batch_visualize_luna_rnn_results.py \
  --result-root "$RESULT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --sample-indices 0 1 2 3
echo "[$(date '+%F %T')] Uniform RNN visualization batch completed"
