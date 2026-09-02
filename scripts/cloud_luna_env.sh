#!/usr/bin/env bash
# Source this before running Luna/PyTorch jobs on the cloud host.
set -euo pipefail

source /private/env_setup.sh

export PYTHONPATH="/usr/local/corex/lib64/python3/dist-packages:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:/usr/local/iluvatar/lib64:${LD_LIBRARY_PATH:-}"

export AI_AR_HCF_REPO="/mnt/AI-AR-HCF-UPPE-Luna-Modeling"
export LUNA_PROJECT="${AI_AR_HCF_REPO}/Luna.jl-master"
export LUNA_LEGACY_DATA_ROOT="/mnt/Luna.jl-master"

