#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/pipeline"
cd "$REPO_ROOT"

source /home/cuhk/miniconda3/etc/profile.d/conda.sh
conda activate KinovaGen3_Easy_Control

export PYTHONPATH="$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH}:$REPO_ROOT/lerobot/src"
export PYTHONPATH="${PYTHONPATH}:$PIPELINE_DIR"
export HF_HUB_OFFLINE=1

: "${HF_USER:=local}"
: "${ROBOT_IP:=192.168.8.10}"
: "${DATASET_ROOT:=$REPO_ROOT/data}"
export HF_USER ROBOT_IP DATASET_ROOT
