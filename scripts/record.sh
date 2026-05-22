#!/usr/bin/env bash
# 采集数据：action 为末端 twist 速度（ee.twist_* + gripper.position）
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# --- 参数（改这里或用环境变量覆盖）---
DATASET_NAME="${DATASET_NAME:-$(date +%m%d-%H%M%S)}"
NUM_EPISODES="${NUM_EPISODES:-20}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
FPS="${FPS:-5}"
SINGLE_TASK="${SINGLE_TASK:-Kinova gamepad demo}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
MAX_LINEAR_VELOCITY_M_S="${MAX_LINEAR_VELOCITY_M_S:-0.1}"
TWIST_DURATION_MS="${TWIST_DURATION_MS:-400}"

export DATASET_NAME
export REPO_ID="${HF_USER}/kinova_${DATASET_NAME//-/_}"
export DATA_ROOT="${DATASET_ROOT}/${DATASET_NAME}"
export NUM_EPISODES EPISODE_TIME_S FPS SINGLE_TASK DISPLAY_DATA MAX_LINEAR_VELOCITY_M_S TWIST_DURATION_MS

exec python "$PIPELINE_DIR/record.py"

# 示例:
#   bash scripts/record.sh
#   DATASET_NAME=0517-220612 bash scripts/record.sh
#   NUM_EPISODES=10 EPISODE_TIME_S=45 DISPLAY_DATA=true bash scripts/record.sh
#   MAX_LINEAR_VELOCITY_M_S=0.08 bash scripts/record.sh
