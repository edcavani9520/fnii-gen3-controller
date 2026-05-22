#!/usr/bin/env bash
# 查看数据集 episode / 帧信息
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# --- 参数（改这里或用环境变量覆盖）---
DATASET_NAME="${1:-${DATASET_NAME:-0517-220612_torque}}"
EPISODE_INDEX="${2:-${EPISODE_INDEX:-0}}"

export DATASET_NAME EPISODE_INDEX
exec python "$PIPELINE_DIR/read.py"

# 示例:
#   bash scripts/read.sh pick_place_torque 0
#   bash scripts/read.sh 0517-220612 0          # 查看原版 twist 数据
#   bash scripts/read.sh                        # 使用默认数据集与 episode 0
