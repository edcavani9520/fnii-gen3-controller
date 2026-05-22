#!/usr/bin/env bash
# 力矩模式回放 *_torque 数据集
set -euo pipefail
source "$(dirname "$0")/_common.sh"

export PYTHONPATH="${PYTHONPATH}:$REPO_ROOT/isbot"

# --- 参数（改这里或用环境变量覆盖）---
DATASET_NAME="${1:-${DATASET_NAME:-0517-220612_torque}}"
EPISODE_INDEX="${2:-${EPISODE_INDEX:-0}}"
SETTLE_S="${SETTLE_S:-2.0}"

export DATASET_NAME EPISODE_INDEX SETTLE_S
exec python "$PIPELINE_DIR/replay.py"

# 示例:
#   bash scripts/replay.sh 0517-220612_torque 0
#   bash scripts/replay.sh                        # 使用默认数据集与 episode 0
#   SETTLE_S=3.0 ROBOT_IP=192.168.8.10 bash scripts/replay.sh 0517-220612_torque 0
