#!/usr/bin/env bash
# twist 数据集 → 关节差分 + 夹爪绝对值 action（输出 {name}_torque）
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# --- 参数（改这里或用环境变量覆盖）---
DATASET_NAME="${1:-${DATASET_NAME:-0517-220612}}"
FORCE="${FORCE:-false}"

export DATASET_NAME FORCE
exec python "$PIPELINE_DIR/convert.py"

# 示例:
#   bash scripts/convert.sh pick_place
#   bash scripts/convert.sh                    # 使用默认 DATASET_NAME=0517-220612
#   FORCE=true bash scripts/convert.sh 0517-220612
