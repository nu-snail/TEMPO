#!/usr/bin/env bash
# ------------------------------------------------------------------
# GRPO Training Script (per-task)
#
# Usage:
#   ./scripts/train.sh configs/qwen/train_grpo_full.yaml --tasks stock
#   ./scripts/train.sh configs/deepseek/train_grpo_leak_only.yaml --tasks salary
#   ./scripts/train.sh configs/qwen/train_grpo_perf_only.yaml --tasks legal --num_epochs 5
#
# Any --key value pair after the config path overrides the YAML value.
# ------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CONFIG="${1:?Usage: $0 <config.yaml> [--override_key value ...]}"
shift

echo "============================================"
echo "  GRPO Training — Temporal Leakage Prevention"
echo "============================================"
echo "  Config : $CONFIG"
echo "  Extras : $*"
echo "  Time   : $(date)"
echo "============================================"

uv run python -m src.training.train --config "$CONFIG" "$@"
