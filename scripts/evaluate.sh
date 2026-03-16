#!/usr/bin/env bash
# ------------------------------------------------------------------
# 4-Phase Evaluation Script (Shapley-DCLR + OLR)
#
# Usage:
#   ./scripts/evaluate.sh configs/qwen/eval_base_model.yaml --tasks stock
#   ./scripts/evaluate.sh configs/deepseek/eval_grpo_full.yaml --tasks legal --tinker_checkpoint <ckpt>
#   ./scripts/evaluate.sh configs/qwen/eval_timespec.yaml --tasks salary
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
echo "  4-Phase Evaluation (TimeSPEC Methodology)"
echo "============================================"
echo "  Config : $CONFIG"
echo "  Extras : $*"
echo "  Time   : $(date)"
echo "============================================"

uv run python -m src.evaluation.evaluate --config "$CONFIG" "$@"
