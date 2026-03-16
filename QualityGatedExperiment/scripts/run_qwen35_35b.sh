#!/usr/bin/env bash
# Convenience wrapper: run all tasks for Qwen3.5-35B-A3B.
#
# Usage:
#   nohup bash scripts/run_qwen35_35b.sh > runtime/results/qwen35_35b/run_full.log 2>&1 &
#   bash scripts/run_qwen35_35b.sh stock        # single task
#   bash scripts/run_qwen35_35b.sh stock salary  # multiple tasks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_full.sh" \
    configs/train_qwen35_35b.yaml \
    configs/eval_qwen35_35b.yaml \
    "$@"
