#!/usr/bin/env bash
# Run all 4 evaluation jobs sequentially:
#   1. Base model on all tasks (350 instances)
#   2. Trained stock model on stock (100 instances)
#   3. Trained salary model on salary (152 instances)
#   4. Trained legal model on legal (98 instances)
#
# Usage:  bash scripts/run_eval_all.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  [1/4] Base model — all tasks"
echo "============================================================"
uv run python -m src.evaluation.evaluate --config configs/qwen/eval_base_all.yaml

echo ""
echo "============================================================"
echo "  [2/4] Trained model — stock"
echo "============================================================"
uv run python -m src.evaluation.evaluate --config configs/qwen/eval_trained_stock.yaml

echo ""
echo "============================================================"
echo "  [3/4] Trained model — salary"
echo "============================================================"
uv run python -m src.evaluation.evaluate --config configs/qwen/eval_trained_salary.yaml

echo ""
echo "============================================================"
echo "  [4/4] Trained model — legal"
echo "============================================================"
uv run python -m src.evaluation.evaluate --config configs/qwen/eval_trained_legal.yaml

echo ""
echo "============================================================"
echo "  All evaluations complete. Running analysis..."
echo "============================================================"
uv run python scripts/analyze_eval.py
