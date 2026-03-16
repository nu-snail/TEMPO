#!/usr/bin/env bash
# Evaluate a single model variant on a single task.
# Usage: run_eval.sh <variant> <trained_task> [checkpoint_path]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".env" ]]; then
  echo "Missing .env file. Copy .env.example to .env and set keys."
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

VARIANT="${1:?Usage: run_eval.sh <variant> <trained_task> [checkpoint_path]}"
TRAINED_TASK="${2:?Usage: run_eval.sh <variant> <trained_task> [checkpoint_path]}"
CHECKPOINT_PATH="${3:-}"
OUTPUT_DIR="runtime/results/eval_${TRAINED_TASK}_${VARIANT}"

uv run python -m full_experiment.evaluate \
  --config configs/eval.yaml \
  --variant "${VARIANT}" \
  --trained_task "${TRAINED_TASK}" \
  --eval_tasks "${TRAINED_TASK}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --output_dir "${OUTPUT_DIR}"
