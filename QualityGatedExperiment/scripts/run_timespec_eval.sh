#!/usr/bin/env bash
# Standalone TimeSPEC evaluation baseline.
#
# Usage:
#   bash scripts/run_timespec_eval.sh configs/eval_timespec_qwen3_30b.yaml
#   bash scripts/run_timespec_eval.sh configs/eval_timespec_qwen3_30b.yaml stock
#   bash scripts/run_timespec_eval.sh configs/eval_timespec_qwen35_35b.yaml stock salary legal
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

EVAL_CONFIG="${1:?Usage: run_timespec_eval.sh <eval_timespec_config> [task1 task2 ...]}"
shift 1

ENV_FILE="${ROOT_DIR}/../.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; source "${ENV_FILE}"; set +a
elif [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a; source "${ROOT_DIR}/.env"; set +a
else
  echo "Missing .env file."
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

PYTHON="/home/pcv1327/RL-Based-Temporal-Prevention/.venv/bin/python"

if [[ $# -gt 0 ]]; then
    TASKS=("$@")
else
    TASKS=("stock" "salary" "legal")
fi

MODEL_TAG=$(basename "${EVAL_CONFIG}" .yaml | sed 's/^eval_timespec_//')
LOG_ROOT="runtime/results/${MODEL_TAG}"

echo "============================================================"
echo "  TimeSPEC Evaluation Baseline"
echo "  Config: ${EVAL_CONFIG}"
echo "  Tasks: ${TASKS[*]}"
echo "  Log root: ${LOG_ROOT}"
echo "  Started at: $(date)"
echo "============================================================"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "===== TimeSPEC evaluation: ${TASK} ====="

    $PYTHON -m quality_gated_experiment.timespec_evaluate \
        --config "${EVAL_CONFIG}" \
        --eval_tasks "${TASK}" \
        --output_dir "${LOG_ROOT}/eval_${TASK}_timespec"

    echo "  Done: ${TASK} at $(date)"
done

echo ""
echo "============================================================"
echo "  TimeSPEC evaluation complete at $(date). Results in ${LOG_ROOT}/"
echo "============================================================"
