#!/usr/bin/env bash
# Evaluate base models on the control dataset (post-knowledge-cutoff).
# Runs all 6 combinations: 2 models x 3 tasks.
#
# Usage:
#   nohup bash scripts/run_control_eval.sh > runtime/results/run_control_eval.log 2>&1 &
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

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
TASKS=("stock" "salary" "legal")

echo "============================================================"
echo "  Control Set Evaluation: Base Models x 3 Tasks"
echo "  Started at: $(date)"
echo "============================================================"

for MODEL_TAG in qwen3_30b qwen35_35b; do
    EVAL_CONFIG="configs/eval_${MODEL_TAG}.yaml"
    LOG_ROOT="runtime/results/${MODEL_TAG}"

    for TASK in "${TASKS[@]}"; do
        OUT_DIR="${LOG_ROOT}/control_${TASK}"
        echo ""
        echo "===== [${MODEL_TAG}] Control eval: ${TASK} ====="
        echo "  Output: ${OUT_DIR}"
        echo "  Started at: $(date)"

        $PYTHON -m quality_gated_experiment.evaluate \
            --config "${EVAL_CONFIG}" \
            --variant base \
            --trained_task "${TASK}" \
            --eval_tasks "${TASK}" \
            --checkpoint_path "" \
            --data_split control \
            --output_dir "${OUT_DIR}"

        echo "  Finished ${MODEL_TAG}/${TASK} at $(date)"
    done
done

echo ""
echo "============================================================"
echo "  All control evaluations complete at $(date)"
echo "============================================================"
