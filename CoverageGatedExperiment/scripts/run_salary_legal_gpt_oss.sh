#!/usr/bin/env bash
# Salary + Legal pipeline for openai/gpt-oss-20b (with reward fixes).
# Stock is already complete; this script runs only the remaining two tasks.
#
# Usage:
#   nohup bash scripts/run_salary_legal_gpt_oss.sh > runtime/results/gpt_oss/run_salary_legal.log 2>&1 &
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".env" ]]; then
  echo "Missing .env file. Copy .env.example to .env and set keys."
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
source "${ROOT_DIR}/.env"

PYTHON="/home/pcv1327/RL-Based-Temporal-Prevention/.venv/bin/python"

TRAIN_CFG="configs/train_gpt_oss.yaml"
EVAL_CFG="configs/eval_gpt_oss.yaml"
LOG_ROOT="runtime/results/gpt_oss"

TASKS=("salary" "legal")

echo "============================================================"
echo "  gpt-oss-20b: Salary + Legal (reward fixes applied)"
echo "  Tasks: ${TASKS[*]}"
echo "  Started at: $(date)"
echo "============================================================"

for TASK in "${TASKS[@]}"; do
    TRAIN_LOG_DIR="${LOG_ROOT}/train_${TASK}"

    TASK_EXTRA_ARGS=""
    if [[ "${TASK}" == "legal" ]]; then
        TASK_EXTRA_ARGS="--num_validations 5"
    fi

    echo ""
    echo "===== [${TASK}] Step 1: Training ====="
    $PYTHON -m cov_gated_experiment.train \
        --config "${TRAIN_CFG}" \
        --train_tasks "${TASK}" \
        --log_dir "${TRAIN_LOG_DIR}" \
        ${TASK_EXTRA_ARGS}

    BEST_CKPT=$($PYTHON -m cov_gated_experiment.checkpoint_io read_best \
        --log_dir "${TRAIN_LOG_DIR}")
    FINAL_CKPT=$($PYTHON -m cov_gated_experiment.checkpoint_io read \
        --log_dir "${TRAIN_LOG_DIR}" --name final)

    echo ""
    echo "===== [${TASK}] Step 2a: Evaluating base model on ${TASK} ====="
    $PYTHON -m cov_gated_experiment.evaluate \
        --config "${EVAL_CFG}" \
        --variant base \
        --trained_task "${TASK}" \
        --eval_tasks "${TASK}" \
        --checkpoint_path "" \
        --output_dir "${LOG_ROOT}/eval_${TASK}_base"

    echo ""
    echo "===== [${TASK}] Step 2b: Evaluating best checkpoint on ${TASK} ====="
    if [[ -n "${BEST_CKPT}" ]]; then
        $PYTHON -m cov_gated_experiment.evaluate \
            --config "${EVAL_CFG}" \
            --variant best \
            --trained_task "${TASK}" \
            --eval_tasks "${TASK}" \
            --checkpoint_path "${BEST_CKPT}" \
            --output_dir "${LOG_ROOT}/eval_${TASK}_best"
    else
        echo "  WARNING: best checkpoint not found, skipping."
    fi

    echo ""
    echo "===== [${TASK}] Step 2c: Evaluating final checkpoint on ${TASK} ====="
    if [[ -n "${FINAL_CKPT}" ]]; then
        $PYTHON -m cov_gated_experiment.evaluate \
            --config "${EVAL_CFG}" \
            --variant final \
            --trained_task "${TASK}" \
            --eval_tasks "${TASK}" \
            --checkpoint_path "${FINAL_CKPT}" \
            --output_dir "${LOG_ROOT}/eval_${TASK}_final"
    else
        echo "  WARNING: final checkpoint not found, skipping."
    fi

    echo ""
    echo "===== [${TASK}] Step 3: Deleting remote checkpoints ====="
    $PYTHON -m cov_gated_experiment.checkpoint_io delete \
        --log_dir "${TRAIN_LOG_DIR}"

    echo ""
    echo "===== [${TASK}] Done at $(date) ====="
done

echo ""
echo "============================================================"
echo "  Salary + Legal complete at $(date). Results in ${LOG_ROOT}/"
echo "============================================================"
