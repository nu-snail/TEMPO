#!/usr/bin/env bash
# Full pipeline: for each task, train -> evaluate (base + best + final) -> delete checkpoints.
#
# Usage:
#   nohup bash scripts/run_full.sh configs/train_qwen35_35b.yaml configs/eval_qwen35_35b.yaml > results/run_full.log 2>&1 &
#   bash scripts/run_full.sh configs/train_qwen35_35b.yaml configs/eval_qwen35_35b.yaml stock
#   bash scripts/run_full.sh configs/train_qwen3_30b.yaml configs/eval_qwen3_30b.yaml stock salary
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TRAIN_CONFIG="${1:?Usage: run_full.sh <train_config> <eval_config> [task1 task2 ...]}"
EVAL_CONFIG="${2:?Usage: run_full.sh <train_config> <eval_config> [task1 task2 ...]}"
shift 2

ENV_FILE="${ROOT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; source "${ENV_FILE}"; set +a
elif [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a; source "${ROOT_DIR}/.env"; set +a
else
  echo "Missing .env file."
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

if [[ $# -gt 0 ]]; then
    TASKS=("$@")
else
    TASKS=("stock" "salary" "legal")
fi

MODEL_TAG=$(basename "${TRAIN_CONFIG}" .yaml | sed 's/^train_//')
LOG_ROOT="results/${MODEL_TAG}"

echo "============================================================"
echo "  TEMPO: Train -> Evaluate -> Delete"
echo "  Model config: ${TRAIN_CONFIG}"
echo "  Tasks: ${TASKS[*]}"
echo "  Log root: ${LOG_ROOT}"
echo "  Started at: $(date)"
echo "============================================================"

for TASK in "${TASKS[@]}"; do
    TRAIN_LOG_DIR="${LOG_ROOT}/train_${TASK}"

    echo ""
    echo "===== [${TASK}] Step 1: Training ====="

    EPOCHS=1
    if [[ "${TASK}" == "legal" ]]; then
        EPOCHS=1
        MAX_STEPS_ARG="--max_steps $($PYTHON -c "import math,json,yaml; cfg=yaml.safe_load(open('${TRAIN_CONFIG}')); n=3160; vf=cfg.get('val_fraction',0.10); bs=cfg.get('batch_size',8); print(math.ceil(n*(1-vf)/bs)//2)")"
    else
        MAX_STEPS_ARG=""
    fi

    $PYTHON -m quality_gated_experiment.train \
        --config "${TRAIN_CONFIG}" \
        --train_tasks "${TASK}" \
        --log_dir "${TRAIN_LOG_DIR}" \
        --num_epochs "${EPOCHS}" \
        ${MAX_STEPS_ARG:-}

    BEST_CKPT=$($PYTHON -m quality_gated_experiment.checkpoint_io read_best \
        --log_dir "${TRAIN_LOG_DIR}" 2>/dev/null || echo "")
    FINAL_CKPT=$($PYTHON -m quality_gated_experiment.checkpoint_io read \
        --log_dir "${TRAIN_LOG_DIR}" --name final 2>/dev/null || echo "")

    echo ""
    echo "===== [${TASK}] Step 2a: Evaluating base model on ${TASK} ====="
    $PYTHON -m quality_gated_experiment.evaluate \
        --config "${EVAL_CONFIG}" \
        --variant base \
        --trained_task "${TASK}" \
        --eval_tasks "${TASK}" \
        --checkpoint_path "" \
        --output_dir "${LOG_ROOT}/eval_${TASK}_base"

    echo ""
    echo "===== [${TASK}] Step 2b: Evaluating best checkpoint on ${TASK} ====="
    if [[ -n "${BEST_CKPT}" ]]; then
        $PYTHON -m quality_gated_experiment.evaluate \
            --config "${EVAL_CONFIG}" \
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
        $PYTHON -m quality_gated_experiment.evaluate \
            --config "${EVAL_CONFIG}" \
            --variant final \
            --trained_task "${TASK}" \
            --eval_tasks "${TASK}" \
            --checkpoint_path "${FINAL_CKPT}" \
            --output_dir "${LOG_ROOT}/eval_${TASK}_final"
    else
        echo "  WARNING: final checkpoint not found, skipping."
    fi

    echo ""
    echo "===== [${TASK}] Step 2d: Evaluating RAG baseline on ${TASK} ====="
    RAG_CONFIG="configs/eval_rag_${MODEL_TAG}.yaml"
    if [[ -f "${RAG_CONFIG}" ]]; then
        $PYTHON -m quality_gated_experiment.rag_evaluate \
            --config "${RAG_CONFIG}" \
            --eval_tasks "${TASK}" \
            --output_dir "${LOG_ROOT}/eval_${TASK}_rag"
    else
        echo "  WARNING: RAG config ${RAG_CONFIG} not found, skipping."
    fi

    echo ""
    echo "===== [${TASK}] Step 2e: Evaluating TimeSPEC baseline on ${TASK} ====="
    TIMESPEC_CONFIG="configs/eval_timespec_${MODEL_TAG}.yaml"
    if [[ -f "${TIMESPEC_CONFIG}" ]]; then
        $PYTHON -m quality_gated_experiment.timespec_evaluate \
            --config "${TIMESPEC_CONFIG}" \
            --eval_tasks "${TASK}" \
            --output_dir "${LOG_ROOT}/eval_${TASK}_timespec"
    else
        echo "  WARNING: TimeSPEC config ${TIMESPEC_CONFIG} not found, skipping."
    fi

    echo ""
    echo "===== [${TASK}] Step 3: Deleting remote checkpoints ====="
    $PYTHON -m quality_gated_experiment.checkpoint_io delete \
        --log_dir "${TRAIN_LOG_DIR}" || echo "  WARNING: checkpoint deletion failed."

    echo ""
    echo "===== [${TASK}] Done at $(date) ====="
done

echo ""
echo "============================================================"
echo "  All tasks complete at $(date). Results in ${LOG_ROOT}/"
echo "============================================================"
