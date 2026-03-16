#!/usr/bin/env bash
# v5 ablation: train + evaluate (base/best/final) for all 3 tasks with trapezoidal gates.
# Skips RAG and TimeSPEC baselines (appendix ablation only).
#
# Usage:
#   nohup bash scripts/run_v5_ablation.sh > runtime/results/qwen35_35b_v5/run_v5.log 2>&1 &
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TRAIN_CONFIG="configs/train_qwen35_35b_v5.yaml"
EVAL_CONFIG="configs/eval_qwen35_35b.yaml"

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
LOG_ROOT="runtime/results/qwen35_35b_v5"

mkdir -p "${LOG_ROOT}"

echo "============================================================"
echo "  v5 Ablation: Trapezoidal Gates (Qwen3.5-35B)"
echo "  Train config: ${TRAIN_CONFIG}"
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
    echo "===== [${TASK}] Step 2a: Evaluating base model ====="
    $PYTHON -m quality_gated_experiment.evaluate \
        --config "${EVAL_CONFIG}" \
        --variant base \
        --trained_task "${TASK}" \
        --eval_tasks "${TASK}" \
        --checkpoint_path "" \
        --output_dir "${LOG_ROOT}/eval_${TASK}_base"

    echo ""
    echo "===== [${TASK}] Step 2b: Evaluating best checkpoint ====="
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
    echo "===== [${TASK}] Step 2c: Evaluating final checkpoint ====="
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
    echo "===== [${TASK}] Step 3: Deleting remote checkpoints ====="
    $PYTHON -m quality_gated_experiment.checkpoint_io delete \
        --log_dir "${TRAIN_LOG_DIR}" || echo "  WARNING: checkpoint deletion failed."

    echo ""
    echo "===== [${TASK}] Done at $(date) ====="
done

echo ""
echo "============================================================"
echo "  v5 ablation complete at $(date). Results in ${LOG_ROOT}/"
echo "============================================================"
