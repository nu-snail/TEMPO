#!/usr/bin/env bash
# Full pipeline for one model: download -> train all tasks -> evaluate all tasks -> delete model.
#
# Usage:
#   bash scripts/run_pipeline.sh qwen35 "Qwen/Qwen3.5-4B" "salary legal stock"
set -euo pipefail

MODEL_KEY="${1:?Usage: run_pipeline.sh <model_key> <hf_model_name> <tasks>}"
MODEL_HF="${2:?Usage: run_pipeline.sh <model_key> <hf_model_name> <tasks>}"
TASKS="${3:-salary legal stock}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".env" ]]; then
    echo "Missing .env file."
    exit 1
fi
source "${ROOT_DIR}/.env"

PYTHON="/home/pcv1327/RL-Based-Temporal-Prevention/.venv/bin/python"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="/shares/bcs516/pcv1327/hf_cache"

TRAIN_CFG="configs/train_${MODEL_KEY}.yaml"
EVAL_CFG="configs/eval_${MODEL_KEY}.yaml"
LOG_ROOT="runtime/results/${MODEL_KEY}"

echo "============================================================"
echo "  TRLExperiment Pipeline: ${MODEL_HF}"
echo "  Tasks: ${TASKS}"
echo "  Started at: $(date)"
echo "============================================================"

echo ""
echo "===== Step 0: Pre-download model weights ====="
$PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_HF}', cache_dir='${HF_HOME}')
print('Download complete.')
"

for TASK in $TASKS; do
    TRAIN_LOG="${LOG_ROOT}/train_${TASK}"

    EXTRA_ARGS=""
    if [[ "${TASK}" == "legal" ]]; then
        EXTRA_ARGS="--num_validations 5"
    fi

    echo ""
    echo "===== [${TASK}] Step 1: Training ====="
    $PYTHON -m trl_experiment.train \
        --config "${TRAIN_CFG}" \
        --train_tasks "${TASK}" \
        --log_dir "${TRAIN_LOG}" \
        ${EXTRA_ARGS}

    BEST_CKPT=$($PYTHON -m trl_experiment.checkpoint_io read_best \
        --log_dir "${TRAIN_LOG}")
    FINAL_CKPT=$($PYTHON -m trl_experiment.checkpoint_io read \
        --log_dir "${TRAIN_LOG}" --name final)

    echo ""
    echo "===== [${TASK}] Step 2a: Evaluate base ====="
    $PYTHON -m trl_experiment.evaluate \
        --config "${EVAL_CFG}" \
        --variant base \
        --trained_task "${TASK}" \
        --eval_tasks "${TASK}" \
        --checkpoint_path "" \
        --output_dir "${LOG_ROOT}/eval_${TASK}_base"

    echo ""
    echo "===== [${TASK}] Step 2b: Evaluate best ====="
    if [[ -n "${BEST_CKPT}" ]]; then
        $PYTHON -m trl_experiment.evaluate \
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
    echo "===== [${TASK}] Step 2c: Evaluate final ====="
    if [[ -n "${FINAL_CKPT}" ]]; then
        $PYTHON -m trl_experiment.evaluate \
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
    echo "===== [${TASK}] Step 3: Delete task checkpoints ====="
    $PYTHON -m trl_experiment.checkpoint_io delete \
        --log_dir "${TRAIN_LOG}"

    echo ""
    echo "===== [${TASK}] Done at $(date) ====="
done

echo ""
echo "===== Step 4: Delete model from HF cache ====="
MODEL_ESCAPED=$(echo "${MODEL_HF}" | tr '/' '--')
$PYTHON -c "
import shutil, os
path = os.path.join('${HF_HOME}', 'hub', 'models--${MODEL_ESCAPED}')
if os.path.exists(path):
    shutil.rmtree(path)
    print(f'Deleted {path}')
else:
    print(f'Not found: {path}')
"

echo ""
echo "============================================================"
echo "  Pipeline complete at $(date). Results in ${LOG_ROOT}/"
echo "============================================================"
