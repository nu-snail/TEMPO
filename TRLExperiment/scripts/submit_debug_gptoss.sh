#!/usr/bin/env bash
# 1-step debug test for gpt-oss-20b on 2 GPUs.
# bf16 model weights are ~43GB; 2 GPUs give ~160GB total for model + KV cache.
#
# Usage (from TRLExperiment/):
#   mkdir -p runtime/results/debug_gptoss && sbatch scripts/submit_debug_gptoss.sh

#SBATCH --job-name=trl-debug-gptoss
#SBATCH --partition=job
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --chdir=/home/pcv1327/RL-Based-Temporal-Prevention/TRLExperiment
#SBATCH --output=runtime/results/debug_gptoss/slurm_%j.log
#SBATCH --error=runtime/results/debug_gptoss/slurm_%j.err

set -euo pipefail

ROOT_DIR="/home/pcv1327/RL-Based-Temporal-Prevention/TRLExperiment"
cd "${ROOT_DIR}"

mkdir -p runtime/results/debug_gptoss

if [[ ! -f ".env" ]]; then
    echo "Missing .env file."
    exit 1
fi
source "${ROOT_DIR}/.env"

PYTHON="/home/pcv1327/RL-Based-Temporal-Prevention/.venv/bin/python"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="/shares/bcs516/pcv1327/hf_cache"

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-unknown}"
nvidia-smi

echo ""
echo "===== Debug: 1-step gpt-oss-20b training on salary (2 GPUs) ====="
echo "Config: reasoning_effort=none, max_tokens=1024, batch=2, group=2"
echo ""
$PYTHON -m trl_experiment.train \
    --config configs/train_gptoss.yaml \
    --train_tasks salary \
    --log_dir runtime/results/debug_gptoss/train_salary \
    --max_steps 1 \
    --batch_size 2 \
    --group_size 2 \
    --max_tokens 1024

echo ""
echo "===== Debug: Output files ====="
ls -la runtime/results/debug_gptoss/train_salary/

echo ""
echo "===== metrics.jsonl ====="
cat runtime/results/debug_gptoss/train_salary/metrics.jsonl 2>/dev/null || echo "NOT FOUND"

echo ""
echo "===== step_log.jsonl (first 50 lines) ====="
head -50 runtime/results/debug_gptoss/train_salary/step_log.jsonl 2>/dev/null || echo "NOT FOUND"

echo ""
echo "Job ${SLURM_JOB_ID} finished at $(date)"
