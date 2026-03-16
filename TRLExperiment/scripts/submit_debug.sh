#!/usr/bin/env bash
# 1-step debug test for Qwen3.5-4B with max_tokens=1024.
# Validates: completion quality with sufficient token budget, timing baseline.
#
# Usage (from TRLExperiment/):
#   mkdir -p runtime/results/debug && sbatch scripts/submit_debug.sh

#SBATCH --job-name=trl-debug-qwen
#SBATCH --partition=job
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/home/pcv1327/RL-Based-Temporal-Prevention/TRLExperiment
#SBATCH --output=runtime/results/debug/slurm_%j.log
#SBATCH --error=runtime/results/debug/slurm_%j.err

set -euo pipefail

ROOT_DIR="/home/pcv1327/RL-Based-Temporal-Prevention/TRLExperiment"
cd "${ROOT_DIR}"

mkdir -p runtime/results/debug

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
echo "===== Debug: 1-step Qwen3.5-4B training on salary ====="
echo "Config: max_tokens=1024, batch=4, group=4"
echo ""
$PYTHON -m trl_experiment.train \
    --config configs/train_qwen35.yaml \
    --train_tasks salary \
    --log_dir runtime/results/debug/train_salary \
    --max_steps 1 \
    --batch_size 4 \
    --group_size 4 \
    --max_tokens 1024

echo ""
echo "===== Debug: Output files ====="
ls -la runtime/results/debug/train_salary/

echo ""
echo "===== metrics.jsonl ====="
cat runtime/results/debug/train_salary/metrics.jsonl 2>/dev/null || echo "NOT FOUND"

echo ""
echo "===== step_log.jsonl (first 50 lines) ====="
head -50 runtime/results/debug/train_salary/step_log.jsonl 2>/dev/null || echo "NOT FOUND"

echo ""
echo "Job ${SLURM_JOB_ID} finished at $(date)"
