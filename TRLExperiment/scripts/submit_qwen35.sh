#!/usr/bin/env bash
# SLURM batch submission for Qwen3.5-4B full pipeline.
#
# Usage:
#   sbatch scripts/submit_qwen35.sh
#
#SBATCH --job-name=trl-qwen35
#SBATCH --partition=job
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --output=runtime/results/qwen35/slurm_%j.log
#SBATCH --error=runtime/results/qwen35/slurm_%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p runtime/results/qwen35

export HF_HOME="/shares/bcs516/pcv1327/hf_cache"
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS:-0,1}"

echo "Job ${SLURM_JOB_ID} started on $(hostname) at $(date)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi

bash scripts/run_pipeline.sh qwen35 "Qwen/Qwen3.5-4B" "salary legal stock"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
