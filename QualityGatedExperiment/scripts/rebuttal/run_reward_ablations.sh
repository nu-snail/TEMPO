#!/usr/bin/env bash
# Rebuttal reward-design ablations (YbBe W3 / PV5k Q3): train + evaluate one arm.
#
# Arms: scalar_mix | leakage_only | performance_only  (optionally scalar_mix_a07)
# Model: Qwen3-30B-A3B (35B retired from Tinker). Task: stock only.
# Identical config to the paper's run (configs/train_qwen3_30b.yaml); only the
# reward computation changes.
#
# Usage (from QualityGatedExperiment/), one shell per arm so they run concurrently:
#   bash scripts/rebuttal/run_reward_ablations.sh smoke              # 1-step sanity check (<$1, ~10 min)
#   nohup bash scripts/rebuttal/run_reward_ablations.sh scalar_mix        > runtime/results/qwen3_30b_ablations/scalar_mix.log 2>&1 &
#   nohup bash scripts/rebuttal/run_reward_ablations.sh leakage_only      > runtime/results/qwen3_30b_ablations/leakage_only.log 2>&1 &
#   nohup bash scripts/rebuttal/run_reward_ablations.sh performance_only  > runtime/results/qwen3_30b_ablations/performance_only.log 2>&1 &
#   nohup bash scripts/rebuttal/run_reward_ablations.sh scalar_mix_a07    > runtime/results/qwen3_30b_ablations/scalar_mix_a07.log 2>&1 &   # optional
#
# Each arm: train (150 steps, ~$120-160, ~7.6 h) -> eval best + final checkpoints
# on the stock eval set -> delete remote checkpoints.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARM="${1:?Usage: run_reward_ablations.sh <smoke|scalar_mix|leakage_only|performance_only|scalar_mix_a07>}"
TASK="stock"
TRAIN_CONFIG="configs/train_qwen3_30b.yaml"
EVAL_CONFIG="configs/eval_qwen3_30b.yaml"
LOG_ROOT="runtime/results/qwen3_30b_ablations"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a; source "${ROOT_DIR}/.env"; set +a
else
  echo "Missing ${ROOT_DIR}/.env"; exit 1
fi
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
PYTHON="${ROOT_DIR}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || PYTHON="python"

mkdir -p "${LOG_ROOT}"

# Map arm name -> (reward_variant, extra flags)
EXTRA_ARGS=()
case "${ARM}" in
  smoke)
    # One training step with scalar_mix to confirm the Tinker training path
    # and the reward-variant code work end to end. Run BEFORE the real arms.
    ${PYTHON} -m quality_gated_experiment.train \
        --config "${TRAIN_CONFIG}" \
        --train_tasks "${TASK}" \
        --log_dir "${LOG_ROOT}/smoke" \
        --num_epochs 1 --max_steps 1 \
        --reward_variant scalar_mix
    echo "SMOKE OK — training path and reward variant work. Safe to launch arms."
    exit 0
    ;;
  scalar_mix)        VARIANT="scalar_mix" ;;
  leakage_only)      VARIANT="leakage_only" ;;
  performance_only)  VARIANT="performance_only" ;;
  scalar_mix_a07)    VARIANT="scalar_mix"; EXTRA_ARGS+=(--scalar_alpha 0.7) ;;
  *) echo "Unknown arm: ${ARM}"; exit 1 ;;
esac

TRAIN_LOG_DIR="${LOG_ROOT}/train_${ARM}"

echo "===== [${ARM}] Training (${TASK}, Qwen3-30B, variant=${VARIANT}) at $(date) ====="
${PYTHON} -m quality_gated_experiment.train \
    --config "${TRAIN_CONFIG}" \
    --train_tasks "${TASK}" \
    --log_dir "${TRAIN_LOG_DIR}" \
    --num_epochs 1 \
    --reward_variant "${VARIANT}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

BEST_CKPT=$(${PYTHON} -m quality_gated_experiment.checkpoint_io read_best \
    --log_dir "${TRAIN_LOG_DIR}" 2>/dev/null || echo "")
FINAL_CKPT=$(${PYTHON} -m quality_gated_experiment.checkpoint_io read \
    --log_dir "${TRAIN_LOG_DIR}" --name final 2>/dev/null || echo "")

for VER in best final; do
    CKPT_VAR="BEST_CKPT"; [[ "${VER}" == "final" ]] && CKPT_VAR="FINAL_CKPT"
    CKPT="${!CKPT_VAR}"
    if [[ -z "${CKPT}" ]]; then
        echo "WARNING: ${VER} checkpoint not found for ${ARM}, skipping eval."
        continue
    fi
    echo "===== [${ARM}] Evaluating ${VER} checkpoint at $(date) ====="
    ${PYTHON} -m quality_gated_experiment.evaluate \
        --config "${EVAL_CONFIG}" \
        --variant "${VER}" \
        --trained_task "${TASK}" \
        --eval_tasks "${TASK}" \
        --checkpoint_path "${CKPT}" \
        --output_dir "${LOG_ROOT}/eval_${ARM}_${VER}"
done

echo "===== [${ARM}] Deleting remote checkpoints at $(date) ====="
${PYTHON} -m quality_gated_experiment.checkpoint_io delete \
    --log_dir "${TRAIN_LOG_DIR}" || echo "WARNING: checkpoint deletion failed."

echo "===== [${ARM}] Done at $(date). Eval results: ${LOG_ROOT}/eval_${ARM}_{best,final}/summary.json ====="
