#!/usr/bin/env bash
# ------------------------------------------------------------------
# Full Experiment Pipeline — Plan B (Main Experiment)
#
# Per-task training: train on task T, evaluate on task T.
# Each model x task produces one metric row.
#
# Default (Plan B) conditions:
#   1. Base Model        — untrained, no temporal hint
#   2. Temporal Hint     — untrained, with temporal hint prompt
#   3. TimeSPEC          — 5-phase inference-time pipeline (KDD paper)
#   6. Our Method        — trained with R_leakage + lambda(t)*R_performance
#
# Ablation conditions (--with-ablation):
#   4. GRPO + Perf Only  — trained with R_performance only
#   5. GRPO + Leak Only  — trained with R_leakage only
#
# Usage:
#   ./scripts/run_all.sh                         # Qwen, all tasks, Plan B
#   ./scripts/run_all.sh --with-deepseek         # Qwen + DeepSeek
#   ./scripts/run_all.sh --with-ablation         # include conditions 4+5
#   ./scripts/run_all.sh --skip-train            # eval only (needs checkpoints)
#   ./scripts/run_all.sh --train-only            # train only (no evaluation)
#   ./scripts/run_all.sh --skip-baselines        # skip conditions 1-3
#   ./scripts/run_all.sh --tasks "stock legal"   # subset of tasks
# ------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SKIP_TRAIN=false
SKIP_EVAL=false
SKIP_BASELINES=false
WITH_DEEPSEEK=false
WITH_ABLATION=false
TASKS="stock salary legal"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train)      SKIP_TRAIN=true; shift ;;
        --train-only)      SKIP_EVAL=true; SKIP_BASELINES=true; shift ;;
        --skip-baselines)  SKIP_BASELINES=true; shift ;;
        --with-deepseek)   WITH_DEEPSEEK=true; shift ;;
        --with-ablation)   WITH_ABLATION=true; shift ;;
        --tasks)           TASKS="$2"; shift 2 ;;
        *)                 shift ;;
    esac
done

MODELS="qwen"
if [ "$WITH_DEEPSEEK" = true ]; then
    MODELS="qwen deepseek"
fi

PLAN="Plan B (4 conditions)"
if [ "$WITH_ABLATION" = true ]; then
    PLAN="Plan A (6 conditions, with ablation)"
fi

STEP_NUM=0
count_steps() {
    local n=0
    local ntasks
    ntasks=$(echo "$TASKS" | wc -w)
    for ms in $MODELS; do
        if [ "$SKIP_BASELINES" = false ] && [ "$SKIP_EVAL" = false ]; then
            n=$((n + ntasks * 3))  # Conditions 1-3
        fi
        if [ "$WITH_ABLATION" = true ]; then
            [ "$SKIP_TRAIN" = false ] && n=$((n + ntasks * 2))
            [ "$SKIP_EVAL" = false ]  && n=$((n + ntasks * 2))
        fi
        [ "$SKIP_TRAIN" = false ] && n=$((n + ntasks))    # Condition 6 train
        [ "$SKIP_EVAL" = false ]  && n=$((n + ntasks))    # Condition 6 eval
    done
    echo "$n"
}
TOTAL_STEPS=$(count_steps)

step() {
    STEP_NUM=$((STEP_NUM + 1))
    echo ""
    echo "━━━ [${STEP_NUM}/${TOTAL_STEPS}] $1 ━━━"
}

echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Full Experiment — ${PLAN}"
echo "│  Models: $MODELS"
echo "│  Tasks:  $TASKS"
echo "│  Total pipeline steps: $TOTAL_STEPS"
echo "│  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""
echo "  Results directory layout:"
echo "    results/<model>/train_grpo_full_<task>/    — checkpoints + search cache"
echo "    results/<model>/eval_<condition>_<task>/    — per-instance + summary"
echo "      ├── completions.json"
echo "      ├── instance_results.json"
echo "      ├── summary.json                         ← final metrics"
echo "      ├── phase1_claims/<id>.json"
echo "      ├── phase2_shapley/<id>.json"
echo "      ├── phase3_leakage/<id>.json"
echo "      └── phase4_metrics/<id>.json"
echo ""

for MS in $MODELS; do
    CFGDIR="configs/${MS}"

    # ==============================================================
    # Conditions 1-3: No-training baselines
    # ==============================================================
    if [ "$SKIP_BASELINES" = false ]; then
        for TASK in $TASKS; do
            step "$MS/$TASK — Eval Cond 1: Base Model"
            bash scripts/evaluate.sh "${CFGDIR}/eval_base_model.yaml" \
                --tasks "$TASK" \
                --output_dir "results/${MS}/eval_base_model_${TASK}"

            step "$MS/$TASK — Eval Cond 2: Temporal Hint"
            bash scripts/evaluate.sh "${CFGDIR}/eval_temporal_hint.yaml" \
                --tasks "$TASK" \
                --output_dir "results/${MS}/eval_temporal_hint_${TASK}"

            step "$MS/$TASK — Eval Cond 3: TimeSPEC"
            bash scripts/evaluate.sh "${CFGDIR}/eval_timespec.yaml" \
                --tasks "$TASK" \
                --output_dir "results/${MS}/eval_timespec_${TASK}"
        done
    fi

    # ==============================================================
    # Conditions 4-5: Ablation (only with --with-ablation)
    # ==============================================================
    if [ "$WITH_ABLATION" = true ]; then
        for TASK in $TASKS; do
            # --- Condition 4: GRPO + Performance Only ---
            CKPT_PERF="results/${MS}/train_grpo_perf_only_${TASK}"
            if [ "$SKIP_TRAIN" = false ]; then
                step "$MS/$TASK — Train Cond 4: GRPO+Perf Only"
                bash scripts/train.sh "${CFGDIR}/train_grpo_perf_only.yaml" \
                    --tasks "$TASK" \
                    --log_path "$CKPT_PERF" \
                    --search_cache_path "${CKPT_PERF}/search_cache.json"
            fi
            if [ "$SKIP_EVAL" = false ]; then
                step "$MS/$TASK — Eval Cond 4: GRPO+Perf Only"
                bash scripts/evaluate.sh "${CFGDIR}/eval_grpo_perf_only.yaml" \
                    --tasks "$TASK" \
                    --output_dir "results/${MS}/eval_grpo_perf_only_${TASK}" \
                    --tinker_checkpoint "${CKPT_PERF}/final"
            fi

            CKPT_LEAK="results/${MS}/train_grpo_leak_only_${TASK}"
            if [ "$SKIP_TRAIN" = false ]; then
                step "$MS/$TASK — Train Cond 5: GRPO+Leak Only"
                bash scripts/train.sh "${CFGDIR}/train_grpo_leak_only.yaml" \
                    --tasks "$TASK" \
                    --log_path "$CKPT_LEAK" \
                    --search_cache_path "${CKPT_LEAK}/search_cache.json"
            fi
            if [ "$SKIP_EVAL" = false ]; then
                step "$MS/$TASK — Eval Cond 5: GRPO+Leak Only"
                bash scripts/evaluate.sh "${CFGDIR}/eval_grpo_leak_only.yaml" \
                    --tasks "$TASK" \
                    --output_dir "results/${MS}/eval_grpo_leak_only_${TASK}" \
                    --tinker_checkpoint "${CKPT_LEAK}/final"
            fi
        done
    fi

    # ==============================================================
    # Condition 6: Our Method (always runs)
    # ==============================================================
    for TASK in $TASKS; do
        CKPT_FULL="results/${MS}/train_grpo_full_${TASK}"
        if [ "$SKIP_TRAIN" = false ]; then
            step "$MS/$TASK — Train Cond 6: Our Method"
            bash scripts/train.sh "${CFGDIR}/train_grpo_full.yaml" \
                --tasks "$TASK" \
                --log_path "$CKPT_FULL" \
                --search_cache_path "${CKPT_FULL}/search_cache.json"
        fi
        if [ "$SKIP_EVAL" = false ]; then
            step "$MS/$TASK — Eval Cond 6: Our Method"
            bash scripts/evaluate.sh "${CFGDIR}/eval_grpo_full.yaml" \
                --tasks "$TASK" \
                --output_dir "results/${MS}/eval_grpo_full_${TASK}" \
                --tinker_checkpoint "${CKPT_FULL}/final"
        fi
    done
done

echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Pipeline complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "│  Steps completed: ${STEP_NUM}/${TOTAL_STEPS}"
echo "│  Results: results/<model>/eval_<condition>_<task>/summary.json"
echo "└─────────────────────────────────────────────────────────────┘"
