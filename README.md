# TEMPO

Code, data, and evaluation outputs for **TEMPO: Temporal Enforcement via Mode-Separated Policy Optimization for Trustworthy LLM Backtesting** (NeurIPS 2026).

TEMPO trains an LLM with GRPO and a two-mode reward so that the evidence behind each prediction is dated before the instance's cutoff. Training and sampling run on the [Tinker](https://tinker-docs.thinkingmachines.ai/) API; the date-plausibility verifier and search run through [OpenRouter](https://openrouter.ai/). No local GPU is needed.

## Setup

```bash
uv sync                     # add --extra data to rebuild the datasets
cp .env.example .env        # then fill in the API keys
```

All commands below run from the repository root.

## Repository layout

| Path | Contents |
|---|---|
| `src/quality_gated_experiment/` | Reward pipeline, GRPO training, evaluation, RAG and TimeSPEC baselines |
| `configs/` | Training and evaluation configs for Qwen3-30B-A3B and Qwen3.5-35B-A3B |
| `scripts/` | Run scripts; `scripts/analysis/` holds the appendix analyses |
| `dataset/` | `train/`, `evaluation/`, `control/` splits with build scripts; `verifier_gold/` holds the 120 human-labeled claims |
| `results/` | Evaluation outputs behind the paper's tables |

## Reproducing the paper

| Paper | Command |
|---|---|
| Dataset checks (App. D) | `python scripts/verify_dataset.py` |
| TEMPO, train + evaluate (Table 2) | `bash scripts/run_qwen3_30b.sh [tasks]`, `bash scripts/run_qwen35_35b.sh [tasks]` |
| RAG baseline (Table 2) | `bash scripts/run_rag_eval.sh configs/eval_rag_<model>.yaml [tasks]` |
| TimeSPEC baseline (Table 2) | `bash scripts/run_timespec_eval.sh configs/eval_timespec_<model>.yaml [tasks]` |
| Control set (Table 2) | `bash scripts/run_control_eval.sh` |
| DCLR (Table 2) | `python scripts/analysis/shapley_dclr.py --task <task> --model_dir <model>` |
| Reward-design ablation (Table 4) | `bash scripts/run_reward_ablations.sh <scalar_mix\|leakage_only\|performance_only>` |
| Verifier validation (App. H.1) | `scripts/analysis/verifier_gold_test.py`, `independent_rescore.py`, `label_noise_sim.py`, `verifier_extensions.py` |
| Outcome-recall probe (App. H.2) | `scripts/analysis/recall_probe.py`, then `analyze_recall_probe.py` and `gap_vs_null_diff.py` |
| Reconstruction test (App. H.3) | `python scripts/analysis/reconstruction_gap.py --task <task> --variant <method>` |

Each analysis script documents its arguments at the top of the file. Training writes to `results/<model>/train_*` and evaluation to `results/<model>/eval_<task>_{best,final}`.

**Note on re-training.** Both base models have since been retired from Tinker for fine-tuning. Evaluation, the baselines, and all analyses still run from the released outputs; re-training requires a provider that serves these models.

## Released outputs

`results/<model>/eval_<task>_<method>/` contains `summary.json` (aggregate metrics) and `eval_step_log.jsonl` (each completion with its claims and verifier verdicts), where `<method>` is `base` (Temporal Hint), `rag`, `timespec`, or `tempo` (the TEMPO checkpoint reported in the paper). `results/<model>/control_<task>/` holds the control-set runs, and `results/analysis/` holds the outputs of the appendix analyses.

## Citation

```bibtex
@inproceedings{zhang2026tempo,
  title     = {{TEMPO}: Temporal Enforcement via Mode-Separated Policy Optimization for Trustworthy {LLM} Backtesting},
  author    = {Zhang, Zeyu and Stadie, Bradly C.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
