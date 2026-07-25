# NeurIPS 2026 rebuttal experiments

Scripts answering the reviewer requests. All run from `QualityGatedExperiment/`
with the project environment (`uv sync`) and a `.env` file (see `.env.example`
one level up).

## Setup on a fresh machine

```bash
git clone <repo> && cd <repo>
bash scripts/bootstrap_data.sh                 # links dataset/ into runtime/data/
cd QualityGatedExperiment
cp .env.example .env                           # fill in real keys
uv sync
```

Two kinds of scripts, with different input needs:

- **Training** (`run_reward_ablations.sh`) needs only the dataset (in the repo)
  and API keys. Runs anywhere — training happens on the Tinker managed API, so
  no local GPU is used.
- **Re-scoring / analysis** (everything else) reads the paper's saved eval
  outputs from `runtime/results/<model_dir>/eval_<task>_<variant>/`, which are
  not tracked (large). Copy them from the machine that produced the paper's
  numbers, e.g.:

```bash
rsync -av paper-machine:RL-Based-Temporal-Prevention/QualityGatedExperiment/runtime/results/qwen35_35b \
    QualityGatedExperiment/runtime/results/
```

## Scripts

| Script | Answers | Inputs | Cost |
| --- | --- | --- | --- |
| `run_reward_ablations.sh` | YbBe W3, PV5k Q3 | dataset only | ~$50–60/arm, ~9 h |
| `shapley_dclr.py` | PV5k W1/Q1 | saved eval outputs | ~$15–30/task, ~2–3 h |
| `independent_rescore.py` | PV5k Q2, X7Nt W3/Q4 | saved eval outputs | ~$10 |
| `label_noise_sim.py` | X7Nt Q4 | re-score outputs | free (offline) |
| `recall_probe.py` + `analyze_recall_probe.py` | YbBe W1 | dataset only | ~$5 |
| `reconstruction_gap.py` | YbBe W2 | saved eval outputs | ~$10 |
| `gap_vs_null_diff.py` | YbBe W1 (CI) | probe outputs | free (offline) |
| `undated_claims_count.py` | X7Nt W1/Q1 | saved eval outputs | free (offline) |
| `verifier_gold_test.py` | X7Nt W3, PV5k Q2 | human gold labels (private, not in repo) | ~$2 |

## Reward-design ablation (the training runs)

```bash
mkdir -p runtime/results/qwen3_30b_ablations
# 1-step sanity check first (<$1, ~10 min):
bash scripts/rebuttal/run_reward_ablations.sh smoke
# then the arms, concurrently (one shell each):
nohup bash scripts/rebuttal/run_reward_ablations.sh scalar_mix       > runtime/results/qwen3_30b_ablations/scalar_mix.log 2>&1 &
nohup bash scripts/rebuttal/run_reward_ablations.sh leakage_only     > runtime/results/qwen3_30b_ablations/leakage_only.log 2>&1 &
nohup bash scripts/rebuttal/run_reward_ablations.sh performance_only > runtime/results/qwen3_30b_ablations/performance_only.log 2>&1 &
```

Each arm trains Qwen3-30B-A3B on stock with the paper's config, changing only
the reward computation (`--reward_variant`), then evaluates the best and final
checkpoints and deletes the remote checkpoints. Results land in
`runtime/results/qwen3_30b_ablations/eval_<arm>_{best,final}/summary.json`.

## Shapley-DCLR re-scoring

```bash
uv run python scripts/rebuttal/shapley_dclr.py --task stock --dry_run   # count calls, no cost
nohup uv run python scripts/rebuttal/shapley_dclr.py --task stock \
    > runtime/results/rebuttal/shapley_dclr/stock.log 2>&1 &
```

Per-instance results are checkpointed to JSONL, so reruns resume where they
stopped. `--budget_c/--budget_cap` trade Shapley precision against cost
(defaults c=4/cap=400; the reference implementation uses c=15/cap=1200).

Note: `verifier_gold_test.py` expects the human-labeled gold set from the
separate leakage-evaluation project (default path `../LeakageEvaluationFramework
copy/rebuttal_validation/...`); that data is private and not distributed here.
