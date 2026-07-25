"""Shapley-DCLR re-scoring of saved eval outputs (PV5k W1/Q1, AC weakness 3).

For each saved completion, the evidence claims are players in a cooperative
game whose value is how well a closed-world model, given only that claim
subset, reconstructs the ORIGINAL prediction. Shapley values are exact for
<= 8 claims and Leverage SHAP (Musco & Witter, ICLR 2025) above; DCLR is the
share of total |Shapley| weight carried by leaked claims. Leakage labels are
the paper verifier's saved verdicts — no re-judging.

Coalition math (compute_budget, leverage sampling, projected regression,
DCLR) follows LeakageEvaluationFramework/leakage_eval/shapley verbatim.

Usage (from QualityGatedExperiment/):
    # smoke test, 2 instances of one cell (~$0.5):
    uv run python scripts/rebuttal/shapley_dclr.py --task stock --variants best --limit 2
    # full run for one task:
    uv run python scripts/rebuttal/shapley_dclr.py --task stock
    # resume is automatic (per-instance results are checkpointed)

Cost control: --budget_c/--budget_cap scale coalition counts per instance.
Defaults (c=4, cap=400) target the $30-80 envelope; the framework's own
defaults (c=15, cap=1200) triple the cost for tighter Shapley estimates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quality_gated_experiment.config import load_env
from quality_gated_experiment.data import load_eval_instances
from quality_gated_experiment.prompts import _build_salary_user, _build_stock_user
from quality_gated_experiment.types import TaskInstance

from recall_probe import make_sampler, parse_json_object
from reconstruction_gap import RECON_SYSTEM, OUTPUT_SPECS, extract_original

logger = logging.getLogger("shapley_dclr")

METHODS = {"base": "Temporal Hint", "rag": "RAG", "timespec": "TimeSPEC", "best": "TEMPO"}


# ---------------------------------------------------------------------------
# Coalition math — vendored from LeakageEvaluationFramework/leakage_eval/shapley
# ---------------------------------------------------------------------------

def compute_budget(n: int, c: float, cap: int, exact_threshold: int) -> int:
    if n <= exact_threshold:
        return 2 ** n
    budget = int(math.floor(c * n * math.log(n)))
    return min(cap, max(2 * n + 2, budget))


def exact_coalitions(n: int) -> list[tuple[int, ...]]:
    return [tuple(int(b) for b in bits) for bits in itertools.product([0, 1], repeat=n)]


def leverage_shap_coalitions(n: int, budget: int, seed: int) -> list[tuple[int, ...]]:
    if n <= 0:
        return [tuple()]
    if budget >= 2 ** n:
        return exact_coalitions(n)
    rng = np.random.default_rng(seed)
    masks: dict[tuple[int, ...], None] = {tuple([0] * n): None, tuple([1] * n): None}
    while len(masks) < budget:
        size = int(rng.integers(1, n))
        chosen = set(rng.choice(n, size=size, replace=False).tolist())
        mask = tuple(1 if i in chosen else 0 for i in range(n))
        masks.setdefault(mask, None)
        if len(masks) < budget:
            masks.setdefault(tuple(1 - b for b in mask), None)
        if len(masks) == 2 ** n:
            break
    return list(masks.keys())


def solve_projected_regression(
    masks: list[tuple[int, ...]], values: list[float], v_full: float
) -> np.ndarray:
    n = len(masks[0])
    rows, targets, weights = [], [], []
    for mask, value in zip(masks, values):
        size = sum(mask)
        if size == 0 or size == n:
            continue
        rows.append(mask)
        targets.append(float(value))
        shap_w = 1.0 / (math.comb(n, size) * size * (n - size))
        samp_p = 1.0 / ((n - 1) * math.comb(n, size))
        weights.append(shap_w / samp_p)
    if not rows:
        return np.full(n, v_full / n, dtype=float)
    z = np.array(rows, dtype=float)
    y = np.array(targets, dtype=float)
    w = np.sqrt(np.array(weights, dtype=float))
    one = np.ones(n)
    projection = np.eye(n) - np.outer(one, one) / n
    base = (float(v_full) / n) * one
    lhs = (z @ projection) * w[:, None]
    rhs = (y - z @ base) * w
    theta, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return projection @ theta + base


def shapley_dclr(phi: np.ndarray, leaked: list[bool]) -> float:
    abs_phi = np.abs(phi)
    total = float(abs_phi.sum())
    if total <= 0:
        return 0.0
    return float(abs_phi[np.array(leaked, dtype=bool)].sum() / total)


# ---------------------------------------------------------------------------
# Closed-world value function (matches reconstruction_gap setup)
# ---------------------------------------------------------------------------

def build_coalition_messages(
    inst: TaskInstance, evidence_subset: list[dict]
) -> list[dict[str, str]]:
    task_block = _build_stock_user(inst) if inst.task == "stock" else _build_salary_user(inst)
    user = (
        f"{task_block}\n\n"
        f"EVIDENCE (the only information you may use):\n"
        f"{json.dumps(evidence_subset, ensure_ascii=False)}\n\n"
        f"Output a single JSON object and nothing else, no markdown fences:\n"
        f"{OUTPUT_SPECS[inst.task]}"
    )
    return [{"role": "system", "content": RECON_SYSTEM}, {"role": "user", "content": user}]


def _rank_rho(ranking_a: list, ranking_b: list) -> float:
    """Spearman rho between two ticker rankings over their common items."""
    a = [str(x).strip().upper() for x in ranking_a]
    b = [str(x).strip().upper() for x in ranking_b]
    common = [t for t in a if t in b]
    if len(common) < 2:
        return 0.0
    pos_b = {t: i for i, t in enumerate(b)}
    ranks_b = [pos_b[t] for t in a if t in pos_b]
    n = len(common)
    d2 = sum((i - rb) ** 2 for i, rb in enumerate(_rerank(ranks_b)))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _rerank(vals: list[int]) -> list[int]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0] * len(vals)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def reconstruction_value(task: str, rebuilt: dict | None, original: dict) -> float:
    """Value in [0,1]: how faithfully the coalition reconstructs the original prediction."""
    if rebuilt is None:
        return 0.0
    if task == "stock":
        rank = rebuilt.get("ranking")
        if not isinstance(rank, list) or not rank:
            return 0.0
        return (_rank_rho(rank, original["ranking"]) + 1.0) / 2.0
    if task == "salary":
        try:
            pred = float(rebuilt.get("predicted_salary"))
        except (TypeError, ValueError):
            return 0.0
        orig = float(original["predicted_salary"])
        if orig <= 0:
            return 1.0 if pred <= 0 else 0.0
        return max(0.0, 1.0 - abs(pred - orig) / orig)
    raise ValueError(f"Unsupported task: {task}")


# ---------------------------------------------------------------------------
# Data loading: evidence + leak labels from saved eval outputs
# ---------------------------------------------------------------------------

def load_jobs(
    results_root: Path, model_dir: str, task: str, suffix: str, data_root: Path
) -> list[dict]:
    """One job per instance: evidence items, leak labels, original prediction."""
    log = results_root / model_dir / f"eval_{task}_{suffix}" / "eval_step_log.jsonl"
    rows: dict[str, dict] = {}
    for line in log.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["instance_id"]] = r  # last run wins (log is append-mode)
    instances = {i.instance_id: i for i in load_eval_instances(data_root, [task])}

    jobs, n_skipped = [], 0
    for iid, row in rows.items():
        inst = instances.get(iid)
        extracted = extract_original(row.get("completion", ""), task) if inst else None
        if extracted is None:
            n_skipped += 1
            continue
        evidence, parsed = extracted
        items = [
            {"fact": e["fact"], "source_date": e.get("source_date")}
            for e in evidence
            if isinstance(e, dict) and e.get("fact")
        ]
        leak_by_text = {
            c["text"].strip(): c.get("leak_verdict") == "leaked"
            for c in row.get("claims", [])
            if c.get("source") == "evidence"
        }
        labels = [leak_by_text.get(it["fact"].strip(), False) for it in items]
        if not items:
            n_skipped += 1
            continue
        jobs.append({
            "instance_id": iid, "inst": inst, "items": items,
            "labels": labels, "original": parsed,
        })
    logger.info("%s/%s: %d instances (%d skipped)", task, suffix, len(jobs), n_skipped)
    return jobs


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_instance(job: dict, submit, collect, cfg: argparse.Namespace, seed: int) -> dict:
    inst, items, labels = job["inst"], job["items"], job["labels"]
    n = len(items)
    budget = compute_budget(n, cfg.budget_c, cfg.budget_cap, cfg.exact_threshold)
    masks = (
        exact_coalitions(n) if n <= cfg.exact_threshold
        else leverage_shap_coalitions(n, budget, seed)
    )
    nonempty = [m for m in masks if sum(m) > 0]
    futures = [
        submit(build_coalition_messages(inst, [it for it, b in zip(items, m) if b]))
        for m in nonempty
    ]
    values_by_mask: dict[tuple[int, ...], float] = {m: 0.0 for m in masks if sum(m) == 0}
    for m, fut in zip(nonempty, futures):
        rebuilt = parse_json_object(collect(fut))
        values_by_mask[m] = reconstruction_value(inst.task, rebuilt, job["original"])
    ordered_masks = list(values_by_mask.keys())
    ordered_values = [values_by_mask[m] for m in ordered_masks]
    v_full = values_by_mask.get(tuple([1] * n), 0.0)
    phi = solve_projected_regression(ordered_masks, ordered_values, v_full)
    return {
        "instance_id": job["instance_id"],
        "n_claims": n,
        "n_coalitions": len(masks),
        "n_leaked": int(sum(labels)),
        "olr_instance": sum(labels) / n,
        "v_full": v_full,
        "phi": [round(float(p), 6) for p in phi],
        "labels": labels,
        "dclr_instance": shapley_dclr(phi, labels),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["stock", "salary"])
    parser.add_argument("--variants", default="base,rag,timespec,best")
    parser.add_argument("--model_dir", default="qwen35_35b")
    parser.add_argument("--value_model", default="qwen/qwen3.5-35b-a3b")
    parser.add_argument("--budget_c", type=float, default=4.0)
    parser.add_argument("--budget_cap", type=int, default=400)
    parser.add_argument("--exact_threshold", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Count coalitions, no API calls.")
    parser.add_argument("--results_root", default="runtime/results")
    parser.add_argument("--data_root", default="runtime/data")
    parser.add_argument("--output_dir", default="runtime/results/rebuttal/shapley_dclr")
    args = parser.parse_args()

    load_env(Path(__file__).resolve().parents[2])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    submit = collect = None
    if not args.dry_run:
        submit, collect = make_sampler(args.value_model, "openrouter")

    grand_total = 0
    for suffix in args.variants.split(","):
        jobs = load_jobs(Path(args.results_root), args.model_dir, args.task, suffix,
                         Path(args.data_root))
        if args.limit:
            jobs = jobs[: args.limit]

        total_coalitions = sum(
            compute_budget(len(j["items"]), args.budget_c, args.budget_cap,
                           args.exact_threshold)
            for j in jobs
        )
        grand_total += total_coalitions
        logger.info("%s/%s: %d instances, ~%d coalition calls",
                    args.task, suffix, len(jobs), total_coalitions)
        if args.dry_run:
            continue

        out_path = out_dir / f"{args.model_dir}_{args.task}_{suffix}.jsonl"
        done = set()
        if out_path.exists():
            done = {json.loads(l)["instance_id"] for l in out_path.read_text().splitlines()
                    if l.strip()}
            logger.info("%s: resuming, %d instances already done", out_path.name, len(done))

        with open(out_path, "a", encoding="utf-8") as f:
            for k, job in enumerate(jobs):
                if job["instance_id"] in done:
                    continue
                result = run_instance(job, submit, collect, args, seed=hash(job["instance_id"]) % 2**31)
                f.write(json.dumps(result) + "\n")
                f.flush()
                if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                    logger.info("%s/%s: %d/%d instances", args.task, suffix, k + 1, len(jobs))

        rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        dclr = sum(r["dclr_instance"] for r in rows) / len(rows)
        olr = sum(r["olr_instance"] for r in rows) / len(rows)
        logger.info("== %s %s (%s): n=%d  macro-OLR=%.2f%%  macro-DCLR=%.2f%% ==",
                    args.task, METHODS.get(suffix, suffix), args.model_dir,
                    len(rows), 100 * olr, 100 * dclr)

    if args.dry_run:
        logger.info("DRY RUN total: ~%d coalition calls for task=%s", grand_total, args.task)


if __name__ == "__main__":
    main()
