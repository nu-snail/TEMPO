"""Analyze the outcome-recall probe: memorization-conditioned accuracy.

Joins per-instance recall scores (from recall_probe.py) with saved
per-instance evaluation results, then reports, for each method, accuracy
on memorized versus non-memorized instances. A method whose accuracy is
concentrated on memorized instances shows the contamination signature
(paper Appendix H.2).

Usage (from the repository root):
    uv run python scripts/analysis/analyze_recall_probe.py \
        --probe_dir results/analysis/recall_probe \
        --output_path results/analysis/recall_probe/analysis.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Memorized: stock if recalled-ranking Spearman rho >= threshold;
# salary if recalled AAV within relative-error threshold.
STOCK_RHO_THRESHOLDS = [0.6, 0.8]
SALARY_RELERR_THRESHOLDS = [0.10, 0.05]
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42

MODELS = {
    "qwen3_30b": "qwen3_30b_a3b",
    "qwen35_35b": "qwen35_35b_a3b",
}
METHODS = {
    "temporal_hint": "eval_{task}_base",
    "rag": "eval_{task}_rag",
    "timespec": "eval_{task}_timespec",
    "tempo": "eval_{task}_tempo",
}
TASKS = ["stock", "salary"]


def load_probe_scores(path: Path) -> dict[str, dict]:
    """Load probe rows keyed by instance_id."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["instance_id"]: r for r in rows}


def load_instance_performance(path: Path) -> dict[str, float]:
    """Load per-instance r_performance keyed by instance_id."""
    results = json.loads(path.read_text())
    return {r["instance_id"]: float(r.get("r_performance", 0.0)) for r in results}


def is_memorized(probe_row: dict, task: str, threshold: float) -> bool | None:
    """Classify one instance as memorized under the given threshold.

    Returns None when the probe produced no scorable recall; such
    instances are excluded from the conditional means (reported separately).
    """
    if not probe_row.get("recall_valid"):
        return None
    if task == "stock":
        return float(probe_row["recall_rho"]) >= threshold
    return float(probe_row["recall_rel_err"]) <= threshold


def recall_score(probe_row: dict, task: str) -> float | None:
    """Continuous memorization score in [0, 1] (higher = better recall)."""
    if not probe_row.get("recall_valid"):
        return None
    if task == "stock":
        return (float(probe_row["recall_rho"]) + 1.0) / 2.0
    return max(0.0, 1.0 - float(probe_row["recall_rel_err"]))


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def bootstrap_gap_ci(
    mem: list[float], non: list[float], rng: random.Random
) -> tuple[float, float] | None:
    """95% bootstrap CI for mean(mem) - mean(non)."""
    if len(mem) < 5 or len(non) < 5:
        return None
    gaps = []
    for _ in range(BOOTSTRAP_SAMPLES):
        m = [rng.choice(mem) for _ in mem]
        n = [rng.choice(non) for _ in non]
        gaps.append(mean(m) - mean(n))
    gaps.sort()
    lo = gaps[int(0.025 * len(gaps))]
    hi = gaps[int(0.975 * len(gaps))]
    return (lo, hi)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman correlation with average ranks for ties."""
    n = len(xs)
    if n < 5:
        return None

    def avg_ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = avg_ranks(xs), avg_ranks(ys)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def analyze_cell(
    probe: dict[str, dict],
    perf: dict[str, float],
    task: str,
    threshold: float,
    rng: random.Random,
) -> dict:
    """Compute memorization-conditioned accuracy for one method/task/model."""
    mem_perf, non_perf, scores, perfs = [], [], [], []
    n_unscorable = 0
    for iid, row in probe.items():
        if iid not in perf:
            continue
        flag = is_memorized(row, task, threshold)
        score = recall_score(row, task)
        if flag is None:
            n_unscorable += 1
            continue
        (mem_perf if flag else non_perf).append(perf[iid])
        scores.append(score)
        perfs.append(perf[iid])

    gap = None
    if mem_perf and non_perf:
        gap = mean(mem_perf) - mean(non_perf)
    ci = bootstrap_gap_ci(mem_perf, non_perf, rng)
    return {
        "n_memorized": len(mem_perf),
        "n_non_memorized": len(non_perf),
        "n_unscorable": n_unscorable,
        "acc_memorized": mean(mem_perf),
        "acc_non_memorized": mean(non_perf),
        "gap": gap,
        "gap_ci95": ci,
        "spearman_recall_vs_perf": spearman(scores, perfs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe_dir", default="results/analysis/recall_probe")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_path", default="results/analysis/recall_probe/analysis.json")
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    results_root = Path(args.results_root)
    rng = random.Random(BOOTSTRAP_SEED)

    analysis: dict = {}
    for model_dir, probe_tag in MODELS.items():
        probe_path = probe_dir / f"{probe_tag}.jsonl"
        if not probe_path.exists():
            print(f"[skip] missing probe file: {probe_path}")
            continue
        probe_all = load_probe_scores(probe_path)
        for task in TASKS:
            probe_task = {k: v for k, v in probe_all.items() if v["task"] == task}
            thresholds = (
                STOCK_RHO_THRESHOLDS if task == "stock" else SALARY_RELERR_THRESHOLDS
            )
            # Memorization rate is method-independent; report once per model/task.
            for thr in thresholds:
                flags = [is_memorized(r, task, thr) for r in probe_task.values()]
                n_mem = sum(1 for f in flags if f)
                n_scored = sum(1 for f in flags if f is not None)
                key = f"{model_dir}/{task}/threshold_{thr}"
                analysis[key] = {
                    "memorization_rate": n_mem / n_scored if n_scored else None,
                    "n_scored": n_scored,
                    "methods": {},
                }
                for method, dir_tpl in METHODS.items():
                    res_path = (
                        results_root / model_dir / dir_tpl.format(task=task) / "instance_results.json"
                    )
                    if not res_path.exists():
                        continue
                    perf = load_instance_performance(res_path)
                    analysis[key]["methods"][method] = analyze_cell(
                        probe_task, perf, task, thr, rng
                    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"Wrote {out_path}")
    _print_table(analysis)


def _print_table(analysis: dict) -> None:
    """Print a compact markdown summary of the primary thresholds."""
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else "--"
    for key, block in analysis.items():
        if "threshold_0.6" not in key and "threshold_0.1" not in key:
            continue
        print(f"\n## {key}  (memorization rate: {fmt(block['memorization_rate'])}, n={block['n_scored']})")
        print("| Method | n mem | n non | Acc(mem) | Acc(non) | Gap | 95% CI | rho(recall,perf) |")
        print("|---|---|---|---|---|---|---|---|")
        for method, cell in block["methods"].items():
            ci = cell["gap_ci95"]
            ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "--"
            print(
                f"| {method} | {cell['n_memorized']} | {cell['n_non_memorized']} "
                f"| {fmt(cell['acc_memorized'])} | {fmt(cell['acc_non_memorized'])} "
                f"| {fmt(cell['gap'])} | {ci_s} | {fmt(cell['spearman_recall_vs_perf'])} |"
            )


if __name__ == "__main__":
    main()
