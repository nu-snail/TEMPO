"""Compare base vs. trained evaluation results across all tasks.

Reads summary.json and instance_results.json from each eval output directory
and produces a formatted comparison table.

Usage:
    uv run python scripts/analyze_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path("results/qwen")

RUNS = {
    "base": RESULTS_ROOT / "eval_base_all",
    "trained_stock": RESULTS_ROOT / "eval_trained_stock",
    "trained_salary": RESULTS_ROOT / "eval_trained_salary",
    "trained_legal": RESULTS_ROOT / "eval_trained_legal",
}

TASKS = ["stock", "salary", "legal"]


def load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_get(summary: dict, task: str, field: str, sub: str = "mean") -> float:
    task_data = summary.get(task, {})
    val = task_data.get(field, {})
    if isinstance(val, dict):
        return val.get(sub, 0.0)
    return float(val) if val else 0.0


def compute_zero_leak_rate(instances: list[dict], task: str) -> float:
    task_rows = [r for r in instances if r["task"] == task]
    if not task_rows:
        return 0.0
    return sum(1 for r in task_rows if r.get("olr", 1.0) == 0.0) / len(task_rows)


def compute_category_breakdown(eval_dir: Path, task: str) -> dict[str, dict]:
    """Aggregate per-category leakage across all instances for a task."""
    metrics_dir = eval_dir / "phase4_metrics"
    if not metrics_dir.exists():
        return {}

    agg: dict[str, dict[str, int]] = {}
    for f in metrics_dir.iterdir():
        if not f.name.endswith(".json"):
            continue
        data = load_json(f)
        cat_bd = data.get("category_breakdown", {})
        for cat, counts in cat_bd.items():
            if cat not in agg:
                agg[cat] = {"total": 0, "leaked": 0}
            agg[cat]["total"] += counts.get("total", 0)
            agg[cat]["leaked"] += counts.get("leaked", 0)
    return agg


def print_separator(char: str = "=", width: int = 90) -> None:
    print(char * width)


def main() -> None:
    summaries: dict[str, dict] = {}
    instances: dict[str, list] = {}

    for name, path in RUNS.items():
        summary_path = path / "summary.json"
        instance_path = path / "instance_results.json"
        if not summary_path.exists():
            print(f"WARNING: {summary_path} not found — skipping {name}")
            continue
        summaries[name] = load_json(summary_path)
        if instance_path.exists():
            instances[name] = load_json(instance_path)

    if not summaries:
        print("No evaluation results found. Run evaluations first.")
        sys.exit(1)

    # --- Main comparison table ---
    print_separator()
    print("  EVALUATION RESULTS: Base Model vs. GRPO-Trained Models")
    print_separator()
    print()

    header = f"{'Task':<10} {'Model':<18} {'OLR':>10} {'Zero-Leak%':>12} {'Performance':>13} {'Parse%':>8} {'N':>5}"
    print(header)
    print("-" * len(header))

    for task in TASKS:
        base_key = "base"
        trained_key = f"trained_{task}"

        for label, key in [("Base", base_key), ("Trained", trained_key)]:
            if key not in summaries:
                continue
            s = summaries[key]
            inst_list = instances.get(key, [])

            olr_mean = safe_get(s, task, "olr", "mean")
            olr_std = safe_get(s, task, "olr", "std")
            perf_mean = safe_get(s, task, "performance", "mean")
            perf_std = safe_get(s, task, "performance", "std")
            parse_rate = s.get(task, {}).get("parse_rate", 0.0)
            n_inst = s.get(task, {}).get("n_instances", 0)
            zero_leak = compute_zero_leak_rate(inst_list, task)

            olr_str = f"{olr_mean:.3f}±{olr_std:.3f}"
            perf_str = f"{perf_mean:.3f}±{perf_std:.3f}"
            zl_str = f"{zero_leak * 100:.1f}%"
            parse_str = f"{parse_rate * 100:.1f}%"

            print(f"{task:<10} {label:<18} {olr_str:>10} {zl_str:>12} {perf_str:>13} {parse_str:>8} {n_inst:>5}")

        print()

    # --- Per-category leakage breakdown ---
    print_separator("-")
    print("  PER-CATEGORY LEAKAGE BREAKDOWN")
    print_separator("-")
    print()

    for task in TASKS:
        base_dir = RUNS["base"]
        trained_dir = RUNS.get(f"trained_{task}")

        print(f"  {task.upper()}")
        cat_header = f"  {'Category':<8} {'Base(leaked/total)':>22} {'Trained(leaked/total)':>24}"
        print(cat_header)

        base_cats = compute_category_breakdown(base_dir, task)
        trained_cats = compute_category_breakdown(trained_dir, task) if trained_dir and trained_dir.exists() else {}

        all_cats = sorted(set(list(base_cats.keys()) + list(trained_cats.keys())))
        for cat in all_cats:
            bc = base_cats.get(cat, {"total": 0, "leaked": 0})
            tc = trained_cats.get(cat, {"total": 0, "leaked": 0})
            print(f"  {cat:<8} {bc['leaked']:>8}/{bc['total']:<12} {tc['leaked']:>8}/{tc['total']:<12}")
        print()

    # --- Delta summary ---
    print_separator("-")
    print("  DELTA: Trained - Base (negative OLR = less leakage = better)")
    print_separator("-")
    print()

    delta_header = f"{'Task':<10} {'ΔOLR':>10} {'ΔPerf':>10} {'ΔZero-Leak%':>14}"
    print(delta_header)
    print("-" * len(delta_header))

    for task in TASKS:
        trained_key = f"trained_{task}"
        if "base" not in summaries or trained_key not in summaries:
            continue

        base_olr = safe_get(summaries["base"], task, "olr", "mean")
        trained_olr = safe_get(summaries[trained_key], task, "olr", "mean")
        base_perf = safe_get(summaries["base"], task, "performance", "mean")
        trained_perf = safe_get(summaries[trained_key], task, "performance", "mean")

        base_zl = compute_zero_leak_rate(instances.get("base", []), task)
        trained_zl = compute_zero_leak_rate(instances.get(trained_key, []), task)

        d_olr = trained_olr - base_olr
        d_perf = trained_perf - base_perf
        d_zl = (trained_zl - base_zl) * 100

        sign_olr = "+" if d_olr >= 0 else ""
        sign_perf = "+" if d_perf >= 0 else ""
        sign_zl = "+" if d_zl >= 0 else ""

        print(f"{task:<10} {sign_olr}{d_olr:>9.3f} {sign_perf}{d_perf:>9.3f} {sign_zl}{d_zl:>13.1f}%")

    print()
    print_separator()

    # Save structured comparison to JSON
    comparison = {}
    for task in TASKS:
        trained_key = f"trained_{task}"
        if "base" not in summaries or trained_key not in summaries:
            continue
        comparison[task] = {
            "base": {
                "olr_mean": safe_get(summaries["base"], task, "olr", "mean"),
                "olr_std": safe_get(summaries["base"], task, "olr", "std"),
                "performance_mean": safe_get(summaries["base"], task, "performance", "mean"),
                "performance_std": safe_get(summaries["base"], task, "performance", "std"),
                "parse_rate": summaries["base"].get(task, {}).get("parse_rate", 0.0),
                "zero_leak_rate": compute_zero_leak_rate(instances.get("base", []), task),
            },
            "trained": {
                "olr_mean": safe_get(summaries[trained_key], task, "olr", "mean"),
                "olr_std": safe_get(summaries[trained_key], task, "olr", "std"),
                "performance_mean": safe_get(summaries[trained_key], task, "performance", "mean"),
                "performance_std": safe_get(summaries[trained_key], task, "performance", "std"),
                "parse_rate": summaries[trained_key].get(task, {}).get("parse_rate", 0.0),
                "zero_leak_rate": compute_zero_leak_rate(instances.get(trained_key, []), task),
            },
        }

    out_path = RESULTS_ROOT / "eval_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Comparison saved to {out_path}")
    print()


if __name__ == "__main__":
    main()
