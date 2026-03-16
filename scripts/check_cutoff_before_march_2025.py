#!/usr/bin/env python3
"""Check that every instance's knowledge cutoff date is before March 2025.

Uses CoverageGatedExperiment data loading (explicit salary cutoffs). Reads from
dataset/ with evaluation/ for eval files (dataset uses 'evaluation' not 'eval').
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root and CoverageGatedExperiment for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CoverageGatedExperiment" / "src"))

from cov_gated_experiment.data import (
    load_train_instances,
    load_eval_instances,
)
from cov_gated_experiment.types import TaskInstance

MARCH_2025 = "2025-03-01"  # cutoff must be < this to be "before March 2025"


def load_eval_from_evaluation_folder(data_root: Path) -> list[TaskInstance]:
    """Load eval instances from dataset/evaluation/ (dataset has evaluation not eval)."""
    from cov_gated_experiment.data import (
        _load_stock,
        _load_salary_eval,
        _load_legal,
    )

    eval_dir = data_root / "evaluation"
    out: list[TaskInstance] = []
    out.extend(_load_stock(eval_dir / "stock_rankings.json", prefix="stock_eval"))
    out.extend(_load_salary_eval(eval_dir / "nba_salaries.csv"))
    out.extend(_load_legal(eval_dir / "legal_cases.json", prefix="legal_eval"))
    return out


def main() -> None:
    data_root = PROJECT_ROOT / "dataset"
    if not data_root.is_dir():
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    train_instances = load_train_instances(data_root, ["stock", "salary", "legal"])
    eval_dir = data_root / "evaluation"
    if eval_dir.is_dir():
        eval_instances = load_eval_from_evaluation_folder(data_root)
    else:
        eval_instances = load_eval_instances(data_root, ["stock", "salary", "legal"])
    all_instances = train_instances + eval_instances

    not_before: list[tuple[str, str, str]] = []  # (task, instance_id, cutoff_date)
    for inst in all_instances:
        if inst.cutoff_date and inst.cutoff_date >= MARCH_2025:
            not_before.append((inst.task, inst.instance_id, inst.cutoff_date))

    total = len(all_instances)
    print(f"Total instances: {total}")
    print(f"Threshold: cutoff date must be before {MARCH_2025}")
    print()

    if not not_before:
        print("All instances have knowledge cutoff before March 2025.")
        return

    print(f"Instances with cutoff on or after March 2025: {len(not_before)}")
    print()
    by_task_cutoff: dict[tuple[str, str], list[str]] = {}
    for task, iid, cutoff in not_before:
        key = (task, cutoff)
        by_task_cutoff.setdefault(key, []).append(iid)

    for (task, cutoff), ids in sorted(by_task_cutoff.items(), key=lambda x: (x[0][0], x[0][1])):
        sample = ids[:5]
        suffix = "..." if len(ids) > 5 else ""
        print(f"  {task}: cutoff={cutoff}  count={len(ids)}  example_ids={sample}{suffix}")
    print()
    print("First 30 violating instances:")
    for task, iid, cutoff in not_before[:30]:
        print(f"  {task}  {iid}  cutoff={cutoff}")
    if len(not_before) > 30:
        print(f"  ... and {len(not_before) - 30} more.")
    sys.exit(1)


if __name__ == "__main__":
    main()
