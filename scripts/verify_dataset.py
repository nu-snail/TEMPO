"""Verify dataset integrity: date precision, distribution, counts, and cutoff compliance.

Checks:
1. All dates are YYYY-MM-DD format (day-level precision)
2. No single year exceeds 30% of a task's data
3. Train/eval answer dates < 2025-03-01
4. Control answer dates > 2026-02-01
5. Instance counts meet targets
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_CUTOFF = "2025-03-01"
CONTROL_CUTOFF = "2026-02-01"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

TARGETS = {
    "stock_train": 1000,
    "stock_eval": 100,
    "stock_control": 100,
    "salary_train": 900,
    "salary_eval": 100,
    "salary_control": 25,
    "legal_train": 800,
    "legal_eval": 80,
    "legal_control": 50,
}


class Checker:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        print(f"  ERROR: {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  WARN:  {msg}")

    def ok(self, msg: str) -> None:
        print(f"  OK:    {msg}")


def read_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("questions", [])
    return data


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def get_answer_date_stock(q: dict) -> str:
    """For stock, the answer is known at end_date."""
    period = q.get("past_period", {})
    return period.get("end_date", period.get("end", ""))


def get_answer_date_salary(row: dict) -> str:
    return row.get("Signing_Date", "") or ""


def get_answer_date_legal(case: dict) -> str:
    return case.get("decision_date", case.get("cutoff_date", ""))


def check_dates(
    checker: Checker, dates: list[str], label: str
) -> None:
    """Check date format and report distribution."""
    invalid = [d for d in dates if d and not DATE_RE.match(d)]
    empty = sum(1 for d in dates if not d)
    if invalid:
        checker.error(f"{label}: {len(invalid)} dates not YYYY-MM-DD: {invalid[:5]}")
    elif empty:
        checker.warn(f"{label}: {empty}/{len(dates)} dates empty")
    else:
        checker.ok(f"{label}: all {len(dates)} dates are YYYY-MM-DD")

    years = Counter(d[:4] for d in dates if d)
    total = sum(years.values())
    if total:
        for y, n in years.most_common(3):
            pct = n / total * 100
            if pct > 30:
                checker.warn(f"{label}: year {y} has {pct:.1f}% ({n}/{total})")


def check_cutoff(
    checker: Checker, dates: list[str], threshold: str, direction: str, label: str
) -> None:
    """Check that all dates are before/after a threshold."""
    violations = []
    for d in dates:
        if not d:
            continue
        if direction == "before" and d >= threshold:
            violations.append(d)
        elif direction == "after" and d < threshold:
            violations.append(d)
    if violations:
        checker.error(
            f"{label}: {len(violations)} dates violate {direction} {threshold}: "
            f"{violations[:5]}"
        )
    else:
        valid = sum(1 for d in dates if d)
        checker.ok(f"{label}: all {valid} dates {direction} {threshold}")


def check_count(checker: Checker, actual: int, target: int, label: str) -> None:
    if actual < target:
        checker.warn(f"{label}: {actual} instances (target: {target})")
    else:
        checker.ok(f"{label}: {actual} instances (target: {target})")


def main() -> None:
    checker = Checker()

    # Stock train
    print("\n=== Stock Training ===")
    stock_train = read_json(DATASET_DIR / "train" / "stock_rankings" / "stock_rankings_train_combined.json")
    check_count(checker, len(stock_train), TARGETS["stock_train"], "stock_train")
    dates = [get_answer_date_stock(q) for q in stock_train]
    check_dates(checker, dates, "stock_train")
    check_cutoff(checker, dates, TRAIN_CUTOFF, "before", "stock_train")

    # Stock eval
    print("\n=== Stock Eval ===")
    stock_eval = read_json(DATASET_DIR / "evaluation" / "stock_rankings.json")
    check_count(checker, len(stock_eval), TARGETS["stock_eval"], "stock_eval")
    dates = [get_answer_date_stock(q) for q in stock_eval]
    check_dates(checker, dates, "stock_eval")
    check_cutoff(checker, dates, TRAIN_CUTOFF, "before", "stock_eval")

    # Stock control
    print("\n=== Stock Control ===")
    stock_ctrl = read_json(DATASET_DIR / "control" / "stock_rankings.json")
    check_count(checker, len(stock_ctrl), TARGETS["stock_control"], "stock_control")
    dates = [get_answer_date_stock(q) for q in stock_ctrl]
    check_dates(checker, dates, "stock_control")
    check_cutoff(checker, dates, CONTROL_CUTOFF, "after", "stock_control")

    # Salary train
    print("\n=== Salary Training ===")
    salary_train = read_csv(DATASET_DIR / "train" / "salary_prediction" / "salary_prediction_train_combined.csv")
    check_count(checker, len(salary_train), TARGETS["salary_train"], "salary_train")
    dates = [get_answer_date_salary(r) for r in salary_train]
    check_dates(checker, dates, "salary_train")
    non_empty = [d for d in dates if d]
    if non_empty:
        check_cutoff(checker, non_empty, TRAIN_CUTOFF, "before", "salary_train")
    else:
        checker.warn("salary_train: no Signing_Date values to check")

    # Salary eval
    print("\n=== Salary Eval ===")
    salary_eval = read_csv(DATASET_DIR / "evaluation" / "nba_salaries.csv")
    check_count(checker, len(salary_eval), TARGETS["salary_eval"], "salary_eval")
    dates = [get_answer_date_salary(r) for r in salary_eval]
    non_empty = [d for d in dates if d]
    if non_empty:
        check_dates(checker, non_empty, "salary_eval")
        check_cutoff(checker, non_empty, TRAIN_CUTOFF, "before", "salary_eval")

    # Salary control
    print("\n=== Salary Control ===")
    salary_ctrl = read_csv(DATASET_DIR / "control" / "salary_transactions.csv")
    check_count(checker, len(salary_ctrl), TARGETS["salary_control"], "salary_control")
    dates = [get_answer_date_salary(r) for r in salary_ctrl]
    check_dates(checker, dates, "salary_control")
    check_cutoff(checker, dates, CONTROL_CUTOFF, "after", "salary_control")

    # Legal train
    print("\n=== Legal Training ===")
    legal_train = read_json(DATASET_DIR / "train" / "legal_cases" / "legal_cases_train.json")
    check_count(checker, len(legal_train), TARGETS["legal_train"], "legal_train")
    dates = [get_answer_date_legal(c) for c in legal_train]
    check_dates(checker, dates, "legal_train")
    check_cutoff(checker, dates, TRAIN_CUTOFF, "before", "legal_train")

    # Legal eval
    print("\n=== Legal Eval ===")
    legal_eval = read_json(DATASET_DIR / "evaluation" / "legal_cases.json")
    check_count(checker, len(legal_eval), TARGETS["legal_eval"], "legal_eval")
    dates = [get_answer_date_legal(c) for c in legal_eval]
    check_dates(checker, dates, "legal_eval")
    check_cutoff(checker, dates, TRAIN_CUTOFF, "before", "legal_eval")

    # Legal control
    print("\n=== Legal Control ===")
    legal_ctrl = read_json(DATASET_DIR / "control" / "legal_cases.json")
    check_count(checker, len(legal_ctrl), TARGETS["legal_control"], "legal_control")
    if legal_ctrl:
        dates = [get_answer_date_legal(c) for c in legal_ctrl]
        check_dates(checker, dates, "legal_control")
        check_cutoff(checker, dates, CONTROL_CUTOFF, "after", "legal_control")
    else:
        checker.warn("legal_control: file empty or missing (run build_legal_control.py)")

    # Summary
    print(f"\n{'='*60}")
    print(f"ERRORS:   {len(checker.errors)}")
    print(f"WARNINGS: {len(checker.warnings)}")
    if checker.errors:
        print("\nErrors:")
        for e in checker.errors:
            print(f"  - {e}")
    sys.exit(1 if checker.errors else 0)


if __name__ == "__main__":
    main()
