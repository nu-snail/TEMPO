"""Merge SCOTUS and appellate cases, split into train and eval sets.

Combines legal_cases_train.json (SCOTUS) and appellate_cases.json (federal
appellate) into a unified training set, then carves out ~100 eval instances.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent
EVAL_DIR = OUT_DIR.parent.parent / "evaluation"
SCOTUS_PATH = OUT_DIR / "legal_cases_train.json"
APPELLATE_PATH = OUT_DIR / "appellate_cases.json"
TRAIN_OUT = OUT_DIR / "legal_cases_train.json"
EVAL_OUT = EVAL_DIR / "legal_cases.json"

MAX_ANSWER_DATE = "2025-03-01"
EVAL_SIZE = 100
SEED = 42


def main() -> None:
    scotus = []
    if SCOTUS_PATH.exists():
        with open(SCOTUS_PATH) as f:
            scotus = json.load(f)
    print(f"SCOTUS cases: {len(scotus)}")

    appellate = []
    if APPELLATE_PATH.exists():
        with open(APPELLATE_PATH) as f:
            appellate = json.load(f)
    print(f"Appellate cases: {len(appellate)}")

    # Deduplicate (guards against re-running after TRAIN_OUT == SCOTUS_PATH)
    seen: set[str] = set()
    combined: list[dict] = []
    for case in scotus + appellate:
        key = (case.get("case_name", ""), case.get("decision_date", ""), case.get("court", ""))
        sig = "|".join(str(k) for k in key)
        if sig in seen:
            continue
        seen.add(sig)
        combined.append(case)

    # Filter: decision_date < MAX_ANSWER_DATE
    all_cases = []
    dropped = 0
    for case in combined:
        dd = case.get("decision_date", "") or case.get("cutoff_date", "")
        if dd and dd >= MAX_ANSWER_DATE:
            dropped += 1
            continue
        all_cases.append(case)

    print(f"After date filter (< {MAX_ANSWER_DATE}): {len(all_cases)} (dropped {dropped})")

    # Split into eval + train
    rng = random.Random(SEED)

    # Stratify eval by year for temporal diversity
    by_year: dict[str, list[dict]] = {}
    for case in all_cases:
        dd = case.get("decision_date", "") or case.get("cutoff_date", "")
        year = dd[:4] if dd else "unknown"
        by_year.setdefault(year, []).append(case)

    eval_cases: list[dict] = []
    train_cases: list[dict] = []

    years = sorted(y for y in by_year if y != "unknown")
    eval_per_year = max(1, (EVAL_SIZE + len(years) - 1) // len(years)) if years else 0

    for year in years:
        pool = by_year[year]
        rng.shuffle(pool)
        n = min(eval_per_year, len(pool) // 2)
        # Balance outcomes within year
        pet = [c for c in pool if c["outcome"] == "PETITIONER"]
        resp = [c for c in pool if c["outcome"] == "RESPONDENT"]
        n_pet = min(n // 2, len(pet))
        n_resp = min(n - n_pet, len(resp))
        eval_from_year = pet[:n_pet] + resp[:n_resp]
        eval_set = set(id(c) for c in eval_from_year)
        eval_cases.extend(eval_from_year)
        train_cases.extend(c for c in pool if id(c) not in eval_set)

    # Handle unknown year
    if "unknown" in by_year:
        train_cases.extend(by_year["unknown"])

    # Trim eval to target if overshot
    if len(eval_cases) > EVAL_SIZE:
        rng.shuffle(eval_cases)
        overflow = eval_cases[EVAL_SIZE:]
        eval_cases = eval_cases[:EVAL_SIZE]
        train_cases.extend(overflow)

    # Reassign IDs
    rng.shuffle(train_cases)
    for i, case in enumerate(train_cases, start=1):
        case["case_id"] = f"train_{i:04d}"
    for i, case in enumerate(eval_cases, start=1):
        case["case_id"] = f"eval_{i:04d}"

    # Save
    field_order = [
        "case_id", "case_name", "petitioner", "respondent", "court",
        "term", "facts", "conclusion", "oral_argument_date",
        "decision_date", "cutoff_date", "outcome", "vote_split",
    ]

    def order_fields(case: dict) -> dict:
        return {k: case.get(k) for k in field_order if k in case}

    train_ordered = [order_fields(c) for c in train_cases]
    eval_ordered = [order_fields(c) for c in eval_cases]

    with open(TRAIN_OUT, "w") as f:
        json.dump(train_ordered, f, indent=2, ensure_ascii=False)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT, "w") as f:
        json.dump(eval_ordered, f, indent=2, ensure_ascii=False)

    print(f"\nTrain: {len(train_ordered)} -> {TRAIN_OUT}")
    print(f"Eval: {len(eval_ordered)} -> {EVAL_OUT}")

    # Stats
    print("\nTrain per-year:")
    for y in sorted(set(c.get("decision_date", "")[:4] for c in train_ordered)):
        n = sum(1 for c in train_ordered if c.get("decision_date", "")[:4] == y)
        print(f"  {y}: {n}")

    print("\nEval per-year:")
    for y in sorted(set(c.get("decision_date", "")[:4] for c in eval_ordered)):
        n = sum(1 for c in eval_ordered if c.get("decision_date", "")[:4] == y)
        print(f"  {y}: {n}")

    print("\nEval outcome distribution:")
    pet = sum(1 for c in eval_ordered if c.get("outcome") == "PETITIONER")
    resp = sum(1 for c in eval_ordered if c.get("outcome") == "RESPONDENT")
    print(f"  PETITIONER: {pet}")
    print(f"  RESPONDENT: {resp}")


if __name__ == "__main__":
    main()
