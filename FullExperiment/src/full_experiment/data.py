"""Dataset loading utilities for FullExperiment."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .types import TaskInstance

STOCK_TRAIN_REL = "train/stock_rankings/stock_rankings_train_combined.json"
SALARY_TRAIN_REL = "train/salary_prediction/salary_prediction_train_combined.csv"
LEGAL_TRAIN_REL = "train/legal_cases/legal_cases_train.json"

STOCK_EVAL_REL = "eval/stock_rankings.json"
SALARY_EVAL_REL = "eval/nba_salaries.csv"
LEGAL_EVAL_REL = "eval/legal_cases.json"


def load_train_instances(data_root: Path, tasks: list[str]) -> list[TaskInstance]:
    """Load train instances for selected tasks."""

    all_instances: list[TaskInstance] = []
    if "stock" in tasks:
        all_instances.extend(_load_stock(data_root / STOCK_TRAIN_REL))
    if "salary" in tasks:
        all_instances.extend(_load_salary(data_root / SALARY_TRAIN_REL))
    if "legal" in tasks:
        all_instances.extend(_load_legal(data_root / LEGAL_TRAIN_REL))
    return all_instances


def split_train_val(
    instances: list[TaskInstance], val_fraction: float, seed: int
) -> tuple[list[TaskInstance], list[TaskInstance]]:
    """Split train and validation sets stratified by (task, cutoff_date)."""

    rng = random.Random(seed)
    by_group: dict[tuple[str, str], list[TaskInstance]] = {}
    for inst in instances:
        key = (inst.task, inst.cutoff_date)
        by_group.setdefault(key, []).append(inst)

    train: list[TaskInstance] = []
    val: list[TaskInstance] = []
    for group_instances in by_group.values():
        local = list(group_instances)
        rng.shuffle(local)
        n_val = max(1, int(len(local) * val_fraction))
        val.extend(local[:n_val])
        train.extend(local[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def load_eval_instances(data_root: Path, tasks: list[str]) -> list[TaskInstance]:
    """Load evaluation instances for selected tasks."""

    all_instances: list[TaskInstance] = []
    if "stock" in tasks:
        all_instances.extend(_load_stock(data_root / STOCK_EVAL_REL, prefix="stock_eval"))
    if "salary" in tasks:
        all_instances.extend(_load_salary_eval(data_root / SALARY_EVAL_REL))
    if "legal" in tasks:
        all_instances.extend(_load_legal(data_root / LEGAL_EVAL_REL, prefix="legal_eval"))
    return all_instances


def _load_salary_eval(path: Path) -> list[TaskInstance]:
    """Load NBA salary evaluation set (no Sport column)."""

    rows = _read_csv(path)
    out: list[TaskInstance] = []
    for idx, row in enumerate(rows):
        year = str(row.get("Year", "")).strip()
        cutoff = f"{year}-01-01" if year else ""
        aav = _parse_aav(row.get("AAV", "0"))
        out.append(
            TaskInstance(
                task="salary",
                instance_id=f"salary_eval_{idx}",
                cutoff_date=cutoff,
                ground_truth={"aav": aav},
                raw={**row, "Sport": "NBA"},
            )
        )
    return out


def _load_stock(path: Path, prefix: str = "stock") -> list[TaskInstance]:
    payload = _read_json(path)
    questions = payload if isinstance(payload, list) else payload.get("questions", [])
    windows = payload.get("metadata", {}).get("windows", {}) if isinstance(payload, dict) else {}

    out: list[TaskInstance] = []
    for idx, q in enumerate(questions):
        window_id = str(q.get("window_id", ""))
        cutoff = (
            q.get("model_knowledge_cutoff")
            or q.get("past_period", {}).get("cutoff")
            or windows.get(window_id, {}).get("cutoff", "")
        )
        out.append(
            TaskInstance(
                task="stock",
                instance_id=f"{prefix}_{q.get('question_id', idx)}",
                cutoff_date=cutoff,
                ground_truth={"ranking": q.get("past_period", {}).get("ranking", [])},
                raw=q,
            )
        )
    return out


def _load_salary(path: Path) -> list[TaskInstance]:
    rows = _read_csv(path)
    out: list[TaskInstance] = []
    for idx, row in enumerate(rows):
        year = str(row.get("Year", "")).strip()
        cutoff = f"{year}-01-01" if year else ""
        out.append(
            TaskInstance(
                task="salary",
                instance_id=f"salary_{idx}",
                cutoff_date=cutoff,
                ground_truth={"aav": _parse_aav(row.get("AAV", "0"))},
                raw=row,
            )
        )
    return out


def _load_legal(path: Path, prefix: str = "legal") -> list[TaskInstance]:
    rows = _read_json(path)
    out: list[TaskInstance] = []
    for idx, row in enumerate(rows):
        out.append(
            TaskInstance(
                task="legal",
                instance_id=row.get("case_id", f"{prefix}_{idx}"),
                cutoff_date=row.get("cutoff_date", ""),
                ground_truth={"outcome": row.get("outcome", "")},
                raw=row,
            )
        )
    return out


def _parse_aav(raw: object) -> float:
    """Safely parse AAV field from various formats ($1,000,000 / 1000000 / None)."""
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).replace("$", "").replace(",", "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

