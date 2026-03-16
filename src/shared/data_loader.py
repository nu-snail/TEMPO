"""Unified data loader for all three prediction tasks.

Loads training and evaluation datasets for stock ranking, salary prediction,
and legal case prediction.  Returns normalised dicts with a common ``task``
field so downstream code can dispatch without knowing file formats.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskInstance:
    """A single prediction instance with all fields needed for training/eval."""

    task: str
    instance_id: str
    cutoff_date: str
    ground_truth: dict[str, Any]
    raw: dict[str, Any]

    # Convenience accessors -------------------------------------------------

    @property
    def question_text(self) -> str:
        """Human-readable question summary used in prompts."""
        return _build_question_text(self.task, self.raw)


@dataclass
class DataSplit:
    """Container for a train / validation split."""

    train: list[TaskInstance]
    val: list[TaskInstance]
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading helpers (one per task)
# ---------------------------------------------------------------------------


def load_stock_train(path: Path | None = None) -> list[TaskInstance]:
    """Load stock ranking training data (combined JSON with all windows)."""
    if path is None:
        path = config.TRAIN_DATA_DIR / config.TASK_INFO["stock"]["train_file"]
    data = _read_json(path)

    questions = data if isinstance(data, list) else data.get("questions", [])
    # Window cutoff dates live in metadata.windows keyed by window_id
    window_cutoffs: dict[str, str] = {}
    if isinstance(data, dict):
        for wid, winfo in data.get("metadata", {}).get("windows", {}).items():
            window_cutoffs[str(wid)] = winfo.get("cutoff", "")

    instances: list[TaskInstance] = []
    for q in questions:
        wid = str(q.get("window_id", ""))
        period = q.get("past_period", {})
        cutoff = (
            q.get("model_knowledge_cutoff")
            or period.get("cutoff")
            or window_cutoffs.get(wid, "")
        )
        gt = {
            "true_ranking": period.get("ranking", q.get("true_ranking", [])),
            "returns": period.get("returns", q.get("returns", {})),
        }
        instances.append(TaskInstance(
            task="stock",
            instance_id=f"stock_train_{q.get('question_id', len(instances))}",
            cutoff_date=cutoff,
            ground_truth=gt,
            raw=q,
        ))
    return instances


def load_salary_train(path: Path | None = None) -> list[TaskInstance]:
    """Load salary prediction training data (combined CSV)."""
    if path is None:
        path = config.TRAIN_DATA_DIR / config.TASK_INFO["salary"]["train_file"]
    rows = _read_csv(path)

    instances: list[TaskInstance] = []
    for i, row in enumerate(rows):
        year = row.get("Year", "")
        cutoff = row.get("Signing_Date", "") or (f"{year}-01-01" if year else "")
        gt = {
            "aav": row.get("AAV", ""),
            "total_value": row.get("Total_Value", ""),
            "years": row.get("Years", ""),
            "to_team": row.get("To_Team", ""),
        }
        instances.append(TaskInstance(
            task="salary",
            instance_id=f"salary_train_{i}",
            cutoff_date=cutoff,
            ground_truth=gt,
            raw=row,
        ))
    return instances


def load_legal_train(path: Path | None = None) -> list[TaskInstance]:
    """Load legal case prediction training data (JSON list)."""
    if path is None:
        path = config.TRAIN_DATA_DIR / config.TASK_INFO["legal"]["train_file"]
    cases = _read_json(path)

    instances: list[TaskInstance] = []
    for case in cases:
        gt = {
            "outcome": case.get("outcome", ""),
            "vote_split": case.get("vote_split", ""),
        }
        instances.append(TaskInstance(
            task="legal",
            instance_id=case.get("case_id", f"legal_train_{len(instances)}"),
            cutoff_date=case.get("cutoff_date", ""),
            ground_truth=gt,
            raw=case,
        ))
    return instances


def load_stock_eval(path: Path | None = None) -> list[TaskInstance]:
    """Load stock ranking evaluation data."""
    if path is None:
        path = config.EVAL_DATA_DIR / config.TASK_INFO["stock"]["eval_file"]
    data = _read_json(path)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    global_cutoff = metadata.get("model_knowledge_cutoff", "")

    instances: list[TaskInstance] = []
    for q in questions:
        period = q.get("past_period", {})
        cutoff = (
            q.get("model_knowledge_cutoff")
            or period.get("cutoff")
            or global_cutoff
        )
        gt = {
            "true_ranking": period.get("ranking", q.get("true_ranking", [])),
            "returns": period.get("returns", q.get("returns", {})),
        }
        instances.append(TaskInstance(
            task="stock",
            instance_id=f"stock_eval_{q.get('question_id', len(instances))}",
            cutoff_date=cutoff,
            ground_truth=gt,
            raw=q,
        ))
    return instances


def load_salary_eval(path: Path | None = None) -> list[TaskInstance]:
    """Load salary prediction evaluation data (NBA CSV)."""
    if path is None:
        path = config.EVAL_DATA_DIR / config.TASK_INFO["salary"]["eval_file"]
    rows = _read_csv(path)

    instances: list[TaskInstance] = []
    for i, row in enumerate(rows):
        year = row.get("Year", row.get("Season", ""))
        cutoff = row.get("Signing_Date", "") or (f"{year}-01-01" if year else "")
        gt = {
            "aav": row.get("AAV", row.get("Salary", "")),
            "total_value": row.get("Total_Value", row.get("Value", "")),
            "years": row.get("Years", ""),
            "to_team": row.get("To_Team", row.get("Team", "")),
        }
        instances.append(TaskInstance(
            task="salary",
            instance_id=f"salary_eval_{i}",
            cutoff_date=cutoff,
            ground_truth=gt,
            raw=row,
        ))
    return instances


def load_legal_eval(path: Path | None = None) -> list[TaskInstance]:
    """Load legal case prediction evaluation data."""
    if path is None:
        path = config.EVAL_DATA_DIR / config.TASK_INFO["legal"]["eval_file"]
    cases = _read_json(path)

    instances: list[TaskInstance] = []
    for case in cases:
        gt = {
            "outcome": case.get("outcome", ""),
            "vote_split": case.get("vote_split", ""),
        }
        instances.append(TaskInstance(
            task="legal",
            instance_id=case.get("case_id", f"legal_eval_{len(instances)}"),
            cutoff_date=case.get("cutoff_date", ""),
            ground_truth=gt,
            raw=case,
        ))
    return instances


# ---------------------------------------------------------------------------
# Aggregate loaders
# ---------------------------------------------------------------------------

_TRAIN_LOADERS = {
    "stock": load_stock_train,
    "salary": load_salary_train,
    "legal": load_legal_train,
}

_EVAL_LOADERS = {
    "stock": load_stock_eval,
    "salary": load_salary_eval,
    "legal": load_legal_eval,
}


def load_all_train(tasks: list[str] | None = None) -> list[TaskInstance]:
    """Load training data for the specified tasks (default: all three).

    Args:
        tasks: Subset of ["stock", "salary", "legal"]. None means all.

    Returns:
        Flat list of TaskInstance objects across requested tasks.
    """
    tasks = tasks or config.VALID_TASKS
    instances: list[TaskInstance] = []
    for t in tasks:
        if t not in _TRAIN_LOADERS:
            raise ValueError(f"Unknown task '{t}'. Valid: {config.VALID_TASKS}")
        instances.extend(_TRAIN_LOADERS[t]())
    return instances


def load_all_eval(tasks: list[str] | None = None) -> list[TaskInstance]:
    """Load evaluation data for the specified tasks (default: all three)."""
    tasks = tasks or config.VALID_TASKS
    instances: list[TaskInstance] = []
    for t in tasks:
        if t not in _EVAL_LOADERS:
            raise ValueError(f"Unknown task '{t}'. Valid: {config.VALID_TASKS}")
        instances.extend(_EVAL_LOADERS[t]())
    return instances


def make_train_val_split(
    instances: list[TaskInstance],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> DataSplit:
    """Stratified train/val split preserving task proportions.

    Args:
        instances: Full training set.
        val_fraction: Fraction held out for validation (default 10%).
        seed: Random seed for reproducibility.

    Returns:
        DataSplit with .train and .val lists.
    """
    rng = random.Random(seed)

    by_task: dict[str, list[TaskInstance]] = {}
    for inst in instances:
        by_task.setdefault(inst.task, []).append(inst)

    train_set: list[TaskInstance] = []
    val_set: list[TaskInstance] = []

    for task, task_instances in by_task.items():
        shuffled = list(task_instances)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_set.extend(shuffled[:n_val])
        train_set.extend(shuffled[n_val:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)

    counts = {t: len(v) for t, v in by_task.items()}
    return DataSplit(
        train=train_set,
        val=val_set,
        meta={
            "total": len(instances),
            "train_size": len(train_set),
            "val_size": len(val_set),
            "val_fraction": val_fraction,
            "task_counts": counts,
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_question_text(task: str, raw: dict[str, Any]) -> str:
    """Build a compact question string for prompt construction."""
    if task == "stock":
        companies = raw.get("companies", [])
        sector = raw.get("sector", "Unknown")
        period = raw.get("past_period", {})
        start = period.get("start_date", period.get("start", ""))
        end = period.get("end_date", period.get("end", ""))
        return (
            f"Rank these {len(companies)} {sector} companies by stock return "
            f"from {start} to {end}: {', '.join(companies)}"
        )

    if task == "salary":
        player = raw.get("Player", "Unknown")
        position = raw.get("Position", "")
        sport = raw.get("Sport", "NBA")
        year = raw.get("Year", raw.get("Season", ""))
        from_team = raw.get("From_Team", raw.get("Previous_Team", ""))
        return (
            f"Predict the free-agent contract for {player} ({position}, {sport}) "
            f"entering the {year} market from {from_team}"
        )

    if task == "legal":
        case_name = raw.get("case_name", "Unknown")
        return f"Predict the outcome of {case_name}"

    return f"[{task}] {raw.get('question_id', 'unknown')}"
