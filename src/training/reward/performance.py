"""Task-specific performance reward computation.

All rewards are normalised to [0, 1] before MO-GRPO variance normalisation.

| Task   | Metric              | R_perf formula         | Range  |
|--------|---------------------|------------------------|--------|
| stock  | Spearman rho        | (rho + 1) / 2         | [0, 1] |
| salary | Relative error (RE) | max(0, 1 - RE)        | [0, 1] |
| legal  | Brier score (BS)    | 1 - BS(p_hat, p_true) | [0, 1] |
"""

from __future__ import annotations

import logging
from typing import Any

from src.shared.data_loader import TaskInstance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_performance_reward(
    parsed: dict[str, Any] | None,
    instance: TaskInstance,
) -> float:
    """Compute the task-specific performance reward in [0, 1].

    Returns 0.0 for unparseable responses (parse failure penalty).
    """
    if parsed is None:
        return 0.0

    dispatch = {
        "stock": _stock_reward,
        "salary": _salary_reward,
        "legal": _legal_reward,
    }
    fn = dispatch.get(instance.task)
    if fn is None:
        logger.warning("No performance reward for task '%s'", instance.task)
        return 0.0

    try:
        return fn(parsed, instance)
    except Exception as exc:
        logger.warning("Performance reward error (%s): %s", instance.task, exc)
        return 0.0


# ---------------------------------------------------------------------------
# Per-task implementations
# ---------------------------------------------------------------------------


def _stock_reward(parsed: dict[str, Any], instance: TaskInstance) -> float:
    """Spearman rank correlation mapped to [0, 1] via (rho + 1) / 2."""
    predicted = parsed.get("ranking", [])
    true_ranking = instance.ground_truth.get("true_ranking", [])
    if not predicted or not true_ranking:
        return 0.0

    common = [t for t in predicted if t in true_ranking]
    if len(common) < 2:
        return 0.0

    true_order = [t for t in true_ranking if t in common]
    pred_order = [t for t in predicted if t in common]

    rho = _spearman(pred_order, true_order)
    return (rho + 1.0) / 2.0


def _salary_reward(parsed: dict[str, Any], instance: TaskInstance) -> float:
    """max(0, 1 - relative_error).  RE = |pred - true| / true."""
    predicted = parsed.get("predicted_salary", 0.0)
    raw_aav = instance.ground_truth.get("aav", 0)
    try:
        true_aav = float(str(raw_aav).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        true_aav = 0.0
    if true_aav <= 0 or predicted <= 0:
        return 0.0

    re = abs(predicted - true_aav) / true_aav
    return max(0.0, 1.0 - re)


def _legal_reward(parsed: dict[str, Any], instance: TaskInstance) -> float:
    """1 - Brier score.  BS = (p_hat - p_true)^2."""
    p_hat = parsed.get("probability_petitioner")
    if p_hat is None:
        return 0.0

    outcome = instance.ground_truth.get("outcome", "").upper()
    p_true = 1.0 if outcome == "PETITIONER" else 0.0

    bs = (p_hat - p_true) ** 2
    return 1.0 - bs


# ---------------------------------------------------------------------------
# Spearman correlation helper
# ---------------------------------------------------------------------------


def _spearman(predicted: list[str], true_order: list[str]) -> float:
    """Spearman rank correlation between two orderings of the same items.

    Both lists must contain exactly the same set of items.
    Returns a float in [-1, 1].
    """
    n = len(predicted)
    if n < 2:
        return 0.0

    true_rank = {item: rank for rank, item in enumerate(true_order)}
    pred_rank = {item: rank for rank, item in enumerate(predicted)}

    d_sq = sum((pred_rank[item] - true_rank[item]) ** 2 for item in predicted)
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))
