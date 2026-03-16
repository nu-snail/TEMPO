"""Metrics calculation (Phase 4 of evaluation).

Computes:
- OLR: Overall Leakage Rate = # leaked / # total
- Shapley-DCLR: sum(|phi_i| * lambda_i) / sum(|phi_i|)
  where lambda_i = 1 if leaked, 0 otherwise

Aggregation across instances via mean +/- std.
Faithfully replicates the TimeSPEC MetricsCalculator.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def calculate_shapley_metrics(
    shapley_values: list[float],
    is_leaked: list[bool],
) -> dict[str, Any]:
    """Compute OLR and Shapley-DCLR for a single instance.

    Args:
        shapley_values: Shapley values [phi_1, phi_2, ...].
        is_leaked: Boolean leak flags aligned with shapley_values.

    Returns:
        Dict with ``olr``, ``shapley_dclr``, ``total_claims``,
        ``leaked_claims``, ``total_shapley_weight``, ``leaked_shapley_weight``.
    """
    n = len(shapley_values)
    if n == 0:
        return _empty_metrics()

    if len(is_leaked) != n:
        raise ValueError(f"Length mismatch: {n} Shapley values vs {len(is_leaked)} leak flags")

    phi = np.array(shapley_values)
    leaked = np.array(is_leaked, dtype=bool)

    leaked_count = int(np.sum(leaked))
    olr = leaked_count / n

    abs_phi = np.abs(phi)
    total_w = float(np.sum(abs_phi))
    if total_w == 0:
        dclr = 0.0
        leaked_w = 0.0
    else:
        leaked_w = float(np.sum(abs_phi[leaked]))
        dclr = leaked_w / total_w

    return {
        "total_claims": n,
        "leaked_claims": leaked_count,
        "olr": float(olr),
        "shapley_dclr": float(dclr),
        "total_shapley_weight": total_w,
        "leaked_shapley_weight": leaked_w,
    }


def aggregate_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-instance metrics with mean +/- std.

    Args:
        metrics_list: List of dicts from ``calculate_shapley_metrics``.

    Returns:
        Aggregated dict with ``olr``, ``shapley_dclr`` (mean/std), totals.
    """
    if not metrics_list:
        return {
            "n_instances": 0,
            "olr": {"mean": 0.0, "std": 0.0},
            "shapley_dclr": {"mean": 0.0, "std": 0.0},
        }

    olrs = [m["olr"] for m in metrics_list]
    dclrs = [m["shapley_dclr"] for m in metrics_list]
    n = len(metrics_list)

    return {
        "n_instances": n,
        "olr": {
            "mean": float(np.mean(olrs)),
            "std": float(np.std(olrs, ddof=1)) if n > 1 else 0.0,
        },
        "shapley_dclr": {
            "mean": float(np.mean(dclrs)),
            "std": float(np.std(dclrs, ddof=1)) if n > 1 else 0.0,
        },
        "total_claims_sum": sum(m["total_claims"] for m in metrics_list),
        "leaked_claims_sum": sum(m["leaked_claims"] for m in metrics_list),
        "total_shapley_weight_sum": sum(m["total_shapley_weight"] for m in metrics_list),
        "leaked_shapley_weight_sum": sum(m["leaked_shapley_weight"] for m in metrics_list),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "total_claims": 0,
        "leaked_claims": 0,
        "olr": 0.0,
        "shapley_dclr": 0.0,
        "total_shapley_weight": 0.0,
        "leaked_shapley_weight": 0.0,
    }
