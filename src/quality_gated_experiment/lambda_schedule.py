"""Cosine annealing schedules for reward weight mixing."""

from __future__ import annotations

import math


def cosine_anneal(step: int, total_steps: int, v_min: float, v_max: float) -> float:
    """Cosine anneal from *v_min* to *v_max* over *total_steps*."""

    if total_steps <= 0:
        return v_min
    ratio = min(max(step / total_steps, 0.0), 1.0)
    return v_min + (v_max - v_min) * 0.5 * (1.0 - math.cos(math.pi * ratio))

