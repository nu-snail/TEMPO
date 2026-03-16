"""Shared dataclasses for the FullExperiment project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskInstance:
    """Single training/evaluation instance."""

    task: str
    instance_id: str
    cutoff_date: str
    ground_truth: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class CompletionReward:
    """Reward breakdown for one completion.

    Training advantage (two-mode, coverage-gated):
      r_eff = r_performance * g(coverage)
      Performance Mode (all_clean): adv = w_perf_clean * (z(r_eff) + baseline)
      Leakage Mode (mixed/leaked):  adv = w_leak * (z(r_leak) + baseline) + cov_bonus

    Hallucination detection is skipped during training; retained for
    validation and evaluation.
    """

    r_format: float
    r_valid: float
    r_performance: float
    leakage_rate: float
    hallucination_rate: float
    coverage: float
    parse_success: bool
    num_claims: int
    num_leaked: int
    num_unsupported: int
    num_undecidable: int
