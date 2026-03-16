"""Adapter bridging CoverageGatedExperiment reward pipeline to TRL's reward function API.

TRL's GRPOTrainer calls reward functions with signature:
    def fn(completions: list[list[dict]], **kwargs) -> list[float]

where completions[i] is [{"role": "assistant", "content": "..."}] and kwargs
contains dataset columns (one value per completion, repeated for each
generation within a group).

This adapter:
1. Extracts completion text from TRL's message format
2. Groups completions by prompt (using instance_id from kwargs)
3. Calls our score_group() for each group
4. Computes a composite scalar reward per completion
5. Logs completion text and per-completion reward breakdown to step_log.jsonl
6. Returns flat list[float] matching TRL's expected output
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from .reward_pipeline import RewardPipeline
from .types import CompletionReward, TaskInstance

logger = logging.getLogger(__name__)

_call_counter = 0


def build_reward_fn(
    pipeline: RewardPipeline,
    cov_target: float,
    cov_floor: float,
    step_log_path: Path | None = None,
) -> callable:
    """Build a TRL-compatible reward function wrapping our RewardPipeline.

    Args:
        pipeline: Initialized RewardPipeline instance.
        cov_target: Coverage target for gating (from config).
        cov_floor: Coverage floor (from config).
        step_log_path: If set, log detailed per-completion diagnostics.

    Returns:
        Callable with TRL reward function signature.
    """

    def reward_fn(
        completions: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[float]:
        """Compute composite rewards for a batch of completions."""
        global _call_counter
        _call_counter += 1

        texts = [c[0]["content"] if c else "" for c in completions]
        instance_ids = kwargs.get("instance_id", [])
        tasks = kwargs.get("task", [])
        cutoff_dates = kwargs.get("cutoff_date", [])
        ground_truth_jsons = kwargs.get("ground_truth_json", [])

        groups: dict[str, dict[str, Any]] = {}
        for i, iid in enumerate(instance_ids):
            if iid not in groups:
                gt = json.loads(ground_truth_jsons[i])
                inst = TaskInstance(
                    task=tasks[i],
                    instance_id=iid,
                    cutoff_date=cutoff_dates[i],
                    ground_truth=gt,
                    raw={},
                )
                groups[iid] = {"instance": inst, "indices": [], "texts": []}
            groups[iid]["indices"].append(i)
            groups[iid]["texts"].append(texts[i])

        rewards = [0.0] * len(completions)
        all_details: list[dict[str, Any]] = [{}] * len(completions)

        for iid, grp in groups.items():
            try:
                group_rewards, claim_detail = pipeline.score_group(
                    grp["texts"], grp["instance"], skip_hallucination=True
                )
                for local_idx, global_idx in enumerate(grp["indices"]):
                    r = group_rewards[local_idx]
                    scalar = _composite_reward(r, cov_target, cov_floor)
                    rewards[global_idx] = scalar
                    all_details[global_idx] = {
                        "instance_id": iid,
                        "parse_success": r.parse_success,
                        "r_valid": r.r_valid,
                        "r_performance": r.r_performance,
                        "leakage_rate": r.leakage_rate,
                        "coverage": r.coverage,
                        "num_claims": r.num_claims,
                        "num_leaked": r.num_leaked,
                        "composite_reward": scalar,
                        "text_preview": grp["texts"][local_idx][:300],
                    }
            except Exception:
                logger.exception("Reward scoring failed for instance %s", iid)

        _log_batch(texts, rewards, all_details, step_log_path)
        return rewards

    return reward_fn


def _log_batch(
    texts: list[str],
    rewards: list[float],
    details: list[dict[str, Any]],
    step_log_path: Path | None,
) -> None:
    """Print and optionally persist completion diagnostics."""
    n_parse_ok = sum(1 for d in details if d.get("parse_success"))
    n_total = len(texts)
    reward_arr = rewards
    mean_r = sum(reward_arr) / len(reward_arr) if reward_arr else 0.0

    logger.info(
        "REWARD BATCH: n=%d  parsed=%d/%d  mean_reward=%.4f  "
        "min=%.4f  max=%.4f",
        n_total, n_parse_ok, n_total, mean_r,
        min(reward_arr) if reward_arr else 0,
        max(reward_arr) if reward_arr else 0,
    )

    for i, d in enumerate(details):
        if not d:
            continue
        logger.info(
            "  [%d] iid=%s parse=%s r_perf=%.3f leak=%.3f cov=%.3f "
            "claims=%d leaked=%d composite=%.4f\n"
            "       text: %.200s",
            i, d.get("instance_id", "?"),
            d.get("parse_success"), d.get("r_performance", 0),
            d.get("leakage_rate", 0), d.get("coverage", 0),
            d.get("num_claims", 0), d.get("num_leaked", 0),
            d.get("composite_reward", 0),
            d.get("text_preview", "")[:200],
        )

    if step_log_path is not None:
        entry = {
            "call": _call_counter,
            "n_total": n_total,
            "n_parsed": n_parse_ok,
            "mean_reward": mean_r,
            "completions": [
                {**d, "reward": rewards[i]}
                for i, d in enumerate(details) if d
            ],
        }
        with open(step_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _composite_reward(
    r: CompletionReward,
    cov_target: float,
    cov_floor: float,
) -> float:
    """Compute a single scalar reward from a CompletionReward.

    Combines leakage penalty, coverage gating, and performance score:
        r_composite = r_performance * coverage_gate * r_leak_exp

    where:
        coverage_gate = cov_floor + (1 - cov_floor) * min(1, coverage / cov_target)
        r_leak_exp = exp(-0.5 * num_leaked)

    Unparseable completions get 0.0.
    """
    if not r.parse_success:
        return 0.0
    cov_gate = cov_floor + (1.0 - cov_floor) * min(1.0, r.coverage / max(cov_target, 1e-8))
    r_leak_exp = math.exp(-0.5 * r.num_leaked)
    return r.r_performance * cov_gate * r_leak_exp
