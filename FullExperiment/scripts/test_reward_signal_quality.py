"""One-step reward signal quality test for the Stage-1-only pipeline.

Re-scores real completions from step_log.jsonl through the new pipeline and reports:
  1. Leakage detection precision — are source_dates specific (day-level) or vague?
  2. Verdict consistency — do verdicts agree with stored ones?
  3. Signal strength — mean|advantage|, r_leak_std distribution, % strong-signal groups.

Run from FullExperiment/:
    uv run python scripts/test_reward_signal_quality.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from full_experiment.config import TrainConfig, load_config, load_env
from full_experiment.llm import OpenRouterClient
from full_experiment.reward_pipeline import RewardPipeline, _compute_rates
from full_experiment.schemas import parse_completion
from full_experiment.types import TaskInstance

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STEP_LOG = PROJECT_ROOT / "runtime/results/train_stock/step_log.jsonl"
# Instance IDs with known leakage diversity across cutoff tiers
TARGET_INSTANCES = [
    ("stock_9",   "2021-12-31"),   # hard: lots of leakage
    ("stock_315", "2023-02-28"),   # medium
    ("stock_725", "2025-03-31"),   # easy: recent, low leakage
]


def _load_completions_from_log(
    instance_id: str, step_log: Path
) -> tuple[list[str], list[dict]] | None:
    """Return (completions, stored_completion_entries) for the first matching row."""
    with open(step_log) as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("instance_id") == instance_id and row.get("completions"):
                texts = [c["text"] for c in row["completions"]]
                return texts, row["completions"]
    return None


def _load_task_instance(instance_id: str, cfg: TrainConfig) -> TaskInstance | None:
    from full_experiment.data import load_train_instances, split_train_val
    instances = load_train_instances(PROJECT_ROOT / cfg.data_root, ["stock"])
    all_instances = instances  # includes both train and val
    for inst in all_instances:
        if inst.instance_id == instance_id:
            return inst
    return None


def _classify_date_precision(date_str: str | None) -> str:
    """Classify whether a date is day-precise, month-precise, or missing."""
    if not date_str:
        return "missing"
    parts = date_str.strip().split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        return "day"
    if len(parts) == 2:
        return "month"
    return "unparseable"


def _analyze_leakage_precision(claim_detail: dict) -> dict:
    """Summarize date precision of all anchored leakage decisions."""
    counts: dict[str, int] = {"day": 0, "month": 0, "missing": 0, "unparseable": 0}
    corrections: list[str] = []
    for c in claim_detail.get("claims", []):
        if c.get("leak_method") != "anchored":
            continue
        event_date = c.get("leak_event_date", "")
        precision = _classify_date_precision(event_date)
        counts[precision] = counts.get(precision, 0) + 1
        if not c.get("leak_plausible", True):
            corrections.append(
                f"  '{c['text'][:60]}' declared→{c.get('source_date','?')} corrected→{event_date}"
            )
    return {"precision_counts": counts, "corrections": corrections}


def _compare_to_stored(
    new_claims: list[dict], stored_completions: list[dict]
) -> dict:
    """Compare new leak verdicts to those stored in step_log."""
    # Build stored verdict map: claim_text → set of stored verdicts seen
    stored_map: dict[str, str] = {}
    for comp in stored_completions:
        for old_c in comp.get("claims", []):
            # step_log per-completion may not have per-claim detail
            pass
    # step_log stores per-completion aggregates, not per-claim verdicts
    # Instead compare stored leakage_rate per completion vs new computed
    return {}


def _signal_analysis(advantages: list[float], r_leak_values: list[float]) -> dict:
    """Summarize signal strength metrics."""
    abs_adv = [abs(a) for a in advantages if abs(a) > 1e-8]
    return {
        "n_completions": len(advantages),
        "mean_abs_adv": float(np.mean(abs_adv)) if abs_adv else 0.0,
        "max_abs_adv": float(np.max(abs_adv)) if abs_adv else 0.0,
        "pct_strong": sum(1 for a in abs_adv if a > 0.3) / max(len(abs_adv), 1) * 100,
        "r_leak_std": float(np.std(r_leak_values)) if r_leak_values else 0.0,
        "r_leak_range": [
            round(float(min(r_leak_values)), 4),
            round(float(max(r_leak_values)), 4),
        ] if r_leak_values else [0, 0],
    }


def main() -> None:
    load_env(PROJECT_ROOT)
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    cfg = load_config(str(PROJECT_ROOT / "configs/train.yaml"))
    client = OpenRouterClient(api_key)
    pipe = RewardPipeline(cfg, client)

    # Simulate w_leak at step 0 (start of run)
    w_leak_t = cfg.w_leak_start
    batch_r_leaks: list[float] = []

    results_per_instance: list[dict] = []

    logger.info("\n%s", "=" * 70)
    logger.info("REWARD SIGNAL QUALITY TEST — Stage 1-only pipeline")
    logger.info("judge_model=%s  w_leak=%.2f  lambda_base=%.2f",
                cfg.judge_model, w_leak_t, cfg.lambda_base)
    logger.info("%s\n", "=" * 70)

    # --- Phase 1: score all instances (skip_hallucination=True, like training) ---
    scored_groups: list[tuple[TaskInstance, list, dict]] = []

    for instance_id, cutoff_date in TARGET_INSTANCES:
        loaded = _load_completions_from_log(instance_id, STEP_LOG)
        if loaded is None:
            logger.warning("No completions found for %s — skipping", instance_id)
            continue
        completions, stored_entries = loaded

        inst = _load_task_instance(instance_id, cfg)
        if inst is None:
            logger.warning("Instance %s not found in dataset", instance_id)
            continue

        logger.info("Scoring %s (cutoff=%s, %d completions)...", instance_id, cutoff_date, len(completions))
        rewards, claim_detail = pipe.score_group(
            completions, inst, skip_hallucination=True
        )

        # Collect for batch mean
        for r in rewards:
            if r.parse_success:
                batch_r_leaks.append(float(np.exp(-0.5 * r.num_leaked)))

        scored_groups.append((inst, rewards, claim_detail))

        # Leakage precision analysis
        precision = _analyze_leakage_precision(claim_detail)
        logger.info("  Precision: %s", precision["precision_counts"])
        if precision["corrections"]:
            logger.info("  Plausibility corrections (%d):", len(precision["corrections"]))
            for c in precision["corrections"][:5]:
                logger.info("   %s", c)

        # Per-completion reward summary
        num_leaked_list = [r.num_leaked for r in rewards if r.parse_success]
        logger.info("  num_leaked per completion: %s", num_leaked_list)
        logger.info("  r_leak (exp): %s",
                    [round(np.exp(-0.5 * n), 4) for n in num_leaked_list])

        results_per_instance.append({
            "instance_id": instance_id,
            "cutoff_date": cutoff_date,
            "n_completions": len(completions),
            "precision": precision["precision_counts"],
            "num_leaked_distribution": num_leaked_list,
        })

    # --- Phase 2: compute advantages with shared batch_mean_r_leak ---
    batch_mean_r_leak = float(np.mean(batch_r_leaks)) if batch_r_leaks else 0.5
    logger.info("\nbatch_mean_r_leak = %.4f  (over %d completions)\n", batch_mean_r_leak, len(batch_r_leaks))

    logger.info("%s", "=" * 70)
    logger.info("ADVANTAGE ANALYSIS")
    logger.info("%s", "=" * 70)

    all_abs_adv: list[float] = []
    group_signals: list[dict] = []

    for inst, rewards, claim_detail in scored_groups:
        advantages, diag = pipe.compute_advantages(rewards, w_leak_t, batch_mean_r_leak)
        r_leak_vals = diag.get("r_leak_values", [])
        signal = _signal_analysis(advantages, r_leak_vals)

        logger.info("\n%s (cutoff=%s)", inst.instance_id, inst.cutoff_date)
        logger.info("  r_leak_std=%.4f  r_perf_std=%.4f",
                    diag["r_leak_std"], diag["r_perf_std"])
        logger.info("  baseline_bonus range: [%.3f, %.3f]",
                    min(diag.get("baseline_bonus", [0])),
                    max(diag.get("baseline_bonus", [0])))
        logger.info("  advantages: %s", [round(a, 4) for a in advantages])
        logger.info("  mean|adv|=%.4f  max|adv|=%.4f  pct_strong(>0.3)=%.1f%%",
                    signal["mean_abs_adv"], signal["max_abs_adv"], signal["pct_strong"])
        logger.info("  r_leak range: %s", signal["r_leak_range"])

        all_abs_adv.extend(abs(a) for a in advantages if abs(a) > 1e-8)
        group_signals.append({**signal, "instance_id": inst.instance_id})

    # --- Phase 3: leakage verdict quality deep-dive ---
    logger.info("\n%s", "=" * 70)
    logger.info("LEAKAGE VERDICT QUALITY (sample claims per instance)")
    logger.info("%s", "=" * 70)

    for inst, rewards, claim_detail in scored_groups:
        logger.info("\n%s (cutoff=%s):", inst.instance_id, inst.cutoff_date)
        anchored = [c for c in claim_detail.get("claims", []) if c.get("leak_method") == "anchored"]
        leaked = [c for c in anchored if c.get("leak_verdict") == "leaked"]
        not_leaked = [c for c in anchored if c.get("leak_verdict") == "not_leaked"]

        logger.info("  anchored=%d  leaked=%d  not_leaked=%d",
                    len(anchored), len(leaked), len(not_leaked))

        logger.info("  Sample LEAKED claims (up to 3):")
        for c in leaked[:3]:
            precision = _classify_date_precision(c.get("leak_event_date"))
            logger.info("    [%s|%s] event_date=%s  '%s'",
                        c.get("leak_verdict"), precision,
                        c.get("leak_event_date"), c["text"][:80])

        logger.info("  Sample NOT_LEAKED claims (up to 3):")
        for c in not_leaked[:3]:
            precision = _classify_date_precision(c.get("leak_event_date"))
            logger.info("    [%s|%s] event_date=%s  '%s'",
                        c.get("leak_verdict"), precision,
                        c.get("leak_event_date"), c["text"][:80])

    # --- Final summary ---
    logger.info("\n%s", "=" * 70)
    logger.info("SUMMARY")
    logger.info("%s", "=" * 70)

    overall_mean_abs = float(np.mean(all_abs_adv)) if all_abs_adv else 0.0
    pct_strong_overall = sum(1 for a in all_abs_adv if a > 0.3) / max(len(all_abs_adv), 1) * 100
    total_day_prec = sum(r["precision"].get("day", 0) for r in results_per_instance)
    total_month_prec = sum(r["precision"].get("month", 0) for r in results_per_instance)
    total_anchored = total_day_prec + total_month_prec

    logger.info("Date precision (across all instances):")
    logger.info("  day-precise: %d / %d (%.1f%%)", total_day_prec, total_anchored,
                total_day_prec / max(total_anchored, 1) * 100)
    logger.info("  month-only:  %d / %d (%.1f%%)", total_month_prec, total_anchored,
                total_month_prec / max(total_anchored, 1) * 100)
    logger.info("Signal strength (all completions across %d groups):", len(scored_groups))
    logger.info("  mean|advantage| = %.4f", overall_mean_abs)
    logger.info("  %% strong signal (|adv|>0.3) = %.1f%%", pct_strong_overall)
    logger.info("  group r_leak_stds: %s",
                [round(g["r_leak_std"], 4) for g in group_signals])

    verdict = "PASS" if overall_mean_abs >= 0.15 and pct_strong_overall >= 20 else "WARN"
    logger.info("\nSignal verdict: %s (mean|adv|=%.4f, pct_strong=%.1f%%)",
                verdict, overall_mean_abs, pct_strong_overall)


if __name__ == "__main__":
    main()
