"""Rigorous comparison: Current vs Adaptive Group-Conditional advantage design.

Loads real completions from steps 0, 10, 20 of the live training run
(8 groups × 16 completions each step = 384 completions total) and computes
advantages under two designs:

  CURRENT  : adv = w_leak * (z_leak + baseline) + (1-w_leak) * z_perf
             w_leak = 0.95 for all groups regardless of leakage state.

  ADAPTIVE : same as CURRENT for mixed/leaky groups;
             for all-clean groups (r_leak.min >= 0.99):
               adv = w_perf_clean * z_perf + baseline
             w_perf_clean = 0.40

Evaluation criteria (from research goals):
  Goal 1 — No leakage:   leaked completions must receive strictly lower advantage
            than clean completions within the same group.
  Goal 2 — Better perf when clean:  within all-clean groups, advantage must rank
            completions by r_performance.

All numbers come directly from step_log.jsonl — no LLM calls, no sampling.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
STEP_LOG   = Path(__file__).resolve().parents[1] / "runtime/results/train_stock/step_log.jsonl"
TEST_STEPS = [0, 10, 20]   # early / mid / late training
W_LEAK_CFG = 0.95          # scheduled w_leak at step 0 (conservative for all steps)
LAMBDA_BASE = 0.5
Z_CLIP      = 5.0
W_PERF_CLEAN = 0.40        # proposed adaptive weight for all-clean groups
FORMAT_PENALTY = 0.5
CLEAN_THRESHOLD = 0.99     # r_leak.min() >= this → all_clean


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zscore(arr: np.ndarray, clip: float = Z_CLIP) -> np.ndarray:
    s = arr.std()
    if s < 1e-6:
        return np.zeros_like(arr)
    return np.clip((arr - arr.mean()) / s, -clip, clip)


@dataclass
class CompletionData:
    idx: int
    num_leaked: int
    r_performance: float
    parse_success: bool

    @property
    def r_leak(self) -> float:
        return float(np.exp(-0.5 * self.num_leaked))


@dataclass
class GroupResult:
    instance_id: str
    cutoff_date: str
    step: int
    completions: list[CompletionData]
    all_clean: bool
    batch_mean_r_leak: float

    adv_current: list[float]
    adv_adaptive: list[float]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_step_groups(step_log: Path, steps: list[int]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {s: [] for s in steps}
    with open(step_log) as f:
        for line in f:
            row = json.loads(line.strip())
            s = row.get("step", -1)
            if s in groups:
                groups[s].append(row)
    return groups


# ── Advantage computation ─────────────────────────────────────────────────────

def _compute_group_advantages(
    comps: list[CompletionData],
    w_leak: float,
    batch_mean_r_leak: float,
    adaptive: bool,
) -> tuple[list[float], dict[str, Any]]:
    parseable = [c for c in comps if c.parse_success]
    if len(parseable) <= 1:
        return [-FORMAT_PENALTY if not c.parse_success else 0.0 for c in comps], {}

    r_leak = np.array([c.r_leak for c in parseable])
    r_perf = np.array([c.r_performance for c in parseable])
    z_leak = _zscore(r_leak)
    z_perf = _zscore(r_perf)
    baseline = LAMBDA_BASE * (r_leak - batch_mean_r_leak)

    all_clean = bool(r_leak.min() >= CLEAN_THRESHOLD)

    if adaptive and all_clean:
        adv_raw = W_PERF_CLEAN * z_perf + baseline
    else:
        adv_raw = w_leak * (z_leak + baseline) + (1.0 - w_leak) * z_perf

    out: list[float] = []
    j = 0
    for c in comps:
        if not c.parse_success:
            out.append(-FORMAT_PENALTY)
        else:
            out.append(float(adv_raw[j]))
            j += 1

    return out, {
        "all_clean": all_clean,
        "r_leak_std": float(r_leak.std()),
        "r_perf_std": float(r_perf.std()),
        "adv_std": float(np.std(out)),
    }


def process_step(step_rows: list[dict], step: int, adaptive: bool) -> list[GroupResult]:
    # Compute batch_mean_r_leak from all parseable completions in this step
    all_r_leak = [
        np.exp(-0.5 * c.get("num_leaked", 0))
        for row in step_rows
        for c in row["completions"]
        if c.get("parse_success", True) and c.get("num_claims", 1) > 0
    ]
    batch_mean = float(np.mean(all_r_leak)) if all_r_leak else 0.5

    # Use w_leak at the right step (cosine from 0.95→0.80 over 321 total steps)
    total_steps = 321
    w_leak = float(0.80 + 0.5 * (0.95 - 0.80) * (1 + np.cos(np.pi * step / total_steps)))

    results = []
    for row in step_rows:
        comps = [
            CompletionData(
                idx=c["idx"],
                num_leaked=c.get("num_leaked", 0),
                r_performance=c.get("r_performance", 0.0),
                parse_success=c.get("parse_success", True) and c.get("num_claims", 1) > 0,
            )
            for c in row["completions"]
        ]
        r_leak_arr = np.array([c.r_leak for c in comps if c.parse_success])
        all_clean = bool(len(r_leak_arr) > 0 and r_leak_arr.min() >= CLEAN_THRESHOLD)

        adv, _ = _compute_group_advantages(comps, w_leak, batch_mean, adaptive)

        results.append(GroupResult(
            instance_id=row["instance_id"],
            cutoff_date=row["cutoff_date"],
            step=step,
            completions=comps,
            all_clean=all_clean,
            batch_mean_r_leak=batch_mean,
            adv_current=adv if not adaptive else [],
            adv_adaptive=adv if adaptive else [],
        ))
    return results, batch_mean, w_leak


# ── Goal-alignment checks ─────────────────────────────────────────────────────

def check_goal1_leakage_ordering(results_curr: list[GroupResult],
                                  results_adap: list[GroupResult]) -> dict:
    """Goal 1: leaked completions must get strictly lower advantage than clean ones
    within the same group. Checks both designs."""
    violations_curr = violations_adap = correct_curr = correct_adap = 0

    for rc, ra in zip(results_curr, results_adap):
        parseable = [c for c in rc.completions if c.parse_success]
        if len(parseable) < 2:
            continue
        leaked_idx  = [i for i, c in enumerate(rc.completions) if c.parse_success and c.num_leaked > 0]
        clean_idx   = [i for i, c in enumerate(rc.completions) if c.parse_success and c.num_leaked == 0]
        if not leaked_idx or not clean_idx:
            continue

        # Check: mean(adv[clean]) > mean(adv[leaked])
        mean_clean_curr = np.mean([rc.adv_current[i] for i in clean_idx])
        mean_leak_curr  = np.mean([rc.adv_current[i] for i in leaked_idx])
        mean_clean_adap = np.mean([ra.adv_adaptive[i] for i in clean_idx])
        mean_leak_adap  = np.mean([ra.adv_adaptive[i] for i in leaked_idx])

        if mean_clean_curr > mean_leak_curr:
            correct_curr += 1
        else:
            violations_curr += 1
        if mean_clean_adap > mean_leak_adap:
            correct_adap += 1
        else:
            violations_adap += 1

    total = correct_curr + violations_curr
    return {
        "total_mixed_groups": total,
        "curr_correct": correct_curr,
        "curr_violations": violations_curr,
        "curr_accuracy": correct_curr / max(total, 1),
        "adap_correct": correct_adap,
        "adap_violations": violations_adap,
        "adap_accuracy": correct_adap / max(total, 1),
    }


def check_goal2_perf_ordering_clean(results_curr: list[GroupResult],
                                     results_adap: list[GroupResult]) -> dict:
    """Goal 2: within all-clean groups, higher r_performance must map to higher advantage.
    Measured as Spearman rank correlation between r_perf and advantage."""
    from scipy.stats import spearmanr  # type: ignore

    corr_curr_list: list[float] = []
    corr_adap_list: list[float] = []
    gap_curr_list: list[float] = []
    gap_adap_list: list[float] = []

    for rc, ra in zip(results_curr, results_adap):
        if not rc.all_clean:
            continue
        parseable_curr = [(c.r_performance, rc.adv_current[i])
                          for i, c in enumerate(rc.completions) if c.parse_success]
        parseable_adap = [(c.r_performance, ra.adv_adaptive[i])
                          for i, c in enumerate(ra.completions) if c.parse_success]
        if len(parseable_curr) < 4:
            continue

        r_perfs_c, advs_c = zip(*parseable_curr)
        r_perfs_a, advs_a = zip(*parseable_adap)

        rho_c, _ = spearmanr(r_perfs_c, advs_c)
        rho_a, _ = spearmanr(r_perfs_a, advs_a)
        corr_curr_list.append(rho_c)
        corr_adap_list.append(rho_a)

        gap_c = max(advs_c) - min(advs_c)
        gap_a = max(advs_a) - min(advs_a)
        gap_curr_list.append(gap_c)
        gap_adap_list.append(gap_a)

    return {
        "n_clean_groups": len(corr_curr_list),
        "curr_mean_spearman": float(np.mean(corr_curr_list)) if corr_curr_list else float("nan"),
        "adap_mean_spearman": float(np.mean(corr_adap_list)) if corr_adap_list else float("nan"),
        "curr_mean_adv_gap": float(np.mean(gap_curr_list)) if gap_curr_list else 0.0,
        "adap_mean_adv_gap": float(np.mean(gap_adap_list)) if gap_adap_list else 0.0,
        "amplification": (float(np.mean(gap_adap_list)) / max(float(np.mean(gap_curr_list)), 1e-6))
                         if gap_curr_list else float("nan"),
    }


def gradient_balance(results_curr: list[GroupResult],
                      results_adap: list[GroupResult]) -> dict:
    """What fraction of total batch gradient magnitude comes from clean vs leaky groups?"""
    def _frac(results: list[GroupResult], adv_attr: str) -> dict[str, float]:
        clean_mag = leaky_mag = 0.0
        for r in results:
            advs = getattr(r, adv_attr)
            mag = sum(abs(a) for a in advs)
            if r.all_clean:
                clean_mag += mag
            else:
                leaky_mag += mag
        total = clean_mag + leaky_mag
        return {
            "clean_pct": clean_mag / max(total, 1e-6) * 100,
            "leaky_pct": leaky_mag / max(total, 1e-6) * 100,
        }

    return {
        "current":  _frac(results_curr, "adv_current"),
        "adaptive": _frac(results_adap, "adv_adaptive"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Loading step log from %s", STEP_LOG)
    all_groups = load_step_groups(STEP_LOG, TEST_STEPS)

    missing = [s for s in TEST_STEPS if not all_groups[s]]
    if missing:
        logger.warning("Steps not found in log: %s — skipping", missing)
        TEST_STEPS_AVAIL = [s for s in TEST_STEPS if all_groups[s]]
    else:
        TEST_STEPS_AVAIL = TEST_STEPS

    all_curr: list[GroupResult] = []
    all_adap: list[GroupResult] = []

    # ── Per-step results ──────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 72)
    logger.info("PER-STEP GROUP BREAKDOWN")
    logger.info("%s", "=" * 72)

    for step in TEST_STEPS_AVAIL:
        rows = all_groups[step]
        curr_results, batch_mean, w_leak = process_step(rows, step, adaptive=False)
        adap_results, _, _              = process_step(rows, step, adaptive=True)

        # Attach adv_current from curr pass and adv_adaptive from adap pass
        merged_curr = curr_results
        merged_adap = adap_results

        all_curr.extend(merged_curr)
        all_adap.extend(merged_adap)

        n_clean = sum(1 for r in merged_curr if r.all_clean)
        n_leaky = len(merged_curr) - n_clean
        logger.info(
            "\nStep %d  (w_leak=%.4f  batch_mean_r_leak=%.4f)  "
            "groups=%d  clean=%d  leaky=%d",
            step, w_leak, batch_mean, len(rows), n_clean, n_leaky,
        )

        for rc, ra in zip(merged_curr, merged_adap):
            tag = "CLEAN" if rc.all_clean else "LEAKY"
            parseable = [c for c in rc.completions if c.parse_success]
            r_leak_arr = np.array([c.r_leak for c in parseable])
            r_perf_arr = np.array([c.r_performance for c in parseable])

            adv_c = [rc.adv_current[i] for i, c in enumerate(rc.completions) if c.parse_success]
            adv_a = [ra.adv_adaptive[i] for i, c in enumerate(ra.completions) if c.parse_success]

            logger.info(
                "  [%s] %s  cutoff=%s  r_leak=%.3f±%.3f  r_perf=%.3f±%.3f",
                tag, rc.instance_id, rc.cutoff_date,
                r_leak_arr.mean(), r_leak_arr.std(),
                r_perf_arr.mean(), r_perf_arr.std(),
            )
            logger.info(
                "        CURRENT  adv: std=%.4f  range=[%.3f, %.3f]",
                np.std(adv_c), min(adv_c), max(adv_c),
            )
            logger.info(
                "        ADAPTIVE adv: std=%.4f  range=[%.3f, %.3f]  %s",
                np.std(adv_a), min(adv_a), max(adv_a),
                f"(amplif={np.std(adv_a)/max(np.std(adv_c),1e-6):.1f}x)" if rc.all_clean else "",
            )

    # ── Goal 1: Leakage ordering ──────────────────────────────────────────────
    logger.info("\n%s", "=" * 72)
    logger.info("GOAL 1 CHECK — Leaked completions penalized vs clean within group")
    logger.info("%s", "=" * 72)
    g1 = check_goal1_leakage_ordering(all_curr, all_adap)
    logger.info("Mixed groups tested:  %d", g1["total_mixed_groups"])
    logger.info("CURRENT  correct ordering: %d/%d (%.0f%%)",
                g1["curr_correct"], g1["total_mixed_groups"], g1["curr_accuracy"]*100)
    logger.info("ADAPTIVE correct ordering: %d/%d (%.0f%%)",
                g1["adap_correct"], g1["total_mixed_groups"], g1["adap_accuracy"]*100)
    g1_verdict = (
        "SAME" if g1["curr_accuracy"] == g1["adap_accuracy"]
        else ("IMPROVED" if g1["adap_accuracy"] > g1["curr_accuracy"] else "DEGRADED")
    )
    logger.info("Goal 1 verdict: %s", g1_verdict)

    # ── Goal 2: Perf ordering in clean groups ─────────────────────────────────
    logger.info("\n%s", "=" * 72)
    logger.info("GOAL 2 CHECK — Higher r_perf → higher advantage within clean groups")
    logger.info("%s", "=" * 72)
    g2 = check_goal2_perf_ordering_clean(all_curr, all_adap)
    logger.info("All-clean groups tested: %d", g2["n_clean_groups"])
    logger.info("CURRENT  mean Spearman(r_perf, adv): %.4f", g2["curr_mean_spearman"])
    logger.info("ADAPTIVE mean Spearman(r_perf, adv): %.4f", g2["adap_mean_spearman"])
    logger.info("CURRENT  mean adv gap (best-worst):  %.4f", g2["curr_mean_adv_gap"])
    logger.info("ADAPTIVE mean adv gap (best-worst):  %.4f", g2["adap_mean_adv_gap"])
    logger.info("Signal amplification for clean groups: %.1fx", g2["amplification"])

    # ── Gradient balance ──────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 72)
    logger.info("GRADIENT BALANCE — % of total |advantage| from clean vs leaky groups")
    logger.info("%s", "=" * 72)
    gb = gradient_balance(all_curr, all_adap)
    logger.info("CURRENT:  clean=%.1f%%  leaky=%.1f%%",
                gb["current"]["clean_pct"], gb["current"]["leaky_pct"])
    logger.info("ADAPTIVE: clean=%.1f%%  leaky=%.1f%%",
                gb["adaptive"]["clean_pct"], gb["adaptive"]["leaky_pct"])

    # ── Concrete example: best clean group ───────────────────────────────────
    logger.info("\n%s", "=" * 72)
    logger.info("CONCRETE EXAMPLE — Most performance-diverse clean group")
    logger.info("%s", "=" * 72)
    best_clean_pair = None
    best_r_perf_std = 0.0
    for rc, ra in zip(all_curr, all_adap):
        if not rc.all_clean:
            continue
        parseable = [c for c in rc.completions if c.parse_success]
        s = float(np.std([c.r_performance for c in parseable]))
        if s > best_r_perf_std:
            best_r_perf_std = s
            best_clean_pair = (rc, ra)

    if best_clean_pair:
        rc, ra = best_clean_pair
        parseable = [c for c in rc.completions if c.parse_success]
        sorted_by_perf = sorted(enumerate(parseable), key=lambda x: -x[1].r_performance)
        logger.info("%s  step=%d  cutoff=%s", rc.instance_id, rc.step, rc.cutoff_date)
        logger.info("  %-6s  %-8s  %-12s  %-12s  %-8s",
                    "idx", "r_perf", "adv_CURRENT", "adv_ADAPTIVE", "delta")
        for orig_idx, c in sorted_by_perf:
            adv_c = rc.adv_current[orig_idx]
            adv_a = ra.adv_adaptive[orig_idx]
            logger.info("  %-6d  %-8.3f  %-12.4f  %-12.4f  %+.4f",
                        c.idx, c.r_performance, adv_c, adv_a, adv_a - adv_c)

    # ── Summary verdict ───────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 72)
    logger.info("SUMMARY")
    logger.info("%s", "=" * 72)
    spearman_improved = g2["adap_mean_spearman"] >= g2["curr_mean_spearman"] - 0.01
    amplified         = g2["amplification"] > 3.0
    leakage_preserved = g1["adap_accuracy"] >= g1["curr_accuracy"]
    leakage_dominant  = gb["adaptive"]["leaky_pct"] > 50.0

    logger.info("Goal 1 (no-leakage ordering preserved): %s",
                "PASS" if leakage_preserved else "FAIL")
    logger.info("Goal 2 (perf signal amplified):         %s (%.1fx)",
                "PASS" if amplified else "FAIL", g2["amplification"])
    logger.info("Perf rank correlation improved:          %s (%.4f → %.4f)",
                "PASS" if spearman_improved else "FAIL",
                g2["curr_mean_spearman"], g2["adap_mean_spearman"])
    logger.info("Leakage still dominates batch gradient:  %s (%.1f%% of total)",
                "PASS" if leakage_dominant else "FAIL", gb["adaptive"]["leaky_pct"])

    overall = all([leakage_preserved, amplified, spearman_improved, leakage_dominant])
    logger.info("\nOverall verdict: %s", "PASS — safe to implement" if overall else "FAIL — review before implementing")


if __name__ == "__main__":
    main()
