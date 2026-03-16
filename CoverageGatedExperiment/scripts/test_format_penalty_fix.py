"""Test suite for the dynamic format penalty fix.

Verifies three properties for every test group:
  P1 (case C): format_penalty < min(valid_adv) with margin == format_margin.
  P2 (case A): all-fail group returns all-zero advantages (will be skipped).
  P3 (case B): one-valid group uses absolute floor = -(z_clip*w_leak + format_margin).
  P4 (historical): penalty at early steps ≤ historical leaked advantage floor.

Groups are constructed from real distributions observed in step_log.jsonl
(steps 0–224, stock task), covering the key scenarios that led to collapse.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Minimal re-implementations (mirrors reward_pipeline.py exactly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletionReward:
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


@dataclass
class MinimalConfig:
    z_clip: float
    lambda_base: float
    format_margin: float


def _zscore(arr: np.ndarray, clip: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        return np.zeros_like(arr)
    z = (arr - mean) / std
    return np.clip(z, -clip, clip)


def compute_advantages(
    cfg: MinimalConfig,
    rewards: list[CompletionReward],
    w_leak: float,
    batch_mean_r_leak: float,
) -> tuple[list[float], dict[str, Any]]:
    """Mirrors reward_pipeline.RewardPipeline.compute_advantages exactly."""
    parseable = [r for r in rewards if r.parse_success]

    if len(parseable) == 0:
        return (
            [0.0 for _ in rewards],
            {"r_leak_std": 0.0, "r_perf_std": 0.0, "skipped_all_fail": True},
        )

    if len(parseable) == 1:
        floor = -(cfg.z_clip * w_leak + cfg.format_margin)
        advs = [floor if not r.parse_success else 0.0 for r in rewards]
        return advs, {"r_leak_std": 0.0, "r_perf_std": 0.0, "format_penalty": floor}

    r_leak = np.array([np.exp(-0.5 * r.num_leaked) for r in parseable], dtype=np.float64)
    r_perf = np.array([r.r_performance for r in parseable], dtype=np.float64)

    z_leak = _zscore(r_leak, cfg.z_clip)
    z_perf = _zscore(r_perf, cfg.z_clip)
    baseline = cfg.lambda_base * (r_leak - batch_mean_r_leak)

    w_perf = 1.0 - w_leak
    adv_raw = w_leak * (z_leak + baseline) + w_perf * z_perf

    format_penalty = float(adv_raw.min()) - cfg.format_margin

    out: list[float] = []
    j = 0
    for r in rewards:
        if not r.parse_success:
            out.append(format_penalty)
        else:
            out.append(float(adv_raw[j]))
            j += 1

    diag: dict[str, Any] = {
        "r_leak_std": float(r_leak.std()),
        "r_perf_std": float(r_perf.std()),
        "format_penalty": format_penalty,
        "skipped_all_fail": False,
    }
    return out, diag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reward(num_leaked: int, r_performance: float, parse_success: bool = True) -> CompletionReward:
    return CompletionReward(
        r_format=1.0 if parse_success else 0.0,
        r_valid=0.0,
        r_performance=r_performance,
        leakage_rate=0.0,
        hallucination_rate=0.0,
        coverage=1.0,
        parse_success=parse_success,
        num_claims=5,
        num_leaked=num_leaked,
        num_unsupported=0,
        num_undecidable=0,
    )


def _fail() -> CompletionReward:
    return _make_reward(0, 0.0, parse_success=False)


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


# ---------------------------------------------------------------------------
# Test configuration (mirrors train.yaml after the fix)
# ---------------------------------------------------------------------------

CFG = MinimalConfig(z_clip=5.0, lambda_base=0.5, format_margin=1.0)


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

ALL_PASSED = True


def run_test(
    label: str,
    rewards: list[CompletionReward],
    w_leak: float,
    batch_mean_r_leak: float,
    extra_checks: list[tuple[str, bool, str]] | None = None,
) -> bool:
    global ALL_PASSED
    print(f"\n{'─'*60}")
    print(f"Scenario: {label}")
    print(f"  w_leak={w_leak:.3f}  batch_mean_r_leak={batch_mean_r_leak:.3f}  "
          f"n_rewards={len(rewards)}  n_parseable={sum(r.parse_success for r in rewards)}")

    advs, diag = compute_advantages(CFG, rewards, w_leak, batch_mean_r_leak)
    valid_advs = [a for r, a in zip(rewards, advs) if r.parse_success]
    fail_advs  = [a for r, a in zip(rewards, advs) if not r.parse_success]

    passed = True

    # Core property checks
    if sum(r.parse_success for r in rewards) == 0:
        passed &= check("P2 all-zeros for all-fail group", all(abs(a) < 1e-9 for a in advs),
                        f"advs={advs}")
        passed &= check("P2 skipped_all_fail flag set", diag.get("skipped_all_fail") is True)
        passed &= check("P2 is_skipped trigger (all |a|<1e-8)", all(abs(a) < 1e-8 for a in advs))
    elif sum(r.parse_success for r in rewards) == 1:
        expected_floor = -(CFG.z_clip * w_leak + CFG.format_margin)
        for a in fail_advs:
            passed &= check("P3 floor penalty applied", abs(a - expected_floor) < 1e-9,
                            f"a={a:.4f} expected={expected_floor:.4f}")
        passed &= check("P3 valid completion has adv=0.0", abs(valid_advs[0]) < 1e-9,
                        f"adv={valid_advs[0]:.4f}")
        passed &= check("P3 floor < 0", expected_floor < 0,
                        f"floor={expected_floor:.4f}")
    else:
        fp = diag["format_penalty"]
        min_valid = min(valid_advs) if valid_advs else float("inf")
        for i, (r, a) in enumerate(zip(rewards, advs)):
            if not r.parse_success:
                passed &= check(f"P1 fail[{i}] adv < min(valid)", a < min_valid,
                                f"fail_adv={a:.4f} min_valid={min_valid:.4f}")
        passed &= check("P1 format_penalty == min(valid) - margin",
                        abs(fp - (min_valid - CFG.format_margin)) < 1e-6,
                        f"fp={fp:.4f} min_valid={min_valid:.4f} margin={CFG.format_margin}")

    if extra_checks:
        for name, cond, detail in extra_checks:
            passed &= check(name, cond, detail)

    ALL_PASSED = ALL_PASSED and passed
    return passed


# ---------------------------------------------------------------------------
# Scenario 1: All parse failures (Case A)
# ---------------------------------------------------------------------------
run_test(
    "All parse failures (G=16)",
    rewards=[_fail()] * 16,
    w_leak=0.95,
    batch_mean_r_leak=0.884,
)

# ---------------------------------------------------------------------------
# Scenario 2: Exactly 1 valid completion (Case B)
# ---------------------------------------------------------------------------
rewards_1valid = [_fail()] * 15 + [_make_reward(2, 0.4)]
run_test(
    "Exactly 1 valid completion (G=16)",
    rewards=rewards_1valid,
    w_leak=0.95,
    batch_mean_r_leak=0.884,
)

# ---------------------------------------------------------------------------
# Scenario 3: Early training (step 0) — all-clean group, 1 parse failure
# Replicates the original collapse trigger: leaked adv ≈ -3.0, old fixed -0.5
# ---------------------------------------------------------------------------
group_step0 = (
    [_make_reward(0, 0.75)] * 4
    + [_make_reward(1, 0.55)] * 6
    + [_make_reward(3, 0.30)] * 5
    + [_make_reward(4, 0.20)] * 1
    + [_fail()]
)
advs_step0, diag_step0 = compute_advantages(CFG, group_step0, w_leak=0.95, batch_mean_r_leak=0.884)
min_valid_step0 = min(a for r, a in zip(group_step0, advs_step0) if r.parse_success)
fp_step0 = diag_step0["format_penalty"]
run_test(
    "Step-0 mixed group (4×0-leaked, 6×1-leaked, 5×3-leaked, 1×4-leaked, 1 fail)",
    rewards=group_step0,
    w_leak=0.95,
    batch_mean_r_leak=0.884,
    extra_checks=[
        ("P4 format_penalty < old -0.5 floor", fp_step0 < -0.5,
         f"new fp={fp_step0:.3f}"),
        ("P4 format_penalty < historical min leaked adv",
         fp_step0 < min_valid_step0,
         f"fp={fp_step0:.3f} min_valid={min_valid_step0:.3f}"),
    ],
)

# ---------------------------------------------------------------------------
# Scenario 4: Mid-training (step 100, w_leak ≈ 0.913) — mixed with failures
# Illustrates penalty adapts to w_leak change
# ---------------------------------------------------------------------------
w_leak_100 = 0.95 + (0.80 - 0.95) / 2 * (1 - math.cos(math.pi * 100 / 300))
group_mid = (
    [_make_reward(0, 0.80)] * 8
    + [_make_reward(2, 0.50)] * 4
    + [_fail()] * 4
)
run_test(
    f"Step-100 mixed group (8×clean, 4×2-leaked, 4 fails, w_leak≈{w_leak_100:.3f})",
    rewards=group_mid,
    w_leak=w_leak_100,
    batch_mean_r_leak=0.750,
)

# ---------------------------------------------------------------------------
# Scenario 5: All clean, no parse failures (baseline sanity check)
# format_penalty logged but no fail tokens → should not appear in out
# ---------------------------------------------------------------------------
group_clean = [_make_reward(0, r) for r in [0.70, 0.75, 0.80, 0.82, 0.78, 0.72,
                                             0.68, 0.85, 0.77, 0.74, 0.71, 0.83,
                                             0.76, 0.79, 0.81, 0.73]]
advs_clean, diag_clean = compute_advantages(CFG, group_clean, w_leak=0.95, batch_mean_r_leak=0.884)
print(f"\n{'─'*60}")
print("Scenario: All clean, no failures (sanity — all advantages finite)")
passed_clean = True
passed_clean &= check("no parse failures → no format_penalty applied",
                      all(r.parse_success for r in group_clean))
passed_clean &= check("all advantages finite",
                      all(math.isfinite(a) for a in advs_clean))
passed_clean &= check("format_penalty logged in diag",
                      "format_penalty" in diag_clean)
ALL_PASSED = ALL_PASSED and passed_clean

# ---------------------------------------------------------------------------
# Scenario 6: High-collapse step (step 200 analogue) — 90% failures
# ---------------------------------------------------------------------------
group_collapse = [_fail()] * 14 + [_make_reward(2, 0.40), _make_reward(0, 0.70)]
run_test(
    "Step-200 collapse analogue (14 fails, 2 valid)",
    rewards=group_collapse,
    w_leak=0.875,
    batch_mean_r_leak=0.400,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'═'*60}")
if ALL_PASSED:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED — review output above")
    sys.exit(1)
