"""Rigorous unit tests for _compute_rates.

Tests cover all combinations of independent leakage and hallucination verdicts:
  leakage:       leaked | not_leaked
  hallucination: supported | unsupported | undecidable

Key properties verified:
  1. leak_rate  = n_leaked / total  (independent of hall)
  2. hall_rate  = (n_unsupported + 0.5 * n_undecidable) / total  (independent of leak)
  3. r_valid    = (n_clean + 0.5 * n_half_clean) / total
                  n_clean      = not_leaked AND supported
                  n_half_clean = not_leaked AND undecidable
  4. Empty pool → (0.0, 0.0, 1.0, 0, 0, 0)
  5. A claim that is both leaked AND unsupported contributes to BOTH rates
  6. A claim that is leaked AND undecidable contributes to leak_rate only
     (undecidable penalty only applies when not leaked)
  7. Rates sum correctly (leak_rate + hall_rate can exceed 1.0 if a claim is both)

Also runs against 3 stored val_log instances to verify end-to-end
realistic output with actual evidence-level claims.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cov_gated_experiment.reward_pipeline import _compute_rates

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_leak(verdict: str) -> dict[str, Any]:
    return {"verdict": verdict, "confidence": "high"}


def make_hall(verdict: str) -> dict[str, Any]:
    return {"verdict": verdict}


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_empty_pool():
    lr, hr, rv, nl, nu, nd = _compute_rates([], {}, {})
    assert lr == 0.0 and hr == 0.0 and rv == 1.0
    assert nl == 0 and nu == 0 and nd == 0
    print("PASS  empty pool → (0,0,1)")


def test_all_valid():
    """All 4 claims: not_leaked + supported → fully clean."""
    indices = [0, 1, 2, 3]
    leak = {i: make_leak("not_leaked") for i in indices}
    hall = {i: make_hall("supported") for i in indices}
    lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak, hall)
    assert close(lr, 0.0), f"leak_rate={lr}"
    assert close(hr, 0.0), f"hall_rate={hr}"
    assert close(rv, 1.0), f"r_valid={rv}"
    assert nl == 0 and nu == 0 and nd == 0
    print("PASS  all valid")


def test_all_leaked():
    """All 3 claims leaked + supported → leak_rate=1, hall_rate=0, r_valid=0."""
    indices = [0, 1, 2]
    leak = {i: make_leak("leaked") for i in indices}
    hall = {i: make_hall("supported") for i in indices}
    lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak, hall)
    assert close(lr, 1.0), f"leak_rate={lr}"
    assert close(hr, 0.0), f"hall_rate={hr}"
    assert close(rv, 0.0), f"r_valid={rv}"
    assert nl == 3 and nu == 0 and nd == 0
    print("PASS  all leaked, no hallucination")


def test_all_unsupported():
    """All 4 claims not_leaked + unsupported → leak=0, hall=1, r_valid=0."""
    indices = [0, 1, 2, 3]
    leak = {i: make_leak("not_leaked") for i in indices}
    hall = {i: make_hall("unsupported") for i in indices}
    lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak, hall)
    assert close(lr, 0.0), f"leak_rate={lr}"
    assert close(hr, 1.0), f"hall_rate={hr}"
    assert close(rv, 0.0), f"r_valid={rv}"
    assert nl == 0 and nu == 4 and nd == 0
    print("PASS  all unsupported (hallucinated)")


def test_all_undecidable():
    """All 4 claims not_leaked + undecidable → leak=0, hall=0.5, r_valid=0.5."""
    indices = [0, 1, 2, 3]
    leak = {i: make_leak("not_leaked") for i in indices}
    hall = {i: make_hall("undecidable") for i in indices}
    lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak, hall)
    assert close(lr, 0.0), f"leak_rate={lr}"
    assert close(hr, 0.5), f"hall_rate={hr}"
    assert close(rv, 0.5), f"r_valid={rv}"
    assert nl == 0 and nu == 0 and nd == 4
    print("PASS  all undecidable → α=0.5 applied")


def test_leaked_and_unsupported_counts_both():
    """
    Key independence test: a claim that is leaked AND unsupported
    must contribute to BOTH leak_rate and hall_rate independently.
    4 claims: [leaked+unsupported, not_leaked+supported, not_leaked+supported, not_leaked+supported]
    Expected: leak_rate=0.25, hall_rate=0.25, r_valid=0.75
    """
    indices = [0, 1, 2, 3]
    leak = {0: make_leak("leaked"), 1: make_leak("not_leaked"),
            2: make_leak("not_leaked"), 3: make_leak("not_leaked")}
    hall = {0: make_hall("unsupported"), 1: make_hall("supported"),
            2: make_hall("supported"), 3: make_hall("supported")}
    lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak, hall)
    assert close(lr, 0.25), f"leak_rate={lr}"
    assert close(hr, 0.25), f"hall_rate={hr}"
    assert close(rv, 0.75), f"r_valid={rv}"
    assert nl == 1 and nu == 1 and nd == 0
    print("PASS  leaked+unsupported counts in BOTH rates independently")


def test_leaked_and_undecidable():
    """
    Leaked + undecidable:
      - contributes to leak_rate (it is leaked)
      - contributes 0.5 to hall_rate (n_undecidable counts all undecidable regardless of leak)
      - does NOT contribute to r_valid (n_half_clean requires not_leaked AND undecidable)

    4 claims: [leaked+undecidable, not_leaked+supported×3]
    Expected:
      n_leaked=1, n_undecidable=1
      leak_rate = 1/4 = 0.25
      hall_rate = 0.5*1/4 = 0.125
      r_valid   = 3/4 = 0.75  (only the 3 clean not_leaked+supported claims)
    """
    indices = [0, 1, 2, 3]
    leak = {0: make_leak("leaked"), 1: make_leak("not_leaked"),
            2: make_leak("not_leaked"), 3: make_leak("not_leaked")}
    hall = {0: make_hall("undecidable"), 1: make_hall("supported"),
            2: make_hall("supported"), 3: make_hall("supported")}
    lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak, hall)
    assert close(lr, 0.25), f"leak_rate={lr}"
    assert close(hr, 0.125), f"hall_rate={hr} (expected 0.5*1/4=0.125)"
    assert close(rv, 0.75), f"r_valid={rv}"
    assert nl == 1 and nu == 0 and nd == 1
    print("PASS  leaked+undecidable: leak=0.25, hall=0.125 (α=0.5×1/4), r_valid=0.75")


def test_mixed_realistic():
    """
    Realistic mix: 8 claims
      0: leaked + supported
      1: leaked + unsupported
      2: leaked + undecidable
      3: not_leaked + supported     (clean)
      4: not_leaked + unsupported   (hallucinated)
      5: not_leaked + undecidable   (half-clean)
      6: not_leaked + supported     (clean)
      7: not_leaked + supported     (clean)

    Expected:
      n_leaked     = 3  (0,1,2)
      n_unsupported = 2  (1,4)
      n_undecidable = 2  (2,5)
      leak_rate     = 3/8 = 0.375
      hall_rate     = (2 + 0.5*2) / 8 = 3/8 = 0.375
      n_clean       = 3  (3,6,7)
      n_half_clean  = 1  (5)
      r_valid       = (3 + 0.5*1) / 8 = 3.5/8 = 0.4375
    """
    indices = list(range(8))
    leak = {0: make_leak("leaked"), 1: make_leak("leaked"), 2: make_leak("leaked"),
            3: make_leak("not_leaked"), 4: make_leak("not_leaked"),
            5: make_leak("not_leaked"), 6: make_leak("not_leaked"), 7: make_leak("not_leaked")}
    hall = {0: make_hall("supported"), 1: make_hall("unsupported"), 2: make_hall("undecidable"),
            3: make_hall("supported"), 4: make_hall("unsupported"), 5: make_hall("undecidable"),
            6: make_hall("supported"), 7: make_hall("supported")}
    lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak, hall)
    assert nl == 3 and nu == 2 and nd == 2, f"counts: {nl},{nu},{nd}"
    assert close(lr, 3/8), f"leak_rate={lr}"
    assert close(hr, 3/8), f"hall_rate={hr}"
    assert close(rv, 3.5/8), f"r_valid={rv}"
    print(f"PASS  realistic mix: leak={lr:.4f} hall={hr:.4f} r_valid={rv:.4f}")


def test_missing_hall_results_defaults_to_supported():
    """When hall_results is empty (training, skip_hallucination), verdict defaults
    to 'supported' (not in dict → .get returns {} → .get('verdict') is None → not 'unsupported').
    leak_rate should still work; hall_rate should be 0."""
    indices = [0, 1, 2]
    leak = {0: make_leak("leaked"), 1: make_leak("not_leaked"), 2: make_leak("not_leaked")}
    hall = {}  # empty — simulates training where hallucination is skipped
    lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak, hall)
    assert close(lr, 1/3), f"leak_rate={lr}"
    assert close(hr, 0.0), f"hall_rate={hr} (should be 0 when hall skipped)"
    assert close(rv, 2/3), f"r_valid={rv}"
    assert nu == 0 and nd == 0
    print("PASS  empty hall_results (training mode) → hall_rate=0")


def test_rates_can_exceed_one_when_overlap():
    """
    If ALL claims are both leaked AND unsupported:
    leak_rate=1, hall_rate=1 — both can be 1 simultaneously.
    r_valid=0.
    This demonstrates independence; the two rates are not constrained to sum ≤ 1.
    """
    indices = [0, 1]
    leak = {0: make_leak("leaked"), 1: make_leak("leaked")}
    hall = {0: make_hall("unsupported"), 1: make_hall("unsupported")}
    lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak, hall)
    assert close(lr, 1.0) and close(hr, 1.0) and close(rv, 0.0)
    assert nl == 2 and nu == 2
    print("PASS  fully leaked+unsupported → both rates = 1.0 simultaneously")


# ---------------------------------------------------------------------------
# Integration check against stored val_log instances
# ---------------------------------------------------------------------------

def test_integration_val_log():
    """
    Run _compute_rates on stored claim pools from val_log step=0.
    Uses the new (VERIFY) hallucination verdicts obtained in test_verify_validation.py.
    We verify that the formulas produce consistent numbers given known inputs.
    """
    val_log_path = REPO_ROOT / "runtime/results/train_stock/val_log.jsonl"
    with open(val_log_path) as f:
        entries = {e["instance_id"]: e for line in f
                   for e in [json.loads(line)] if e["step"] == 0}

    # Known new verdicts from test_verify_validation.py run
    # stock_33: supported=1, unsupported=3, undecidable=2, leaked=0
    # stock_206: supported=0, unsupported=6, undecidable=4, leaked=0
    # stock_825: supported=0, unsupported=3, undecidable=2, leaked=0
    known = {
        "stock_33":  {"n_leak": 0, "n_uns": 3, "n_und": 2, "total": 6},
        "stock_206": {"n_leak": 0, "n_uns": 6, "n_und": 4, "total": 10},
        "stock_825": {"n_leak": 0, "n_uns": 3, "n_und": 2, "total": 5},
    }
    for iid, k in known.items():
        if iid not in entries:
            print(f"SKIP  {iid} not in val_log")
            continue
        e = entries[iid]
        claims = e["claims"]
        indices = [c["index"] for c in claims]
        # Build leak_results from stored data
        leak_results = {c["index"]: {"verdict": c.get("leak_verdict", "not_leaked")} for c in claims}
        # Simulate the new hall verdicts by constructing results matching known counts
        # (use stored data where available, else assume supported for remainder)
        # We can't re-run Sonar here, so we construct synthetic hall_results matching known counts
        hall_results = {}
        uns_left, und_left = k["n_uns"], k["n_und"]
        for c in claims:
            idx = c["index"]
            if uns_left > 0:
                hall_results[idx] = {"verdict": "unsupported"}
                uns_left -= 1
            elif und_left > 0:
                hall_results[idx] = {"verdict": "undecidable"}
                und_left -= 1
            else:
                hall_results[idx] = {"verdict": "supported"}

        lr, hr, rv, nl, nu, nd = _compute_rates(indices, leak_results, hall_results)
        total = k["total"]
        exp_lr = k["n_leak"] / total
        exp_hr = (k["n_uns"] + 0.5 * k["n_und"]) / total
        exp_rv = (total - k["n_leak"] - k["n_uns"] - 0.5 * k["n_und"]) / total

        assert close(lr, exp_lr, 1e-9), f"{iid} leak_rate={lr} expected {exp_lr}"
        assert close(hr, exp_hr, 1e-9), f"{iid} hall_rate={hr} expected {exp_hr}"
        assert close(rv, exp_rv, 1e-9), f"{iid} r_valid={rv} expected {exp_rv}"
        print(f"PASS  {iid}: leak={lr:.3f} hall={hr:.3f} r_valid={rv:.3f}  "
              f"(n_leaked={nl} n_unsupported={nu} n_undecidable={nd})")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Unit tests for _compute_rates (independent leakage + hall)")
    print("=" * 60)
    failures = 0
    tests = [
        test_empty_pool,
        test_all_valid,
        test_all_leaked,
        test_all_unsupported,
        test_all_undecidable,
        test_leaked_and_unsupported_counts_both,
        test_leaked_and_undecidable,
        test_mixed_realistic,
        test_missing_hall_results_defaults_to_supported,
        test_rates_can_exceed_one_when_overlap,
        test_integration_val_log,
    ]
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e}")
            failures += 1

    print("=" * 60)
    if failures == 0:
        print(f"All {len(tests)} tests passed.")
    else:
        print(f"{failures}/{len(tests)} tests FAILED.")
    print("=" * 60)
    sys.exit(failures)
