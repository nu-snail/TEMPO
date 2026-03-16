"""Validation pipeline comparison: old parametric Gemini vs new VERIFY-style.

Uses stored claims from step=0 val_log.jsonl (no re-extraction needed).
Tests 3 representative instances:
  - stock_33  (2021 cutoff, 6 claims, old hall=0.50)
  - stock_206 (2021 cutoff, 10 claims, old hall=0.90 — extreme case)
  - stock_825 (2025 cutoff, 5 claims, old hall=0.60 — where parametric fails)

Runs the new _detect_hallucination from evaluate.py on the stored claim pools,
then compares verdict-by-verdict and re-computes r_valid.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT.parent / ".env", override=True)

from cov_gated_experiment.llm import OpenRouterClient
from cov_gated_experiment.evaluate import _detect_hallucination
from cov_gated_experiment.reward_pipeline import _merge_labels, _count_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JUDGE_MODEL = "google/gemini-3-flash-preview"
SEARCH_MODEL = "perplexity/sonar"
VAL_LOG_PATH = REPO_ROOT / "runtime/results/train_stock/val_log.jsonl"
TARGET_IDS = {"stock_33", "stock_206", "stock_825"}

# ---------------------------------------------------------------------------
# Load stored val_log data
# ---------------------------------------------------------------------------
def load_val_step0() -> dict[str, dict]:
    """Load step=0 validation entries from val_log.jsonl."""
    with open(VAL_LOG_PATH) as f:
        entries = [json.loads(line) for line in f]
    step0 = {e["instance_id"]: e for e in entries if e["step"] == 0}
    return step0


def build_claim_pool(stored_claims: list[dict]) -> list[dict]:
    """Reconstruct claim_pool format from stored val_log claims."""
    pool = []
    for c in stored_claims:
        pool.append({
            "index": c["index"],
            "text": c["text"],
            "source": c.get("source", "evidence"),
            "source_date": None,  # not needed for hallucination detection
            "key": c["text"].lower().strip(),
        })
    return pool


def compute_r_valid(claim_pool: list[dict], leak_results: dict, hall_results: dict) -> tuple[float, dict]:
    """Compute r_valid = n_valid / total using new hallucination results."""
    # For this test: use stored leak verdicts, only swap hallucination results
    merged = _merge_labels(claim_pool, leak_results, hall_results)
    labels = [merged[c["index"]] for c in claim_pool]
    n_valid, n_leaked, n_hall, n_both = _count_labels(labels)
    total = n_valid + n_leaked + n_hall + n_both
    r_valid = n_valid / total if total > 0 else 1.0
    hall_rate = (n_hall + n_both) / max(total, 1)
    return r_valid, {
        "n_valid": n_valid, "n_leaked": n_leaked,
        "n_hall": n_hall, "n_both": n_both,
        "hall_rate": hall_rate, "merged": merged,
    }


def run_test():
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")
    client = OpenRouterClient(api_key)

    all_entries = load_val_step0()
    logger.info("Loaded %d step=0 validation instances", len(all_entries))

    for instance_id in sorted(TARGET_IDS):
        if instance_id not in all_entries:
            logger.warning("Instance %s not found in val_log", instance_id)
            continue

        e = all_entries[instance_id]
        stored_claims = e["claims"]
        cutoff = e["cutoff_date"]
        old_hall_rate = e["hallucination_rate"]
        old_r_valid = e["r_valid"]

        print(f"\n{'='*72}")
        print(f"INSTANCE: {instance_id}  cutoff={cutoff}  n_claims={len(stored_claims)}")
        print(f"OLD RESULTS: hall_rate={old_hall_rate:.3f}  r_valid={old_r_valid:.3f}")
        print(f"{'='*72}")

        # Show old verdict distribution
        from collections import Counter
        old_dist = Counter(c.get("hall_verdict", "?") for c in stored_claims)
        print(f"\nOld (parametric Gemini) verdicts: {dict(old_dist)}")

        # Build claim pool
        claim_pool = build_claim_pool(stored_claims)

        # Build leak_results from stored data (preserve original leakage verdicts)
        leak_results = {}
        for c in stored_claims:
            leak_results[c["index"]] = {
                "verdict": c.get("leak_verdict", "not_leaked"),
                "confidence": c.get("leak_confidence", "high"),
            }

        # Run new VERIFY-style hallucination detection
        print(f"\nRunning new VERIFY-style hallucination detection...")
        t0 = time.time()
        new_hall_results = _detect_hallucination(
            claim_pool, client, JUDGE_MODEL, SEARCH_MODEL
        )
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s")

        # Show new verdict distribution
        new_dist = Counter(v.get("verdict", "?") for v in new_hall_results.values())
        print(f"New (VERIFY: Sonar+Gemini) verdicts: {dict(new_dist)}")

        # Compute new r_valid
        new_r_valid, new_counts = compute_r_valid(claim_pool, leak_results, new_hall_results)
        print(f"\nNEW RESULTS: hall_rate={new_counts['hall_rate']:.3f}  r_valid={new_r_valid:.3f}")
        print(f"  Δhall_rate = {new_counts['hall_rate'] - old_hall_rate:+.3f}")
        print(f"  Δr_valid   = {new_r_valid - old_r_valid:+.3f}")

        # Side-by-side verdict comparison per claim
        print(f"\nPer-claim verdict comparison:")
        print(f"  {'IDX':>3}  {'OLD':12}  {'NEW':12}  CHANGED?  CLAIM (first 70 chars)")
        print(f"  {'-'*3}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*70}")
        for c in stored_claims:
            idx = c["index"]
            old_v = c.get("hall_verdict", "?")
            new_entry = new_hall_results.get(idx, {})
            new_v = new_entry.get("verdict", "missing")
            changed = "✓ CHANGED" if old_v != new_v else ""
            reason = new_entry.get("reason", "")
            print(f"  {idx:>3}  {old_v:12}  {new_v:12}  {changed:9}  {c['text'][:70]}")
            if reason:
                print(f"       {'':12}  {'':12}  {'':9}  → reason: {reason[:80]}")

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print("\nInstance          OLD r_valid  NEW r_valid  Δ         OLD hall  NEW hall  Δ")
    print(f"{'-'*80}")
    for instance_id in sorted(TARGET_IDS):
        if instance_id not in all_entries:
            continue
        e = all_entries[instance_id]
        # We already printed these; re-read would need re-running, so just print what we have
        print(f"(see detailed output above for {instance_id})")


if __name__ == "__main__":
    run_test()
