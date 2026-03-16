"""Validate the Stage-1-only pipeline design before applying production changes.

Three isolated parts (no production files modified):
  Part 1: Unit tests — verify _build_claim_pool_v2 and _detect_leakage_v1 logic
  Part 2: Integration — re-run Stage 1 on stored completions, compare verdicts
  Part 3: Generation — compare source_date quality: old STOCK_SYSTEM_PROMPT vs new V2

Run from FullExperiment/:
    uv run python scripts/test_stage1_only_design.py            # all parts
    uv run python scripts/test_stage1_only_design.py --parts 1  # only Part 1
    uv run python scripts/test_stage1_only_design.py --parts 1 2 3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cov_gated_experiment.config import load_config, load_env
from cov_gated_experiment.llm import OpenRouterClient
from cov_gated_experiment.prompts import DATE_PLAUSIBILITY_PROMPT, STOCK_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# New pipeline variants (inline — no production files modified)
# ---------------------------------------------------------------------------

_JUDGE_CHUNK_SIZE = 80
_MAX_TOKENS_JUDGE = 8192

# Inline copy of the parser from reward_pipeline.py
def _parse_json_array_raw(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _build_claim_pool_v2(
    parseable: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Stage-1-only claim pool: evidence claims only, no reasoning extraction."""
    claim_pool: list[dict[str, Any]] = []
    claim_to_completions: dict[str, list[int]] = {}

    for i, p in enumerate(parseable):
        for ev in p.get("evidence", []):
            fact = str(ev.get("fact", "")).strip()
            if not fact:
                continue
            key = fact.lower()
            if key not in claim_to_completions:
                claim_to_completions[key] = []
                sd = ev.get("source_date")
                source_date = str(sd).strip() if sd else None
                claim_pool.append({
                    "text": fact,
                    "key": key,
                    "source": "evidence",
                    "source_date": source_date,
                })
            if i not in claim_to_completions[key]:
                claim_to_completions[key].append(i)

    for idx, c in enumerate(claim_pool):
        c["index"] = idx

    return claim_pool, claim_to_completions


def _detect_leakage_v1(
    claim_pool: list[dict[str, Any]],
    cutoff_date: str,
    client: OpenRouterClient,
    judge_model: str,
) -> dict[int, dict[str, Any]]:
    """Stage 1 only: DATE_PLAUSIBILITY_PROMPT + deterministic date comparison.

    Evidence items without source_date are skipped (treated as clean).
    No Stage 2 / LEAKAGE_INFER_PROMPT calls.
    """
    anchored = [
        c for c in claim_pool
        if c.get("source") == "evidence" and c.get("source_date")
    ]
    results: dict[int, dict[str, Any]] = {}

    for chunk_start in range(0, len(anchored), _JUDGE_CHUNK_SIZE):
        chunk = anchored[chunk_start : chunk_start + _JUDGE_CHUNK_SIZE]
        items_payload = [
            {
                "index": c["index"],
                "fact": c["text"],
                "declared_source_date": c["source_date"],
            }
            for c in chunk
        ]
        text, _ = client.call_text(
            judge_model,
            DATE_PLAUSIBILITY_PROMPT.format(
                cutoff=cutoff_date,
                items_json=json.dumps(items_payload, indent=2, ensure_ascii=False),
            ),
            temperature=0.0,
            max_tokens=_MAX_TOKENS_JUDGE,
        )
        verifier_by_idx = {
            r["index"]: r for r in _parse_json_array_raw(text) if "index" in r
        }
        for c in chunk:
            idx = c["index"]
            declared = c["source_date"]
            vr = verifier_by_idx.get(idx, {})
            plausible = bool(vr.get("plausible", True))
            corrected = vr.get("corrected_date")
            effective = corrected if (not plausible and corrected) else declared
            try:
                leaked = bool(effective > cutoff_date)
            except TypeError:
                leaked = False
            results[idx] = {
                "verdict": "leaked" if leaked else "not_leaked",
                "event_date": effective,
                "method": "anchored",
                "plausible": plausible,
            }

    return results


# ---------------------------------------------------------------------------
# New STOCK_SYSTEM_PROMPT_V2 with strengthened source_date guidance
# ---------------------------------------------------------------------------

STOCK_SYSTEM_PROMPT_V2 = STOCK_SYSTEM_PROMPT.replace(
    "- For EACH fact, include a source_date field: the specific date (YYYY-MM-DD)\n"
    "  when this information was FIRST publicly announced or released (e.g.,\n"
    "  earnings report date, SEC filing date, press release date). Use the\n"
    "  EARLIEST date the information was publicly available. If the exact day is\n"
    "  uncertain, use the first day of the announcing month.",
    "- For EACH fact, include a source_date field: the YYYY-MM-DD date when this\n"
    "  information was FIRST PUBLICLY ANNOUNCED (earnings release date, press release\n"
    "  date, SEC filing date). This must be the ANNOUNCEMENT date, NOT the period end.\n"
    "  WRONG: source_date \"2023-12-31\" for Q4 2023 earnings (period end, not yet public)\n"
    "  RIGHT: source_date \"2024-02-15\" for Q4 2023 earnings (actual earnings release)\n"
    "  If the exact announcement day is uncertain, use the 1st of the announcing month.",
)

_EARNINGS_KEYWORDS = frozenset(["q1", "q2", "q3", "q4", "earnings", "revenue",
                                  "ebitda", "eps", "net income", "annual", "fy",
                                  "full-year", "full year", "fiscal year", "results"])
_QUARTER_END_DAYS = {(3, 31), (6, 30), (9, 30), (12, 31)}


def _classify_source_date(fact: str, source_date: str) -> str:
    """Classify a source_date as 'period_end', 'announcement', or 'other'.

    A 'period_end' date is the last day of a fiscal quarter used incorrectly
    as a source_date for an earnings claim.
    """
    if not source_date:
        return "missing"
    try:
        parts = source_date.split("-")
        month, day = int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return "unparseable"

    fact_lower = fact.lower()
    has_earnings = any(kw in fact_lower for kw in _EARNINGS_KEYWORDS)
    is_quarter_end = (month, day) in _QUARTER_END_DAYS

    if has_earnings and is_quarter_end:
        return "period_end"
    return "announcement" if has_earnings else "other"


# ---------------------------------------------------------------------------
# Part 1: Unit tests
# ---------------------------------------------------------------------------

class TestClaimPoolV2(unittest.TestCase):

    def _make_parseable(self, with_reasoning_text: str = "extra claim") -> list[dict]:
        return [{
            "evidence": [
                {"fact": "Company A reported Q3 2021 earnings on Oct 28 2021.",
                 "source_date": "2021-10-28"},
                {"fact": "Company B launched product in June 2021.",
                 "source_date": "2021-06-15"},
                {"fact": "Company C had no source date.", "source_date": None},
            ],
            "reasoning": f"Strong results [1][2]. Also: {with_reasoning_text}",
        }]

    def test_only_evidence_claims_in_pool(self) -> None:
        """_build_claim_pool_v2 must never include reasoning-derived claims."""
        parseable = self._make_parseable("extra claim from reasoning not in evidence")
        pool, _ = _build_claim_pool_v2(parseable)
        sources = {c["source"] for c in pool}
        self.assertEqual(sources, {"evidence"}, f"Unexpected sources: {sources}")
        self.assertEqual(len(pool), 3)

    def test_evidence_without_source_date_is_skipped_by_leakage(self) -> None:
        """Claims with source_date=None must not appear in leakage results."""
        parseable = self._make_parseable()
        pool, _ = _build_claim_pool_v2(parseable)
        no_date_claims = [c for c in pool if not c.get("source_date")]
        self.assertEqual(len(no_date_claims), 1)

        mock_client = MagicMock()
        mock_client.call_text.return_value = ("[]", None)
        results = _detect_leakage_v1(pool, "2022-01-01", mock_client, "test-model")
        # Claim with no source_date has no result entry
        missing_indices = {c["index"] for c in no_date_claims}
        for idx in missing_indices:
            self.assertNotIn(idx, results, f"index {idx} should not appear in results")

    def test_no_leakage_infer_prompt_calls(self) -> None:
        """_detect_leakage_v1 must never call with LEAKAGE_INFER_PROMPT content."""
        parseable = self._make_parseable()
        pool, _ = _build_claim_pool_v2(parseable)
        mock_client = MagicMock()
        mock_client.call_text.return_value = ('[{"index":0,"plausible":true,"corrected_date":null}]', None)
        _detect_leakage_v1(pool, "2022-01-01", mock_client, "test-model")
        for call in mock_client.call_text.call_args_list:
            prompt_arg = call.args[1] if len(call.args) > 1 else call.kwargs.get("prompt", "")
            self.assertNotIn(
                "LEAKAGE_INFER_PROMPT", prompt_arg,
                "LEAKAGE_INFER_PROMPT content found in a call — Stage 2 must be removed",
            )
            # Also check it doesn't contain Stage 2's unique phrasing
            self.assertNotIn("earliest_public_date", prompt_arg)

    def test_deterministic_date_comparison(self) -> None:
        """leaked when effective_date > cutoff_date (string comparison)."""
        parseable = [{
            "evidence": [
                {"fact": "Some Q4 2022 result.", "source_date": "2023-02-10"},
                {"fact": "Some Q2 2021 result.", "source_date": "2021-07-30"},
            ],
        }]
        pool, _ = _build_claim_pool_v2(parseable)
        mock_client = MagicMock()
        # Gemini says both plausible — effective date = declared
        mock_client.call_text.return_value = (
            '[{"index":0,"plausible":true,"corrected_date":null},'
            ' {"index":1,"plausible":true,"corrected_date":null}]',
            None,
        )
        results = _detect_leakage_v1(pool, "2023-01-31", mock_client, "test-model")
        idx0 = pool[0]["index"]
        idx1 = pool[1]["index"]
        self.assertEqual(results[idx0]["verdict"], "leaked",
                         "2023-02-10 > 2023-01-31 should be leaked")
        self.assertEqual(results[idx1]["verdict"], "not_leaked",
                         "2021-07-30 < 2023-01-31 should be not_leaked")

    def test_plausibility_correction_applied(self) -> None:
        """When plausible=False, effective_date uses corrected_date for comparison."""
        parseable = [{
            "evidence": [{"fact": "Q4 2021 earnings.", "source_date": "2021-12-31"}],
        }]
        pool, _ = _build_claim_pool_v2(parseable)
        mock_client = MagicMock()
        # Gemini corrects Dec 31 to Feb 2022 (Q4 can't be announced on its last day)
        mock_client.call_text.return_value = (
            '[{"index":0,"plausible":false,"corrected_date":"2022-02-01"}]',
            None,
        )
        results = _detect_leakage_v1(pool, "2021-12-31", mock_client, "test-model")
        idx0 = pool[0]["index"]
        self.assertEqual(results[idx0]["verdict"], "leaked",
                         "corrected to 2022-02-01 > 2021-12-31 should be leaked")
        self.assertFalse(results[idx0]["plausible"])


# ---------------------------------------------------------------------------
# Part 2: Integration test on stored completions
# ---------------------------------------------------------------------------

_VAL_LOG = PROJECT_ROOT / "runtime/results/train_stock/val_log.jsonl"
_N_INSTANCES = 5  # number of instances to re-score


def _load_stored_instances() -> list[dict]:
    """Load step=-1 val_log rows that have anchored evidence claims."""
    if not _VAL_LOG.exists():
        return []
    rows = []
    with open(_VAL_LOG) as f:
        for line in f:
            row = json.loads(line.strip())
            if row.get("step") != -1:
                continue
            anchored = [
                c for c in row.get("claims", [])
                if c.get("source") == "evidence" and c.get("leak_method") == "anchored"
            ]
            if len(anchored) >= 3:
                rows.append(row)
    return rows[:_N_INSTANCES]


def _extract_evidence_from_completion(
    completion_text: str,
) -> list[dict[str, Any]]:
    """Parse the stored completion JSON and return its evidence list."""
    m = re.search(r"\{.*\}", completion_text, re.DOTALL)
    if not m:
        return []
    try:
        comp = json.loads(m.group())
        return comp.get("evidence", [])
    except json.JSONDecodeError:
        return []


def run_part2(client: OpenRouterClient, judge_model: str) -> bool:
    """Re-run Stage 1 on stored completions and compare to stored verdicts."""
    logger.info("\n=== Part 2: Integration test on stored completions ===")

    rows = _load_stored_instances()
    if not rows:
        logger.error("No suitable instances in val_log.jsonl — skipping Part 2")
        return False

    total_claims = 0
    agreements = 0
    disagreements: list[dict] = []

    for row in rows:
        inst_id = row["instance_id"]
        cutoff = row["cutoff_date"]
        evidence = _extract_evidence_from_completion(row.get("completion", ""))
        if not evidence:
            logger.warning("Could not parse completion for %s — skipping", inst_id)
            continue

        parseable = [{"evidence": evidence, "reasoning": ""}]
        pool, _ = _build_claim_pool_v2(parseable)
        new_results = _detect_leakage_v1(pool, cutoff, client, judge_model)

        # Compare against stored verdicts
        stored_by_text = {
            c["text"].lower(): c["leak_verdict"]
            for c in row.get("claims", [])
            if c.get("source") == "evidence" and c.get("leak_method") == "anchored"
        }
        for c in pool:
            if not c.get("source_date"):
                continue
            stored_verdict = stored_by_text.get(c["text"].lower())
            new_verdict = new_results.get(c["index"], {}).get("verdict")
            if stored_verdict is None or new_verdict is None:
                continue
            total_claims += 1
            if stored_verdict == new_verdict:
                agreements += 1
            else:
                disagreements.append({
                    "instance": inst_id,
                    "claim": c["text"][:80],
                    "stored": stored_verdict,
                    "new": new_verdict,
                    "event_date": new_results[c["index"]].get("event_date"),
                })

    if total_claims == 0:
        logger.error("No comparable claims found")
        return False

    agreement_rate = agreements / total_claims
    logger.info("Claims compared: %d  Agreements: %d  Rate: %.1f%%",
                total_claims, agreements, agreement_rate * 100)

    if disagreements:
        logger.info("Disagreements (%d):", len(disagreements))
        for d in disagreements:
            logger.info("  [%s] '%s' stored=%s new=%s event_date=%s",
                        d["instance"], d["claim"], d["stored"], d["new"], d["event_date"])
    else:
        logger.info("No disagreements — Stage 1 results are stable")

    passed = agreement_rate >= 0.90
    logger.info("Part 2: %s (agreement=%.1f%% threshold=90%%)",
                "PASS" if passed else "FAIL", agreement_rate * 100)
    return passed


# ---------------------------------------------------------------------------
# Part 3: Generation prompt comparison
# ---------------------------------------------------------------------------

_N_GEN_INSTANCES = 3
_GEN_MODEL = "google/gemini-3-flash-preview"  # fast proxy; tests prompt behavior


def _generate_completion(client: OpenRouterClient, system: str, user: str) -> str:
    """Generate one completion with system+user messages via OpenRouter."""
    prompt = f"<SYSTEM>\n{system}\n</SYSTEM>\n\n<USER>\n{user}\n</USER>\n\nAssistant:"
    text, _ = client.call_text(
        _GEN_MODEL,
        prompt,
        temperature=0.0,
        max_tokens=4096,
    )
    return text


def _analyze_source_dates(evidence: list[dict]) -> dict[str, int]:
    """Count period_end vs announcement source_dates in evidence list."""
    counts: dict[str, int] = {"period_end": 0, "announcement": 0, "other": 0, "missing": 0}
    for ev in evidence:
        label = _classify_source_date(
            ev.get("fact", ""),
            ev.get("source_date", "") or "",
        )
        counts[label] = counts.get(label, 0) + 1
    return counts


def run_part3(client: OpenRouterClient) -> bool:
    """Compare source_date quality: current STOCK_SYSTEM_PROMPT vs V2.

    Uses a capable proxy model (Gemini Flash) to generate completions for
    both prompts on the same instances. Checks that V2 produces fewer
    period_end_date errors.
    """
    logger.info("\n=== Part 3: Generation prompt comparison (old vs V2) ===")

    load_env(PROJECT_ROOT)
    cfg = load_config(str(PROJECT_ROOT / "configs/train.yaml"))

    from cov_gated_experiment.data import load_train_instances, split_train_val
    instances = load_train_instances(PROJECT_ROOT / cfg.data_root, ["stock"])
    train_set, _ = split_train_val(instances, cfg.val_fraction, cfg.data_seed)
    test_instances = train_set[:_N_GEN_INSTANCES]

    old_totals: dict[str, int] = {"period_end": 0, "announcement": 0, "other": 0}
    new_totals: dict[str, int] = {"period_end": 0, "announcement": 0, "other": 0}

    for inst in test_instances:
        from cov_gated_experiment.prompts import _build_stock_user
        user_msg = _build_stock_user(inst)

        logger.info("Generating for %s (cutoff=%s)...", inst.instance_id, inst.cutoff_date)

        # Old prompt
        sys_old = STOCK_SYSTEM_PROMPT.format(cutoff_date=inst.cutoff_date)
        raw_old = _generate_completion(client, sys_old, user_msg)
        m = re.search(r"\{.*\}", raw_old, re.DOTALL)
        ev_old = json.loads(m.group()).get("evidence", []) if m else []
        counts_old = _analyze_source_dates(ev_old)

        # New V2 prompt
        sys_new = STOCK_SYSTEM_PROMPT_V2.format(cutoff_date=inst.cutoff_date)
        raw_new = _generate_completion(client, sys_new, user_msg)
        m = re.search(r"\{.*\}", raw_new, re.DOTALL)
        ev_new = json.loads(m.group()).get("evidence", []) if m else []
        counts_new = _analyze_source_dates(ev_new)

        logger.info("  [%s] old: %s  |  new: %s",
                    inst.instance_id, counts_old, counts_new)
        for k in old_totals:
            old_totals[k] += counts_old.get(k, 0)
            new_totals[k] += counts_new.get(k, 0)

    total_old = sum(old_totals.values()) or 1
    total_new = sum(new_totals.values()) or 1
    old_pct = old_totals["period_end"] / total_old * 100
    new_pct = new_totals["period_end"] / total_new * 100

    logger.info("Totals — old prompt: %s (period_end=%.1f%%)", old_totals, old_pct)
    logger.info("Totals — new prompt: %s (period_end=%.1f%%)", new_totals, new_pct)

    passed = new_pct <= old_pct
    logger.info("Part 3: %s (new_period_end=%.1f%% vs old=%.1f%%)",
                "PASS" if passed else "FAIL", new_pct, old_pct)
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_client() -> tuple[OpenRouterClient, str]:
    load_env(PROJECT_ROOT)
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set — cannot run integration tests")
    cfg = load_config(str(PROJECT_ROOT / "configs/train.yaml"))
    return OpenRouterClient(api_key), cfg.judge_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parts", nargs="+", type=int, default=[1, 2, 3],
        help="Parts to run (default: all three). E.g. --parts 1 2",
    )
    args = parser.parse_args()
    run_parts = set(args.parts)

    results: dict[int, bool] = {}

    # ---- Part 1: unit tests ----
    if 1 in run_parts:
        logger.info("\n=== Part 1: Unit tests ===")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestClaimPoolV2)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        results[1] = result.wasSuccessful()

    # ---- Parts 2 & 3: need live API ----
    if 2 in run_parts or 3 in run_parts:
        client, judge_model = _build_client()

    if 2 in run_parts:
        results[2] = run_part2(client, judge_model)

    if 3 in run_parts:
        results[3] = run_part3(client)

    # ---- Summary ----
    logger.info("\n=== Test summary ===")
    all_passed = True
    for part, passed in sorted(results.items()):
        status = "PASS" if passed else "FAIL"
        logger.info("  Part %d: %s", part, status)
        all_passed = all_passed and passed

    sys.exit(0 if all_passed else 1)
