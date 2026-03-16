#!/usr/bin/env python3
"""Send the ACTUAL claims from step-0 completions through the pipeline
to see exactly what Gemini Tier 1 and the categorizer return."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from cov_gated_experiment.llm import OpenRouterClient
from cov_gated_experiment.prompts import (
    CATEGORIZE_PROMPT,
    CONFIDENT_LEVELS,
    LLM_DATE_CHECK_PROMPT,
    SOURCE_B_PROMPT,
    build_completions_block,
)
from cov_gated_experiment.schemas import (
    CategorizationResult,
    DateVerifyResult,
    SourceBResult,
    parse_completion,
    parse_json_array,
)
from cov_gated_experiment.reward_pipeline import Claim, _is_leaked_from_date

CUTOFF = "2021-12-31"
JUDGE = "google/gemini-2.5-flash"


def sep(title: str) -> None:
    print(f"\n{'='*72}\n  {title}\n{'='*72}\n")


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set"); return
    client = OpenRouterClient(api_key)

    diag_path = Path(__file__).resolve().parent.parent / "runtime/results/train_stock/diag_completions_step0.json"
    with open(diag_path) as f:
        diag = json.load(f)

    raw_comp = diag["completions"][0]
    parsed = parse_completion("stock", raw_comp)
    if parsed is None:
        print("PARSE FAILED"); return

    evidence = parsed.get("evidence", [])
    sep(f"EVIDENCE FROM COMPLETION (cutoff={CUTOFF})")
    for ev in evidence:
        print(f"  [{ev['id']}] {ev['fact']}")

    # Source B extraction
    sb_prompt = SOURCE_B_PROMPT.format(completions_block=build_completions_block([parsed]))
    sb_text, _ = client.call_text(JUDGE, sb_prompt, temperature=0.0, max_tokens=4096)
    sb_rows = parse_json_array(sb_text, SourceBResult)
    source_b_claims = []
    for row in sb_rows:
        source_b_claims.extend([c.claim for c in row.extra_claims])

    sep("SOURCE B CLAIMS")
    for b in source_b_claims:
        print(f"  B: {b}")

    # Build full claim list
    claims: list[Claim] = []
    for ev in evidence:
        claims.append(Claim(text=str(ev.get("fact", "")), owner_idx=0, source="A"))
    for b in source_b_claims:
        claims.append(Claim(text=b.strip(), owner_idx=0, source="B"))

    sep(f"ALL CLAIMS ({len(claims)} total)")
    for i, c in enumerate(claims):
        print(f"  [{i}] ({c.source}) {c.text[:90]}")

    # Categorize
    sep("CATEGORIZE")
    payload = [{"index": i, "claim": c.text, "source": c.source} for i, c in enumerate(claims)]
    cat_text, _ = client.call_text(JUDGE, CATEGORIZE_PROMPT.format(
        claims_json=json.dumps(payload, ensure_ascii=False, indent=2)), temperature=0.0, max_tokens=4096)
    cat_rows = parse_json_array(cat_text, CategorizationResult)
    by_idx = {r.index: r for r in cat_rows}
    categorized = [by_idx.get(i, CategorizationResult(index=i, category="A1", category_reason="fallback", search_query=None))
                   for i in range(len(claims))]

    from collections import Counter
    cat_dist = Counter(c.category for c in categorized)
    print(f"  Distribution: {dict(cat_dist)}")
    for i, (cl, cat) in enumerate(zip(claims, categorized)):
        print(f"  [{i}] {cat.category}: {cl.text[:70]}  reason={cat.category_reason[:50]}")

    # Determine which need verification
    to_verify = []
    auto_leaked = []
    auto_safe = []
    for i, cat in enumerate(categorized):
        if cat.category in {"A4", "A5"}:
            auto_leaked.append(i)
        elif cat.category in {"B1", "B2"}:
            auto_safe.append(i)
        else:
            to_verify.append(i)

    print(f"\n  Auto-leaked (A4/A5): {auto_leaked}")
    print(f"  Auto-safe (B1/B2): {auto_safe}")
    print(f"  Need verify (A1/A2/A3): {to_verify}")

    if not to_verify:
        print("\n  ALL claims auto-resolved. Pipeline CANNOT detect leakage in these.")
        return

    # Tier 1: LLM date check
    sep("TIER 1: LLM DATE CHECK")
    llm_payload = [{"index": j, "claim": claims[idx].text, "category": categorized[idx].category}
                   for j, idx in enumerate(to_verify)]
    llm_prompt = LLM_DATE_CHECK_PROMPT.format(
        claims_json=json.dumps(llm_payload, ensure_ascii=False, indent=2))
    llm_raw, _ = client.call_text(JUDGE, llm_prompt, temperature=0.0, max_tokens=4096)
    llm_rows = parse_json_array(llm_raw, DateVerifyResult)
    llm_by_idx = {r.index: r for r in llm_rows}

    needs_search = []
    results = {}
    for j, idx in enumerate(to_verify):
        hit = llm_by_idx.get(j)
        if hit is None:
            print(f"  [{idx}] MISSING from Tier 1 response!")
            needs_search.append(j)
            continue

        resolved = hit.confidence in CONFIDENT_LEVELS and hit.event_date is not None
        if resolved:
            leaked = _is_leaked_from_date(hit.event_date, CUTOFF)
            results[idx] = {"leaked": leaked, "factual_status": hit.factual_status}
            tag = "LEAKED" if leaked else "OK"
        else:
            tag = "→ SONAR"
            needs_search.append(j)

        print(f"  [{idx}] {tag}")
        print(f"       claim: {claims[idx].text[:70]}")
        print(f"       date={hit.event_date}  conf={hit.confidence}  status={hit.factual_status}")
        print(f"       reasoning: {hit.reasoning[:100] if hit.reasoning else 'none'}")
        print()

    n_leaked = sum(1 for v in results.values() if v.get("leaked"))
    n_contra = sum(1 for v in results.values() if v.get("factual_status") == "contradicted")
    print(f"  Tier 1 resolved: {len(to_verify) - len(needs_search)}/{len(to_verify)}")
    print(f"  Leaked: {n_leaked}  Contradicted: {n_contra}  Need Sonar: {len(needs_search)}")


if __name__ == "__main__":
    main()
