"""Diagnostic: fake GRPO completions → full two-tier reward pipeline with verbose tracing.

Creates two completions for a stock task with cutoff 2021-12-31:
  Completion 0: OBVIOUS post-cutoff leakage (2022-2023 events)
  Completion 1: CLEAN pre-cutoff facts only (2020-2021)

Traces the two-tier verification flow:
  Tier 1: LLM parametric date check (batched, cheap)
  Tier 2: Sonar web search (only for low-confidence claims)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from cov_gated_experiment.config import TrainConfig
from cov_gated_experiment.llm import OpenRouterClient
from cov_gated_experiment.reward_pipeline import (
    Claim,
    RewardPipeline,
    _is_leaked_from_date,
    _rates,
)
from cov_gated_experiment.schemas import (
    CategorizationResult,
    DateVerifyResult,
    parse_json_array,
)
from cov_gated_experiment.prompts import (
    CATEGORIZE_PROMPT,
    CONFIDENT_LEVELS,
    DATE_VERIFY_PROMPT,
    FALLBACK_PROMPT,
    LLM_DATE_CHECK_PROMPT,
    SOURCE_B_PROMPT,
    build_completions_block,
)
from cov_gated_experiment.types import TaskInstance

CUTOFF = "2021-12-31"

LEAKED_COMPLETION = json.dumps({
    "evidence": [
        {"id": 1, "fact": "Cisco Systems (CSCO) reported Q1 FY2023 revenue of $13.6 billion in November 2022."},
        {"id": 2, "fact": "Super Micro Computer (SMCI) stock price rose over 300% during 2023 due to AI server demand."},
        {"id": 3, "fact": "Accenture (ACN) acquired Avanade's cloud practice in March 2022."},
        {"id": 4, "fact": "Hewlett Packard Enterprise (HPE) completed the acquisition of Juniper Networks in early 2024."},
        {"id": 5, "fact": "Western Digital (WDC) announced a planned split into two companies in October 2023."},
    ],
    "reasoning": (
        "Based on [1], Cisco showed strong revenue growth into FY2023, signaling robust enterprise demand. "
        "[2] demonstrates SMCI's explosive growth driven by the AI infrastructure boom in 2023. "
        "[3] shows Accenture expanding its cloud capabilities through the 2022 Avanade acquisition. "
        "[4] indicates HPE's major strategic move with the Juniper acquisition in 2024. "
        "[5] shows WDC pursuing a corporate restructuring. "
        "Given these factors, SMCI's AI-driven momentum makes it the top pick."
    ),
    "ranking": ["SMCI", "CSCO", "ACN", "HPE", "WDC"],
})

CLEAN_COMPLETION = json.dumps({
    "evidence": [
        {"id": 1, "fact": "Cisco Systems (CSCO) reported FY2021 total revenue of $49.8 billion."},
        {"id": 2, "fact": "Super Micro Computer (SMCI) had annual revenue of approximately $3.9 billion in FY2021."},
        {"id": 3, "fact": "Accenture (ACN) reported FY2021 revenue of $50.5 billion with 14% growth."},
        {"id": 4, "fact": "Hewlett Packard Enterprise (HPE) reported Q4 FY2021 revenue of $7.35 billion."},
        {"id": 5, "fact": "Western Digital (WDC) shipped 77 exabytes of HDD capacity in Q4 2021."},
    ],
    "reasoning": (
        "Analyzing pre-cutoff fundamentals: [1] shows Cisco maintaining steady enterprise revenue. "
        "[2] indicates SMCI has a small but growing server business as of 2021. "
        "[3] demonstrates Accenture's strong consulting growth trajectory. "
        "[4] shows HPE stabilizing its enterprise business. "
        "[5] indicates WDC benefits from continued storage demand. "
        "ACN's growth rate leads the group."
    ),
    "ranking": ["ACN", "CSCO", "SMCI", "HPE", "WDC"],
})


def sep(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main() -> None:
    import yaml
    with open(Path(__file__).resolve().parent.parent / "configs" / "train.yaml") as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set"); return
    client = OpenRouterClient(api_key)

    from cov_gated_experiment.schemas import parse_completion
    completions = [LEAKED_COMPLETION, CLEAN_COMPLETION]
    parsed = [parse_completion("stock", c) for c in completions]
    parseable = [p for p in parsed if p is not None]

    sep("STAGE 1: PARSE + SOURCE-B + BUILD CLAIM POOL")
    sb_prompt = SOURCE_B_PROMPT.format(completions_block=build_completions_block(parseable))
    sb_text, _ = client.call_text(cfg.judge_model, sb_prompt, temperature=0.0, max_tokens=4096)
    from cov_gated_experiment.schemas import SourceBResult
    sb_rows = parse_json_array(sb_text, SourceBResult)
    source_b: dict[int, list[str]] = {i: [] for i in range(len(parseable))}
    for row in sb_rows:
        source_b[row.completion_idx] = [c.claim for c in row.extra_claims]

    claim_pool: dict[int, list[Claim]] = {}
    for i, p in enumerate(parseable):
        claims: list[Claim] = []
        for ev in p.get("evidence", []):
            fact = str(ev.get("fact", "")).strip()
            if fact:
                claims.append(Claim(text=fact, owner_idx=i, source="A"))
        for b in source_b.get(i, []):
            if b.strip():
                claims.append(Claim(text=b.strip(), owner_idx=i, source="B"))
        claim_pool[i] = claims
    for i, cls in claim_pool.items():
        print(f"  Completion {i}: {len(cls)} claims")

    seen: set[str] = set()
    dedup: list[Claim] = []
    for cls in claim_pool.values():
        for c in cls:
            key = c.text.lower().strip()
            if key not in seen:
                seen.add(key)
                dedup.append(c)
    print(f"  Total unique claims: {len(dedup)}")

    sep("STAGE 2: CATEGORIZE")
    payload = [{"index": i, "claim": c.text, "source": c.source} for i, c in enumerate(dedup)]
    cat_text, _ = client.call_text(cfg.judge_model, CATEGORIZE_PROMPT.format(
        claims_json=json.dumps(payload, ensure_ascii=False, indent=2)), temperature=0.0, max_tokens=4096)
    cat_rows = parse_json_array(cat_text, CategorizationResult)
    by_idx = {r.index: r for r in cat_rows}
    categorized = [by_idx.get(i, CategorizationResult(index=i, category="A1", category_reason="fallback", search_query=None)) for i in range(len(dedup))]

    from collections import Counter
    cat_counts = Counter(c.category for c in categorized)
    print(f"  Category distribution: {dict(cat_counts)}")

    to_verify = [i for i, c in enumerate(categorized) if c.category not in {"A4", "A5", "B1", "B2"}]
    auto_leaked = [i for i, c in enumerate(categorized) if c.category in {"A4", "A5"}]
    auto_safe = [i for i, c in enumerate(categorized) if c.category in {"B1", "B2"}]
    print(f"  Auto-leaked (A4/A5): {len(auto_leaked)}")
    print(f"  Auto-safe (B1/B2): {len(auto_safe)}")
    print(f"  Need verification: {len(to_verify)}")

    # --- Tier 1: LLM parametric date check ---
    sep("STAGE 3 (TIER 1): LLM PARAMETRIC DATE CHECK")
    llm_payload = [{"index": j, "claim": dedup[idx].text, "category": categorized[idx].category} for j, idx in enumerate(to_verify)]
    llm_prompt = LLM_DATE_CHECK_PROMPT.format(claims_json=json.dumps(llm_payload, ensure_ascii=False, indent=2))
    llm_text, _ = client.call_text(cfg.judge_model, llm_prompt, temperature=0.0, max_tokens=4096)
    print(f"  Raw LLM response (first 2000 chars):\n{llm_text[:2000]}")
    llm_rows = parse_json_array(llm_text, DateVerifyResult)
    llm_by_idx = {r.index: r for r in llm_rows}

    results: dict[int, dict] = {}
    for i in auto_leaked:
        results[i] = {"leaked": True, "factual_status": "supported"}
    for i in auto_safe:
        results[i] = {"leaked": False, "factual_status": "supported"}

    needs_search = []
    tier1_resolved = 0
    for j, idx in enumerate(to_verify):
        hit = llm_by_idx.get(j)
        if hit is not None and hit.confidence in CONFIDENT_LEVELS and hit.event_date is not None:
            leaked = _is_leaked_from_date(hit.event_date, CUTOFF)
            results[idx] = {"leaked": leaked, "factual_status": hit.factual_status}
            tier1_resolved += 1
            print(f"    [{idx}] TIER1 RESOLVED: date={hit.event_date} status={hit.factual_status} conf={hit.confidence} → leaked={leaked}")
        else:
            conf = hit.confidence if hit else "missing"
            date = hit.event_date if hit else None
            print(f"    [{idx}] TIER1 UNRESOLVED: date={date} conf={conf} → sending to Sonar")
            needs_search.append(j)

    print(f"\n  Tier 1 resolved: {tier1_resolved}/{len(to_verify)}")
    print(f"  Need Sonar search: {len(needs_search)}")

    # --- Tier 2: Sonar search for low-confidence ---
    if needs_search:
        sep("STAGE 4 (TIER 2): SONAR WEB SEARCH FALLBACK")
        search_queries = []
        search_claim_indices = []
        for j in needs_search:
            idx = to_verify[j]
            query = categorized[idx].search_query or dedup[idx].text
            search_queries.append(query)
            search_claim_indices.append(idx)

        search_results_map = {}
        for k, query in enumerate(search_queries):
            text, _ = client.call_search_single(cfg.search_model, query)
            search_results_map[k] = text
            print(f"    Search [{k}] (claim {search_claim_indices[k]}): {text[:120]}...")

        data = [{"index": k, "claim": dedup[search_claim_indices[k]].text,
                 "category": categorized[search_claim_indices[k]].category,
                 "search_result": search_results_map.get(k, "")} for k in range(len(search_queries))]
        dv_prompt = DATE_VERIFY_PROMPT.format(data_json=json.dumps(data, ensure_ascii=False, indent=2))
        dv_text, _ = client.call_text(cfg.judge_model, dv_prompt, temperature=0.0, max_tokens=4096)
        dv_rows = parse_json_array(dv_text, DateVerifyResult)
        dv_by_idx = {r.index: r for r in dv_rows}

        for k, claim_idx in enumerate(search_claim_indices):
            hit = dv_by_idx.get(k)
            if hit is not None:
                leaked = _is_leaked_from_date(hit.event_date, CUTOFF)
                results[claim_idx] = {"leaked": leaked, "factual_status": hit.factual_status}
                print(f"    [{claim_idx}] TIER2: date={hit.event_date} status={hit.factual_status} → leaked={leaked}")
            else:
                print(f"    [{claim_idx}] TIER2: no response from date verifier")
    else:
        print("\n  No Sonar search needed — all claims resolved by Tier 1!")

    for i in range(len(dedup)):
        if i not in results:
            results[i] = {"leaked": False, "factual_status": "unsupported"}

    # --- Map back to completions ---
    sep("STAGE 5: MAP RESULTS BACK TO COMPLETIONS")
    dedup_index = {c.text.lower().strip(): idx for idx, c in enumerate(dedup)}
    for comp_idx in range(len(parseable)):
        my_texts = {c.text.lower().strip() for c in claim_pool.get(comp_idx, [])}
        my_results = []
        for text_key in my_texts:
            didx = dedup_index.get(text_key)
            if didx is not None:
                my_results.append(results.get(didx, {"leaked": False, "factual_status": "unsupported"}))

        leak_rate, hall_rate, num_claims, num_leaked, num_unsupported = _rates(my_results)
        expected = "LEAKED" if comp_idx == 0 else "CLEAN"
        print(f"  Completion {comp_idx} ({expected}):")
        print(f"    claims={num_claims} leaked={num_leaked} contradicted={num_unsupported}")
        print(f"    leakage_rate={leak_rate:.3f}  hallucination_rate={hall_rate:.3f}")

    sep("DIAGNOSIS COMPLETE")


if __name__ == "__main__":
    main()
