#!/usr/bin/env python3
"""Trace exactly what happens for ONE real training instance.

Generates completions from a saved step-0 instance, runs them through
the full reward pipeline with verbose tracing at every stage, and reports
where leakage/hallucination detection succeeds or fails.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import yaml
from full_experiment.config import TrainConfig
from full_experiment.data import load_train_instances
from full_experiment.llm import OpenRouterClient
from full_experiment.prompts import (
    CATEGORIZE_PROMPT,
    CONFIDENT_LEVELS,
    LLM_DATE_CHECK_PROMPT,
    DATE_VERIFY_PROMPT,
    SOURCE_B_PROMPT,
    build_completions_block,
    build_generation_prompt,
)
from full_experiment.reward_pipeline import (
    Claim,
    RewardPipeline,
    _is_leaked_from_date,
    _rates,
)
from full_experiment.schemas import (
    CategorizationResult,
    DateVerifyResult,
    parse_completion,
    parse_json_array,
)

def sep(title: str) -> None:
    print(f"\n{'='*72}\n  {title}\n{'='*72}\n")


def main() -> None:
    with open(Path(__file__).resolve().parent.parent / "configs" / "train.yaml") as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set"); return
    client = OpenRouterClient(api_key)

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / cfg.data_root
    tasks = [t.strip() for t in cfg.train_tasks.split(",") if t.strip()]
    all_instances = load_train_instances(data_root, tasks)
    print(f"Loaded {len(all_instances)} instances")

    # Pick one instance with cutoff 2021-12-31 (most likely to show leakage)
    target = None
    for inst in all_instances:
        if inst.cutoff_date == "2021-12-31":
            target = inst
            break
    if target is None:
        print(f"No instance with cutoff 2021-12-31 found")
        print(f"Sample cutoffs: {[i.cutoff_date for i in all_instances[:5]]}")
        return

    sep(f"INSTANCE: {target.instance_id}  cutoff={target.cutoff_date}  task={target.task}")
    print(f"  Sector: {target.raw.get('sector', '?')}")
    print(f"  Companies: {target.raw.get('companies', [])}")

    # Generate ONE completion using the same prompt as training
    sys_prompt, user_prompt = build_generation_prompt(target)
    print(f"\n  System prompt (first 200 chars): {sys_prompt[:200]}...")
    print(f"  User prompt (first 200 chars): {user_prompt[:200]}...")

    sep("STAGE 0: GENERATE COMPLETION (temperature=0.8)")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Use the same model but via OpenRouter to simulate what Tinker generates
    gen_text, _ = client.call_text(
        "qwen/qwen3-30b-a3b", 
        sys_prompt + "\n\n" + user_prompt,
        temperature=0.8, max_tokens=4096
    )
    print(f"  Raw completion (first 500 chars):\n{gen_text[:500]}")

    parsed = parse_completion(target.task, gen_text)
    if parsed is None:
        print("\n  PARSE FAILED — cannot continue")
        return

    evidence = parsed.get("evidence", [])
    reasoning = parsed.get("reasoning", "")
    print(f"\n  Parsed successfully: {len(evidence)} evidence items")
    for ev in evidence:
        print(f"    [{ev.get('id')}] {ev.get('fact', '')[:100]}")

    sep("STAGE 1: SOURCE-B EXTRACTION")
    sb_prompt = SOURCE_B_PROMPT.format(completions_block=build_completions_block([parsed]))
    sb_text, _ = client.call_text(cfg.judge_model, sb_prompt, temperature=0.0, max_tokens=4096)
    from full_experiment.schemas import SourceBResult
    sb_rows = parse_json_array(sb_text, SourceBResult)
    source_b: dict[int, list[str]] = {0: []}
    for row in sb_rows:
        source_b[row.completion_idx] = [c.claim for c in row.extra_claims]
    print(f"  Source B claims: {len(source_b.get(0, []))}")
    for b in source_b.get(0, []):
        print(f"    B: {b[:100]}")

    sep("STAGE 2: BUILD CLAIM POOL")
    claims: list[Claim] = []
    for ev in evidence:
        fact = str(ev.get("fact", "")).strip()
        if fact:
            claims.append(Claim(text=fact, owner_idx=0, source="A"))
    for b in source_b.get(0, []):
        if b.strip():
            claims.append(Claim(text=b.strip(), owner_idx=0, source="B"))
    print(f"  Total claims: {len(claims)}")

    sep("STAGE 3: CATEGORIZE")
    payload = [{"index": i, "claim": c.text, "source": c.source} for i, c in enumerate(claims)]
    cat_text, _ = client.call_text(cfg.judge_model, CATEGORIZE_PROMPT.format(
        claims_json=json.dumps(payload, ensure_ascii=False, indent=2)), temperature=0.0, max_tokens=4096)
    cat_rows = parse_json_array(cat_text, CategorizationResult)
    by_idx = {r.index: r for r in cat_rows}
    categorized = [by_idx.get(i, CategorizationResult(index=i, category="A1", category_reason="fallback", search_query=None)) for i in range(len(claims))]

    from collections import Counter
    cat_counts = Counter(c.category for c in categorized)
    print(f"  Category distribution: {dict(cat_counts)}")
    for i, (cl, cat) in enumerate(zip(claims, categorized)):
        print(f"  [{i}] {cat.category}: {cl.text[:80]}...")

    to_verify = [i for i, c in enumerate(categorized) if c.category not in {"A4", "A5", "B1", "B2"}]
    print(f"\n  Need verification: {len(to_verify)} claims")

    if not to_verify:
        print("  All claims are auto-resolved (A4/A5/B1/B2). No leakage detection possible.")
        return

    sep("STAGE 4 (TIER 1): LLM DATE CHECK")
    llm_payload = [{"index": j, "claim": claims[idx].text, "category": categorized[idx].category}
                   for j, idx in enumerate(to_verify)]
    llm_prompt = LLM_DATE_CHECK_PROMPT.format(
        claims_json=json.dumps(llm_payload, ensure_ascii=False, indent=2))
    llm_text, _ = client.call_text(cfg.judge_model, llm_prompt, temperature=0.0, max_tokens=4096)
    llm_rows = parse_json_array(llm_text, DateVerifyResult)
    llm_by_idx = {r.index: r for r in llm_rows}

    results = {}
    needs_search = []
    for j, idx in enumerate(to_verify):
        hit = llm_by_idx.get(j)
        if hit is not None and hit.confidence in CONFIDENT_LEVELS and hit.event_date is not None:
            leaked = _is_leaked_from_date(hit.event_date, target.cutoff_date)
            results[idx] = {"leaked": leaked, "factual_status": hit.factual_status}
            status = f"RESOLVED date={hit.event_date} status={hit.factual_status} conf={hit.confidence} leaked={leaked}"
        else:
            needs_search.append(j)
            conf = hit.confidence if hit else "missing"
            date = hit.event_date if hit else None
            fstat = hit.factual_status if hit else "?"
            status = f"UNRESOLVED date={date} status={fstat} conf={conf} → SONAR"
        print(f"  [{idx}] {claims[idx].text[:70]}...")
        print(f"       {status}")

    print(f"\n  Tier 1 resolved: {len(to_verify) - len(needs_search)}/{len(to_verify)}")
    print(f"  Need Sonar: {len(needs_search)}")

    tier1_leaked = sum(1 for v in results.values() if v["leaked"])
    tier1_contradicted = sum(1 for v in results.values() if v["factual_status"] == "contradicted")
    print(f"  Tier 1 leaked: {tier1_leaked}")
    print(f"  Tier 1 contradicted: {tier1_contradicted}")

    if needs_search:
        sep("STAGE 5 (TIER 2): SONAR SEARCH")
        for j in needs_search:
            idx = to_verify[j]
            query = categorized[idx].search_query or claims[idx].text
            text, _ = client.call_search_single(cfg.search_model, query)
            print(f"  [{idx}] {claims[idx].text[:60]}...")
            print(f"       Sonar: {text[:120]}...")
            print()

    sep("FINAL: COMPUTE RATES")
    # Auto-resolved
    for i, cat in enumerate(categorized):
        if i not in results:
            if cat.category in {"A4", "A5"}:
                results[i] = {"leaked": True, "factual_status": "supported"}
            elif cat.category in {"B1", "B2"}:
                results[i] = {"leaked": False, "factual_status": "supported"}
            else:
                results[i] = {"leaked": False, "factual_status": "unsupported"}

    all_results = [results.get(i, {"leaked": False, "factual_status": "unsupported"})
                   for i in range(len(claims))]
    leak_rate, hall_rate, num_claims, num_leaked, num_unsupported = _rates(all_results)
    print(f"  claims={num_claims} leaked={num_leaked} contradicted={num_unsupported}")
    print(f"  leakage_rate={leak_rate:.3f}  hallucination_rate={hall_rate:.3f}")

    if num_leaked == 0 and num_unsupported == 0:
        print("\n  *** STILL ZERO! Possible explanations:")
        print("  1. Model genuinely generates only pre-cutoff, factually accurate claims")
        print("  2. Gemini Tier 1 is incorrectly marking claims as 'supported' + pre-cutoff")
        print("  3. Claims are mostly B1/B2 (auto-safe) and the few A1/A2 are legitimate")


if __name__ == "__main__":
    main()
