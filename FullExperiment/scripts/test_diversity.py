"""Test completion diversity at temperature=0.8 with Qwen via Tinker.

Generates 8 completions for one stock instance and measures:
  - Evidence fact overlap (Jaccard similarity)
  - Claim count variance
  - Ranking variance
  - Raw text similarity (character-level)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT / ".env")

import tinker
from tinker import types as tinker_types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from full_experiment.data import load_train_instances
from full_experiment.prompts import build_generation_prompt
from full_experiment.schemas import parse_completion

MODEL = "Qwen/Qwen3-30B-A3B"
NUM_SAMPLES = 8
TEMPERATURE = 0.8
MAX_TOKENS = 4096

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0

def run():
    data_root = PROJECT / "runtime" / "data"
    instances = load_train_instances(data_root, ["stock"])
    inst = instances[25]  # BG, CPB, KMB, MKC, TGT

    print(f"Instance: {inst.instance_id}")
    print(f"Companies: {inst.raw.get('companies', [])}")
    print(f"Cutoff: {inst.cutoff_date}")
    print(f"Model: {MODEL} | temp={TEMPERATURE} | samples={NUM_SAMPLES}")
    print()

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sampling_params = tinker_types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=renderer.get_stop_sequences(),
    )

    system, user = build_generation_prompt(inst)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    model_input = renderer.build_generation_prompt(messages)

    print("Generating completions...", flush=True)
    result = sampling_client.sample(
        prompt=model_input, num_samples=NUM_SAMPLES, sampling_params=sampling_params,
    ).result()

    completions = []
    for seq in result.sequences:
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        text = renderers.get_text_content(parsed_msg)
        completions.append(text)

    print(f"Got {len(completions)} completions\n")

    # Parse and analyze
    parsed_list = []
    for i, comp in enumerate(completions):
        p = parse_completion("stock", comp)
        parsed_list.append(p)
        if p is None:
            print(f"[{i}] PARSE FAIL | len={len(comp)} chars")
        else:
            ev_count = len(p.get("evidence", []))
            ranking = p.get("ranking", [])
            print(f"[{i}] OK | {ev_count} evidence items | ranking={ranking}")

    print()

    # Evidence fact sets
    fact_sets = []
    for i, p in enumerate(parsed_list):
        if p is None:
            fact_sets.append(set())
            continue
        facts = set()
        for ev in p.get("evidence", []):
            facts.add(ev.get("fact", "").strip().lower()[:80])
        fact_sets.append(facts)

    # Pairwise Jaccard similarity
    print("=== Evidence Fact Overlap (Jaccard) ===")
    jaccards = []
    for i in range(len(fact_sets)):
        for j in range(i + 1, len(fact_sets)):
            if parsed_list[i] is None or parsed_list[j] is None:
                continue
            j_val = jaccard(fact_sets[i], fact_sets[j])
            jaccards.append(j_val)
            if j_val > 0.5:
                print(f"  [{i}] vs [{j}]: {j_val:.3f}  (HIGH overlap)")

    if jaccards:
        import numpy as np
        arr = np.array(jaccards)
        print(f"\n  Mean Jaccard: {arr.mean():.3f}")
        print(f"  Max Jaccard:  {arr.max():.3f}")
        print(f"  Min Jaccard:  {arr.min():.3f}")
        print(f"  Std Jaccard:  {arr.std():.3f}")
    else:
        print("  (no valid pairs)")

    # Ranking diversity
    print("\n=== Ranking Diversity ===")
    rankings = [tuple(p.get("ranking", [])) for p in parsed_list if p is not None]
    unique_rankings = set(rankings)
    print(f"  Total rankings: {len(rankings)}")
    print(f"  Unique rankings: {len(unique_rankings)}")
    for r in unique_rankings:
        count = rankings.count(r)
        print(f"    {list(r)} x{count}")

    # Evidence count variance
    print("\n=== Evidence Count Distribution ===")
    ev_counts = [len(p.get("evidence", [])) for p in parsed_list if p is not None]
    if ev_counts:
        import numpy as np
        arr = np.array(ev_counts)
        print(f"  Counts: {ev_counts}")
        print(f"  Mean: {arr.mean():.1f}  Std: {arr.std():.1f}")

    # Print first 200 chars of each evidence list for visual inspection
    print("\n=== First 3 Evidence Items per Completion ===")
    for i, p in enumerate(parsed_list):
        if p is None:
            continue
        evs = p.get("evidence", [])[:3]
        print(f"\n  [{i}]:")
        for ev in evs:
            fact = ev.get("fact", "")[:120]
            print(f"    - {fact}")

if __name__ == "__main__":
    run()
