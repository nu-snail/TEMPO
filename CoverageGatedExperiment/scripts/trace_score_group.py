#!/usr/bin/env python3
"""Trace the FULL score_group method with 2 fake completions.

Uses the same flow as training: parse → source_b → claim pool → dedup →
categorize → verify (two-tier) → _claims_for_completion → _rates.

Exposes bugs in the dedup-to-per-completion mapping.
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
from cov_gated_experiment.config import TrainConfig
from cov_gated_experiment.llm import OpenRouterClient
from cov_gated_experiment.reward_pipeline import (
    Claim,
    RewardPipeline,
    _rates,
)
from cov_gated_experiment.types import TaskInstance

CUTOFF = "2021-12-31"

LEAKED_COMP = json.dumps({
    "evidence": [
        {"id": 1, "fact": "Cisco Systems (CSCO) reported Q1 FY2023 revenue of $13.6 billion in November 2022."},
        {"id": 2, "fact": "Super Micro Computer (SMCI) stock price rose over 300% during 2023 due to AI server demand."},
        {"id": 3, "fact": "Hewlett Packard Enterprise (HPE) completed the acquisition of Juniper Networks in early 2024."},
    ],
    "reasoning": (
        "Based on [1], Cisco showed strong growth into FY2023. "
        "[2] demonstrates SMCI's AI-driven momentum in 2023. "
        "[3] shows HPE's strategic move with Juniper. "
        "SMCI's momentum makes it the top pick."
    ),
    "ranking": ["SMCI", "CSCO", "HPE", "AAPL", "MSFT"],
})

CLEAN_COMP = json.dumps({
    "evidence": [
        {"id": 1, "fact": "Cisco Systems (CSCO) reported FY2021 total revenue of $49.8 billion."},
        {"id": 2, "fact": "Accenture (ACN) reported FY2021 revenue of $50.5 billion with 14% growth."},
        {"id": 3, "fact": "Apple (AAPL) reported Q4 FY2021 revenue of $83.4 billion in October 2021."},
    ],
    "reasoning": (
        "[1] shows Cisco maintaining steady revenue. "
        "[2] shows Accenture's strong growth trajectory. "
        "[3] shows Apple's massive Q4 performance. "
        "ACN's growth rate leads the group."
    ),
    "ranking": ["ACN", "AAPL", "CSCO", "MSFT", "HPE"],
})


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    with open(project_root / "configs" / "train.yaml") as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set"); return
    client = OpenRouterClient(api_key)

    instance = TaskInstance(
        instance_id="test_trace",
        task="stock",
        cutoff_date=CUTOFF,
        ground_truth={"ranking": ["SMCI", "CSCO", "ACN", "AAPL", "HPE"]},
        raw={},
    )

    pipeline = RewardPipeline(cfg, client)
    completions = [LEAKED_COMP, CLEAN_COMP]

    print("="*72)
    print("  RUNNING score_group with 2 completions")
    print(f"  Cutoff: {CUTOFF}")
    print("  Completion 0: LEAKED (post-2021 facts)")
    print("  Completion 1: CLEAN (pre-2022 facts)")
    print("="*72)

    rewards = pipeline.score_group(completions, instance)

    print()
    for i, r in enumerate(rewards):
        expected = "LEAKED" if i == 0 else "CLEAN"
        print(f"Completion {i} ({expected}):")
        print(f"  parse_success={r.parse_success}")
        print(f"  num_claims={r.num_claims}  num_leaked={r.num_leaked}  num_unsupported={r.num_unsupported}")
        print(f"  leakage_rate={r.leakage_rate:.3f}  hallucination_rate={r.hallucination_rate:.3f}")
        print(f"  r_valid={r.r_valid:.3f}  r_performance={r.r_performance:.3f}")
        print()

    # Verify expectations
    ok = True
    if rewards[0].num_leaked == 0:
        print("*** BUG: Completion 0 (LEAKED) has num_leaked=0!")
        ok = False
    else:
        print(f"OK: Completion 0 has {rewards[0].num_leaked} leaked claims")

    if rewards[1].num_leaked > 0:
        print(f"*** UNEXPECTED: Completion 1 (CLEAN) has {rewards[1].num_leaked} leaked claims")
    else:
        print("OK: Completion 1 has 0 leaked claims")

    if ok:
        print("\n*** Pipeline mapping is CORRECT ***")
    else:
        print("\n*** Pipeline mapping has a BUG ***")


if __name__ == "__main__":
    main()
