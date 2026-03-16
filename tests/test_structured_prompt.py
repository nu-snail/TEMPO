"""Test: Does a structured evidence-reasoning-prediction prompt help
Qwen3-30B-A3B produce cleaner, more explicit outputs without any training?

Runs one example per task (stock, salary, legal) with:
  (A) Current free-form prompt
  (B) Proposed structured prompt (evidence list → cited reasoning → prediction)

Saves full outputs to tests/structured_prompt_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared.data_loader import load_all_train
from src.shared.prompts.builder import build_prompt
from src.shared.llm_client import LLMClient

MODEL = "qwen/qwen3-30b-a3b"
OUT_PATH = Path(__file__).parent / "structured_prompt_results_v2.json"


# ── Structured prompts ──

STRUCTURED_STOCK_SYSTEM = """\
You are an expert stock analyst. Your task is to rank a set of companies by
their expected stock return over a specified evaluation period.

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}.
Do NOT use any knowledge of events, earnings reports, market movements,
announcements, or developments from on or after {cutoff_date}.
If uncertain whether a piece of information predates the cutoff, omit it.

You MUST follow this three-step procedure:

STEP 1 — EVIDENCE:
List every verifiable fact you will use. Requirements:
- Each fact must be concrete and specific (a reported number, a completed
  event, a published announcement) — not an opinion or prediction.
- Each fact must have been publicly known before {cutoff_date}.
- Write each fact as a plain statement. Do NOT include citation markers
  like [1] or [2] in the evidence text.
- Include as many facts as your analysis requires — no fixed number.

STEP 2 — REASONING:
Write a reasoning paragraph that cites evidence by number (e.g. [1], [3]).
Rules:
- You MUST cite every evidence item at least once.
- Do NOT introduce any new facts, statistics, events, or entity names
  that are not listed in your evidence. Only analysis and inference
  from cited evidence are allowed.
- Show a clear logical path from cited evidence to your ranking.

STEP 3 — PREDICTION:
Give your final ranking based solely on the reasoning above.

OUTPUT FORMAT — a single JSON object, no markdown fences:
{
  "evidence": [
    {"id": 1, "fact": "<plain verifiable fact>"},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "ranking": ["BEST_TICKER", ..., "WORST_TICKER"]
}"""

STRUCTURED_SALARY_SYSTEM = """\
You are an expert sports salary analyst. Your task is to predict a free
agent's upcoming contract value (Average Annual Value in USD).

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}.
Do NOT use any knowledge of signings, trades, salary-cap changes,
or performance data from on or after {cutoff_date}.
If uncertain whether a piece of information predates the cutoff, omit it.

You MUST follow this three-step procedure:

STEP 1 — EVIDENCE:
List every verifiable fact you will use. Requirements:
- Each fact must be concrete and specific (a player stat, a signed contract,
  a reported market condition) — not an opinion or prediction.
- Each fact must have been publicly known before {cutoff_date}.
- Write each fact as a plain statement. Do NOT include citation markers
  like [1] or [2] in the evidence text.
- Include as many facts as your analysis requires — no fixed number.

STEP 2 — REASONING:
Write a reasoning paragraph that cites evidence by number (e.g. [1], [3]).
Rules:
- You MUST cite every evidence item at least once.
- Do NOT introduce any new facts, statistics, player details, or contract
  information not listed in your evidence. Only analysis and inference
  from cited evidence are allowed.
- Show how comparable contracts, player performance, and market conditions
  lead to your salary estimate.

STEP 3 — PREDICTION:
Give your predicted AAV based solely on the reasoning above.

OUTPUT FORMAT — a single JSON object, no markdown fences:
{
  "evidence": [
    {"id": 1, "fact": "<plain verifiable fact>"},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "predicted_salary": <integer USD>
}"""

STRUCTURED_LEGAL_SYSTEM = """\
You are an expert Supreme Court analyst. Your task is to predict the outcome
of a case based on its facts, the legal questions presented, and relevant
precedent.

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}.
Do NOT use any knowledge of the case's actual decision, subsequent rulings,
commentary, or legislative responses from on or after {cutoff_date}.
If uncertain whether a piece of information predates the cutoff, omit it.

You MUST follow this three-step procedure:

STEP 1 — EVIDENCE:
List every verifiable fact and legal precedent you will use. Requirements:
- Each item must be a specific case fact, or a decided case with its
  holding — not an opinion or prediction.
- Each item must have been publicly known before {cutoff_date}.
- Write each item as a plain statement. Do NOT include citation markers
  like [1] or [2] in the evidence text.
- Include as many items as your analysis requires — no fixed number.

STEP 2 — REASONING:
Write a reasoning paragraph that cites evidence by number (e.g. [1], [3]).
Rules:
- You MUST cite every evidence item at least once.
- Do NOT introduce any new facts, case names, legal principles, or holdings
  not listed in your evidence. Only analysis and inference from cited
  evidence are allowed.
- Analyse each side's arguments using only cited precedent and facts.

STEP 3 — PREDICTION:
Give your predicted outcome based solely on the reasoning above.

OUTPUT FORMAT — a single JSON object, no markdown fences:
{
  "evidence": [
    {"id": 1, "fact": "<plain verifiable fact or case holding>"},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "probability_petitioner": <float 0.0 to 1.0>,
  "predicted_winner": "<PETITIONER or RESPONDENT>"
}"""

STRUCTURED_SYSTEMS = {
    "stock": STRUCTURED_STOCK_SYSTEM,
    "salary": STRUCTURED_SALARY_SYSTEM,
    "legal": STRUCTURED_LEGAL_SYSTEM,
}


def get_one_instance_per_task():
    """Load one instance per task for testing."""
    all_data = load_all_train()
    picked = {}
    for inst in all_data:
        if inst.task not in picked:
            picked[inst.task] = inst
        if len(picked) == 3:
            break
    return picked


def call_model(client, system: str, user: str) -> str:
    result = client.call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=MODEL,
        temperature=0.0,
        max_tokens=4096,
    )
    return result["content"] or ""


def main():
    client = LLMClient(provider="openrouter")
    instances = get_one_instance_per_task()

    results = {}
    for task, inst in instances.items():
        print(f"\n{'='*60}")
        print(f"  Task: {task} | Instance: {inst.instance_id}")
        print(f"  Cutoff: {inst.cutoff_date}")
        print(f"{'='*60}")

        # (A) Current prompt
        system_a, user_a = build_prompt(inst)
        print(f"\n  [A] Current prompt — calling {MODEL}...")
        output_a = call_model(client, system_a, user_a)
        print(f"  [A] Done. Length: {len(output_a)} chars")

        # (B) Structured prompt
        system_b_template = STRUCTURED_SYSTEMS[task]
        system_b = system_b_template.replace("{cutoff_date}", inst.cutoff_date)
        _, user_b = build_prompt(inst)  # same user prompt
        print(f"\n  [B] Structured prompt — calling {MODEL}...")
        output_b = call_model(client, system_b, user_b)
        print(f"  [B] Done. Length: {len(output_b)} chars")

        results[task] = {
            "instance_id": inst.instance_id,
            "cutoff_date": inst.cutoff_date,
            "user_prompt": user_b,
            "current_prompt": {
                "system": system_a,
                "output": output_a,
                "length": len(output_a),
            },
            "structured_prompt": {
                "system": system_b,
                "output": output_b,
                "length": len(output_b),
            },
        }

    # Save results
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUT_PATH}")

    # Print side-by-side summary
    for task, r in results.items():
        print(f"\n{'='*70}")
        print(f"  TASK: {task.upper()} ({r['instance_id']})")
        print(f"{'='*70}")
        print(f"\n  --- [A] CURRENT OUTPUT ({r['current_prompt']['length']} chars) ---")
        print(r["current_prompt"]["output"][:2000])
        if len(r["current_prompt"]["output"]) > 2000:
            print("  ... [truncated]")
        print(f"\n  --- [B] STRUCTURED OUTPUT ({r['structured_prompt']['length']} chars) ---")
        print(r["structured_prompt"]["output"][:2000])
        if len(r["structured_prompt"]["output"]) > 2000:
            print("  ... [truncated]")


if __name__ == "__main__":
    main()
