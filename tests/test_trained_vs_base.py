"""Compare untrained (base) vs trained Qwen3-30B-A3B on stock and salary tasks.

Uses the structured evidence-reasoning-prediction prompt format.
Runs inference via Tinker for both base and trained (LoRA) models.

Outputs: tests/trained_vs_base_results.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from src.shared.data_loader import load_all_train
from src.shared.prompts.builder import build_prompt

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
OUT_PATH = Path(__file__).parent / "trained_vs_base_results.json"

STOCK_SAMPLER = "tinker://84dfa2c4-8c2a-5e7b-9d74-186d7743d28a:train:0/sampler_weights/final"
SALARY_SAMPLER = "tinker://9bf33d63-c072-5d41-87ab-7dcd7436099b:train:0/sampler_weights/final"

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

STRUCTURED_SYSTEMS = {
    "stock": STRUCTURED_STOCK_SYSTEM,
    "salary": STRUCTURED_SALARY_SYSTEM,
}


def make_tinker_sampler(sampler_path: str | None):
    """Create a Tinker sampling client directly (inference only, no training state).
    sampler_path=None -> base model, otherwise -> trained LoRA sampler weights.
    """
    service_client = tinker.ServiceClient()
    if sampler_path:
        return service_client.create_sampling_client(model_path=sampler_path)
    return service_client.create_sampling_client(base_model=MODEL_NAME)


def call_tinker(sampling_client, renderer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    model_input = renderer.build_generation_prompt(messages)
    sampling_params = tinker.types.SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )
    result = sampling_client.sample(
        prompt=model_input, num_samples=1, sampling_params=sampling_params,
    ).result()
    parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
    return renderers.get_text_content(parsed_msg)


def main():
    random.seed(42)
    all_data = load_all_train()
    stock_instances = [d for d in all_data if d.task == "stock"]
    salary_instances = [d for d in all_data if d.task == "salary"]
    stock_inst = random.choice(stock_instances)
    salary_inst = random.choice(salary_instances)

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    tasks = {
        "stock": {"instance": stock_inst, "sampler": STOCK_SAMPLER},
        "salary": {"instance": salary_inst, "sampler": SALARY_SAMPLER},
    }

    results = {}
    print("Initializing base model sampler...")
    base_sampler = make_tinker_sampler(None)
    print("Base sampler ready.\n")

    for task_name, cfg in tasks.items():
        inst = cfg["instance"]
        sampler_path = cfg["sampler"]

        print(f"\n{'='*60}")
        print(f"  Task: {task_name} | ID: {inst.instance_id}")
        print(f"  Cutoff: {inst.cutoff_date}")
        print(f"  Ground truth: {inst.ground_truth}")
        print(f"{'='*60}")

        system_template = STRUCTURED_SYSTEMS[task_name]
        system = system_template.replace("{cutoff_date}", inst.cutoff_date)
        _, user = build_prompt(inst)

        # (A) Base model (untrained)
        print(f"\n  [A] Calling base model...")
        output_base = call_tinker(base_sampler, renderer, system, user)
        print(f"  [A] Done. Length: {len(output_base)} chars")

        # (B) Trained model
        print(f"\n  [B] Loading trained sampler weights...")
        trained_sampler = make_tinker_sampler(sampler_path)
        print(f"  [B] Calling trained model...")
        output_trained = call_tinker(trained_sampler, renderer, system, user)
        print(f"  [B] Done. Length: {len(output_trained)} chars")

        results[task_name] = {
            "instance_id": inst.instance_id,
            "cutoff_date": inst.cutoff_date,
            "ground_truth": inst.ground_truth,
            "user_prompt": user,
            "system_prompt": system,
            "base_model": {
                "output": output_base,
                "length": len(output_base),
            },
            "trained_model": {
                "sampler": sampler_path,
                "output": output_trained,
                "length": len(output_trained),
            },
        }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUT_PATH}")

    for task_name, r in results.items():
        print(f"\n{'='*70}")
        print(f"  TASK: {task_name.upper()} ({r['instance_id']})")
        print(f"  Cutoff: {r['cutoff_date']}")
        print(f"  Ground truth: {r['ground_truth']}")
        print(f"{'='*70}")
        print(f"\n  --- [A] BASE MODEL ({r['base_model']['length']} chars) ---")
        print(r["base_model"]["output"][:3000])
        if len(r["base_model"]["output"]) > 3000:
            print("  ... [truncated]")
        print(f"\n  --- [B] TRAINED MODEL ({r['trained_model']['length']} chars) ---")
        print(r["trained_model"]["output"][:3000])
        if len(r["trained_model"]["output"]) > 3000:
            print("  ... [truncated]")


if __name__ == "__main__":
    main()
