"""Compare untrained (base) vs trained Qwen3-30B-A3B using the ORIGINAL
training prompts (temporal_hint mode via build_prompt).

Tests one random instance per task (stock, salary, legal).
Outputs: tests/trained_vs_base_original_prompt.json
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

MODEL = "Qwen/Qwen3-30B-A3B"
OUT_PATH = Path(__file__).parent / "trained_vs_base_original_prompt.json"

SAMPLERS = {
    "stock": "tinker://84dfa2c4-8c2a-5e7b-9d74-186d7743d28a:train:0/sampler_weights/final",
    "salary": "tinker://9bf33d63-c072-5d41-87ab-7dcd7436099b:train:0/sampler_weights/final",
    "legal": "tinker://6e52fd84-ab74-5969-b7a8-e1c4bf029626:train:0/sampler_weights/best",
}


def call_tinker(sampler, renderer, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    model_input = renderer.build_generation_prompt(messages)
    sampling_params = tinker.types.SamplingParams(
        max_tokens=4096, temperature=0.0, stop=renderer.get_stop_sequences(),
    )
    result = sampler.sample(
        prompt=model_input, num_samples=1, sampling_params=sampling_params,
    ).result()
    parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
    return renderers.get_text_content(parsed_msg)


def main():
    random.seed(42)
    all_data = load_all_train()
    by_task: dict[str, list] = {}
    for d in all_data:
        by_task.setdefault(d.task, []).append(d)
    instances = {t: random.choice(v) for t, v in by_task.items() if t in SAMPLERS}

    tokenizer = get_tokenizer(MODEL)
    rname = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(rname, tokenizer)

    sc = tinker.ServiceClient()
    print("Creating base sampler...")
    base_sampler = sc.create_sampling_client(base_model=MODEL)
    print("Base sampler ready.\n")

    results = {}
    for task in ["stock", "salary", "legal"]:
        inst = instances[task]
        system, user = build_prompt(inst, prompt_mode="temporal_hint")

        print(f"{'='*60}")
        print(f"  Task: {task} | ID: {inst.instance_id} | Cutoff: {inst.cutoff_date}")
        print(f"  Ground truth: {inst.ground_truth}")
        print(f"{'='*60}")

        print(f"\n  [A] Base model...")
        out_base = call_tinker(base_sampler, renderer, system, user)
        print(f"  [A] Done. {len(out_base)} chars")

        print(f"\n  [B] Trained model ({SAMPLERS[task].split('/')[-2]})...")
        trained_sampler = sc.create_sampling_client(model_path=SAMPLERS[task])
        out_trained = call_tinker(trained_sampler, renderer, system, user)
        print(f"  [B] Done. {len(out_trained)} chars\n")

        results[task] = {
            "instance_id": inst.instance_id,
            "cutoff_date": inst.cutoff_date,
            "ground_truth": inst.ground_truth,
            "system_prompt": system,
            "user_prompt": user,
            "base_model": {"output": out_base, "length": len(out_base)},
            "trained_model": {
                "sampler": SAMPLERS[task],
                "output": out_trained,
                "length": len(out_trained),
            },
        }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUT_PATH}")

    for task, r in results.items():
        print(f"\n{'='*70}")
        print(f"  TASK: {task.upper()} ({r['instance_id']})")
        print(f"  Cutoff: {r['cutoff_date']} | GT: {r['ground_truth']}")
        print(f"{'='*70}")
        print(f"\n  --- [A] BASE ({r['base_model']['length']} chars) ---")
        print(r["base_model"]["output"][:3000])
        if r["base_model"]["length"] > 3000:
            print("  ... [truncated]")
        print(f"\n  --- [B] TRAINED ({r['trained_model']['length']} chars) ---")
        print(r["trained_model"]["output"][:3000])
        if r["trained_model"]["length"] > 3000:
            print("  ... [truncated]")


if __name__ == "__main__":
    main()
