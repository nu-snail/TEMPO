"""Sample 8 completions per Tinker-supported model on one salary instance.

Evaluates parse success rate to identify models that produce structured JSON
without the uncontrolled thinking/reasoning problem.

Usage (from CoverageGatedExperiment/):
    uv run python scripts/compare_models.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

os.environ["HF_HOME"] = "/shares/bcs516/pcv1327/hf_cache"

from dotenv import load_dotenv

load_dotenv(PROJECT / ".env")

import tinker
from tinker import types as tinker_types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cov_gated_experiment.data import load_train_instances
from cov_gated_experiment.prompts import build_generation_prompt
from cov_gated_experiment.schemas import parse_completion

NUM_SAMPLES = 8
TEMPERATURE = 0.6
MAX_TOKENS = 4096

# Models grouped by renderer (shared tokenizer). Skip Base models.
RENDERER_GROUPS: dict[str, list[tuple[str, str]]] = {
    "qwen3_instruct": [
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", "instruction"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "instruction"),
        ("Qwen/Qwen3-4B-Instruct-2507", "instruction"),
    ],
    "qwen3": [
        ("Qwen/Qwen3-30B-A3B", "hybrid"),
        ("Qwen/Qwen3-32B", "hybrid"),
        ("Qwen/Qwen3-8B", "hybrid"),
    ],
    "llama3": [
        ("meta-llama/Llama-3.3-70B-Instruct", "instruction"),
        ("meta-llama/Llama-3.1-8B-Instruct", "instruction"),
    ],
    "gpt_oss_no_sysprompt": [
        ("openai/gpt-oss-120b", "reasoning"),
        ("openai/gpt-oss-20b", "reasoning"),
    ],
    "deepseekv3": [
        ("deepseek-ai/DeepSeek-V3.1", "hybrid"),
    ],
    "kimi_k2": [
        ("moonshotai/Kimi-K2-Thinking", "reasoning"),
    ],
}

# Representative tokenizer model for each renderer group
TOKENIZER_MODEL: dict[str, str] = {
    "qwen3_instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3": "Qwen/Qwen3-30B-A3B",
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "gpt_oss_no_sysprompt": "openai/gpt-oss-20b",
    "deepseekv3": "deepseek-ai/DeepSeek-V3.1",
    "kimi_k2": "moonshotai/Kimi-K2-Thinking",
}

OUTPUT_PATH = PROJECT / "runtime" / "results" / "model_comparison.jsonl"


def _sample_one_model(
    model_name: str,
    model_type: str,
    renderer_obj: object,
    renderer_name: str,
    sampling_params: tinker_types.SamplingParams,
    model_input: object,
    service_client: tinker.ServiceClient,
) -> dict:
    """Sample NUM_SAMPLES completions and evaluate parse success."""
    print(f"\n{'='*70}")
    print(f"Model: {model_name} ({model_type}) | renderer={renderer_name}")
    print(f"{'='*70}")

    try:
        sampling_client = service_client.create_sampling_client(base_model=model_name)
    except Exception as e:
        print(f"  SKIP: {e}")
        return {"model": model_name, "type": model_type, "error": str(e),
                "n_sampled": 0, "n_parsed": 0, "parse_rate": 0.0}

    t0 = time.time()
    try:
        result = sampling_client.sample(
            prompt=model_input, num_samples=NUM_SAMPLES,
            sampling_params=sampling_params,
        ).result()
    except Exception as e:
        print(f"  SKIP: sampling failed in {time.time()-t0:.1f}s: {e}")
        return {"model": model_name, "type": model_type, "error": str(e),
                "n_sampled": 0, "n_parsed": 0, "parse_rate": 0.0}
    elapsed = time.time() - t0

    completions = []
    for seq in result.sequences:
        parsed_msg, _ = renderer_obj.parse_response(seq.tokens)
        text = renderers.get_text_content(parsed_msg)
        completions.append(text)

    n_parsed = 0
    details = []
    for i, text in enumerate(completions):
        parsed = parse_completion("salary", text)
        is_ok = parsed is not None
        if is_ok:
            n_parsed += 1

        has_think = "<think>" in text.lower() or "we need to" in text[:100].lower()
        details.append({
            "idx": i, "parse_success": is_ok, "text_len": len(text),
            "has_reasoning_preamble": has_think,
            "starts_with_json": text.lstrip().startswith("{"),
            "preview": text[:300],
        })

        status = "OK" if is_ok else "FAIL"
        flag = " [REASONING_PREAMBLE]" if has_think else ""
        print(f"  [{i}] {status} | len={len(text):,}{flag}")
        if not is_ok:
            print(f"       {text[:120]}...")

    rate = n_parsed / len(completions) if completions else 0.0
    print(f"\n  => {n_parsed}/{len(completions)} parsed ({rate:.0%}) in {elapsed:.1f}s")

    return {
        "model": model_name, "type": model_type, "renderer": renderer_name,
        "n_sampled": len(completions), "n_parsed": n_parsed,
        "parse_rate": rate, "elapsed_s": round(elapsed, 1),
        "completions": details,
    }


def main():
    data_root = PROJECT / "runtime" / "data"
    instances = load_train_instances(data_root, ["salary"])
    inst = instances[10]

    print(f"Instance: {inst.instance_id}")
    print(f"Player: {inst.raw.get('Player', '')} | Year: {inst.raw.get('Year', '')}")
    print(f"Cutoff: {inst.cutoff_date}")
    print(f"Config: {NUM_SAMPLES} samples | temp={TEMPERATURE} | max_tokens={MAX_TOKENS}")

    service_client = tinker.ServiceClient()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    system, user = build_generation_prompt(inst)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    results = []

    for renderer_name, model_list in RENDERER_GROUPS.items():
        tok_model = TOKENIZER_MODEL[renderer_name]
        print(f"\n--- Loading tokenizer for renderer '{renderer_name}' from {tok_model} ---")
        tokenizer = get_tokenizer(tok_model)
        renderer_obj = renderers.get_renderer(renderer_name, tokenizer)

        sampling_params = tinker_types.SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=renderer_obj.get_stop_sequences(),
        )
        model_input = renderer_obj.build_generation_prompt(messages)

        for model_name, model_type in model_list:
            r = _sample_one_model(
                model_name, model_type, renderer_obj, renderer_name,
                sampling_params, model_input, service_client,
            )
            results.append(r)
            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<45} {'Type':<12} {'Parsed':>8} {'Rate':>6} {'Time':>6}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: (-x["parse_rate"], x.get("elapsed_s", 999))):
        print(
            f"{r['model']:<45} {r['type']:<12} "
            f"{r['n_parsed']:>3}/{r['n_sampled']:<3}  "
            f"{r['parse_rate']:>5.0%} "
            f"{r.get('elapsed_s', 0):>5.1f}s"
        )


if __name__ == "__main__":
    main()
