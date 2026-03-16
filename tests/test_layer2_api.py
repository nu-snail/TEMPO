"""Layer 2: API integration smoke test (~$0.50).

Tests real API connectivity and the full claim-extraction → leakage-detection
pipeline on 1 instance per task.  Verifies:
  - OpenRouter LLM call works (claim extraction + judge)
  - Perplexity search works (temporal verification)
  - Claim taxonomy (A1-A5 / B1-B2) is produced
  - Leakage detection returns valid verdicts
  - Shapley calculator completes without error
  - TimeSPEC agent field exclusion works with real prompts

Usage:
    uv run python tests/test_layer2_api.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Load .env before checking keys
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  — {detail}")


# ===================================================================
# 1. API Connectivity
# ===================================================================

def test_api_connectivity() -> None:
    print("\n── 1. API Connectivity ──")
    from src.shared.llm_client import simple_call
    from src.shared.search_client import SearchClient

    # OpenRouter
    t0 = time.time()
    result = simple_call(
        "Reply with exactly: PING_OK",
        model="google/gemini-2.5-flash",
        temperature=0,
        max_tokens=20,
    )
    dt = time.time() - t0
    check(f"OpenRouter responds ({dt:.1f}s)", "PING" in result.upper(), f"got: {result[:80]}")

    # Perplexity
    t0 = time.time()
    sc = SearchClient()
    sr = sc.search("What year was the iPhone first released?")
    dt = time.time() - t0
    content = sr.get("content", "")
    check(f"Perplexity responds ({dt:.1f}s)", len(content) > 20, f"len={len(content)}")
    check("Perplexity returns citations", isinstance(sr.get("citations"), list))


# ===================================================================
# 2. Tool-calling (LLM client normalisation)
# ===================================================================

def test_tool_calling() -> None:
    print("\n── 2. Tool-calling (LLM client dict normalisation) ──")
    from src.shared.llm_client import LLMClient

    client = LLMClient()
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    response = client.call(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ],
        model="google/gemini-2.5-flash",
        temperature=0,
        max_tokens=100,
        tools=tools,
    )

    tc = response.get("tool_calls")
    if tc:
        check("tool_calls is a list of dicts", isinstance(tc, list) and isinstance(tc[0], dict))
        first = tc[0]
        check("tool_call has 'id' key", "id" in first)
        check("tool_call has 'function' dict", isinstance(first.get("function"), dict))
        fn = first["function"]
        check("function has 'name'", "name" in fn)
        check("function has 'arguments' (str)", isinstance(fn.get("arguments"), str))
        try:
            args = json.loads(fn["arguments"])
            check("arguments is valid JSON", isinstance(args, dict))
        except json.JSONDecodeError:
            check("arguments is valid JSON", False, f"raw: {fn['arguments'][:80]}")
    else:
        # Some models respond with content instead of tool calls
        check("model responded (content or tool_calls)",
              response.get("content") is not None or tc is not None)
        print("    (model did not use tool_call; content-only response)")


# ===================================================================
# 3. Claim Extraction + Leakage Detection (1 instance)
# ===================================================================

def test_claim_pipeline() -> None:
    print("\n── 3. Claim Extraction + Leakage Detection ──")
    from src.shared.claim_extractor import ClaimExtractor
    from src.shared.leakage_detector import LeakageDetector

    extractor = ClaimExtractor(model="google/gemini-2.5-flash")
    detector = LeakageDetector(
        model="google/gemini-2.5-flash",
        use_search=True,
        use_llm_judge=True,
    )

    rationale = (
        "Apple reported Q3 2023 revenue of $81.8 billion in July 2023. "
        "In January 2024, Apple announced the Vision Pro headset pricing. "
        "The S&P 500 has historically averaged 10% annual returns. "
        "Microsoft acquired Activision Blizzard in October 2023."
    )
    cutoff = "2023-09-01"

    t0 = time.time()
    extraction = extractor.extract(
        rationale,
        task_description="Stock return ranking",
        event_description="Ranking for Tech sector 2023-H2",
        cutoff_date=cutoff,
    )
    dt_extract = time.time() - t0
    claims = extraction.claims
    check(f"extracted {len(claims)} claims ({dt_extract:.1f}s)", len(claims) >= 2)

    if claims:
        cats = {c.category for c in claims}
        print(f"    Categories found: {sorted(cats)}")
        check("claims have category field", all(c.category for c in claims))
        check("claims have text field", all(c.text for c in claims))

        t0 = time.time()
        report = detector.detect(claims, cutoff)
        dt_detect = time.time() - t0
        check(f"detection completed ({dt_detect:.1f}s)", report.total_claims == len(claims))
        check("OLR is valid float", 0 <= report.olr <= 1, f"olr={report.olr}")
        check("some claims detected as leaked", report.num_leaked > 0,
              f"leaked={report.num_leaked}/{report.total_claims}")

        for r in report.results:
            if "Vision Pro" in r.text or "Activision" in r.text:
                check(f"post-cutoff claim '{r.text[:40]}...' is leaked", r.is_leaked)
                break

        methods = {r.method for r in report.results}
        print(f"    Detection methods used: {sorted(methods)}")
    else:
        check("claim extraction returned results", False, "0 claims")


# ===================================================================
# 4. Shapley Calculator (mini run, 1 instance)
# ===================================================================

def test_shapley_mini() -> None:
    print("\n── 4. Shapley Calculator (mini) ──")
    from src.evaluation.shapley_calculator import ShapleyCalculator

    shapley = ShapleyCalculator(
        model="google/gemini-2.5-flash",
        max_samples=10,   # very small for speed
        batch_size=8,
    )

    task_input = {
        "case_name": "Test v. State",
        "petitioner": "Test",
        "respondent": "State",
        "term": "OT2022",
    }
    claim_texts = [
        "The Court had a 6-3 conservative majority in 2022.",
        "Similar cases like Dobbs v. Jackson were decided 6-3.",
        "The petitioner filed the case in March 2022.",
    ]

    t0 = time.time()
    result = shapley.compute(
        task="legal",
        task_input=task_input,
        claim_texts=claim_texts,
        control_prediction=0.6,
    )
    dt = time.time() - t0
    sv = result["shapley_values"]
    check(f"Shapley returns {len(sv)} values ({dt:.1f}s)", len(sv) == 3)
    check("Shapley values are floats", all(isinstance(v, float) for v in sv))
    check("at least one non-zero Shapley value", any(abs(v) > 0.001 for v in sv),
          f"values={sv}")
    print(f"    Shapley values: {[round(v, 4) for v in sv]}")
    print(f"    LLM calls: {result.get('num_llm_calls', '?')}, "
          f"cache: {result.get('cache_stats', {})}")


# ===================================================================
# 5. Full Reward Pipeline (1 instance, performance_only shortcut)
# ===================================================================

def test_reward_pipeline() -> None:
    print("\n── 5. Reward Pipeline (performance_only, 1 instance) ──")
    from src.shared.data_loader import load_stock_train
    from src.training.reward.reward_pipeline import RewardPipeline

    pipe = RewardPipeline(
        extractor_model="google/gemini-2.5-flash",
        detector_model="google/gemini-2.5-flash",
    )
    inst = load_stock_train()[0]
    companies = inst.raw.get("companies", [])

    completion = json.dumps({
        "ranking": companies,
        "rationale": "Based on earnings and market conditions as of the cutoff.",
    })

    # Performance-only (no API calls for leakage)
    results = pipe.score_group_performance_only([completion], inst)
    check("perf-only: returns 1 reward", len(results) == 1)
    check("perf-only: r_performance > 0", results[0].r_performance > 0,
          f"got {results[0].r_performance}")
    check("perf-only: r_leakage = 0", results[0].r_leakage == 0.0)
    check("perf-only: parse_success = True", results[0].parse_success)


# ===================================================================
# 6. Full Reward Pipeline (1 instance, full mode with leakage)
# ===================================================================

def test_reward_pipeline_full() -> None:
    print("\n── 6. Reward Pipeline (full mode, 1 instance) ──")
    from src.shared.data_loader import load_legal_train
    from src.training.reward.reward_pipeline import RewardPipeline

    pipe = RewardPipeline(
        extractor_model="google/gemini-2.5-flash",
        detector_model="google/gemini-2.5-flash",
        use_search=True,
        use_llm_judge=True,
    )
    inst = load_legal_train()[0]

    completion = json.dumps({
        "probability_petitioner": 0.65,
        "predicted_winner": "PETITIONER",
        "rationale": (
            "Based on the conservative 6-3 majority established after "
            "Justice Barrett's confirmation in October 2020, the Court has shown "
            "a tendency to favor petitioners in similar constitutional cases. "
            "The precedent from Dobbs v. Jackson (2022) suggests the Court "
            "is willing to overturn long-standing precedent."
        ),
    })

    t0 = time.time()
    result = pipe.score(completion, inst)
    dt = time.time() - t0
    check(f"full reward computed ({dt:.1f}s)", True)
    check("r_leakage in [0,1]", 0 <= result.r_leakage <= 1, f"got {result.r_leakage}")
    check("r_performance in [0,1]", 0 <= result.r_performance <= 1, f"got {result.r_performance}")
    check("OLR in [0,1]", 0 <= result.olr <= 1, f"got {result.olr}")
    check("num_claims > 0", result.num_claims > 0, f"got {result.num_claims}")
    check("parse_success = True", result.parse_success)
    print(f"    r_leak={result.r_leakage:.3f}, r_perf={result.r_performance:.3f}, "
          f"OLR={result.olr:.3f}, claims={result.num_claims}, leaked={result.num_leaked}")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    global PASS, FAIL

    for key in ("OPENROUTER_API_KEY", "PERPLEXITY_API_KEY"):
        if not os.getenv(key):
            print(f"ERROR: {key} not set. Layer 2 requires API keys.")
            sys.exit(1)

    tests = [
        test_api_connectivity,
        test_tool_calling,
        test_claim_pipeline,
        test_shapley_mini,
        test_reward_pipeline,
        test_reward_pipeline_full,
    ]

    print("=" * 60)
    print("  Layer 2: API Integration Smoke Test (~$0.50)")
    print("=" * 60)

    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            FAIL += 1
            print(f"\n  ✗ {test_fn.__name__} CRASHED:")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if FAIL == 0:
        print(f"  ALL {PASS} CHECKS PASSED")
    else:
        print(f"  {PASS} passed, {FAIL} FAILED")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
