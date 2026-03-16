"""TimeSPEC baseline: multi-phase generation with claim supervision and aggregation.

Phases:
  0. Pre-retrieval  -- reuse RAG pipeline (Perplexity Search API, date-filtered)
  1. Generator      -- Qwen + context (identical to RAG)
  2. Supervisor     -- claim extraction + taxonomy (Gemini Flash) + search verification
  3. Regenerator    -- Qwen re-generation avoiding leaked claims (conditional)
  4. Resupervisor   -- re-verify regenerated output (conditional)
  5. Aggregator     -- Qwen closed-world generation from validated claims only

Usage:
    uv run python -m quality_gated_experiment.timespec_evaluate \
        --config configs/eval_timespec_qwen3_30b.yaml \
        --eval_tasks stock \
        --output_dir runtime/results/qwen3_30b/eval_stock_timespec
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import get_perplexity_api_key, load_env, require_api_keys
from .data import load_eval_instances
from .evaluate import (
    EvalConfig,
    InstanceResult,
    _SearchCache,
    _aggregate_task,
    _append_jsonl,
    _build_summary,
    _evaluate_instance,
    _make_generator,
    _print_summary,
    _result_to_dict,
    _save_json,
    load_eval_config,
)
from .llm import OpenRouterClient
from .prompts import (
    CLAIM_EXTRACTION_PROMPT,
    TIMESPEC_CLAIM_VERIFY_PROMPT,
    build_generation_prompt,
    build_rag_context_block,
    build_rag_generation_prompt,
    build_rationale_text,
    build_timespec_aggregator_prompt,
    build_timespec_regeneration_prompt,
    event_description,
    get_rag_query_range,
    task_description,
)
from .rag_evaluate import (
    PerplexitySearchClient,
    RAGConfig,
    _estimate_tokens,
    _parse_query_json,
    generate_search_queries,
    retrieve_and_build_context,
)
from .reward_pipeline import _parse_json_array_raw
from .schemas import parse_completion
from .types import TaskInstance

logger = logging.getLogger(__name__)

_MAX_TOKENS_JUDGE = 4096
_VERIFY_SEARCH_MAX_RESULTS = 3


# ======================================================================
# Config
# ======================================================================

@dataclass
class TimeSPECConfig(RAGConfig):
    timespec_max_regen_cycles: int = 1


def load_timespec_config(path: str) -> TimeSPECConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TimeSPECConfig(
        **{k: v for k, v in raw.items() if k in TimeSPECConfig.__dataclass_fields__}
    )


# ======================================================================
# Phase 2: Supervisor -- claim extraction + verification
# ======================================================================

def extract_claims_with_taxonomy(
    rationale: str,
    instance: TaskInstance,
    client: OpenRouterClient,
    judge_model: str,
) -> list[dict[str, Any]]:
    """Extract claims from rationale using CLAIM_EXTRACTION_PROMPT (1 Gemini call).

    Returns list of dicts with keys: claim_id, text, original_span,
    temporal_ref, category, category_reason.
    """
    prompt = CLAIM_EXTRACTION_PROMPT.format(
        task_description=task_description(instance.task),
        event_description=event_description(instance),
        cutoff_date=instance.cutoff_date,
        rationale=rationale,
    )
    text, _ = client.call_text(
        judge_model, prompt, temperature=0.0, max_tokens=_MAX_TOKENS_JUDGE,
    )
    # Parse JSON response
    text = text.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        logger.warning("Failed to parse claim extraction JSON")
        return []
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from claim extraction")
        return []
    claims = parsed.get("claims", [])
    if not isinstance(claims, list):
        return []
    valid: list[dict[str, Any]] = []
    for c in claims:
        if isinstance(c, dict) and "text" in c and "category" in c:
            valid.append(c)
    return valid


def _classify_claims(
    claims: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split claims into (needs_search, auto_leaked, auto_clean)."""
    needs_search: list[dict[str, Any]] = []
    auto_leaked: list[dict[str, Any]] = []
    auto_clean: list[dict[str, Any]] = []
    for c in claims:
        cat = str(c.get("category", "")).upper()
        if cat in ("A4", "A5"):
            auto_leaked.append(c)
        elif cat in ("B1", "B2"):
            auto_clean.append(c)
        elif cat in ("A1", "A2", "A3"):
            needs_search.append(c)
        else:
            auto_clean.append(c)
    return needs_search, auto_leaked, auto_clean


def verify_claims_via_search(
    claims_to_verify: list[dict[str, Any]],
    cutoff_date: str,
    search_client: PerplexitySearchClient,
    client: OpenRouterClient,
    judge_model: str,
    claim_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Verify A1-A3 claims via batched Perplexity Search + single Gemini judgment.

    Searches are batched in groups of 5 by PerplexitySearchClient (O4).
    All (claim, search_result) pairs are judged in one Gemini call (O5).
    Uses claim_cache to skip already-verified claims (O7).
    """
    if not claims_to_verify:
        return {}

    results: dict[int, dict[str, Any]] = {}
    to_search: list[tuple[int, dict[str, Any]]] = []

    for i, claim in enumerate(claims_to_verify):
        key = claim["text"].strip().lower()
        if claim_cache and key in claim_cache:
            results[i] = claim_cache[key]
            continue
        to_search.append((i, claim))

    if not to_search:
        return results

    # O3: construct search queries directly from claim text
    queries = [
        f"When was this first publicly reported: {c['text']}"
        for _, c in to_search
    ]

    # O4: batch search (groups of 5 internally), O8: unfiltered (no date filter)
    # Search per-claim for targeted results. The batch mechanism inside
    # PerplexitySearchClient.search() handles grouping, but the API returns
    # a single flat deduplicated list, losing per-query attribution.
    # So we search one claim at a time but rely on caching for efficiency.
    per_claim_snippets: dict[int, str] = {}
    for list_idx, (orig_idx, claim) in enumerate(to_search):
        query = queries[list_idx]
        try:
            per_results = search_client.search(
                [query], cutoff_date=None, max_results=_VERIFY_SEARCH_MAX_RESULTS,
            )
        except RuntimeError:
            per_results = []
        snippet_parts = []
        for r in per_results:
            title = r.get("title", "")
            snippet = r.get("snippet") or r.get("content") or ""
            snippet_parts.append(f"{title}: {snippet}")
        per_claim_snippets[orig_idx] = "\n".join(snippet_parts) if snippet_parts else "No results found."

    # O5: batch verification judgment -- all claims in one Gemini call
    items_json = json.dumps([
        {
            "index": orig_idx,
            "claim": claim["text"],
            "search_result": per_claim_snippets.get(orig_idx, "No results found."),
        }
        for orig_idx, claim in to_search
    ], indent=2, ensure_ascii=False)

    verify_text, _ = client.call_text(
        judge_model,
        TIMESPEC_CLAIM_VERIFY_PROMPT.format(cutoff=cutoff_date, items_json=items_json),
        temperature=0.0, max_tokens=_MAX_TOKENS_JUDGE,
    )

    for entry in _parse_json_array_raw(verify_text):
        if isinstance(entry, dict) and "index" in entry:
            idx = entry["index"]
            verdict = {
                "verdict": entry.get("verdict", "not_leaked"),
                "event_date": entry.get("event_date"),
                "method": "timespec_search",
            }
            results[idx] = verdict
            claim_for_idx = next((c for i, c in to_search if i == idx), None)
            if claim_for_idx and claim_cache is not None:
                claim_cache[claim_for_idx["text"].strip().lower()] = verdict

    for orig_idx, _ in to_search:
        if orig_idx not in results:
            results[orig_idx] = {
                "verdict": "not_leaked",
                "event_date": None,
                "method": "timespec_search_default",
            }

    return results


def run_supervisor(
    completion_text: str,
    instance: TaskInstance,
    search_client: PerplexitySearchClient,
    client: OpenRouterClient,
    judge_model: str,
    claim_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the full supervisor pipeline on one completion.

    Returns (all_claims_with_verdicts, leaked_claims, validated_claims).
    Each claim dict has 'text', 'category', 'verdict' fields.
    """
    parsed = parse_completion(instance.task, completion_text)
    if parsed is None:
        return [], [], []

    rationale = build_rationale_text(parsed)
    claims = extract_claims_with_taxonomy(rationale, instance, client, judge_model)

    if not claims:
        return [], [], []

    needs_search, auto_leaked, auto_clean = _classify_claims(claims)

    # Mark auto-leaked
    for c in auto_leaked:
        c["verdict"] = "leaked"
        c["event_date"] = None
        c["method"] = "taxonomy_auto"

    # Mark auto-clean
    for c in auto_clean:
        c["verdict"] = "not_leaked"
        c["event_date"] = None
        c["method"] = "taxonomy_auto"

    # Verify A1-A3 claims
    search_verdicts = verify_claims_via_search(
        needs_search, instance.cutoff_date, search_client, client, judge_model,
        claim_cache=claim_cache,
    )
    for i, c in enumerate(needs_search):
        v = search_verdicts.get(i, {"verdict": "not_leaked", "event_date": None, "method": "default"})
        c["verdict"] = v["verdict"]
        c["event_date"] = v.get("event_date")
        c["method"] = v.get("method", "timespec_search")

    all_claims = auto_leaked + needs_search + auto_clean
    leaked = [c for c in all_claims if c.get("verdict") == "leaked"]
    validated = [c for c in all_claims if c.get("verdict") != "leaked"]

    return all_claims, leaked, validated


# ======================================================================
# Main pipeline
# ======================================================================

def run_timespec_evaluation(
    cfg: TimeSPECConfig,
    project_root: Path,
    *,
    eval_tasks_str: str,
    output_dir: str,
    max_instances: int | None = None,
) -> None:
    """Run the full TimeSPEC evaluation pipeline."""
    out_dir = project_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pplx_key = get_perplexity_api_key()
    if not pplx_key:
        logger.error("PERPLEXITY_API_KEY not set. Skipping TimeSPEC evaluation.")
        print("  WARNING: PERPLEXITY_API_KEY not set, skipping TimeSPEC evaluation.", flush=True)
        return

    _save_json(out_dir / "config.json", {
        "model_name": cfg.model_name,
        "lora_rank": cfg.lora_rank,
        "variant": "timespec",
        "checkpoint_path": "",
        "temperature": cfg.temperature,
        "eval_tasks": eval_tasks_str,
        "judge_model": cfg.judge_model,
        "search_model": cfg.search_model,
        "rag_context_budget": cfg.rag_context_budget,
        "rag_max_results_per_query": cfg.rag_max_results_per_query,
        "timespec_max_regen_cycles": cfg.timespec_max_regen_cycles,
    })

    data_root = project_root / cfg.data_root
    tasks = [t.strip() for t in eval_tasks_str.split(",") if t.strip()]
    instances = load_eval_instances(data_root, tasks)

    if max_instances is not None and max_instances > 0:
        instances = instances[:max_instances]

    _, tinker_key = require_api_keys()
    import os
    os.environ["TINKER_API_KEY"] = tinker_key
    router_key = os.getenv("OPENROUTER_API_KEY", "")

    or_client = OpenRouterClient(router_key)
    search_client = PerplexitySearchClient(pplx_key)
    search_cache = _SearchCache(project_root / cfg.search_cache_path)

    # Print banner
    print(f"\n{'=' * 60}", flush=True)
    print(f"  Evaluation -- timespec (TimeSPEC baseline)", flush=True)
    print(f"  Model: {cfg.model_name}", flush=True)
    print(f"  Checkpoint: base (no checkpoint)", flush=True)
    print(f"  Tasks: {eval_tasks_str} | Instances: {len(instances)}", flush=True)
    print(f"  Judge: {cfg.judge_model} | Search: {cfg.search_model}", flush=True)
    print(f"  Retrieval: Perplexity Search API (date-filtered)", flush=True)
    print(f"  Context budget: {cfg.rag_context_budget} tokens", flush=True)
    print(f"  Max regen cycles: {cfg.timespec_max_regen_cycles}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    # ------------------------------------------------------------------
    # Phase 0: Retrieval (identical to RAG)
    # ------------------------------------------------------------------
    retrieval_log_path = out_dir / "retrieval_log.jsonl"
    context_map: dict[str, str] = {}
    retrieval_t0 = time.time()

    for idx, inst in enumerate(instances):
        t0 = time.time()
        queries = generate_search_queries(or_client, cfg.judge_model, inst)
        context_block, raw_results, num_raw, num_dedup = retrieve_and_build_context(
            search_client, queries, inst.cutoff_date,
            budget=cfg.rag_context_budget,
            max_results_per_query=cfg.rag_max_results_per_query,
        )
        context_tokens = _estimate_tokens(context_block)
        elapsed = time.time() - t0

        context_map[inst.instance_id] = context_block

        _append_jsonl(retrieval_log_path, {
            "instance_id": inst.instance_id,
            "task": inst.task,
            "cutoff_date": inst.cutoff_date,
            "num_queries": len(queries),
            "queries": queries,
            "num_results_raw": num_raw,
            "num_results_dedup": num_dedup,
            "context_tokens": context_tokens,
            "search_time_s": round(elapsed, 2),
        })

        done = idx + 1
        if done % 10 == 0 or done == len(instances):
            logger.info(
                "[Retrieval %d/%d] %s | queries=%d | results=%d | context=%d tok | %.1fs",
                done, len(instances), inst.task,
                len(queries), num_dedup, context_tokens, elapsed,
            )

    retrieval_total = time.time() - retrieval_t0
    logger.info("  Retrieval complete: %d instances | %.1fs total", len(instances), retrieval_total)

    # ------------------------------------------------------------------
    # Phase 1: Generator (identical to RAG)
    # ------------------------------------------------------------------
    def _rag_prompt_builder(inst: TaskInstance) -> tuple[str, str]:
        ctx = context_map.get(inst.instance_id, "")
        return build_rag_generation_prompt(inst, ctx)

    _, generate_batch = _make_generator(
        cfg, checkpoint_path="", prompt_builder=_rag_prompt_builder,
    )

    gen_batch_size = 8
    logger.info("  Phase 1: Generating initial completions (batch=%d) ...", gen_batch_size)
    initial_texts = generate_batch(instances, batch_size=gen_batch_size)

    # ------------------------------------------------------------------
    # Phases 2-5: Supervisor -> [Regenerator -> Resupervisor] -> Aggregator
    # ------------------------------------------------------------------
    # Build a single Tinker generator with a mutable prompt builder to avoid
    # creating a new Tinker client for each instance/phase.
    _dynamic_prompt: dict[str, Any] = {"fn": build_rag_generation_prompt}

    def _dynamic_builder(inst_inner: TaskInstance) -> tuple[str, str]:
        return _dynamic_prompt["fn"](inst_inner)

    dynamic_one, _ = _make_generator(
        cfg, checkpoint_path="", prompt_builder=_dynamic_builder,
    )

    supervisor_log_path = out_dir / "supervisor_log.jsonl"
    claim_cache: dict[str, dict[str, Any]] = {}

    final_texts: list[str] = []

    for idx, inst in enumerate(instances):
        t0 = time.time()
        completion = initial_texts[idx]
        context_block = context_map.get(inst.instance_id, "")

        # Phase 2: Supervisor
        all_claims, leaked, validated = run_supervisor(
            completion, inst, search_client, or_client, cfg.judge_model,
            claim_cache=claim_cache,
        )

        regenerated = False
        regen_claims_info: list[dict[str, Any]] = []
        final_validated = validated
        final_leaked = leaked

        # Phase 3-4: Regeneration + Resupervision (conditional)
        if leaked and cfg.timespec_max_regen_cycles > 0:
            leaked_snapshot = list(leaked)
            ctx_snapshot = context_block
            _dynamic_prompt["fn"] = lambda inst_inner, _ctx=ctx_snapshot, _lk=leaked_snapshot: (
                build_timespec_regeneration_prompt(inst_inner, _ctx, _lk)
            )
            completion = dynamic_one(inst)
            regenerated = True

            # Phase 4: Resupervisor
            regen_claims_info_raw, regen_leaked, regen_validated = run_supervisor(
                completion, inst, search_client, or_client, cfg.judge_model,
                claim_cache=claim_cache,
            )
            regen_claims_info = regen_claims_info_raw
            final_validated = regen_validated
            final_leaked = regen_leaked

        # Phase 5: Aggregator (always runs)
        if final_validated:
            validated_snapshot = list(final_validated)
            _dynamic_prompt["fn"] = lambda inst_inner, _v=validated_snapshot: (
                build_timespec_aggregator_prompt(inst_inner, _v)
            )
            final_completion = dynamic_one(inst)
        else:
            final_completion = completion

        final_texts.append(final_completion)
        elapsed = time.time() - t0

        # Log supervisor details
        _append_jsonl(supervisor_log_path, {
            "instance_id": inst.instance_id,
            "task": inst.task,
            "cutoff_date": inst.cutoff_date,
            "phase2_claims": [
                {"text": c.get("text", ""), "category": c.get("category", ""),
                 "verdict": c.get("verdict", ""), "method": c.get("method", "")}
                for c in all_claims
            ],
            "phase2_leaked_count": len(leaked),
            "phase2_validated_count": len(validated),
            "regenerated": regenerated,
            "phase4_claims": [
                {"text": c.get("text", ""), "category": c.get("category", ""),
                 "verdict": c.get("verdict", ""), "method": c.get("method", "")}
                for c in regen_claims_info
            ] if regenerated else [],
            "final_leaked_count": len(final_leaked),
            "final_validated_count": len(final_validated),
            "time_s": round(elapsed, 2),
        })

        done = idx + 1
        n_a = sum(1 for c in all_claims if c.get("category", "").upper().startswith("A"))
        n_b = len(all_claims) - n_a
        if done % 5 == 0 or done == len(instances):
            logger.info(
                "[TimeSPEC %d/%d] %s | claims=%d (A:%d B:%d) | leaked=%d | regen=%s | %.1fs",
                done, len(instances), inst.task,
                len(all_claims), n_a, n_b, len(final_leaked),
                "Y" if regenerated else "N", elapsed,
            )

    # ------------------------------------------------------------------
    # Scoring (identical pipeline as all baselines)
    # ------------------------------------------------------------------
    completions_log = [
        {"instance_id": inst.instance_id, "task": inst.task, "completion": text}
        for inst, text in zip(instances, final_texts)
    ]

    workers = max(1, cfg.concurrent_workers)
    all_results: list[InstanceResult] = []
    step_log_path = out_dir / "eval_step_log.jsonl"
    _jsonl_lock = threading.Lock()

    def _eval_one(eval_idx: int) -> InstanceResult:
        t0 = time.time()
        inst = instances[eval_idx]
        completion = completions_log[eval_idx]["completion"]
        result, detail = _evaluate_instance(
            inst, completion, cfg=cfg, client=or_client,
            search_cache=search_cache,
        )
        result.time_s = round(time.time() - t0, 2)
        with _jsonl_lock:
            _append_jsonl(step_log_path, {
                "instance_id": inst.instance_id,
                "task": inst.task,
                "cutoff_date": inst.cutoff_date,
                "completion": completion,
                "claims": detail.get("claims", []),
                "metrics": {
                    "parse_success": result.parse_success,
                    "r_format": result.r_format,
                    "leakage_rate": result.leakage_rate,
                    "coverage": result.coverage,
                    "r_performance": result.r_performance,
                    "total_claims": result.total_claims,
                    "leaked_claims": result.leaked_claims,
                    "sonar_verified": result.sonar_verified,
                },
            })
        return result

    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("  Scoring %d completions ...", len(instances))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_one, i): i for i in range(len(instances))}
        completed_count = 0
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(instances):
                done_task = [r for r in all_results if r.task == result.task]
                avg_olr = np.mean([r.leakage_rate for r in done_task]) if done_task else 0
                avg_perf = np.mean([r.r_performance for r in done_task]) if done_task else 0
                logger.info(
                    "[Scoring %d/%d] %s | OLR=%.3f | Perf=%.3f | %.1fs",
                    completed_count, len(instances), result.task,
                    avg_olr, avg_perf, result.time_s,
                )

    id_order = {inst.instance_id: i for i, inst in enumerate(instances)}
    all_results.sort(key=lambda r: id_order.get(r.instance_id, 0))

    search_cache.save()
    _save_json(out_dir / "completions.json", completions_log)
    _save_json(out_dir / "instance_results.json", [_result_to_dict(r) for r in all_results])
    summary = _build_summary(all_results, cfg, "timespec", "none")
    _save_json(out_dir / "summary.json", summary)

    print(f"\n  Results saved to {out_dir / 'summary.json'}", flush=True)
    _print_summary(summary)


# ======================================================================
# CLI entrypoint
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TimeSPEC evaluation baseline.")
    parser.add_argument("--config", required=True, help="Path to TimeSPEC eval YAML config.")
    parser.add_argument("--eval_tasks", required=True,
                        help="Comma-separated task(s) to evaluate on.")
    parser.add_argument("--output_dir", required=True, help="Output directory for results.")
    parser.add_argument("--max_instances", type=int, default=None,
                        help="Limit to first N instances per run (for smoke testing).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    project_root = Path(__file__).resolve().parents[2]
    load_env(project_root)
    cfg = load_timespec_config(args.config)
    run_timespec_evaluation(
        cfg, project_root,
        eval_tasks_str=args.eval_tasks,
        output_dir=args.output_dir,
        max_instances=args.max_instances,
    )


if __name__ == "__main__":
    main()
