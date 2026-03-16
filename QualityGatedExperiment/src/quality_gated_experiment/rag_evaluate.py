"""RAG evaluation baseline using Perplexity Search API with hard date filtering.

Generates adaptive search queries via Gemini Flash, retrieves temporally-
filtered evidence from Perplexity, augments the user prompt with retrieved
context, and scores using the identical pipeline as evaluate.py.

Usage:
    uv run python -m quality_gated_experiment.rag_evaluate \
        --config configs/eval_rag_qwen3_30b.yaml \
        --eval_tasks stock \
        --output_dir runtime/results/qwen3_30b/eval_stock_rag
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
import requests
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
    QUERY_GENERATION_PROMPT,
    build_generation_prompt,
    build_rag_context_block,
    build_rag_generation_prompt,
    get_rag_query_range,
)
from .types import TaskInstance

logger = logging.getLogger(__name__)

_PERPLEXITY_SEARCH_URL = "https://api.perplexity.ai/search"
_MAX_BATCH_QUERIES = 5
_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 2.0


# ======================================================================
# RAG config
# ======================================================================

@dataclass
class RAGConfig(EvalConfig):
    rag_context_budget: int = 3000
    rag_max_results_per_query: int = 5


def load_rag_config(path: str) -> RAGConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RAGConfig(
        **{k: v for k, v in raw.items() if k in RAGConfig.__dataclass_fields__}
    )


# ======================================================================
# Perplexity Search API client
# ======================================================================

def _to_mm_dd_yyyy(date_str: str) -> str:
    """Convert YYYY-MM-DD to MM/DD/YYYY for Perplexity's date filter."""
    parts = date_str.split("-")
    if len(parts) == 3:
        return f"{parts[1]}/{parts[2]}/{parts[0]}"
    return date_str


def _chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class PerplexitySearchClient:
    """Thin REST wrapper around Perplexity Search API."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def search(
        self,
        queries: list[str],
        cutoff_date: str | None = None,
        max_results: int = 5,
        max_tokens_per_page: int = 1024,
    ) -> list[dict[str, Any]]:
        """Execute batched search, optionally with date filtering.

        Queries are chunked into groups of 5 (API limit). Results are
        returned as a flat deduplicated list ordered by relevance.

        Args:
            cutoff_date: YYYY-MM-DD date filter. If ``None``, no date
                filter is applied (used by TimeSPEC verification searches).
        """
        all_results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for batch in _chunks(queries, _MAX_BATCH_QUERIES):
            query_param = batch if len(batch) > 1 else batch[0]
            payload: dict[str, Any] = {
                "query": query_param,
                "max_results": max_results,
                "max_tokens_per_page": max_tokens_per_page,
            }
            if cutoff_date is not None:
                payload["search_before_date_filter"] = _to_mm_dd_yyyy(cutoff_date)
            resp_data = self._request(payload)
            for item in resp_data.get("results", []):
                url = item.get("url", "")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                all_results.append(item)

        return all_results

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    _PERPLEXITY_SEARCH_URL,
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_SEC * attempt)
        raise RuntimeError(
            f"Perplexity Search API failed after {_MAX_RETRIES} attempts: {last_error}"
        ) from last_error


# ======================================================================
# Query generation
# ======================================================================

def _parse_query_json(text: str) -> list[str]:
    """Extract query strings from LLM-generated JSON array."""
    text = text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        arr = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    queries: list[str] = []
    for item in arr:
        if isinstance(item, dict):
            q = item.get("query", "")
        elif isinstance(item, str):
            q = item
        else:
            continue
        q = q.strip()
        if q:
            queries.append(q)
    return queries


def generate_search_queries(
    client: OpenRouterClient,
    judge_model: str,
    instance: TaskInstance,
) -> list[str]:
    """Generate adaptive search queries for one instance via Gemini Flash."""
    _, user_prompt = build_generation_prompt(instance)
    min_q, max_q = get_rag_query_range(instance.task)

    prompt = QUERY_GENERATION_PROMPT.format(
        cutoff_date=instance.cutoff_date,
        user_prompt=user_prompt,
        min_queries=min_q,
        max_queries=max_q,
    )
    text, _ = client.call_text(
        judge_model, prompt, temperature=0.0, max_tokens=1024,
    )
    queries = _parse_query_json(text)

    if len(queries) < min_q:
        logger.warning(
            "Query generation for %s returned %d queries (min=%d), using all available",
            instance.instance_id, len(queries), min_q,
        )
    if len(queries) > max_q:
        queries = queries[:max_q]

    return queries


# ======================================================================
# Context assembly
# ======================================================================

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def retrieve_and_build_context(
    search_client: PerplexitySearchClient,
    queries: list[str],
    cutoff_date: str,
    budget: int = 3000,
    max_results_per_query: int = 5,
) -> tuple[str, list[dict[str, Any]], int, int]:
    """Search, deduplicate, truncate, and format context.

    Returns:
        (context_block, raw_results, num_results_raw, num_results_dedup)
    """
    raw_results = search_client.search(
        queries, cutoff_date, max_results=max_results_per_query,
    )
    num_raw = len(raw_results)

    # Deduplicate by URL (already done in client, but guard against edge cases)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in raw_results:
        url = r.get("url", "")
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        deduped.append(r)
    num_dedup = len(deduped)

    # Truncate to token budget
    selected: list[dict[str, Any]] = []
    total_tokens = 0
    for r in deduped:
        snippet = r.get("snippet") or r.get("content") or ""
        title = r.get("title", "")
        entry_tokens = _estimate_tokens(f"{title}\n{snippet}")
        if total_tokens + entry_tokens > budget and selected:
            break
        selected.append(r)
        total_tokens += entry_tokens

    context_block = build_rag_context_block(selected, cutoff_date)
    return context_block, deduped, num_raw, num_dedup


# ======================================================================
# Main RAG evaluation pipeline
# ======================================================================

def run_rag_evaluation(
    cfg: RAGConfig,
    project_root: Path,
    *,
    eval_tasks_str: str,
    output_dir: str,
    max_instances: int | None = None,
) -> None:
    """Run the full RAG evaluation pipeline."""
    out_dir = project_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check Perplexity API key
    pplx_key = get_perplexity_api_key()
    if not pplx_key:
        logger.error("PERPLEXITY_API_KEY not set. Skipping RAG evaluation.")
        print("  WARNING: PERPLEXITY_API_KEY not set, skipping RAG evaluation.", flush=True)
        return

    _save_json(out_dir / "config.json", {
        "model_name": cfg.model_name,
        "lora_rank": cfg.lora_rank,
        "variant": "rag",
        "checkpoint_path": "",
        "temperature": cfg.temperature,
        "eval_tasks": eval_tasks_str,
        "judge_model": cfg.judge_model,
        "search_model": cfg.search_model,
        "rag_context_budget": cfg.rag_context_budget,
        "rag_max_results_per_query": cfg.rag_max_results_per_query,
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
    print(f"  Evaluation -- rag (RAG baseline)", flush=True)
    print(f"  Model: {cfg.model_name}", flush=True)
    print(f"  Checkpoint: base (no checkpoint)", flush=True)
    print(f"  Tasks: {eval_tasks_str} | Instances: {len(instances)}", flush=True)
    print(f"  Judge: {cfg.judge_model} | Search: {cfg.search_model}", flush=True)
    print(f"  Retrieval: Perplexity Search API (date-filtered)", flush=True)
    print(f"  Context budget: {cfg.rag_context_budget} tokens", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    # Phase 1: Retrieval
    retrieval_log_path = out_dir / "retrieval_log.jsonl"
    context_map: dict[str, str] = {}  # instance_id -> context_block
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

        # Log retrieval details
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
            "results": [
                {
                    "query_idx": i,
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": (r.get("snippet") or "")[:200],
                    "date": r.get("date") or r.get("last_updated") or "",
                }
                for i, r in enumerate(raw_results)
            ],
        })

        done = idx + 1
        if done % 10 == 0 or done == len(instances):
            logger.info(
                "[Retrieval %d/%d] %s | queries=%d | results=%d | context=%d tok | %.1fs",
                done, len(instances), inst.task,
                len(queries), num_dedup, context_tokens, elapsed,
            )

    retrieval_total = time.time() - retrieval_t0
    avg_queries = np.mean([
        len(_parse_query_json(""))  # placeholder
        for _ in instances
    ]) if not instances else 0
    # Re-read retrieval log for stats
    logger.info(
        "  Retrieval complete: %d instances | %.1fs total",
        len(instances), retrieval_total,
    )

    # Phase 2: Generation with augmented prompts
    def _rag_prompt_builder(inst: TaskInstance) -> tuple[str, str]:
        ctx = context_map.get(inst.instance_id, "")
        return build_rag_generation_prompt(inst, ctx)

    _, generate_batch = _make_generator(
        cfg, checkpoint_path="", prompt_builder=_rag_prompt_builder,
    )

    gen_batch_size = 8
    logger.info("  Generating completions in batches of %d ...", gen_batch_size)
    texts = generate_batch(instances, batch_size=gen_batch_size)
    completions_log = [
        {"instance_id": inst.instance_id, "task": inst.task, "completion": text}
        for inst, text in zip(instances, texts)
    ]

    # Phase 3: Scoring (identical pipeline)
    workers = max(1, cfg.concurrent_workers)
    all_results: list[InstanceResult] = []
    step_log_path = out_dir / "eval_step_log.jsonl"
    _jsonl_lock = threading.Lock()

    def _eval_one(idx: int) -> InstanceResult:
        t0 = time.time()
        inst = instances[idx]
        completion = completions_log[idx]["completion"]
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

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_one, idx): idx for idx in range(len(instances))}
        completed_count = 0
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(instances):
                done = [r for r in all_results if r.task == result.task]
                avg_olr = np.mean([r.leakage_rate for r in done]) if done else 0
                avg_perf = np.mean([r.r_performance for r in done]) if done else 0
                logger.info(
                    "[%d/%d] %s | OLR=%.3f | Perf=%.3f | %.1fs",
                    completed_count, len(instances), result.task,
                    avg_olr, avg_perf, result.time_s,
                )

    # Sort to match instance order
    id_order = {inst.instance_id: i for i, inst in enumerate(instances)}
    all_results.sort(key=lambda r: id_order.get(r.instance_id, 0))

    search_cache.save()
    _save_json(out_dir / "completions.json", completions_log)
    _save_json(out_dir / "instance_results.json", [_result_to_dict(r) for r in all_results])
    summary = _build_summary(all_results, cfg, "rag", "none")
    _save_json(out_dir / "summary.json", summary)

    print(f"\n  Results saved to {out_dir / 'summary.json'}", flush=True)
    _print_summary(summary)


# ======================================================================
# CLI entrypoint
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation baseline.")
    parser.add_argument("--config", required=True, help="Path to RAG eval YAML config.")
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
    cfg = load_rag_config(args.config)
    run_rag_evaluation(
        cfg, project_root,
        eval_tasks_str=args.eval_tasks,
        output_dir=args.output_dir,
        max_instances=args.max_instances,
    )


if __name__ == "__main__":
    main()
