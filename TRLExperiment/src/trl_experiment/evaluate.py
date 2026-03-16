"""Post-training evaluation pipeline using vLLM with constrained decoding.

Mirrors CoverageGatedExperiment/src/cov_gated_experiment/evaluate.py in
evaluation protocol, metrics, and output formats. The only difference is the
generation backend: vLLM offline inference replaces Tinker.

Usage:
    python -m trl_experiment.evaluate \
        --config configs/eval_qwen35.yaml \
        --variant base \
        --trained_task stock \
        --eval_tasks stock \
        --output_dir runtime/results/qwen35/eval_stock_base
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig, load_env, load_eval_config, require_api_keys
from .data import load_eval_instances
from .llm import OpenRouterClient
from .prompts import (
    DATE_PLAUSIBILITY_PROMPT,
    SONAR_LEAK_VERIFY_PROMPT,
    build_generation_prompt,
    build_hall_batch_prompt,
    build_hall_sonar_prompt,
)
from .reward_pipeline import (
    _build_claim_detail,
    _compute_rates,
    _coverage,
    _parse_json_array_raw,
    _parse_sonar_json,
    _performance_score,
)
from .schemas import (
    LegalCompletion,
    SalaryCompletion,
    StockCompletion,
    compute_format_score,
    parse_completion,
)
from .types import TaskInstance

logger = logging.getLogger(__name__)

_JUDGE_CHUNK_SIZE = 40
_MAX_TOKENS_JUDGE = 4096

TASK_SCHEMA_MAP = {
    "salary": SalaryCompletion,
    "stock": StockCompletion,
    "legal": LegalCompletion,
}


@dataclass
class InstanceResult:
    instance_id: str
    task: str
    parse_success: bool
    r_format: float
    total_claims: int
    leaked_claims: int
    hallucinated_claims: int
    undecidable_claims: int
    leakage_rate: float
    hallucination_rate: float
    coverage: float
    r_valid: float
    r_performance: float
    sonar_verified: int
    time_s: float


# ======================================================================
# Main orchestration
# ======================================================================


def run_evaluation(
    cfg: EvalConfig,
    project_root: Path,
    *,
    variant: str,
    trained_task: str,
    checkpoint_path: str,
    output_dir: str,
    eval_tasks_override: str | None = None,
) -> None:
    """Run full evaluation: generate -> score -> summarize."""
    out_dir = project_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_tasks_str = eval_tasks_override if eval_tasks_override else cfg.eval_tasks

    _save_json(out_dir / "config.json", {
        "model_name": cfg.model_name,
        "variant": variant,
        "trained_task": trained_task,
        "checkpoint_path": checkpoint_path,
        "temperature": cfg.temperature,
        "eval_tasks": eval_tasks_str,
        "judge_model": cfg.judge_model,
        "search_model": cfg.search_model,
    })

    data_root = project_root / cfg.data_root
    tasks = [t.strip() for t in eval_tasks_str.split(",") if t.strip()]
    instances = load_eval_instances(data_root, tasks)
    logger.info(
        "Evaluating %d instances | variant=%s | trained_task=%s | tasks=%s",
        len(instances), variant, trained_task, tasks,
    )

    require_api_keys()
    import os
    router_key = os.getenv("OPENROUTER_API_KEY", "")

    generate_batch = _make_generator(cfg, checkpoint_path, tasks)
    client = OpenRouterClient(router_key)
    search_cache = _SearchCache(project_root / cfg.search_cache_path)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Evaluation -- {variant} (trained on {trained_task})", flush=True)
    print(f"  Model: {cfg.model_name}", flush=True)
    print(f"  Checkpoint: {checkpoint_path or 'base (no checkpoint)'}", flush=True)
    print(f"  Tasks: {eval_tasks_str} | Instances: {len(instances)}", flush=True)
    print(f"  Judge: {cfg.judge_model} | Search: {cfg.search_model}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    all_results: list[InstanceResult] = []
    step_log_path = out_dir / "eval_step_log.jsonl"
    _jsonl_lock = threading.Lock()

    gen_batch_size = 8
    logger.info("Generating completions in batches of %d ...", gen_batch_size)
    texts = generate_batch(instances, batch_size=gen_batch_size)
    completions_log = [
        {"instance_id": inst.instance_id, "task": inst.task, "completion": text}
        for inst, text in zip(instances, texts)
    ]

    workers = max(1, cfg.concurrent_workers)

    def _eval_one(idx: int) -> InstanceResult:
        t0 = time.time()
        inst = instances[idx]
        completion = completions_log[idx]["completion"]
        result, detail = _evaluate_instance(
            inst, completion, cfg=cfg, client=client,
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
                    "hallucination_rate": result.hallucination_rate,
                    "coverage": result.coverage,
                    "r_valid": result.r_valid,
                    "r_performance": result.r_performance,
                    "total_claims": result.total_claims,
                    "leaked_claims": result.leaked_claims,
                    "hallucinated_claims": result.hallucinated_claims,
                    "undecidable_claims": result.undecidable_claims,
                    "sonar_verified": result.sonar_verified,
                },
            })
        return result

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

    id_order = {inst.instance_id: i for i, inst in enumerate(instances)}
    all_results.sort(key=lambda r: id_order.get(r.instance_id, 0))

    search_cache.save()
    _save_json(out_dir / "completions.json", completions_log)
    _save_json(out_dir / "instance_results.json", [_result_to_dict(r) for r in all_results])
    summary = _build_summary(all_results, cfg, variant, trained_task)
    _save_json(out_dir / "summary.json", summary)

    print(f"\n  Results saved to {out_dir / 'summary.json'}", flush=True)
    _print_summary(summary)


# ======================================================================
# Instance-level evaluation (unchanged from CoverageGatedExperiment)
# ======================================================================


def _evaluate_instance(
    inst: TaskInstance,
    completion: str,
    *,
    cfg: EvalConfig,
    client: OpenRouterClient,
    search_cache: _SearchCache,
) -> tuple[InstanceResult, dict[str, Any]]:
    """Evaluate one instance: parse -> extract claims -> detect leakage + hallucination."""
    parsed = parse_completion(inst.task, completion)
    r_format = compute_format_score(inst.task, parsed)

    if parsed is None:
        return InstanceResult(
            instance_id=inst.instance_id, task=inst.task, parse_success=False,
            r_format=r_format, total_claims=0, leaked_claims=0,
            hallucinated_claims=0, undecidable_claims=0,
            leakage_rate=0.0, hallucination_rate=0.0,
            coverage=0.0, r_valid=0.0, r_performance=0.0,
            sonar_verified=0, time_s=0.0,
        ), {"claims": []}

    claim_pool = _extract_claims_config_b([parsed])

    if not claim_pool:
        cov = _coverage(parsed.get("reasoning", ""), parsed.get("evidence", []))
        r_perf = _performance_score(inst.task, parsed, inst.ground_truth)
        return InstanceResult(
            instance_id=inst.instance_id, task=inst.task, parse_success=True,
            r_format=r_format, total_claims=0, leaked_claims=0,
            hallucinated_claims=0, undecidable_claims=0,
            leakage_rate=0.0, hallucination_rate=0.0,
            coverage=cov, r_valid=1.0,
            r_performance=float(np.clip(r_perf, 0.0, 1.0)),
            sonar_verified=0, time_s=0.0,
        ), {"claims": []}

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_leak = pool.submit(
            _detect_leakage, claim_pool, inst.cutoff_date, client, cfg.judge_model,
        )
        future_hall = pool.submit(
            _detect_hallucination, claim_pool, client, cfg.judge_model,
            cfg.search_model, inst.task,
        )
        leak_results = future_leak.result()
        hall_results = future_hall.result()

    sonar_count = 0
    if cfg.use_search:
        uncertain_indices = [
            c["index"] for c in claim_pool
            if leak_results.get(c["index"], {}).get("confidence") == "low"
            or c["index"] not in leak_results
        ]
        if uncertain_indices:
            sonar_count = len(uncertain_indices)
            sonar_updates = _sonar_verify_leakage(
                claim_pool, uncertain_indices, inst.cutoff_date,
                client, cfg, search_cache,
            )
            for idx, update in sonar_updates.items():
                leak_results[idx] = update

    claim_to_completions: dict[str, list[int]] = {c["key"]: [0] for c in claim_pool}
    detail = _build_claim_detail(claim_pool, claim_to_completions, leak_results, hall_results)

    all_indices = [c["index"] for c in claim_pool]
    total = len(all_indices)
    leak_rate, hall_rate, r_valid, n_leaked, n_unsupported, n_undecidable = (
        _compute_rates(all_indices, leak_results, hall_results)
    )
    cov = _coverage(parsed.get("reasoning", ""), parsed.get("evidence", []))
    r_perf = _performance_score(inst.task, parsed, inst.ground_truth)

    return InstanceResult(
        instance_id=inst.instance_id, task=inst.task, parse_success=True,
        r_format=r_format, total_claims=total,
        leaked_claims=n_leaked, hallucinated_claims=n_unsupported,
        undecidable_claims=n_undecidable,
        leakage_rate=leak_rate, hallucination_rate=hall_rate,
        coverage=cov, r_valid=float(r_valid),
        r_performance=float(np.clip(r_perf, 0.0, 1.0)),
        sonar_verified=sonar_count, time_s=0.0,
    ), detail


# ======================================================================
# Claim extraction (same as CoverageGatedExperiment)
# ======================================================================


def _extract_claims_config_b(
    parseable: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build claim pool from evidence items only."""
    claim_pool: list[dict[str, Any]] = []
    claim_keys: dict[str, int] = {}
    for p in parseable:
        for ev in p.get("evidence", []):
            fact = str(ev.get("fact", "")).strip()
            if not fact:
                continue
            key = fact.lower()
            if key not in claim_keys:
                claim_keys[key] = len(claim_pool)
                sd = ev.get("source_date")
                source_date = str(sd).strip() if sd else None
                claim_pool.append({
                    "text": fact,
                    "key": key,
                    "source": "evidence",
                    "source_date": source_date,
                })
    for idx, c in enumerate(claim_pool):
        c["index"] = idx
    return claim_pool


# ======================================================================
# Leakage / hallucination detection (same as CoverageGatedExperiment)
# ======================================================================


def _detect_leakage(
    claim_pool: list[dict[str, Any]],
    cutoff_date: str,
    client: OpenRouterClient,
    judge_model: str,
) -> dict[int, dict[str, Any]]:
    """Leakage detection via DATE_PLAUSIBILITY_PROMPT + deterministic date comparison."""
    anchored = [c for c in claim_pool if c.get("source") == "evidence" and c.get("source_date")]
    results: dict[int, dict[str, Any]] = {}

    if anchored:
        for chunk_start in range(0, len(anchored), _JUDGE_CHUNK_SIZE):
            chunk = anchored[chunk_start: chunk_start + _JUDGE_CHUNK_SIZE]
            items_payload = [
                {"index": c["index"], "fact": c["text"], "declared_source_date": c["source_date"]}
                for c in chunk
            ]
            text, _ = client.call_text(
                judge_model,
                DATE_PLAUSIBILITY_PROMPT.format(
                    cutoff=cutoff_date,
                    items_json=json.dumps(items_payload, indent=2, ensure_ascii=False),
                ),
                temperature=0.0, max_tokens=_MAX_TOKENS_JUDGE,
            )
            verifier_by_idx = {r["index"]: r for r in _parse_json_array_raw(text) if "index" in r}
            for c in chunk:
                idx = c["index"]
                declared = c["source_date"]
                vr = verifier_by_idx.get(idx, {})
                plausible = bool(vr.get("plausible", True))
                corrected = vr.get("corrected_date")
                effective = corrected if (not plausible and corrected) else declared
                try:
                    leaked = bool(effective > cutoff_date)
                except TypeError:
                    leaked = False
                results[idx] = {
                    "index": idx,
                    "verdict": "leaked" if leaked else "not_leaked",
                    "event_date": effective,
                    "confidence": "high",
                    "method": "anchored",
                    "plausible": plausible,
                }
    return results


def _detect_hallucination(
    claim_pool: list[dict[str, Any]],
    client: OpenRouterClient,
    judge_model: str,
    search_model: str,
    task: str,
) -> dict[int, dict[str, Any]]:
    """VERIFY-style hallucination detection: per-claim Sonar retrieval + batch Gemini judgment."""
    if not claim_pool:
        return {}

    queries = [build_hall_sonar_prompt(task, c["text"]) for c in claim_pool]
    raw_sonar, _ = client.call_search_batch(search_model, queries, cutoff_date="")
    sonar_parsed: dict[int, dict[str, Any]] = {}
    for i, raw in raw_sonar.items():
        sonar_parsed[i] = _parse_sonar_json(raw)

    results: dict[int, dict[str, Any]] = {}
    for chunk_start in range(0, len(claim_pool), _JUDGE_CHUNK_SIZE):
        chunk = claim_pool[chunk_start: chunk_start + _JUDGE_CHUNK_SIZE]
        items = []
        for local_i, c in enumerate(chunk):
            global_i = chunk_start + local_i
            sr = sonar_parsed.get(global_i, {})
            found = sr.get("found_value") or ""
            hint = sr.get("source_hint") or ""
            reason = sr.get("reason") or ""
            evidence = (
                f"{reason}. Found: {found}. Source: {hint}."
                if reason else "No evidence retrieved."
            )
            items.append({"index": c["index"], "claim": c["text"], "search_evidence": evidence})
        text, _ = client.call_text(
            judge_model,
            build_hall_batch_prompt(task, json.dumps(items, indent=2, ensure_ascii=False)),
            temperature=0.0, max_tokens=_MAX_TOKENS_JUDGE,
        )
        for entry in _parse_json_array_raw(text):
            if isinstance(entry, dict) and "index" in entry:
                idx = entry["index"]
                pos = next(
                    (chunk_start + j for j, c in enumerate(chunk) if c["index"] == idx),
                    None,
                )
                sr = sonar_parsed.get(pos, {}) if pos is not None else {}
                results[idx] = {
                    "verdict": entry.get("verdict", "undecidable"),
                    "known_value": sr.get("found_value"),
                    "reason": entry.get("reason", ""),
                }
    return results


# ======================================================================
# Sonar fallback for uncertain leakage claims
# ======================================================================


def _sonar_verify_leakage(
    claim_pool: list[dict[str, Any]],
    uncertain_indices: list[int],
    cutoff_date: str,
    client: OpenRouterClient,
    cfg: EvalConfig,
    search_cache: _SearchCache,
) -> dict[int, dict[str, Any]]:
    """Search via Sonar for specific dates on uncertain leakage claims."""
    idx_to_claim = {c["index"]: c for c in claim_pool}
    queries: list[str] = []
    query_to_idx: list[int] = []

    for idx in uncertain_indices:
        claim = idx_to_claim.get(idx)
        if claim is None:
            continue
        query = f"When was this publicly announced or available? Specific date: {claim['text']}"
        queries.append(query)
        query_to_idx.append(idx)

    if not queries:
        return {}

    search_results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        def _search(i: int) -> tuple[int, str]:
            q = queries[i]
            cached = search_cache.get(q, cutoff_date)
            if cached is not None:
                return i, cached
            text, _ = client.call_search_single(cfg.search_model, q)
            search_cache.put(q, cutoff_date, text)
            return i, text

        futures = {pool.submit(_search, i): i for i in range(len(queries))}
        for future in as_completed(futures):
            i, text = future.result()
            search_results[i] = text

    data_for_verify = [
        {
            "index": query_to_idx[i],
            "claim": idx_to_claim[query_to_idx[i]]["text"],
            "search_result": search_results.get(i, "No results"),
        }
        for i in range(len(queries))
    ]
    verify_text, _ = client.call_text(
        cfg.judge_model,
        SONAR_LEAK_VERIFY_PROMPT.format(
            cutoff=cutoff_date,
            data_json=json.dumps(data_for_verify, indent=2, ensure_ascii=False),
        ),
        temperature=0.0, max_tokens=_MAX_TOKENS_JUDGE,
    )

    updates: dict[int, dict[str, Any]] = {}
    for r in _parse_json_array_raw(verify_text):
        if isinstance(r, dict) and "index" in r:
            updates[r["index"]] = {
                "verdict": r.get("verdict", "not_leaked"),
                "event_date": r.get("event_date"),
                "confidence": "high",
                "method": "sonar",
            }
    return updates


# ======================================================================
# vLLM Generator (replaces Tinker _make_generator)
# ======================================================================


def _make_generator(cfg: EvalConfig, checkpoint_path: str, tasks: list[str]):
    """Build batched generation closure using vLLM offline inference.

    For base variant (no checkpoint), loads model directly.
    For best/final, merges PEFT adapter from checkpoint_path.

    Returns:
        generate_batch callable matching CoverageGatedExperiment interface.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    json_schemas = {t: TASK_SCHEMA_MAP[t].model_json_schema() for t in tasks}

    if checkpoint_path:
        logger.info("Loading base model %s + adapter from %s", cfg.model_name, checkpoint_path)
        base = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype="auto")
        model = PeftModel.from_pretrained(base, checkpoint_path)
        model = model.merge_and_unload()
        merged_path = Path(checkpoint_path) / "_merged"
        merged_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged_path))
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        tokenizer.save_pretrained(str(merged_path))
        llm = LLM(model=str(merged_path), max_model_len=cfg.max_tokens + 2048)
    else:
        logger.info("Loading base model %s (no adapter)", cfg.model_name)
        llm = LLM(model=cfg.model_name, max_model_len=cfg.max_tokens + 2048)

    tokenizer = llm.get_tokenizer()

    def generate_batch(instances: list[TaskInstance], batch_size: int = 8) -> list[str]:
        """Generate completions in batches using vLLM offline inference."""
        all_completions: list[str] = [""] * len(instances)

        for start in range(0, len(instances), batch_size):
            batch = instances[start: start + batch_size]
            prompts = []
            for inst in batch:
                system, user = build_generation_prompt(inst)
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt_text)

            task = batch[0].task
            schema = json_schemas[task]
            params = SamplingParams(
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                repetition_penalty=cfg.repetition_penalty,
                structured_outputs=StructuredOutputsParams(json=schema),
            )

            outputs = llm.generate(prompts, params)
            for i, output in enumerate(outputs):
                text = output.outputs[0].text if output.outputs else ""
                all_completions[start + i] = text

            logger.info(
                "Generated %d / %d completions",
                min(start + batch_size, len(instances)),
                len(instances),
            )
        return all_completions

    return generate_batch


# ======================================================================
# Summary + output helpers (unchanged from CoverageGatedExperiment)
# ======================================================================


def _build_summary(
    results: list[InstanceResult],
    cfg: EvalConfig,
    variant: str,
    trained_task: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": {
            "model": cfg.model_name,
            "variant": variant,
            "trained_task": trained_task,
            "temperature": cfg.temperature,
            "judge_model": cfg.judge_model,
        },
    }
    by_task: dict[str, list[InstanceResult]] = {}
    for r in results:
        by_task.setdefault(r.task, []).append(r)
    for task_name in sorted(by_task.keys()):
        summary[task_name] = _aggregate_task(by_task[task_name])
    summary["overall"] = _aggregate_task(results)
    return summary


def _aggregate_task(rows: list[InstanceResult]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n_instances": 0}

    def _stat(values: list[float]) -> dict[str, float]:
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }

    return {
        "n_instances": n,
        "parse_rate": sum(r.parse_success for r in rows) / n,
        "total_claims_sum": sum(r.total_claims for r in rows),
        "leaked_claims_sum": sum(r.leaked_claims for r in rows),
        "hallucinated_claims_sum": sum(r.hallucinated_claims for r in rows),
        "undecidable_claims_sum": sum(r.undecidable_claims for r in rows),
        "sonar_verified_sum": sum(r.sonar_verified for r in rows),
        "r_format": _stat([r.r_format for r in rows]),
        "leakage_rate": _stat([r.leakage_rate for r in rows]),
        "hallucination_rate": _stat([r.hallucination_rate for r in rows]),
        "coverage": _stat([r.coverage for r in rows]),
        "r_valid": _stat([r.r_valid for r in rows]),
        "r_performance": _stat([r.r_performance for r in rows]),
    }


def _result_to_dict(r: InstanceResult) -> dict[str, Any]:
    return {
        "instance_id": r.instance_id,
        "task": r.task,
        "parse_success": r.parse_success,
        "r_format": r.r_format,
        "total_claims": r.total_claims,
        "leaked_claims": r.leaked_claims,
        "hallucinated_claims": r.hallucinated_claims,
        "undecidable_claims": r.undecidable_claims,
        "leakage_rate": r.leakage_rate,
        "hallucination_rate": r.hallucination_rate,
        "coverage": r.coverage,
        "r_valid": r.r_valid,
        "r_performance": r.r_performance,
        "sonar_verified": r.sonar_verified,
        "time_s": r.time_s,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\n{'=' * 80}", flush=True)
    print(f"  {'Task':<10} {'N':>4} {'Parse%':>7} {'OLR':>7} {'Hall.':>7} "
          f"{'Cov.':>7} {'R_val':>7} {'R_perf':>7} {'Sonar':>6}", flush=True)
    print(f"  {'-' * 68}", flush=True)
    for key in ["stock", "salary", "legal", "overall"]:
        if key not in summary or key == "config":
            continue
        s = summary[key]
        if not isinstance(s, dict) or "n_instances" not in s:
            continue
        n = s["n_instances"]
        pr = s.get("parse_rate", 0)
        olr = s.get("leakage_rate", {}).get("mean", 0)
        hall = s.get("hallucination_rate", {}).get("mean", 0)
        cov = s.get("coverage", {}).get("mean", 0)
        rv = s.get("r_valid", {}).get("mean", 0)
        rp = s.get("r_performance", {}).get("mean", 0)
        sonar = s.get("sonar_verified_sum", 0)
        label = key.upper() if key != "overall" else "ALL"
        print(f"  {label:<10} {n:>4} {pr:>7.1%} {olr:>7.3f} {hall:>7.3f} "
              f"{cov:>7.3f} {rv:>7.3f} {rp:>7.3f} {sonar:>6}", flush=True)
    print(f"{'=' * 80}\n", flush=True)


class _SearchCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, str] = {}
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data = loaded
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt search cache at %s, starting fresh", self.path)

    def get(self, query: str, cutoff_date: str) -> str | None:
        with self._lock:
            return self.data.get(self._key(query, cutoff_date))

    def put(self, query: str, cutoff_date: str, result: str) -> None:
        with self._lock:
            self.data[self._key(query, cutoff_date)] = result

    def save(self) -> None:
        with self._lock:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f)

    @staticmethod
    def _key(query: str, cutoff_date: str) -> str:
        return f"{cutoff_date}::{query.strip().lower()}"


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    """CLI for TRLExperiment evaluation pipeline."""
    parser = argparse.ArgumentParser(description="TRLExperiment evaluation pipeline.")
    parser.add_argument("--config", required=True, help="Path to eval YAML config.")
    parser.add_argument("--variant", required=True, choices=["base", "best", "final"])
    parser.add_argument("--trained_task", required=True)
    parser.add_argument("--eval_tasks", default=None)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    eval_tasks = args.eval_tasks if args.eval_tasks else args.trained_task

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    project_root = Path(__file__).resolve().parents[2]
    load_env(project_root)
    cfg = load_eval_config(args.config)
    run_evaluation(
        cfg, project_root,
        variant=args.variant,
        trained_task=args.trained_task,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        eval_tasks_override=eval_tasks,
    )


if __name__ == "__main__":
    main()
