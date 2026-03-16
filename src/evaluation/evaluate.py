"""4-phase evaluation pipeline, faithfully replicating TimeSPEC methodology.

Phase 1: Claim extraction from prediction rationales.
Phase 2: Shapley value calculation for claim importance.
Phase 3: Leakage detection (search + LLM judge).
Phase 4: Metrics calculation (OLR, Shapley-DCLR).

Usage:
    uv run python -m src.evaluation.evaluate --config configs/eval_base_model.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.shared.claim_extractor import ClaimExtractor
from src.shared.data_loader import TaskInstance, load_all_eval
from src.shared.leakage_detector import LeakageDetector
from src.shared.llm_client import simple_call
from src.shared.progress import ProgressBar
from src.shared.prompts.builder import build_messages, build_prompt, parse_response
from src.shared.search_client import PersistentSearchCache
from src.training.reward.performance import compute_performance_reward

from .metrics_calculator import aggregate_metrics, calculate_shapley_metrics
from .shapley_calculator import ShapleyCalculator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """All settings for an evaluation run."""

    # Model to evaluate (Tinker model id or OpenRouter model name)
    model_name: str = "Qwen/Qwen3-30B-A3B"

    # Generation backend
    use_tinker: bool = True
    tinker_checkpoint: str = ""  # legacy: training-state path (slow)
    tinker_sampler_path: str = ""  # preferred: direct sampler weights path
    lora_rank: int = 32

    # Prompt mode: "temporal_hint" or "base" (no temporal constraint)
    prompt_mode: str = "temporal_hint"

    # TimeSPEC agent mode (overrides simple generation)
    use_timespec: bool = False

    # Generation
    temperature: float = 0.0
    max_tokens: int = 4096

    # Claim extraction + leakage detection
    detector_model: str = "google/gemini-2.5-flash"
    use_search: bool = True
    use_llm_judge: bool = True
    search_cache_path: str = "results/eval/search_cache.json"

    # Shapley computation
    skip_shapley: bool = False
    shapley_model: str = "google/gemini-2.5-flash"
    shapley_max_samples: int = 100
    shapley_batch_size: int = 32

    # Output
    output_dir: str = "results/eval"
    tasks: str = "stock,salary,legal"

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Per-instance 4-phase evaluation
# ---------------------------------------------------------------------------

def evaluate_instance(
    instance: TaskInstance,
    completion: str,
    *,
    extractor: ClaimExtractor,
    shapley: ShapleyCalculator | None,
    detector: LeakageDetector,
    out_dir: Path,
    skip_shapley: bool = False,
) -> dict[str, Any]:
    """Run the full 4-phase evaluation on one instance."""
    iid = instance.instance_id
    parsed = parse_response(completion, instance)
    rationale = _get_rationale(parsed, completion)
    prediction = _get_prediction(parsed, instance)

    if not rationale:
        logger.warning("%s: empty rationale, skipping phases 2-4", iid)
        return {"success": True, "id": iid, "metrics": _zero_metrics()}

    # Phase 1: Claim extraction
    extraction = extractor.extract(
        rationale,
        task_description=_task_desc(instance),
        event_description=_event_desc(instance),
        cutoff_date=instance.cutoff_date,
    )
    claims = extraction.claims
    claim_texts = [c.text for c in claims]

    _save_json(out_dir / "phase1_claims" / f"{iid}.json", {
        "claims": [{"claim_id": c.claim_id, "text": c.text,
                     "category": c.category, "temporal_ref": c.temporal_ref}
                    for c in claims],
        "total": len(claims),
        "group_a": len(extraction.group_a),
        "group_b": len(extraction.group_b),
    })

    if not claims:
        logger.info("%s: 0 claims extracted", iid)
        return {"success": True, "id": iid, "metrics": _zero_metrics()}

    # Phase 2: Shapley value calculation (optional — very expensive)
    if skip_shapley or shapley is None:
        n = len(claims)
        shapley_values = [1.0 / n] * n
    else:
        shapley_result = shapley.compute(
            task=instance.task,
            task_input=instance.raw,
            claim_texts=claim_texts,
            control_prediction=prediction,
        )
        shapley_values = shapley_result["shapley_values"]

        _save_json(out_dir / "phase2_shapley" / f"{iid}.json", {
            "claims": claim_texts,
            "shapley_values": shapley_values,
            "control_prediction": prediction,
            "cache_stats": shapley_result.get("cache_stats", {}),
            "num_llm_calls": shapley_result.get("num_llm_calls", 0),
        })

    # Phase 3: Leakage detection
    leak_report = detector.detect(claims, instance.cutoff_date)

    leak_details = [
        {"claim_id": r.claim_id, "text": r.text, "category": r.category,
         "is_leaked": r.is_leaked, "method": r.method,
         "determination_date": r.determination_date,
         "reason": r.reason}
        for r in leak_report.results
    ]
    _save_json(out_dir / "phase3_leakage" / f"{iid}.json", {
        "cutoff_date": instance.cutoff_date,
        "results": leak_details,
        "summary": {"total": leak_report.total_claims, "leaked": leak_report.num_leaked},
    })

    # Phase 4: Metrics (OLR + Shapley-DCLR)
    is_leaked = [r.is_leaked for r in leak_report.results]
    metrics = calculate_shapley_metrics(shapley_values, is_leaked)

    cat_breakdown: dict[str, dict[str, int]] = {}
    for r in leak_report.results:
        cat_breakdown.setdefault(r.category, {"total": 0, "leaked": 0})
        cat_breakdown[r.category]["total"] += 1
        if r.is_leaked:
            cat_breakdown[r.category]["leaked"] += 1
    metrics["category_breakdown"] = cat_breakdown

    r_perf = compute_performance_reward(parsed, instance)
    metrics["r_performance"] = r_perf
    metrics["parse_success"] = parsed is not None

    _save_json(out_dir / "phase4_metrics" / f"{iid}.json", metrics)

    logger.debug(
        "%s: %d claims (A:%d/B:%d), %d leaked, OLR=%.3f, DCLR=%.3f, Perf=%.3f",
        iid, len(claims), len(extraction.group_a), len(extraction.group_b),
        leak_report.num_leaked, metrics["olr"], metrics["shapley_dclr"], r_perf,
    )

    return {"success": True, "id": iid, "metrics": metrics}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(cfg: EvalConfig) -> None:
    """Run the full evaluation pipeline on all eval instances."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility
    from dataclasses import asdict
    _save_json(out_dir / "config.json", asdict(cfg))

    task_list = [t.strip() for t in cfg.tasks.split(",")]
    all_eval = load_all_eval()
    instances = [inst for inst in all_eval if inst.task in task_list]
    logger.info(
        "Evaluating %d instances | tasks=%s | model=%s | prompt=%s | tinker=%s (ckpt=%s)",
        len(instances), task_list, cfg.model_name, cfg.prompt_mode,
        cfg.use_tinker, cfg.tinker_checkpoint or "base",
    )

    generate_fn = _make_generator(cfg)
    extractor = ClaimExtractor(model=cfg.detector_model)
    cache = PersistentSearchCache(Path(cfg.search_cache_path)) if cfg.search_cache_path else None
    detector = LeakageDetector(
        model=cfg.detector_model,
        use_search=cfg.use_search,
        use_llm_judge=cfg.use_llm_judge,
        persistent_cache=cache,
    )
    shapley: ShapleyCalculator | None = None
    if not cfg.skip_shapley:
        shapley = ShapleyCalculator(
            model=cfg.shapley_model,
            max_samples=cfg.shapley_max_samples,
            batch_size=cfg.shapley_batch_size,
        )

    condition = _infer_condition(cfg)
    sampler_label = cfg.tinker_sampler_path or cfg.tinker_checkpoint or "base"
    print(f"\n{'='*60}", flush=True)
    print(f"  Evaluation — {condition}", flush=True)
    print(f"  Model: {cfg.model_name} | Prompt: {cfg.prompt_mode}", flush=True)
    print(f"  Sampler: {sampler_label}", flush=True)
    print(f"  Tasks: {cfg.tasks} | Instances: {len(instances)}", flush=True)
    print(f"  Skip Shapley: {cfg.skip_shapley}", flush=True)
    print(f"  Output: {cfg.output_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    all_results: list[dict[str, Any]] = []
    completions_log: list[dict[str, Any]] = []
    running_olr: list[float] = []
    running_perf: list[float] = []

    pb = ProgressBar(total=len(instances), prefix=f"Eval[{condition}]")

    for i, inst in enumerate(instances):
        t0 = time.time()

        completion = generate_fn(inst)
        completions_log.append({
            "instance_id": inst.instance_id, "task": inst.task, "completion": completion,
        })

        result = evaluate_instance(
            inst, completion,
            extractor=extractor, shapley=shapley, detector=detector,
            out_dir=out_dir, skip_shapley=cfg.skip_shapley,
        )
        result["time_s"] = round(time.time() - t0, 2)
        all_results.append(result)

        m = result.get("metrics", {})
        running_olr.append(m.get("olr", 0))
        running_perf.append(m.get("r_performance", 0))
        avg_olr = sum(running_olr) / len(running_olr)
        avg_perf = sum(running_perf) / len(running_perf)
        pb.update(i + 1, task=inst.task, OLR=avg_olr, Perf=avg_perf)

    pb.finish(OLR=avg_olr, Perf=avg_perf)

    if cache:
        cache.save()

    _save_aggregate(all_results, completions_log, out_dir, cfg)
    print(f"  Results saved to {out_dir}/summary.json\n", flush=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _save_aggregate(
    results: list[dict],
    completions: list[dict],
    out_dir: Path,
    cfg: EvalConfig,
) -> None:
    _save_json(out_dir / "completions.json", completions)

    task_map: dict[str, str] = {}
    for c in completions:
        task_map[c["instance_id"]] = c["task"]

    instance_rows: list[dict[str, Any]] = []
    for r in results:
        if r.get("success") and "metrics" in r:
            m = r["metrics"]
            instance_rows.append({
                "id": r["id"],
                "task": task_map.get(r["id"], "unknown"),
                "olr": m.get("olr", 0),
                "shapley_dclr": m.get("shapley_dclr", 0),
                "r_performance": m.get("r_performance", 0),
                "total_claims": m.get("total_claims", 0),
                "leaked_claims": m.get("leaked_claims", 0),
                "parse_success": m.get("parse_success", False),
                "time_s": r.get("time_s", 0),
            })
    _save_json(out_dir / "instance_results.json", instance_rows)

    by_task: dict[str, list[dict]] = {}
    for row in instance_rows:
        by_task.setdefault(row["task"], []).append(row)

    summary: dict[str, Any] = {
        "config": {
            "model": cfg.model_name,
            "prompt_mode": cfg.prompt_mode,
            "use_tinker": cfg.use_tinker,
            "tinker_sampler_path": cfg.tinker_sampler_path or "base",
            "skip_shapley": cfg.skip_shapley,
        },
    }
    for task, rows in sorted(by_task.items()):
        metrics_for_agg = [
            {"olr": r["olr"], "shapley_dclr": r["shapley_dclr"],
             "total_claims": r["total_claims"], "leaked_claims": r["leaked_claims"],
             "total_shapley_weight": 0, "leaked_shapley_weight": 0}
            for r in rows
        ]
        agg = aggregate_metrics(metrics_for_agg)
        perfs = [r["r_performance"] for r in rows]
        agg["performance"] = {
            "mean": float(np.mean(perfs)) if perfs else 0,
            "std": float(np.std(perfs, ddof=1)) if len(perfs) > 1 else 0,
        }
        agg["parse_rate"] = sum(r["parse_success"] for r in rows) / max(len(rows), 1)
        summary[task] = agg

    all_metrics = [
        {"olr": r["olr"], "shapley_dclr": r["shapley_dclr"],
         "total_claims": r["total_claims"], "leaked_claims": r["leaked_claims"],
         "total_shapley_weight": 0, "leaked_shapley_weight": 0}
        for r in instance_rows
    ]
    summary["overall"] = aggregate_metrics(all_metrics)

    _save_json(out_dir / "summary.json", summary)

    logger.info("=== Evaluation Summary ===")
    for key, val in summary.items():
        if key == "config":
            continue
        if isinstance(val, dict) and "olr" in val:
            olr = val["olr"]
            dclr = val["shapley_dclr"]
            logger.info(
                "  %s: OLR=%.4f(+/-%.4f) | DCLR=%.4f(+/-%.4f) | n=%d",
                key,
                olr.get("mean", 0), olr.get("std", 0),
                dclr.get("mean", 0), dclr.get("std", 0),
                val.get("n_instances", 0),
            )


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------

def _make_generator(cfg: EvalConfig):
    """Build a completion function matching the configured backend.

    - ``use_timespec=True``: 5-phase TimeSPEC agent (returns prediction+rationale dict).
    - ``use_tinker=True, tinker_checkpoint=""``: Tinker base model (no LoRA).
    - ``use_tinker=True, tinker_checkpoint="<path>"``: Tinker + trained LoRA.
    - ``use_tinker=False``: OpenRouter API.
    """
    if cfg.use_timespec:
        return _make_timespec_generator(cfg)
    if cfg.use_tinker:
        return _make_tinker_generator(cfg)
    return _make_openrouter_generator(cfg)


def _make_timespec_generator(cfg: EvalConfig):
    from .timespec_agent import TimeSpecAgent

    cache = PersistentSearchCache(Path(cfg.search_cache_path)) if cfg.search_cache_path else None
    agent = TimeSpecAgent(
        model=cfg.model_name,
        detector_model=cfg.detector_model,
        search_cache=cache,
    )

    def generate(inst: TaskInstance) -> str:
        result = agent.run(inst)
        pred = result.get("prediction")
        rat = result.get("rationale", "")
        if inst.task == "stock" and isinstance(pred, list):
            return json.dumps({"ranking": pred, "rationale": rat})
        if inst.task == "salary":
            return json.dumps({"predicted_salary": pred, "rationale": rat})
        if inst.task == "legal":
            return json.dumps({
                "probability_petitioner": pred,
                "predicted_winner": "PETITIONER" if (pred or 0) >= 0.5 else "RESPONDENT",
                "rationale": rat,
            })
        return json.dumps({"prediction": pred, "rationale": rat})

    return generate


def _make_openrouter_generator(cfg: EvalConfig):
    pm = cfg.prompt_mode

    def generate(inst: TaskInstance) -> str:
        msgs = build_messages(inst, prompt_mode=pm)
        return simple_call(
            msgs[1]["content"],
            model=cfg.model_name,
            system_prompt=msgs[0]["content"],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    return generate


def _make_tinker_generator(cfg: EvalConfig):
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service_client = tinker.ServiceClient()

    if cfg.tinker_sampler_path:
        sampling_client = service_client.create_sampling_client(
            model_path=cfg.tinker_sampler_path,
        )
    elif cfg.tinker_checkpoint:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            cfg.tinker_checkpoint,
        )
        sampling_client = training_client.save_weights_and_get_sampling_client(name="eval")
    else:
        sampling_client = service_client.create_sampling_client(
            base_model=cfg.model_name,
        )

    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sampling_params = tinker.types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=renderer.get_stop_sequences(),
    )
    pm = cfg.prompt_mode

    def generate(inst: TaskInstance) -> str:
        system, user = build_prompt(inst, prompt_mode=pm)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        model_input = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        ).result()
        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        return renderers.get_text_content(parsed_msg)

    return generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_condition(cfg: EvalConfig) -> str:
    """Infer a short human-readable condition name from the config."""
    if cfg.use_timespec:
        return "TimeSPEC"
    out = cfg.output_dir.lower()
    if "base_all" in out or "base_model" in out:
        return "Base Model"
    if "trained_stock" in out:
        return "Trained (Stock)"
    if "trained_salary" in out:
        return "Trained (Salary)"
    if "trained_legal" in out:
        return "Trained (Legal)"
    if "temporal_hint" in out:
        return "Temporal Hint"
    if "perf_only" in out:
        return "GRPO+Perf"
    if "leak_only" in out:
        return "GRPO+Leak"
    if "grpo_full" in out:
        return "Our Method"
    if cfg.prompt_mode == "base":
        return "Base Model"
    return cfg.prompt_mode


def _get_rationale(parsed: dict[str, Any] | None, raw: str) -> str:
    if parsed:
        r = parsed.get("rationale", parsed.get("ranking_rationale", ""))
        if r:
            return r
    return raw


def _get_prediction(parsed: dict[str, Any] | None, inst: TaskInstance) -> Any:
    if parsed is None:
        return _default_prediction(inst)
    if inst.task == "legal":
        return parsed.get("probability_petitioner", 0.5)
    if inst.task == "salary":
        return parsed.get("predicted_salary", 15_000_000)
    if inst.task == "stock":
        return parsed.get("ranking", inst.raw.get("companies", []))
    return None


def _default_prediction(inst: TaskInstance) -> Any:
    if inst.task == "legal":
        return 0.5
    if inst.task == "salary":
        return 15_000_000
    if inst.task == "stock":
        return inst.raw.get("companies", [])
    return None


def _task_desc(inst: TaskInstance) -> str:
    descs = {
        "stock": "Stock return ranking prediction",
        "salary": "Free-agent salary prediction",
        "legal": "Supreme Court case outcome prediction",
    }
    return descs.get(inst.task, "Prediction task")


def _event_desc(inst: TaskInstance) -> str:
    if inst.task == "stock":
        p = inst.raw.get("past_period", {})
        return f"Stock ranking {p.get('start_date', '')} to {p.get('end_date', '')}"
    if inst.task == "salary":
        return f"Contract for {inst.raw.get('Player', '')} ({inst.raw.get('Year', '')})"
    if inst.task == "legal":
        return f"Outcome of {inst.raw.get('case_name', '')}"
    return inst.question_text


def _zero_metrics() -> dict[str, Any]:
    return {
        "total_claims": 0, "leaked_claims": 0,
        "olr": 0.0, "shapley_dclr": 0.0,
        "total_shapley_weight": 0.0, "leaked_shapley_weight": 0.0,
    }


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="4-phase evaluation (TimeSPEC methodology)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    for name, fld in EvalConfig.__dataclass_fields__.items():
        default = fld.default
        parser.add_argument(
            f"--{name}",
            type=type(default) if default is not None else str,
            default=None,
        )
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.config)
    for name in EvalConfig.__dataclass_fields__:
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)

    run_evaluation(cfg)


if __name__ == "__main__":
    main()
