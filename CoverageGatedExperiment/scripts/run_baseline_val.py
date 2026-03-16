"""Run a single validation pass on the *untrained* base model.

Creates a fresh LoRA training client (no gradient steps applied), gets the
base-model sampling client immediately, runs the full validation pipeline, and
writes results in exactly the same format as the training-loop validation:

  val_log.jsonl  — one row per instance (appended, step=-1)
  metrics.jsonl  — one row with aggregate metrics (appended, step=-1)

Usage (from FullExperiment/):
    uv run python scripts/run_baseline_val.py --config configs/train.yaml

Override log_dir to redirect output:
    uv run python scripts/run_baseline_val.py --config configs/train.yaml --log_dir runtime/results/train_stock
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cov_gated_experiment.config import load_config, load_env, require_api_keys
from cov_gated_experiment.data import load_train_instances, split_train_val
from cov_gated_experiment.llm import OpenRouterClient
from cov_gated_experiment.prompts import build_generation_prompt
from cov_gated_experiment.reward_pipeline import RewardPipeline

logger = logging.getLogger(__name__)

_VAL_MAX_RETRIES = 3
_VAL_RETRY_TEMPERATURE = 0.5
_BASELINE_STEP = -1   # step label used in all log files


# ------------------------------------------------------------------ helpers
def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


# ------------------------------------------------------------------ core
def run_baseline_validation(cfg_path: str, log_dir_override: str | None, train_tasks_override: str | None = None) -> None:
    import os
    import tinker
    from tinker import types as tinker_types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook import renderers as renderer_utils

    load_env(PROJECT_ROOT)
    cfg = load_config(cfg_path)
    if log_dir_override:
        cfg.log_dir = log_dir_override
    if train_tasks_override:
        cfg.train_tasks = train_tasks_override

    log_dir = PROJECT_ROOT / cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    val_log_path = log_dir / "val_log.jsonl"
    metrics_path = log_dir / "metrics.jsonl"

    # ----- check if already done -----
    if val_log_path.exists():
        with open(val_log_path) as f:
            existing_steps = {json.loads(l).get("step") for l in f if l.strip()}
        if _BASELINE_STEP in existing_steps:
            logger.info("Baseline validation (step=%d) already present in %s — skipping.",
                        _BASELINE_STEP, val_log_path)
            return

    # ----- data -----
    data_root = PROJECT_ROOT / cfg.data_root
    tasks = [t.strip() for t in cfg.train_tasks.split(",") if t.strip()]
    all_train = load_train_instances(data_root, tasks)
    _, val_set = split_train_val(all_train, cfg.val_fraction, cfg.data_seed)
    logger.info("Val set: %d instances  tasks=%s", len(val_set), tasks)

    # ----- clients -----
    _, tinker_key = require_api_keys()
    os.environ["TINKER_API_KEY"] = tinker_key
    router_key = os.getenv("OPENROUTER_API_KEY", "")
    judge_client = OpenRouterClient(router_key)
    reward_pipe = RewardPipeline(cfg, judge_client)

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=cfg.model_name, rank=cfg.lora_rank,
    )
    # Get sampling client immediately — NO training steps applied.
    # This is the pure base model.
    sampling_client = training_client.save_weights_and_get_sampling_client()
    logger.info("Obtained base-model sampling client (no gradient steps applied)")

    # ----- sampling params -----
    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    greedy_params = tinker_types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=0.0,
        stop=renderer.get_stop_sequences(),
    )
    retry_params = tinker_types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=_VAL_RETRY_TEMPERATURE,
        stop=renderer.get_stop_sequences(),
    )

    def _build_input(inst: Any) -> Any:
        system, user = build_generation_prompt(inst)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return renderer.build_generation_prompt(msgs)

    def _sample_and_score(inst: Any, params: Any) -> tuple[str, Any, dict]:
        model_input = _build_input(inst)
        result = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=params,
        ).result()
        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        text = renderer_utils.get_text_content(parsed_msg)
        rewards, claim_detail = reward_pipe.score_group([text], inst)
        return text, rewards[0], claim_detail

    # ----- first pass: submit all greedy samples in parallel -----
    pending: list[tuple[Any, Any]] = []  # (inst, future)
    for inst in val_set:
        model_input = _build_input(inst)
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=greedy_params,
        )
        pending.append((inst, future))

    # ----- collect + retry + score -----
    leaks: list[float] = []
    halls: list[float] = []
    covs: list[float] = []
    perfs: list[float] = []
    n_parse_failed = 0

    for i, (inst, future) in enumerate(pending):
        logger.info("[%d/%d] %s", i + 1, len(val_set), inst.instance_id)
        result = future.result()
        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        text = renderer_utils.get_text_content(parsed_msg)
        rewards, claim_detail = reward_pipe.score_group([text], inst)
        r = rewards[0]

        attempt = 0
        while not r.parse_success and attempt < _VAL_MAX_RETRIES:
            attempt += 1
            logger.warning("  parse failed (attempt %d/%d), retrying with T=%.1f",
                           attempt, _VAL_MAX_RETRIES, _VAL_RETRY_TEMPERATURE)
            text, r, claim_detail = _sample_and_score(inst, retry_params)

        parse_failed = not r.parse_success
        if parse_failed:
            n_parse_failed += 1
            logger.warning("  all %d attempts failed to parse — excluded from metrics",
                           _VAL_MAX_RETRIES + 1)
        else:
            leaks.append(r.leakage_rate)
            halls.append(r.hallucination_rate)
            covs.append(r.coverage)
            perfs.append(r.r_performance)
            logger.info("  leak=%.3f hall=%.3f perf=%.3f claims=%d leaked=%d",
                        r.leakage_rate, r.hallucination_rate, r.r_performance,
                        r.num_claims, r.num_leaked)

        _append_jsonl(val_log_path, {
            "step": _BASELINE_STEP,
            "instance_id": inst.instance_id,
            "task": inst.task,
            "cutoff_date": inst.cutoff_date,
            "parse_failed": parse_failed,
            "completion": text,
            "claims": claim_detail.get("claims", []),
            "leakage_rate": r.leakage_rate,
            "hallucination_rate": r.hallucination_rate,
            "coverage": r.coverage,
            "r_valid": r.r_valid,
            "r_performance": r.r_performance,
            "num_claims": r.num_claims,
            "num_leaked": r.num_leaked,
            "num_unsupported": r.num_unsupported,
            "num_undecidable": r.num_undecidable,
        })

    # ----- aggregate -----
    summary = {
        "step": _BASELINE_STEP,
        "val_leakage_rate": _mean(leaks),
        "val_leakage_rate_std": _std(leaks),
        "val_hallucination_rate": _mean(halls),
        "val_hallucination_rate_std": _std(halls),
        "val_coverage": _mean(covs),
        "val_coverage_std": _std(covs),
        "val_performance": _mean(perfs),
        "val_performance_std": _std(perfs),
        "val_n_instances": len(leaks),
        "val_n_parse_failed": n_parse_failed,
    }
    _append_jsonl(metrics_path, summary)
    logger.info("Baseline validation complete.")
    logger.info("  instances scored: %d / %d  (failed: %d)",
                len(leaks), len(val_set), n_parse_failed)
    logger.info("  leak_rate=%.4f  hall_rate=%.4f  perf=%.4f  coverage=%.4f",
                summary["val_leakage_rate"], summary["val_hallucination_rate"],
                summary["val_performance"], summary["val_coverage"])
    logger.info("Results written to: %s", log_dir)


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Validate untrained base model.")
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/train.yaml)")
    parser.add_argument("--log_dir", default=None,
                        help="Override log_dir from config (default: use config value)")
    parser.add_argument("--train_tasks", default=None,
                        help="Override train_tasks from config (e.g. 'stock')")
    args = parser.parse_args()
    run_baseline_validation(args.config, args.log_dir, args.train_tasks)
