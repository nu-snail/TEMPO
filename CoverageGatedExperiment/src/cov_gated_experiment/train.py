"""Main GRPO training entrypoint for the independent FullExperiment project."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrainConfig, load_config, load_env, require_api_keys
from .data import load_train_instances, split_train_val
from .lambda_schedule import cosine_anneal
from .llm import OpenRouterClient
from .prompts import build_generation_prompt
from .reward_pipeline import RewardPipeline

logger = logging.getLogger(__name__)


def run_training(cfg: TrainConfig, project_root: Path) -> None:
    """Run GRPO training with reward pipeline defined in plan documents."""

    import tinker
    import torch
    from tinker import types as tinker_types
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import checkpoint_utils, model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.utils import ml_log

    log_dir = project_root / cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    data_root = project_root / cfg.data_root
    tasks = [t.strip() for t in cfg.train_tasks.split(",") if t.strip()]
    all_train = load_train_instances(data_root, tasks)
    train_set, val_set = split_train_val(all_train, cfg.val_fraction, cfg.data_seed)
    logger.info("Loaded train=%d val=%d tasks=%s", len(train_set), len(val_set), tasks)

    _, tinker_key = require_api_keys()
    import os

    os.environ["TINKER_API_KEY"] = tinker_key
    router_key = os.getenv("OPENROUTER_API_KEY", "")
    judge_client = OpenRouterClient(router_key)

    reward_pipe = RewardPipeline(cfg, judge_client)

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=cfg.model_name, rank=cfg.lora_rank)

    tokenizer = get_tokenizer(cfg.model_name)
    if cfg.renderer_name:
        renderer_name = cfg.renderer_name
    else:
        renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sampling_params = tinker_types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = tinker_types.AdamParams(
        learning_rate=cfg.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    loss_fn_config = {"clip_low_threshold": cfg.clip_low, "clip_high_threshold": cfg.clip_high}

    ml_logger = ml_log.setup_logging(
        log_dir=str(log_dir),
        wandb_project=None,
        wandb_name=None,
        config=cfg,
        do_configure_logging_module=True,
    )

    total_steps = math.ceil(len(train_set) / cfg.batch_size) * cfg.num_epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    val_steps: set[int] = set()
    if cfg.num_validations > 0 and total_steps > 0:
        for i in range(1, cfg.num_validations + 1):
            step = max(0, round(i * total_steps / cfg.num_validations) - 1)
            val_steps.add(min(step, total_steps - 1))
    logger.info("total_steps=%d  val_steps=%s (num_validations=%d)", total_steps, sorted(val_steps), cfg.num_validations)
    global_step = 0
    best_val_leak = float("inf")
    metrics_path = log_dir / "metrics.jsonl"
    step_log_path = log_dir / "step_log.jsonl"
    val_log_path = log_dir / "val_log.jsonl"

    def _signal_handler(signum: int, frame: Any) -> None:
        logger.warning("Received signal %d at step %d — exiting gracefully", signum, global_step)
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        for epoch in range(cfg.num_epochs):
            epoch_items = _difficulty_balanced_shuffle(train_set, cfg.data_seed + epoch, cfg.batch_size)
            for start in range(0, len(epoch_items), cfg.batch_size):
                batch = epoch_items[start : start + cfg.batch_size]
                if not batch:
                    continue
                t0 = time.time()
                sampling_client = training_client.save_weights_and_get_sampling_client()
                sample_data = _sample_group_batch(
                    batch,
                    sampling_client,
                    renderer,
                    sampling_params,
                    cfg.group_size,
                )
                w_leak_t = cosine_anneal(global_step, total_steps, cfg.w_leak_start, cfg.w_leak_end)
                logger.info(
                    "Step %d/%d  epoch=%d  batch=%d  sampled in %.1fs",
                    global_step, total_steps, epoch, len(batch), time.time() - t0,
                )

                all_datums: list[tinker_types.Datum] = []
                step_rewards: list[float] = []
                step_perf: list[float] = []
                step_r_leak: list[float] = []
                step_leak: list[float] = []
                step_cov: list[float] = []
                step_nclaims: list[int] = []
                step_advantages: list[float] = []
                group_rleak_stds: list[float] = []
                group_rperf_stds: list[float] = []
                groups_low_variance = 0
                skipped_groups = 0
                n_perf_mode_groups = 0
                n_leak_mode_groups = 0
                step_r_eff: list[float] = []

                scored_raw: list[tuple[dict, list, dict]] = []
                workers = max(1, getattr(cfg, "concurrent_workers", 4))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(
                            reward_pipe.score_group, sd["completions"], sd["instance"],
                            skip_hallucination=True,
                        ): sd
                        for sd in sample_data
                    }
                    for future in as_completed(futures):
                        sd = futures[future]
                        rewards, claim_detail = future.result()
                        scored_raw.append((sd, rewards, claim_detail))

                _all_parseable: list[Any] = []
                for _, rewards, _ in scored_raw:
                    for r in rewards:
                        if r.parse_success:
                            _all_parseable.append(r)

                all_r_leak_batch = [np.exp(-0.5 * r.num_leaked) for r in _all_parseable]
                batch_mean_r_leak = float(np.mean(all_r_leak_batch)) if all_r_leak_batch else 0.5

                _clean = [r for r in _all_parseable if r.num_leaked == 0]
                if _clean:
                    _cov_gate = [min(1.0, r.coverage / cfg.cov_target) for r in _clean]
                    _r_eff = [
                        r.r_performance * (cfg.cov_floor + (1 - cfg.cov_floor) * g)
                        for r, g in zip(_clean, _cov_gate)
                    ]
                    batch_mean_r_eff_clean = float(np.mean(_r_eff))
                else:
                    batch_mean_r_eff_clean = 0.5 * cfg.cov_floor

                scored: list[tuple[dict, list, list, dict, dict]] = []
                for sd, rewards, claim_detail in scored_raw:
                    advantages, adv_diag = reward_pipe.compute_advantages(
                        rewards, w_leak_t, batch_mean_r_leak, batch_mean_r_eff_clean,
                    )
                    if cfg.overlong_decay > 0:
                        for i, r in enumerate(rewards):
                            if not r.parse_success:
                                text_len = len(sd["completions"][i])
                                len_ratio = min(1.0, text_len / cfg.max_tokens_chars)
                                scale = max(1.0 - cfg.overlong_decay * len_ratio, cfg.overlong_floor)
                                advantages[i] = advantages[i] * scale
                    scored.append((sd, rewards, advantages, claim_detail, adv_diag))

                _n_parseable = len(_all_parseable)
                _n_clean = len(_clean)
                _mode_counts = {"perf": 0, "leak": 0}
                for _, _, _, _, _d in scored:
                    _mode_counts[
                        "perf" if _d.get("adv_mode") == "performance" else "leak"
                    ] += 1
                logger.info(
                    "  Scored %d groups: %d perf-mode, %d leak-mode | "
                    "parseable=%d clean=%d | "
                    "batch_mean_r_leak=%.3f  batch_mean_r_eff_clean=%.3f",
                    len(scored), _mode_counts["perf"], _mode_counts["leak"],
                    _n_parseable, _n_clean,
                    batch_mean_r_leak, batch_mean_r_eff_clean,
                )

                for sd, rewards, advantages, claim_detail, adv_diag in scored:
                    parseable = [r for r in rewards if r.parse_success]
                    p_rperf = [r.r_performance for r in parseable]

                    rl_std = adv_diag.get("r_leak_std", 0.0)
                    rp_std = adv_diag.get("r_perf_std", 0.0)
                    if len(parseable) > 1:
                        group_rleak_stds.append(rl_std)
                        group_rperf_stds.append(rp_std)
                        if rl_std < 0.05:
                            groups_low_variance += 1

                    if adv_diag.get("adv_mode") == "performance":
                        n_perf_mode_groups += 1
                        step_r_eff.extend(adv_diag.get("r_eff_values", []))
                    elif adv_diag.get("adv_mode") == "leakage":
                        n_leak_mode_groups += 1

                    is_skipped = all(abs(a) < 1e-8 for a in advantages)

                    adv_mode = adv_diag.get("adv_mode", "unknown")
                    z_leak_list = adv_diag.get("z_leak", [])
                    z_perf_list = adv_diag.get("z_perf", [])
                    baseline_list = adv_diag.get("baseline_bonus", [])
                    r_leak_vals = adv_diag.get("r_leak_values", [])
                    r_eff_vals = adv_diag.get("r_eff_values", [])
                    cov_vals = adv_diag.get("cov_values", [])
                    cov_bonus_list = adv_diag.get("cov_bonus", [])

                    def _completion_entry(i: int, r: Any) -> dict[str, Any]:
                        entry: dict[str, Any] = {
                            "idx": i,
                            "text": sd["completions"][i],
                            "parse_success": r.parse_success,
                            "r_valid": r.r_valid,
                            "leakage_rate": r.leakage_rate,
                            "coverage": r.coverage,
                            "num_claims": r.num_claims,
                            "num_leaked": r.num_leaked,
                            "r_performance": r.r_performance,
                            "advantage": advantages[i],
                        }
                        if not r.parse_success:
                            return entry
                        p_idx = sum(1 for rr in rewards[:i] if rr.parse_success)
                        if adv_mode == "performance":
                            if p_idx < len(r_eff_vals):
                                entry["r_eff"] = round(r_eff_vals[p_idx], 4)
                            if p_idx < len(z_perf_list):
                                entry["z_eff"] = round(z_perf_list[p_idx], 4)
                            if p_idx < len(baseline_list):
                                entry["perf_baseline"] = round(baseline_list[p_idx], 4)
                        elif adv_mode == "leakage":
                            if p_idx < len(r_leak_vals):
                                entry["r_leak_exp"] = round(r_leak_vals[p_idx], 4)
                            if p_idx < len(z_leak_list):
                                entry["z_leak"] = round(z_leak_list[p_idx], 4)
                            if p_idx < len(baseline_list):
                                entry["leak_baseline"] = round(baseline_list[p_idx], 4)
                            if p_idx < len(cov_bonus_list):
                                entry["cov_bonus"] = round(cov_bonus_list[p_idx], 4)
                        return entry

                    _append_jsonl(
                        step_log_path,
                        {
                            "step": global_step,
                            "instance_id": sd["instance"].instance_id,
                            "task": sd["instance"].task,
                            "cutoff_date": sd["instance"].cutoff_date,
                            "skipped": is_skipped,
                            "adv_mode": adv_mode,
                            "all_clean": adv_diag.get("all_clean"),
                            "batch_mean_r_leak": round(batch_mean_r_leak, 4),
                            "batch_mean_r_eff_clean": round(batch_mean_r_eff_clean, 4),
                            "r_leak_std": round(rl_std, 4),
                            "r_perf_std": round(rp_std, 4),
                            "format_penalty": adv_diag.get("format_penalty"),
                            "claims": claim_detail.get("claims", []),
                            "completions": [
                                _completion_entry(i, r) for i, r in enumerate(rewards)
                            ],
                        },
                    )

                    if is_skipped:
                        skipped_groups += 1
                        continue

                    step_rewards.extend([r.r_valid for r in parseable])
                    step_perf.extend(p_rperf)
                    step_r_leak.extend(
                        [np.exp(-0.5 * r.num_leaked) for r in parseable]
                    )
                    step_leak.extend([r.leakage_rate for r in parseable])
                    step_cov.extend([r.coverage for r in parseable])
                    step_nclaims.extend([r.num_claims for r in parseable])
                    step_advantages.extend(advantages)

                    datum_list = _build_datums(sd, advantages, tinker_types, TensorData, torch)
                    all_datums.extend(datum_list)

                if not all_datums:
                    global_step += 1
                    continue

                training_client.forward_backward(
                    all_datums,
                    loss_fn="ppo",
                    loss_fn_config=loss_fn_config,
                ).result()
                opt_result = training_client.optim_step(adam_params).result()

                adv_arr = np.array(step_advantages) if step_advantages else np.zeros(1)
                metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "R_valid_mean": float(np_mean(step_rewards)),
                    "R_leakage_mean": float(np_mean(step_r_leak)),
                    "R_performance_mean": float(np_mean(step_perf)),
                    "leakage_rate_mean": float(np_mean(step_leak)),
                    "coverage_mean": float(np_mean(step_cov)),
                    "num_claims_mean": float(np_mean(step_nclaims)),
                    "batch_mean_r_leak": batch_mean_r_leak,
                    "batch_mean_r_eff_clean": batch_mean_r_eff_clean,
                    "n_perf_mode_groups": n_perf_mode_groups,
                    "n_leak_mode_groups": n_leak_mode_groups,
                    "r_eff_mean": float(np_mean(step_r_eff)),
                    "adv_mean": float(adv_arr.mean()),
                    "adv_std": float(adv_arr.std()),
                    "adv_abs_mean": float(np.abs(adv_arr).mean()),
                    "adv_max": float(adv_arr.max()),
                    "adv_min": float(adv_arr.min()),
                    "group_rleak_std_mean": float(np_mean(group_rleak_stds)),
                    "group_rperf_std_mean": float(np_mean(group_rperf_stds)),
                    "groups_low_variance": groups_low_variance,
                    "skipped_groups": skipped_groups,
                    "time/step": time.time() - t0,
                    **(opt_result.metrics or {}),
                }
                ml_logger.log_metrics(metrics, step=global_step)
                logger.info(
                    "  Step %d done in %.1fs | adv=[%.3f, %.3f] std=%.3f | "
                    "cov=%.2f perf=%.3f leak_rate=%.3f | skipped=%d",
                    global_step, metrics["time/step"],
                    metrics["adv_min"], metrics["adv_max"], metrics["adv_std"],
                    metrics["coverage_mean"], metrics["R_performance_mean"],
                    metrics["leakage_rate_mean"], skipped_groups,
                )

                if global_step in val_steps and val_set:
                    val_metrics = _run_validation(
                        val_set, sampling_client, renderer, reward_pipe,
                        sampling_params, val_log_path, global_step,
                    )
                    _append_jsonl(metrics_path, {"step": global_step, **val_metrics})
                    logger.info(
                        "  Validation at step %d: leak=%.3f hall=%.3f cov=%.2f perf=%.3f (n=%d, parse_fail=%d)",
                        global_step,
                        val_metrics["val_leakage_rate"],
                        val_metrics["val_hallucination_rate"],
                        val_metrics["val_coverage"],
                        val_metrics["val_performance"],
                        val_metrics["val_n_instances"],
                        val_metrics["val_n_parse_failed"],
                    )
                    val_leak = val_metrics["val_leakage_rate"]
                    if val_leak < best_val_leak:
                        best_val_leak = val_leak
                        # Use a step-stamped unique name so each save writes to a
                        # distinct GCS path — avoids "Resource already exists" errors.
                        best_name = f"best_s{global_step:04d}"
                        _save_checkpoint_safe(
                            checkpoint_utils, training_client,
                            name=best_name,
                            log_path=str(log_dir),
                            loop_state={"step": global_step, "val_leakage_rate": val_leak},
                        )
                        # Delete all older step-stamped best checkpoints to free GCS space.
                        _delete_prev_best_checkpoints(str(log_dir), best_name)

                global_step += 1
                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    logger.info("Reached max_steps=%d, stopping early.", cfg.max_steps)
                    break
            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                break

        _save_checkpoint_safe(
            checkpoint_utils, training_client,
            name="final",
            log_path=str(log_dir),
            loop_state={"step": global_step, "epoch": cfg.num_epochs},
        )
    except Exception:
        logger.exception("Training failed at step %d", global_step)
        raise
    finally:
        ml_logger.close()


def _sample_group_batch(
    batch: list,
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    group_size: int,
) -> list[dict[str, Any]]:
    """Sample completions for all instances in batch with parallel Tinker calls."""
    from tinker_cookbook import renderers as renderer_utils

    pending: list[tuple[Any, Any, Any]] = []
    for inst in batch:
        system, user = build_generation_prompt(inst)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        model_input = renderer.build_generation_prompt(messages)
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=group_size,
            sampling_params=sampling_params,
        )
        pending.append((inst, model_input, future))

    data: list[dict[str, Any]] = []
    for inst, model_input, future in pending:
        sampled = future.result()
        completions: list[str] = []
        tokens_g: list[list[int]] = []
        logprobs_g: list[list[float]] = []
        for seq in sampled.sequences:
            parsed_msg, _ = renderer.parse_response(seq.tokens)
            completions.append(renderer_utils.get_text_content(parsed_msg))
            tokens_g.append(seq.tokens)
            logprobs_g.append(seq.logprobs)
        data.append(
            {
                "instance": inst,
                "model_input": model_input,
                "tokens_G": tokens_g,
                "logprobs_G": logprobs_g,
                "completions": completions,
            }
        )
    return data


def _build_datums(
    sampled: dict[str, Any],
    advantages: list[float],
    tinker_types: Any,
    tensor_data_cls: Any,
    torch_mod: Any,
) -> list[Any]:
    model_input = sampled["model_input"]
    ob_len = model_input.length - 1
    out: list[Any] = []
    for tokens, logprobs, adv in zip(sampled["tokens_G"], sampled["logprobs_G"], advantages):
        full_input = model_input.append(tinker_types.EncodedTextChunk(tokens=tokens[:-1]))
        target = [0] * ob_len + tokens
        padded_lp = [0.0] * ob_len + logprobs
        padded_adv = [0.0] * ob_len + [adv] * (full_input.length - ob_len)
        out.append(
            tinker_types.Datum(
                model_input=full_input,
                loss_fn_inputs={
                    "target_tokens": tensor_data_cls.from_torch(torch_mod.tensor(target)),
                    "logprobs": tensor_data_cls.from_torch(torch_mod.tensor(padded_lp)),
                    "advantages": tensor_data_cls.from_torch(torch_mod.tensor(padded_adv)),
                },
            )
        )
    return out


def _difficulty_balanced_shuffle(
    items: list, seed: int, batch_size: int
) -> list:
    """Build epoch order so each batch has a balanced mix of difficulty tiers.

    Tiers by cutoff_date: hard (<=2021-12-31), medium (<=2023-02-28), easy (rest).
    Each batch gets ~batch_size/3 from each tier via round-robin.
    """

    tiers: list[list] = [[], [], []]
    for it in items:
        if it.cutoff_date <= "2021-12-31":
            tiers[0].append(it)
        elif it.cutoff_date <= "2023-02-28":
            tiers[1].append(it)
        else:
            tiers[2].append(it)

    rng = random.Random(seed)
    for t in tiers:
        rng.shuffle(t)

    idx = [0, 0, 0]
    n_tiers = len(tiers)
    slots_per_tier = batch_size // n_tiers
    extra = batch_size % n_tiers

    result: list = []
    while any(idx[i] < len(tiers[i]) for i in range(n_tiers)):
        batch: list = []
        for i in range(n_tiers):
            n_slots = slots_per_tier + (1 if i < extra else 0)
            for _ in range(n_slots):
                if idx[i] < len(tiers[i]):
                    batch.append(tiers[i][idx[i]])
                    idx[i] += 1
        if not batch:
            break
        rng.shuffle(batch)
        result.extend(batch)

    return result


def _delete_checkpoint_by_name(name: str, log_path: str) -> None:
    """Delete an existing named checkpoint from Tinker using paths in checkpoints.jsonl."""
    import tinker
    from .checkpoint_io import read_checkpoints_jsonl

    rows = read_checkpoints_jsonl(log_path)
    existing = [r for r in rows if r.get("name") == name]
    if not existing:
        logger.warning("No checkpoint named '%s' found in checkpoints.jsonl — cannot delete", name)
        return

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    for key in ("state_path", "sampler_path"):
        tp = existing[-1].get(key)
        if tp:
            try:
                rest_client.delete_checkpoint_from_tinker_path(tp).result()
                logger.info("Deleted existing checkpoint at %s", tp)
            except Exception:
                logger.warning("Failed to delete %s (may already be gone)", tp, exc_info=True)


def _delete_prev_best_checkpoints(log_path: str, current_name: str) -> None:
    """Delete all prior step-stamped best checkpoints except the one just saved.

    Best checkpoints are named  best_s<step>  (e.g. best_s0031).  After saving a
    new best, we delete every older best_s* entry to avoid accumulating GCS blobs.
    Uses the same _delete_checkpoint_by_name helper which tolerates already-gone paths.
    """
    from .checkpoint_io import read_checkpoints_jsonl

    rows = read_checkpoints_jsonl(log_path)
    prev_names = {
        r["name"] for r in rows
        if r.get("name", "").startswith("best_s") and r["name"] != current_name
    }
    for old_name in prev_names:
        logger.info("Deleting old best checkpoint '%s'", old_name)
        _delete_checkpoint_by_name(old_name, log_path)


def _save_checkpoint_safe(
    checkpoint_utils: Any,
    training_client: Any,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
) -> None:
    """Save a checkpoint using a name guaranteed to be unique in GCS.

    Root cause of earlier crashes: Tinker stores weights in GCS under the path
      gs://tml-tinker-prod-us-east4/<model_id>/weights/<name>
    Calling delete_checkpoint_from_tinker_path removes only the metadata record;
    the GCS blob remains.  A subsequent save_checkpoint with the same name then
    fails with "Resource already exists" at the GCS layer, which poisons the
    training session.

    Fix: callers must pass a name that is unique per save (e.g. best_s0031).
    Each unique name maps to a distinct GCS path, so save_checkpoint never
    encounters a storage conflict.  Old step-stamped checkpoints are deleted
    separately via _delete_prev_best_checkpoints after a successful save.

    The "final" checkpoint is written once at end-of-training and needs no special
    handling.
    """
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=name,
        log_path=log_path,
        kind="both",
        loop_state=loop_state,
    )


_VAL_MAX_RETRIES = 3
_VAL_RETRY_TEMPERATURE = 0.5


def _run_validation(
    val_set: list,
    sampling_client: Any,
    renderer: Any,
    reward_pipe: RewardPipeline,
    sampling_params: Any,
    val_log_path: Path,
    global_step: int,
) -> dict[str, float]:
    """Run one full validation pass with up to _VAL_MAX_RETRIES retries per parse failure.

    First attempt uses temperature=0.0 (greedy/deterministic).
    Retry attempts use temperature=_VAL_RETRY_TEMPERATURE to vary output.
    Instances that still fail to parse after all retries are logged with
    parse_failed=True and excluded from the aggregate metrics.
    """
    from tinker import types as tinker_types
    from tinker_cookbook import renderers as renderer_utils

    greedy_params = tinker_types.SamplingParams(
        max_tokens=sampling_params.max_tokens,
        temperature=0.0,
        stop=sampling_params.stop,
    )
    retry_params = tinker_types.SamplingParams(
        max_tokens=sampling_params.max_tokens,
        temperature=_VAL_RETRY_TEMPERATURE,
        stop=sampling_params.stop,
    )

    def _build_model_input(inst: Any) -> Any:
        system, user = build_generation_prompt(inst)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return renderer.build_generation_prompt(messages)

    def _sample_one(inst: Any, params: Any) -> tuple[str, Any, dict]:
        """Sample and score one completion. Returns (text, reward, claim_detail)."""
        model_input = _build_model_input(inst)
        result = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=params,
        ).result()
        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        text = renderer_utils.get_text_content(parsed_msg)
        rewards, claim_detail = reward_pipe.score_group([text], inst)
        return text, rewards[0], claim_detail

    # First pass: submit all greedy samples in parallel, then collect
    pending_first: list[tuple[Any, Any, Any]] = []  # (inst, model_input, future)
    for inst in val_set:
        model_input = _build_model_input(inst)
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=greedy_params,
        )
        pending_first.append((inst, model_input, future))

    # Collect first-pass results; retry failures sequentially
    leaks: list[float] = []
    halls: list[float] = []
    covs: list[float] = []
    perfs: list[float] = []
    n_parse_failed = 0

    for inst, _model_input, future in pending_first:
        result = future.result()
        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        text = renderer_utils.get_text_content(parsed_msg)
        rewards, claim_detail = reward_pipe.score_group([text], inst)
        r = rewards[0]

        # Retry up to _VAL_MAX_RETRIES times if parse failed
        attempt = 0
        while not r.parse_success and attempt < _VAL_MAX_RETRIES:
            attempt += 1
            logger.warning(
                "val step=%d inst=%s parse failed (attempt %d/%d), retrying",
                global_step, inst.instance_id, attempt, _VAL_MAX_RETRIES,
            )
            text, r, claim_detail = _sample_one(inst, retry_params)

        parse_failed = not r.parse_success
        if parse_failed:
            n_parse_failed += 1
            logger.warning(
                "val step=%d inst=%s: all %d attempts failed to parse — skipping from metrics",
                global_step, inst.instance_id, _VAL_MAX_RETRIES + 1,
            )
        else:
            leaks.append(r.leakage_rate)
            halls.append(r.hallucination_rate)
            covs.append(r.coverage)
            perfs.append(r.r_performance)

        _append_jsonl(val_log_path, {
            "step": global_step,
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

    if n_parse_failed:
        logger.warning("val step=%d: %d/%d instances excluded (parse failed after retries)",
                       global_step, n_parse_failed, len(val_set))

    return {
        "val_leakage_rate": float(np_mean(leaks)),
        "val_leakage_rate_std": float(_std(leaks)),
        "val_hallucination_rate": float(np_mean(halls)),
        "val_hallucination_rate_std": float(_std(halls)),
        "val_coverage": float(np_mean(covs)),
        "val_coverage_std": float(_std(covs)),
        "val_performance": float(np_mean(perfs)),
        "val_performance_std": float(_std(perfs)),
        "val_n_instances": len(leaks),
        "val_n_parse_failed": n_parse_failed,
    }


def np_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    """CLI wrapper with optional per-field overrides."""

    parser = argparse.ArgumentParser(description="Run FullExperiment GRPO training.")
    parser.add_argument("--config", required=True, help="Path to YAML config under FullExperiment.")
    parser.add_argument("--train_tasks", default=None, help="Override train_tasks in config.")
    parser.add_argument("--log_dir", default=None, help="Override log_dir in config.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num_epochs in config.")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop training after this many steps.")
    parser.add_argument("--group_size", type=int, default=None, help="Override group_size in config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size in config.")
    parser.add_argument("--num_validations", type=int, default=None, help="Override num_validations in config.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    project_root = Path(__file__).resolve().parents[2]
    load_env(project_root)
    cfg = load_config(args.config)
    if args.train_tasks is not None:
        cfg.train_tasks = args.train_tasks
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.group_size is not None:
        cfg.group_size = args.group_size
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_validations is not None:
        cfg.num_validations = args.num_validations
    run_training(cfg, project_root)


if __name__ == "__main__":
    main()

