"""GRPO training loop for temporal leakage prevention.

Self-contained training script using the Tinker API.  Follows the
``rl_loop.py`` pattern from tinker-cookbook: sample G completions per
prompt, compute MO-GRPO advantages, then do an importance-sampling
policy gradient update.

Usage:
    uv run python -m src.training.train --config configs/train_qwen.yaml

The script reads a YAML config, builds the dataset, initialises the
Tinker training client, and runs the main loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from tinker import types as tinker_types
from tinker.types.tensor_data import TensorData

from src.shared import config as app_config
from src.shared.data_loader import load_all_train, make_train_val_split
from src.shared.progress import ProgressBar
from src.shared.prompts.builder import build_prompt, parse_response
from src.training.reward.lambda_schedule import LambdaSchedule
from src.training.reward.reward_pipeline import (
    CompletionReward,
    RewardPipeline,
    mo_grpo_advantages,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters for a training run."""

    # Model
    model_name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32

    # GRPO (Section 3.2 of EXPERIMENT_PLAN.md)
    group_size: int = 8
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 1e-6
    max_tokens: int = 4096
    soft_length_penalty_start: int = 3072
    temperature: float = 0.8
    kl_coeff: float = 0.05      # beta: KL penalty coefficient
    tis_cap: float = 5.0        # rho: truncated importance sampling cap

    # Reward mode: "full" | "leakage_only" | "performance_only"
    reward_mode: str = "full"

    # Loss function (DAPO clip-higher: thresholds = 1 +/- epsilon)
    loss_fn: str = "ppo"
    clip_low: float = 0.8       # 1 - epsilon_low = 1 - 0.2
    clip_high: float = 1.28     # 1 + epsilon_high = 1 + 0.28

    # Lambda schedule (only used when reward_mode == "full")
    lambda_min: float = 0.1
    lambda_max: float = 0.3
    theta_high: float = 0.3
    theta_low: float = 0.1
    ema_alpha: float = 0.1

    # Judge LLM for claim extraction + leakage detection in reward pipeline
    judge_model: str = "google/gemini-2.5-flash"
    use_search: bool = True
    use_llm_judge: bool = True
    search_cache_path: str = "results/search_cache.json"

    # Data
    tasks: str = "stock,salary,legal"  # comma-separated subset of tasks to train on
    val_fraction: float = 0.1
    data_seed: int = 42

    # Checkpointing
    log_path: str = "results/train"
    save_every: int = 10
    eval_every: int = 5

    # Dynamic sampling: skip groups where all advantages are zero
    dynamic_sampling: bool = True

    # Parallelism for reward computation (I/O-bound API calls)
    reward_workers: int = 8  # threads per instance (for G completions)
    batch_workers: int = 4   # concurrent instances within a batch

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _run_validation(
    sampling_client: Any,
    val_instances: list,
    reward_pipe: RewardPipeline,
    renderer: Any,
    sampling_params: Any,
) -> dict[str, float]:
    """Compute mean OLR on the full validation set.

    Samples 1 greedy completion per instance (G=1, temperature=0)
    for deterministic, reproducible evaluation.

    Args:
        sampling_client: Tinker sampling client with current weights.
        val_instances: Full validation set.
        reward_pipe: Reward pipeline for scoring.
        renderer: Model renderer for prompt building.
        sampling_params: Base sampling parameters (temperature is overridden to 0).

    Returns:
        Dict with val/olr_mean, val/r_leakage, val/r_performance, val/n.
    """
    from tinker_cookbook import renderers as _renderers

    if not val_instances:
        return {"val/olr_mean": 0.5, "val/r_leakage": 0.5, "val/r_performance": 0.0, "val/n": 0}

    val_sampling_params = tinker_types.SamplingParams(
        max_tokens=sampling_params.max_tokens,
        temperature=0.0,
        stop=sampling_params.stop,
    )

    olrs: list[float] = []
    r_leaks: list[float] = []
    r_perfs: list[float] = []

    for inst in val_instances:
        system, user = build_prompt(inst)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        model_input = renderer.build_generation_prompt(messages)

        try:
            result = sampling_client.sample(
                prompt=model_input, num_samples=1, sampling_params=val_sampling_params,
            ).result()
        except Exception as exc:
            logger.warning("Validation sampling failed for %s: %s", inst.instance_id, exc)
            continue

        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        text = _renderers.get_text_content(parsed_msg)

        try:
            reward = reward_pipe.score(text, inst)
            olrs.append(reward.olr)
            r_leaks.append(reward.r_leakage)
            r_perfs.append(reward.r_performance)
        except Exception as exc:
            logger.warning("Validation scoring failed for %s: %s", inst.instance_id, exc)

    if not olrs:
        return {"val/olr_mean": 0.5, "val/r_leakage": 0.5, "val/r_performance": 0.0, "val/n": 0}

    return {
        "val/olr_mean": sum(olrs) / len(olrs),
        "val/r_leakage": sum(r_leaks) / len(r_leaks),
        "val/r_performance": sum(r_perfs) / len(r_perfs),
        "val/n": len(olrs),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> None:
    """Main training entry point."""
    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.utils import ml_log

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    log_dir = Path(cfg.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility
    from dataclasses import asdict
    with open(log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    ml_logger = ml_log.setup_logging(
        log_dir=str(log_dir),
        wandb_project=None,
        wandb_name=None,
        config=cfg,
        do_configure_logging_module=True,
    )

    # --- Data ---
    task_list = [t.strip() for t in cfg.tasks.split(",")]
    logger.info("Loading training data for tasks: %s", task_list)
    all_train = load_all_train()
    all_train = [inst for inst in all_train if inst.task in task_list]
    logger.info("Filtered to %d instances across %s", len(all_train), task_list)
    split = make_train_val_split(all_train, val_fraction=cfg.val_fraction, seed=cfg.data_seed)
    logger.info("Train: %d, Val: %d, Meta: %s", len(split.train), len(split.val), split.meta)

    # --- Tinker client ---
    logger.info("Initialising Tinker training client for %s...", cfg.model_name)
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=cfg.model_name, rank=cfg.lora_rank,
    )
    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # --- Reward pipeline ---
    reward_pipe = RewardPipeline(
        extractor_model=cfg.judge_model,
        detector_model=cfg.judge_model,
        use_search=cfg.use_search,
        use_llm_judge=cfg.use_llm_judge,
        cache_path=cfg.search_cache_path,
    )

    # --- Lambda schedule ---
    total_steps = math.ceil(len(split.train) / cfg.batch_size) * cfg.num_epochs
    lam_sched = LambdaSchedule(
        total_steps=total_steps,
        lambda_min=cfg.lambda_min,
        lambda_max=cfg.lambda_max,
        theta_high=cfg.theta_high,
        theta_low=cfg.theta_low,
        ema_alpha=cfg.ema_alpha,
    )

    sampling_params = tinker_types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = tinker_types.AdamParams(
        learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    loss_fn_config: dict[str, Any] = {}
    if cfg.loss_fn == "ppo":
        loss_fn_config = {"clip_low_threshold": cfg.clip_low, "clip_high_threshold": cfg.clip_high}

    # --- Metrics log (JSONL file for post-hoc analysis / plotting) ---
    metrics_log_path = log_dir / "metrics.jsonl"
    reward_detail_path = log_dir / "reward_details.jsonl"

    def _append_jsonl(path: Path, obj: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")

    # --- Main loop ---
    global_step = 0
    val_olr: float | None = None
    best_val_score: float = -float("inf")
    best_val_step: int = -1
    task_tag = cfg.tasks.replace(",", "+")
    print(f"\n{'='*60}", flush=True)
    print(f"  GRPO Training — {cfg.model_name}", flush=True)
    print(f"  Tasks: {cfg.tasks} | Mode: {cfg.reward_mode}", flush=True)
    print(f"  Steps/epoch: {math.ceil(len(split.train) / cfg.batch_size)} | "
          f"Total steps: {total_steps} | Epochs: {cfg.num_epochs}", flush=True)
    print(f"  Saves: {cfg.log_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    pb = ProgressBar(total=total_steps, prefix=f"Train[{task_tag}]")

    for epoch in range(cfg.num_epochs):
        instances = list(split.train)
        random.Random(cfg.data_seed + epoch).shuffle(instances)

        for batch_start in range(0, len(instances), cfg.batch_size):
            batch = instances[batch_start : batch_start + cfg.batch_size]
            if not batch:
                continue

            t0 = time.time()
            metrics: dict[str, float] = {
                "progress/epoch": epoch,
                "progress/step": global_step,
                "progress/done_frac": global_step / max(total_steps, 1),
                "optim/lr": cfg.learning_rate,
            }

            # ============================================================
            # Phase A: Sample G completions per instance (sequential —
            #   must use the same sampling_client snapshot)
            # ============================================================
            sampling_client = training_client.save_weights_and_get_sampling_client()

            sample_data: list[dict[str, Any]] = []
            t_sample = time.time()
            for inst in batch:
                system, user = build_prompt(inst)
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                model_input = renderer.build_generation_prompt(messages)

                sample_result = sampling_client.sample(
                    prompt=model_input,
                    num_samples=cfg.group_size,
                    sampling_params=sampling_params,
                ).result()

                completions: list[str] = []
                tokens_G: list[list[int]] = []
                logprobs_G: list[list[float]] = []

                for seq in sample_result.sequences:
                    parsed_msg, _ = renderer.parse_response(seq.tokens)
                    text = renderers.get_text_content(parsed_msg)
                    completions.append(text)
                    tokens_G.append(seq.tokens)
                    logprobs_G.append(seq.logprobs)

                sample_data.append({
                    "inst": inst,
                    "model_input": model_input,
                    "completions": completions,
                    "tokens_G": tokens_G,
                    "logprobs_G": logprobs_G,
                    "sequences": sample_result.sequences,
                })

            logger.debug("Step %d: sampled %d instances in %.1fs",
                         global_step, len(batch), time.time() - t_sample)

            # ============================================================
            # Phase B: Compute rewards in parallel across instances
            # ============================================================
            def _compute_rewards_for_instance(sd: dict) -> dict:
                inst = sd["inst"]
                comps = sd["completions"]
                if cfg.reward_mode == "performance_only":
                    return {"rewards": reward_pipe.score_group_performance_only(comps, inst)}
                return {"rewards": reward_pipe.score_group_batched(
                    comps, inst, max_workers=cfg.reward_workers,
                )}

            t_reward = time.time()
            reward_results: list[list[CompletionReward] | None] = [None] * len(sample_data)

            with ThreadPoolExecutor(max_workers=cfg.batch_workers) as pool:
                futures = {
                    pool.submit(_compute_rewards_for_instance, sd): i
                    for i, sd in enumerate(sample_data)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        reward_results[idx] = fut.result()["rewards"]
                    except Exception as exc:
                        logger.warning("Reward computation failed for instance %d: %s", idx, exc)
                        g = cfg.group_size
                        reward_results[idx] = [
                            CompletionReward(
                                r_leakage=0.5, r_performance=0.0, olr=0.5,
                                num_claims=0, num_leaked=0, parse_success=False,
                            )
                        ] * g

            logger.debug("Step %d: rewards computed in %.1fs",
                         global_step, time.time() - t_reward)

            # ============================================================
            # Phase C: Build datums (advantages, overlong filtering)
            # ============================================================
            if cfg.reward_mode == "leakage_only":
                lambda_t = 0.0
            elif cfg.reward_mode == "performance_only":
                lambda_t = 1.0
            else:
                lambda_t = lam_sched.get_lambda(global_step, val_olr=val_olr)

            all_datums: list[tinker_types.Datum] = []
            batch_rewards: list[float] = []
            batch_olrs: list[float] = []
            batch_r_leaks: list[float] = []
            batch_r_perfs: list[float] = []
            batch_n_claims: list[int] = []
            batch_n_leaked: list[int] = []
            batch_resp_lens: list[int] = []
            batch_parse_ok: list[bool] = []
            skipped = 0

            for sd, rewards in zip(sample_data, reward_results):
                if rewards is None:
                    skipped += 1
                    continue

                # DAPO overlong filtering: exclude truncated completions
                # from advantage computation and gradient update entirely
                valid_idx = [
                    i for i, seq in enumerate(sd["sequences"])
                    if len(seq.tokens) < cfg.max_tokens
                ]
                if not valid_idx:
                    skipped += 1
                    continue

                valid_rewards = [rewards[i] for i in valid_idx]
                advantages = mo_grpo_advantages(valid_rewards, lambda_t)

                if cfg.dynamic_sampling and all(abs(a) < 1e-8 for a in advantages):
                    skipped += 1
                    continue

                # Soft length penalty: linearly decay advantage for
                # completions between soft_length_penalty_start and max_tokens
                if cfg.soft_length_penalty_start > 0:
                    penalty_range = cfg.max_tokens - cfg.soft_length_penalty_start
                    if penalty_range > 0:
                        for k, vi in enumerate(valid_idx):
                            seq_len = len(sd["sequences"][vi].tokens)
                            if seq_len > cfg.soft_length_penalty_start:
                                frac = (seq_len - cfg.soft_length_penalty_start) / penalty_range
                                advantages[k] *= max(0.0, 1.0 - frac)

                # Metrics: report over ALL G completions for monitoring
                g = len(rewards)
                mean_olr = sum(r.olr for r in rewards) / g
                mean_r_leak = sum(r.r_leakage for r in rewards) / g
                mean_r_perf = sum(r.r_performance for r in rewards) / g
                mean_total = (mean_r_leak + lambda_t * mean_r_perf) / (1.0 + lambda_t)
                batch_rewards.append(mean_total)
                batch_olrs.append(mean_olr)
                batch_r_leaks.append(mean_r_leak)
                batch_r_perfs.append(mean_r_perf)
                batch_n_claims.extend(r.num_claims for r in rewards)
                batch_n_leaked.extend(r.num_leaked for r in rewards)
                batch_resp_lens.extend(len(t) for t in sd["tokens_G"])
                batch_parse_ok.extend(r.parse_success for r in rewards)

                # Build datums ONLY for valid (non-overlong) completions
                model_input = sd["model_input"]
                ob_len = model_input.length - 1
                for vi, adv in zip(valid_idx, advantages):
                    tok = sd["tokens_G"][vi]
                    lp = sd["logprobs_G"][vi]
                    full_input = model_input.append(
                        tinker_types.EncodedTextChunk(tokens=tok[:-1])
                    )
                    target = [0] * ob_len + tok
                    padded_lp = [0.0] * ob_len + lp
                    padded_adv = [0.0] * ob_len + [adv] * (full_input.length - ob_len)

                    all_datums.append(tinker_types.Datum(
                        model_input=full_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_lp)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_adv)),
                        },
                    ))

            if not all_datums:
                logger.warning("Step %d: all groups filtered by dynamic sampling", global_step)
                global_step += 1
                continue

            # Step 2: Forward-backward + optimizer step
            fwd_future = training_client.forward_backward(
                all_datums, loss_fn=cfg.loss_fn,
                **({"loss_fn_config": loss_fn_config} if loss_fn_config else {}),
            )
            opt_future = training_client.optim_step(adam_params)
            fwd_result = fwd_future.result()
            opt_result = opt_future.result()

            if opt_result.metrics:
                metrics.update(opt_result.metrics)

            # Logging — per-component + health metrics (Section 3.3)
            n_groups = max(len(batch_rewards), 1)
            n_completions = max(len(batch_resp_lens), 1)
            metrics["time/step"] = time.time() - t0
            metrics["reward/total"] = sum(batch_rewards) / n_groups
            metrics["reward/r_leakage"] = sum(batch_r_leaks) / n_groups
            metrics["reward/r_performance"] = sum(batch_r_perfs) / n_groups
            metrics["reward/olr_mean"] = sum(batch_olrs) / n_groups
            metrics["reward/lambda"] = lambda_t
            metrics["training/num_datums"] = len(all_datums)
            metrics["training/skipped_groups"] = skipped
            metrics["training/dynamic_skip_rate"] = skipped / max(len(batch), 1)
            metrics["claims/mean_total"] = sum(batch_n_claims) / n_completions
            metrics["claims/mean_leaked"] = sum(batch_n_leaked) / n_completions
            metrics["generation/mean_length"] = sum(batch_resp_lens) / n_completions
            metrics["generation/parse_rate"] = sum(batch_parse_ok) / n_completions

            ml_logger.log_metrics(metrics, step=global_step)

            # Save per-completion reward breakdown at eval intervals
            detail_interval = cfg.eval_every if cfg.eval_every > 0 else 10
            if global_step % detail_interval == 0:
                for sd, rewards in zip(sample_data, reward_results):
                    if rewards is None:
                        continue
                    for j, r in enumerate(rewards):
                        _append_jsonl(reward_detail_path, {
                            "step": global_step,
                            "instance_id": sd["inst"].instance_id,
                            "task": sd["inst"].task,
                            "completion_idx": j,
                            "r_leakage": r.r_leakage,
                            "r_performance": r.r_performance,
                            "olr": r.olr,
                            "num_claims": r.num_claims,
                            "num_leaked": r.num_leaked,
                            "parse_success": r.parse_success,
                        })

            pb.update(
                global_step + 1,
                step_time=metrics["time/step"],
                ep=f"{epoch+1}/{cfg.num_epochs}",
                R=metrics["reward/total"],
                OLR=metrics["reward/olr_mean"],
                lam=lambda_t,
                skip=f"{skipped}/{len(batch)}",
            )

            # Periodic checkpoint (Tinker storage is expensive — disable with save_every=0)
            if cfg.save_every > 0 and global_step % cfg.save_every == 0 and global_step > 0:
                from tinker_cookbook import checkpoint_utils
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"step_{global_step:06d}",
                    log_path=cfg.log_path,
                    kind="state",
                    loop_state={"step": global_step, "epoch": epoch},
                )

            # Periodic validation for adaptive lambda + overfitting monitoring
            if cfg.eval_every > 0 and global_step % cfg.eval_every == 0 and split.val:
                t_val = time.time()
                val_metrics = _run_validation(
                    sampling_client=sampling_client,
                    val_instances=split.val,
                    reward_pipe=reward_pipe,
                    renderer=renderer,
                    sampling_params=sampling_params,
                )
                val_olr = val_metrics["val/olr_mean"]
                val_metrics["val/time"] = time.time() - t_val
                _append_jsonl(metrics_log_path, {"step": global_step, **val_metrics})
                reward_pipe.save_cache()
                val_score = val_metrics["val/r_leakage"]
                is_best = val_score > best_val_score
                best_tag = ""
                if is_best:
                    best_val_score = val_score
                    best_val_step = global_step
                    best_tag = " ★ NEW BEST"
                    from tinker_cookbook import checkpoint_utils as _ckpt
                    _ckpt.save_checkpoint(
                        training_client=training_client,
                        name="best",
                        log_path=cfg.log_path,
                        kind="both",
                        loop_state={
                            "step": global_step, "epoch": epoch,
                            "val_score": val_score,
                            "val_olr": val_olr,
                        },
                    )

                print(
                    f"  VAL step={global_step} | OLR={val_olr:.3f} "
                    f"r_leak={val_metrics['val/r_leakage']:.3f} "
                    f"r_perf={val_metrics['val/r_performance']:.3f} "
                    f"n={val_metrics['val/n']:.0f} "
                    f"({val_metrics['val/time']:.0f}s)"
                    f"{best_tag}",
                    flush=True,
                )

            global_step += 1

    # Final save
    pb.finish(R=metrics.get("reward/total", 0), OLR=metrics.get("reward/olr_mean", 0))
    if best_val_step >= 0:
        print(f"\n  Best model was at step {best_val_step} "
              f"(val_score={best_val_score:.4f}), saved to {cfg.log_path}/best",
              flush=True)
    print(f"  Saving final checkpoint to {cfg.log_path}/final ...", flush=True)
    from tinker_cookbook import checkpoint_utils
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=cfg.log_path,
        kind="both",
        loop_state={"step": global_step, "epoch": cfg.num_epochs},
    )
    reward_pipe.save_cache()
    ml_logger.close()
    print(f"  Training complete. {global_step} steps, checkpoint: {cfg.log_path}/final\n", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for temporal leakage prevention")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    # Allow CLI overrides for any config field
    for name, f in TrainConfig.__dataclass_fields__.items():
        parser.add_argument(f"--{name}", type=type(f.default) if f.default is not None else str, default=None)
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    # Apply CLI overrides
    for name in TrainConfig.__dataclass_fields__:
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, val)

    logger.info("Config: %s", cfg)
    run_training(cfg)


if __name__ == "__main__":
    main()
