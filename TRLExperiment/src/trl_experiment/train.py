"""GRPO training via TRL GRPOTrainer with vLLM.

Mirrors CoverageGatedExperiment/src/cov_gated_experiment/train.py in CLI,
metrics, and output file formats, but replaces Tinker with TRL + vLLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .checkpoint_io import save_checkpoint
from .config import load_env, load_train_config, require_api_keys
from .data import load_train_instances, split_train_val
from .llm import OpenRouterClient
from .prompts import build_generation_prompt
from .reward_adapter import build_reward_fn
from .reward_pipeline import RewardPipeline

logger = logging.getLogger(__name__)


def _build_dataset(instances: list, task: str) -> Dataset:
    """Convert TaskInstance list to HuggingFace Dataset for GRPOTrainer.

    Each row has a 'prompt' column (list of message dicts) plus metadata
    columns that TRL passes through to the reward function via **kwargs.
    """
    rows: list[dict[str, Any]] = []
    for inst in instances:
        system, user = build_generation_prompt(inst)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        rows.append({
            "prompt": messages,
            "task": inst.task,
            "instance_id": inst.instance_id,
            "cutoff_date": inst.cutoff_date,
            "ground_truth_json": json.dumps(inst.ground_truth),
        })
    return Dataset.from_list(rows)


class MetricsCallback(TrainerCallback):
    """Logs per-step metrics to metrics.jsonl matching CoverageGatedExperiment format."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.metrics_path = log_dir / "metrics.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step, **logs}
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


class DebugCallback(TrainerCallback):
    """Prints detailed per-step diagnostics for debugging training dynamics."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss", "N/A")
        grad = logs.get("grad_norm", "N/A")
        reward_mean = logs.get("reward", "N/A")
        reward_std = logs.get("reward_std", "N/A")
        entropy = logs.get("entropy", "N/A")
        logp_diff_mean = logs.get("sampling/sampling_logp_difference/mean", "N/A")
        logp_diff_max = logs.get("sampling/sampling_logp_difference/max", "N/A")
        is_mean = logs.get("sampling/importance_sampling_ratio/mean", "N/A")
        is_min = logs.get("sampling/importance_sampling_ratio/min", "N/A")
        is_max = logs.get("sampling/importance_sampling_ratio/max", "N/A")
        clip_low = logs.get("clip_ratio/low_mean", "N/A")
        clip_high = logs.get("clip_ratio/high_mean", "N/A")
        comp_mean_len = logs.get("completions/mean_length", "N/A")

        logger.info(
            "\n========== STEP %d DEBUG ==========\n"
            "  loss=%.6e  grad_norm=%.6e\n"
            "  reward: mean=%.4f  std=%.4f\n"
            "  entropy=%.4f\n"
            "  logp_diff: mean=%.4f  max=%.4f\n"
            "  importance_sampling: min=%.4e  mean=%.4e  max=%.4e\n"
            "  clip_ratio: low=%.4f  high=%.4f\n"
            "  completion_mean_length=%.1f\n"
            "===================================",
            step,
            float(loss) if loss != "N/A" else 0,
            float(grad) if grad != "N/A" else 0,
            float(reward_mean) if reward_mean != "N/A" else 0,
            float(reward_std) if reward_std != "N/A" else 0,
            float(entropy) if entropy != "N/A" else 0,
            float(logp_diff_mean) if logp_diff_mean != "N/A" else 0,
            float(logp_diff_max) if logp_diff_max != "N/A" else 0,
            float(is_min) if is_min != "N/A" else 0,
            float(is_mean) if is_mean != "N/A" else 0,
            float(is_max) if is_max != "N/A" else 0,
            float(clip_low) if clip_low != "N/A" else 0,
            float(clip_high) if clip_high != "N/A" else 0,
            float(comp_mean_len) if comp_mean_len != "N/A" else 0,
        )


def run_training(cfg: TrainConfig, project_root: Path) -> None:
    """Execute the full TRL GRPO training pipeline."""
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    tasks = [t.strip() for t in cfg.train_tasks.split(",") if t.strip()]
    if len(tasks) != 1:
        raise ValueError(
            f"TRL training expects exactly one task per run, got: {tasks}. "
            "Pass --train_tasks with a single task."
        )
    task = tasks[0]

    all_train = load_train_instances(project_root / cfg.data_root, tasks)
    train_set, val_set = split_train_val(all_train, cfg.val_fraction, cfg.data_seed)
    logger.info("Loaded %d train, %d val instances for task '%s'", len(train_set), len(val_set), task)

    router_key = require_api_keys()
    judge_client = OpenRouterClient(router_key)
    pipeline = RewardPipeline(cfg, judge_client)

    step_log_path = log_dir / "step_log.jsonl"
    reward_fn = build_reward_fn(pipeline, cfg.cov_target, cfg.cov_floor, step_log_path)
    dataset = _build_dataset(train_set, task)

    total_steps = math.ceil(len(train_set) / cfg.batch_size) * cfg.num_epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.effective_lora_alpha,
        target_modules=cfg.lora_target_modules,
        task_type="CAUSAL_LM",
    )

    num_gpus = torch.cuda.device_count()
    training_args = GRPOConfig(
        output_dir=str(log_dir / "trl_output"),
        num_generations=cfg.group_size,
        max_completion_length=cfg.max_tokens,
        temperature=cfg.temperature,
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps if cfg.max_steps else -1,
        learning_rate=cfg.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        epsilon=cfg.clip_low_epsilon,
        epsilon_high=cfg.clip_high_epsilon,
        chat_template_kwargs=cfg.chat_template_kwargs or None,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=num_gpus,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        log_level="info",
    )

    trainer_ref: list[GRPOTrainer] = []
    metrics_cb = MetricsCallback(log_dir)
    debug_cb = DebugCallback()

    trainer = GRPOTrainer(
        model=cfg.model_name,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[metrics_cb, debug_cb],
    )
    trainer_ref.append(trainer)

    logger.info(
        "Starting GRPO training: %d steps, task=%s, model=%s, "
        "epsilon=%.2f, epsilon_high=%.2f, adam_beta2=%.3f",
        total_steps, task, cfg.model_name,
        cfg.clip_low_epsilon, cfg.clip_high_epsilon, 0.95,
    )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info("Training complete in %.1fs", elapsed)

    save_checkpoint(
        trainer, "final", log_dir,
        loop_state={"step": total_steps, "epoch": cfg.num_epochs},
    )
    logger.info("Final checkpoint saved to %s/final", log_dir)


def main() -> None:
    """CLI wrapper with optional per-field overrides."""
    parser = argparse.ArgumentParser(description="TRL GRPO training.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--train_tasks", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_validations", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None, help="Override max_tokens in config.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    project_root = Path(__file__).resolve().parents[2]
    load_env(project_root)
    cfg = load_train_config(args.config)

    for field in ("train_tasks", "log_dir", "num_epochs", "max_steps",
                  "group_size", "batch_size", "num_validations", "max_tokens"):
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)

    run_training(cfg, project_root)


if __name__ == "__main__":
    main()
