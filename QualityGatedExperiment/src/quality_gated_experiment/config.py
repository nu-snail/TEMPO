"""Configuration loading for QualityGatedExperiment (reward v4)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class TrainConfig:
    """Top-level training configuration (v4)."""

    model_name: str
    lora_rank: int
    group_size: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    max_tokens: int
    temperature: float
    clip_low: float
    clip_high: float
    w_leak_start: float
    w_leak_end: float
    lambda_base: float
    format_margin: float
    z_clip: float
    w_perf_clean: float
    cov_target: float
    cov_floor: float
    w_cov_leak: float
    train_tasks: str
    val_fraction: float
    data_seed: int
    judge_model: str
    search_model: str
    num_validations: int
    log_dir: str
    data_root: str
    concurrent_workers: int = 4
    max_steps: int | None = None
    renderer_name: str | None = None

    # Format penalty parameters (unchanged from v3)
    max_neg_ratio: float = 5.0
    overlong_decay: float = 0.5
    overlong_floor: float = 0.3
    max_tokens_chars: int = 32000

    # --- v4: Quality gate parameters ---
    hall_floor: float = 0.3
    ev_target: int = 8
    reason_target: int = 120
    w_quality_leak: float = 0.02

    # --- v4: Loss function ---
    loss_fn: str = "ppo"
    loss_fn_config: dict[str, Any] = field(default_factory=dict)

    # --- v4: KL divergence regularization ---
    kl_penalty_coef: float = 0.05
    kl_discount_factor: float = 0.0

    # --- v5: Gate shape selection (v4 backward compat) ---
    gate_shape: str = "linear"  # "linear" = v4 original, "trapezoidal" = v5

    # --- v5: Trapezoidal gate parameters (used when gate_shape="trapezoidal") ---
    ev_cap: int = 15
    ev_decay: float = 0.05
    ev_floor: float = 0.0
    reason_cap: int = 250
    reason_decay: float = 0.002
    reason_floor: float = 0.0

    # --- Rebuttal ablations: reward-design variant ---
    # two_mode (default) | scalar_mix | leakage_only | performance_only
    reward_variant: str = "two_mode"
    scalar_alpha: float = 0.5  # weight on the leakage term in scalar_mix


def load_config(path: str) -> TrainConfig:
    """Load training config from YAML file.

    Fields not present in the YAML fall back to dataclass defaults.
    Extra YAML keys (e.g. for loss_fn_config) are correctly handled.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    valid_fields = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return TrainConfig(**filtered)


def load_env(project_root: Path) -> None:
    """Load `.env` from project root or its parent (repo root) if present."""
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    parent_env = project_root.parent / ".env"
    if parent_env.exists():
        load_dotenv(parent_env, override=True)


def require_api_keys() -> tuple[str, str]:
    """Read mandatory API keys from environment.

    Returns:
        Tuple of `(openrouter_api_key, tinker_api_key)`.

    Raises:
        ValueError: If required keys are missing.
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    tinker_api_key = os.getenv("TINKER_API_KEY", "").strip()
    if not openrouter_api_key:
        raise ValueError("Missing OPENROUTER_API_KEY in .env.")
    if not tinker_api_key:
        raise ValueError("Missing TINKER_API_KEY in .env.")
    return openrouter_api_key, tinker_api_key


def get_perplexity_api_key() -> str | None:
    """Read optional Perplexity API key from environment.

    Returns:
        The API key string, or None if not set.
    """
    key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    return key or None
