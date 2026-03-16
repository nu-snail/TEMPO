"""Configuration loading for CoverageGatedExperiment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class TrainConfig:
    """Top-level training configuration."""

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

    # Reward fix parameters (Section 10 of training_degeneration_analysis.md)
    max_neg_ratio: float = 5.0
    overlong_decay: float = 0.5
    overlong_floor: float = 0.3
    max_tokens_chars: int = 32000


def load_config(path: str) -> TrainConfig:
    """Load training config from YAML file."""

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def load_env(project_root: Path) -> None:
    """Load `.env` from project root if present."""

    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


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
