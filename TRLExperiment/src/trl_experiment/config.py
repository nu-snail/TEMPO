"""Configuration loading for TRLExperiment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class TrainConfig:
    """Training configuration for TRL GRPOTrainer.

    Field names and defaults match CoverageGatedExperiment/config.py exactly
    except where TRL requires translation (clip_low/clip_high -> epsilon).
    """

    model_name: str
    lora_rank: int
    group_size: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    max_tokens: int
    temperature: float

    # GRPO clipping (matching CoverageGated: clip_low=0.9, clip_high=2.0)
    clip_low: float
    clip_high: float

    # Reward hyperparameters (unchanged from CoverageGatedExperiment)
    w_leak_start: float
    w_leak_end: float
    lambda_base: float
    format_margin: float
    z_clip: float
    w_perf_clean: float
    cov_target: float
    cov_floor: float
    w_cov_leak: float

    # Data
    train_tasks: str
    val_fraction: float
    data_seed: int

    # External models
    judge_model: str
    search_model: str

    # Logging / output
    num_validations: int
    log_dir: str
    data_root: str
    concurrent_workers: int = 4
    max_steps: int | None = None

    # LoRA config
    lora_alpha: int | None = None
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Model-specific chat template kwargs passed to GRPOConfig.
    # Qwen: {"enable_thinking": false}
    # gpt-oss: {"reasoning_effort": "none"}
    chat_template_kwargs: dict = field(default_factory=dict)

    # Reward fix parameters (kept for composite reward compatibility)
    max_neg_ratio: float = 5.0
    overlong_decay: float = 0.5
    overlong_floor: float = 0.3
    max_tokens_chars: int = 32000

    @property
    def clip_low_epsilon(self) -> float:
        """Convert clip_low to TRL epsilon: clip_low=0.9 -> epsilon=0.1."""
        return 1.0 - self.clip_low

    @property
    def clip_high_epsilon(self) -> float:
        """Convert clip_high to TRL epsilon_high: clip_high=2.0 -> epsilon_high=1.0."""
        return self.clip_high - 1.0

    @property
    def effective_lora_alpha(self) -> int:
        """Return lora_alpha, defaulting to 2*lora_rank if not set."""
        if self.lora_alpha is not None:
            return self.lora_alpha
        return 2 * self.lora_rank


@dataclass
class EvalConfig:
    """Evaluation configuration for vLLM offline inference."""

    model_name: str
    temperature: float
    max_tokens: int
    judge_model: str
    search_model: str
    use_search: bool
    eval_tasks: str
    data_root: str
    search_cache_path: str
    concurrent_workers: int = 4
    repetition_penalty: float = 1.0


def load_train_config(path: str) -> TrainConfig:
    """Load training config from YAML, ignoring unknown keys."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    valid = {k: v for k, v in raw.items() if k in TrainConfig.__dataclass_fields__}
    return TrainConfig(**valid)


def load_eval_config(path: str) -> EvalConfig:
    """Load evaluation config from YAML, ignoring unknown keys."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    valid = {k: v for k, v in raw.items() if k in EvalConfig.__dataclass_fields__}
    return EvalConfig(**valid)


def load_env(project_root: Path) -> None:
    """Load .env from project root if present."""
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)


def require_api_keys() -> str:
    """Read mandatory OpenRouter API key from environment.

    Returns:
        The OpenRouter API key string.

    Raises:
        ValueError: If the key is missing.
    """
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError("Missing OPENROUTER_API_KEY in .env.")
    return key
