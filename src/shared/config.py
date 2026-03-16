"""Configuration for RL-based temporal leakage prevention.

Loads API keys from environment / .env, defines provider mappings,
and exposes helper functions used by both training and evaluation.
"""

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Provider selection (overridable via env vars)
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter")
SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "openrouter")

# ---------------------------------------------------------------------------
# Model name mappings  (logical name -> provider-specific identifier)
# ---------------------------------------------------------------------------

MODEL_MAPPINGS: dict[str, dict[str, str]] = {
    "openrouter": {
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "claude-3.5-haiku": "anthropic/claude-3.5-haiku",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
    },
    "anthropic": {
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
    },
    "openai": {
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4o": "gpt-4o-2024-11-20",
    },
}

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")
TINKER_API_KEY: str = os.getenv("TINKER_API_KEY", "")

# ---------------------------------------------------------------------------
# LLM / Search default settings
# ---------------------------------------------------------------------------

LLM_SETTINGS: dict[str, Any] = {
    "temperature": 0,
    "max_tokens": 4096,
    "timeout": 120,
}

SEARCH_SETTINGS: dict[str, Any] = {
    "max_results": 5,
    "timeout": 60,
    "search_model": "perplexity/sonar",
    "search_batch_size": 50,
}

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
TRAIN_DATA_DIR: Path = DATASET_DIR / "train"
EVAL_DATA_DIR: Path = DATASET_DIR / "evaluation"

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_INFO: dict[str, dict[str, Any]] = {
    "stock": {
        "name": "Stock Return Ranking",
        "train_file": "stock_rankings/stock_rankings_train_combined.json",
        "eval_file": "stock_rankings.json",
        "format": "json",
    },
    "salary": {
        "name": "Salary Prediction",
        "train_file": "salary_prediction/salary_prediction_train_combined.csv",
        "eval_file": "nba_salaries.csv",
        "format": "csv",
    },
    "legal": {
        "name": "Legal Case Prediction",
        "train_file": "legal_cases/legal_cases_train.json",
        "eval_file": "legal_cases.json",
        "format": "json",
    },
}

VALID_TASKS = list(TASK_INFO.keys())

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_model_name(logical_name: str, provider: str | None = None) -> str:
    """Resolve a logical model name to a provider-specific identifier.

    Args:
        logical_name: e.g. "gpt-4o-mini" or a direct path like "openai/gpt-4o-mini".
        provider: Override for LLM_PROVIDER.

    Returns:
        Provider-specific model string.

    Raises:
        ValueError: If model or provider is unknown.
    """
    if provider is None:
        provider = LLM_PROVIDER
    if "/" in logical_name:
        return logical_name
    if provider not in MODEL_MAPPINGS:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {list(MODEL_MAPPINGS)}"
        )
    mappings = MODEL_MAPPINGS[provider]
    if logical_name not in mappings:
        raise ValueError(
            f"Model '{logical_name}' not in {provider} mappings. "
            f"Available: {list(mappings)}"
        )
    return mappings[logical_name]


def get_api_key(provider: str | None = None) -> str:
    """Return the API key for the given LLM provider.

    Raises:
        ValueError: If the key is empty or the provider is unknown.
    """
    if provider is None:
        provider = LLM_PROVIDER
    keys = {
        "openrouter": OPENROUTER_API_KEY,
        "anthropic": ANTHROPIC_API_KEY,
        "openai": OPENAI_API_KEY,
    }
    if provider not in keys:
        raise ValueError(f"Unknown LLM provider '{provider}'")
    key = keys[provider]
    if not key:
        raise ValueError(
            f"API key for '{provider}' not set. "
            f"Add {provider.upper()}_API_KEY to your .env file."
        )
    return key


def get_search_api_key(provider: str | None = None) -> str:
    """Return the API key for the given search provider.

    Raises:
        ValueError: If the key is empty or the provider is unknown.
    """
    if provider is None:
        provider = SEARCH_PROVIDER
    keys = {
        "perplexity": PERPLEXITY_API_KEY,
        "openrouter": OPENROUTER_API_KEY,
    }
    if provider not in keys:
        raise ValueError(f"Unknown search provider '{provider}'")
    key = keys[provider]
    if not key:
        raise ValueError(
            f"API key for search provider '{provider}' not set. "
            f"Add {provider.upper()}_API_KEY to your .env file."
        )
    return key


def validate_config() -> None:
    """Validate that all required API keys and providers are set.

    Raises:
        ValueError: On any misconfiguration.
    """
    get_api_key(LLM_PROVIDER)
    get_search_api_key(SEARCH_PROVIDER)
