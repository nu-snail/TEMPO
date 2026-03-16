"""Unified prompt builder and response parser for all prediction tasks.

Supports two prompt modes:
- ``temporal_hint``: includes strict temporal constraint (default for training / eval).
- ``base``: no temporal constraint (zero-shot baseline).

Usage::

    from src.shared.prompts.builder import build_prompt, parse_response

    system, user = build_prompt(instance, prompt_mode="temporal_hint")
    parsed = parse_response(completion_text, instance)
"""

from __future__ import annotations

from typing import Any

from ..data_loader import TaskInstance
from . import legal, salary, stock

_TASK_MODULES = {
    "stock": stock,
    "salary": salary,
    "legal": legal,
}

VALID_PROMPT_MODES = ("temporal_hint", "base")


def build_prompt(
    instance: TaskInstance,
    prompt_mode: str = "temporal_hint",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a TaskInstance.

    Args:
        instance: The prediction instance.
        prompt_mode: ``"temporal_hint"`` (with temporal constraint) or
            ``"base"`` (no temporal constraint).

    Raises:
        ValueError: If the task type or prompt_mode is unknown.
    """
    if prompt_mode not in VALID_PROMPT_MODES:
        raise ValueError(
            f"Unknown prompt_mode '{prompt_mode}'. Valid: {VALID_PROMPT_MODES}"
        )
    mod = _get_module(instance.task)

    if prompt_mode == "base":
        system = mod.BASE_SYSTEM_PROMPT
    else:
        system = mod.SYSTEM_PROMPT.format(cutoff_date=instance.cutoff_date)

    user = mod.build_user_prompt(instance)
    return system, user


def parse_response(text: str, instance: TaskInstance) -> dict[str, Any] | None:
    """Parse a model completion into a structured prediction dict.

    Returns None when the output cannot be parsed.
    """
    mod = _get_module(instance.task)
    return mod.parse_response(text, instance)


def build_messages(
    instance: TaskInstance,
    prompt_mode: str = "temporal_hint",
) -> list[dict[str, str]]:
    """Return a chat-messages list ready for ``llm_client.call``."""
    system, user = build_prompt(instance, prompt_mode=prompt_mode)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _get_module(task: str):
    if task not in _TASK_MODULES:
        raise ValueError(f"Unknown task '{task}'. Valid: {list(_TASK_MODULES)}")
    return _TASK_MODULES[task]
