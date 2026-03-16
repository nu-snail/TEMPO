"""Unified LLM client for reward computation and evaluation.

Wraps the OpenAI-compatible API (OpenRouter, OpenAI, etc.) with retry logic,
structured output support, and a convenience ``simple_call`` helper used
throughout the codebase for claim extraction and leakage judging.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAI

from . import config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """OpenAI-compatible LLM client with automatic provider resolution."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider or config.LLM_PROVIDER
        api_key = config.get_api_key(self.provider)

        base_urls = {
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": None,  # uses default
        }
        if self.provider not in base_urls:
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. "
                f"Supported: {list(base_urls)}"
            )
        url = base_urls[self.provider]
        self.client = OpenAI(base_url=url, api_key=api_key) if url else OpenAI(api_key=api_key)

    def call(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat-completion request and return a normalised result dict.

        Args:
            messages: Chat messages (system / user / assistant).
            model: Logical model name (resolved via config.get_model_name).
            temperature: Sampling temperature (default from config).
            max_tokens: Max completion tokens (default from config).
            tools: Optional tool definitions (OpenAI function-calling format).
            tool_choice: Optional tool-choice directive.

        Returns:
            Dict with keys ``content``, ``tool_calls``, ``usage``, ``finish_reason``.

        Raises:
            RuntimeError: After exhausting all retries.
        """
        provider_model = config.get_model_name(model, self.provider)
        if temperature is None:
            temperature = config.LLM_SETTINGS["temperature"]
        if max_tokens is None:
            max_tokens = config.LLM_SETTINGS["max_tokens"]

        params: dict[str, Any] = {
            "model": provider_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
            if tool_choice:
                params["tool_choice"] = tool_choice

        last_exc: BaseException | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(**params)
                msg = resp.choices[0].message
                usage = (
                    {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                    if resp.usage
                    else None
                )
                tool_calls_data = None
                if msg.tool_calls:
                    tool_calls_data = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                return {
                    "content": msg.content,
                    "tool_calls": tool_calls_data,
                    "finish_reason": resp.choices[0].finish_reason,
                    "usage": usage,
                }
            except Exception as exc:  # noqa: BLE001 – broad catch with re-raise below
                last_exc = exc
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE**attempt
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %ds...",
                        attempt, MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {MAX_RETRIES} attempts "
            f"(model={model}, provider={self.provider})"
        ) from last_exc


# ---------------------------------------------------------------------------
# Module-level singleton and convenience wrappers
# ---------------------------------------------------------------------------

_default_client: LLMClient | None = None


def get_client(provider: str | None = None) -> LLMClient:
    """Return (and lazily create) the module-level LLM client singleton."""
    global _default_client
    if _default_client is None or (provider and provider != _default_client.provider):
        _default_client = LLMClient(provider=provider)
    return _default_client


def call_llm(
    messages: list[dict[str, str]],
    model: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper around the default client's ``call`` method."""
    return get_client().call(messages, model, **kwargs)


def simple_call(
    prompt: str,
    model: str,
    *,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> str:
    """Fire-and-forget text completion — returns the content string only.

    Args:
        prompt: User message.
        model: Logical model name.
        system_prompt: Optional system message.

    Returns:
        The assistant's text response (empty string if none).
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    result = call_llm(messages, model, **kwargs)
    return result["content"] or ""
