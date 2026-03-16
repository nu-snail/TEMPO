"""OpenRouter client helpers for judge and search calls."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SEC = 2.0
SEARCH_CONCURRENCY = 8


@dataclass(frozen=True)
class UsageStat:
    """Token usage accounting."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenRouterClient:
    """Thin OpenRouter chat client with structured-output support."""

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def call_text(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 3072,
        response_schema: type[BaseModel] | None = None,
    ) -> tuple[str, UsageStat | None]:
        """Run a single text call with optional JSON-schema constraint."""

        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": response_schema.model_json_schema(),
                },
            }
        return self._execute(params)

    def call_search_single(self, model: str, query: str) -> tuple[str, UsageStat | None]:
        """Run one search query via Sonar and return the full response text."""
        return self._execute({
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.1,
            "max_tokens": 512,
        })

    def call_search_batch(
        self, model: str, queries: list[str], cutoff_date: str
    ) -> tuple[dict[int, str], UsageStat | None]:
        """Search claims via individual parallel Sonar calls (one query per API call).

        Sonar performs one web search per request, so batching multiple claims
        into a single call only searches for the first one. Individual calls
        parallelised with a thread pool are both faster and more reliable.
        """
        if not queries:
            return {}, None

        results: dict[int, str] = {}
        total_usage = _MutableUsage()

        def _search_one(idx: int, query: str) -> tuple[int, str]:
            text, usage = self.call_search_single(model, query)
            if usage:
                total_usage.add(usage)
            return idx, text

        with ThreadPoolExecutor(max_workers=SEARCH_CONCURRENCY) as pool:
            futures = {pool.submit(_search_one, i, q): i for i, q in enumerate(queries)}
            for future in as_completed(futures):
                try:
                    idx, text = future.result()
                    results[idx] = text
                except Exception:
                    idx = futures[future]
                    logger.warning("Search failed for query %d, using empty result", idx)
                    results[idx] = ""

        agg = total_usage.to_stat() if total_usage.total > 0 else None
        return results, agg

    def _execute(self, params: dict[str, Any]) -> tuple[str, UsageStat | None]:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(**params)
                content = resp.choices[0].message.content or ""
                usage = None
                if resp.usage is not None:
                    usage = UsageStat(
                        prompt_tokens=resp.usage.prompt_tokens,
                        completion_tokens=resp.usage.completion_tokens,
                        total_tokens=resp.usage.total_tokens,
                    )
                return content, usage
            except Exception as exc:  # nosec - provider/network errors are handled and retried
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC * attempt)
        raise RuntimeError(f"OpenRouter request failed after {MAX_RETRIES} attempts: {last_error}") from last_error


class _MutableUsage:
    """Thread-safe accumulator for token usage across parallel calls."""

    def __init__(self) -> None:
        self.prompt = 0
        self.completion = 0
        self.total = 0
        import threading
        self._lock = threading.Lock()

    def add(self, stat: UsageStat) -> None:
        with self._lock:
            self.prompt += stat.prompt_tokens
            self.completion += stat.completion_tokens
            self.total += stat.total_tokens

    def to_stat(self) -> UsageStat:
        return UsageStat(self.prompt, self.completion, self.total)

