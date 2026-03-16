"""Temporal search client for claim verification.

Routes search calls through OpenRouter's perplexity/sonar endpoint,
consolidating all API costs under a single billing account.

Supports both single-query search and batched search (multiple claims
per API call) to reduce the per-request search surcharge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from . import config

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF = 3


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------

def _build_client(api_key: str | None = None) -> tuple[OpenAI, str]:
    """Build an OpenAI-compatible client for the configured search provider."""
    provider = config.SEARCH_PROVIDER
    if api_key is None:
        api_key = config.get_search_api_key(provider)

    if provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
    else:
        base_url = "https://api.perplexity.ai"

    model = config.SEARCH_SETTINGS.get("search_model", "perplexity/sonar")
    return OpenAI(base_url=base_url, api_key=api_key), model


# ---------------------------------------------------------------------------
# Search client
# ---------------------------------------------------------------------------

class SearchClient:
    """Search client routed through OpenRouter (perplexity/sonar)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.client, self.model = _build_client(api_key)
        self._cache: dict[str, dict[str, Any]] = {}

    def search(
        self,
        query: str,
        *,
        cutoff_date: str = "",
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """Execute a single web search query.

        Returns:
            Dict with ``content`` (str) and ``citations`` (list[str]).
        """
        if max_results is None:
            max_results = config.SEARCH_SETTINGS["max_results"]

        cache_key = _cache_key(query, cutoff_date)
        if cache_key in self._cache:
            return self._cache[cache_key]

        system = (
            "You are a precise research assistant. "
            "Return only factual information with specific dates when available."
        )
        if cutoff_date:
            system += (
                f" Focus on information that was publicly available before {cutoff_date}. "
                "For each fact, state the date it became publicly known."
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

        last_exc: BaseException | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.2,
                )
                content = resp.choices[0].message.content or ""
                citations: list[str] = getattr(resp, "citations", []) or []
                result = {"content": content, "citations": citations}
                self._cache[cache_key] = result
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Search failed (attempt %d/%d): %s",
                        attempt, MAX_RETRIES, exc,
                    )
                    time.sleep(RETRY_BACKOFF)

        logger.error("Search failed after %d attempts for query: %s", MAX_RETRIES, query[:80])
        raise RuntimeError(
            f"Search failed after {MAX_RETRIES} attempts"
        ) from last_exc

    def batch_search(
        self,
        claims: list[str],
        *,
        cutoff_date: str = "",
        batch_size: int | None = None,
    ) -> dict[int, str]:
        """Search for multiple claims in batched API calls.

        Groups claims into batches of ``batch_size`` and makes one API call
        per batch, dramatically reducing the per-request search surcharge.

        Args:
            claims: List of claim texts to search for.
            cutoff_date: Knowledge cutoff date (YYYY-MM-DD).
            batch_size: Claims per API call (default from config).

        Returns:
            Dict mapping claim index -> search answer text.
        """
        if batch_size is None:
            batch_size = config.SEARCH_SETTINGS.get("search_batch_size", 10)

        answers: dict[int, str] = {}
        uncached_indices: list[int] = []

        for i, claim in enumerate(claims):
            cache_key = _cache_key(claim, cutoff_date)
            if cache_key in self._cache:
                answers[i] = self._cache[cache_key].get("content", "")[:800]
            else:
                uncached_indices.append(i)

        for batch_start in range(0, len(uncached_indices), batch_size):
            batch_idx = uncached_indices[batch_start:batch_start + batch_size]
            batch_claims = [claims[i] for i in batch_idx]
            batch_result = self._search_batch(batch_claims, cutoff_date)

            for j, idx in enumerate(batch_idx):
                answer = batch_result.get(j, "")
                answers[idx] = answer[:800]
                self._cache[_cache_key(claims[idx], cutoff_date)] = {
                    "content": answer, "citations": [],
                }

        return answers

    def _search_batch(
        self, claims: list[str], cutoff_date: str,
    ) -> dict[int, str]:
        """Make a single API call covering multiple claims.

        All claims are packed into one request to minimise the per-request
        search surcharge ($0.005).  The prompt requests date-only output
        to keep responses concise even for large batches.
        """
        numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))

        system = (
            "You are a temporal fact-checker. For EACH numbered claim, "
            "search for the EXACT DATE (precise to the day) when the claim "
            "first became publicly verifiable. "
            "Reply with EXACTLY one line per claim: [index] YYYY-MM-DD "
            "or [index] UNKNOWN. Nothing else."
        )
        if cutoff_date:
            system += f" Context: knowledge cutoff is {cutoff_date}."

        user_msg = (
            f"For each claim, find the exact date (YYYY-MM-DD) it became "
            f"publicly verifiable (e.g. announced, reported, decided, signed).\n"
            f"Format: [index] YYYY-MM-DD\n"
            f"If undeterminable: [index] UNKNOWN\n\n"
            f"{numbered}"
        )

        last_exc: BaseException | None = None
        tokens_per_claim = 30
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=max(256, tokens_per_claim * len(claims)),
                    temperature=0.1,
                )
                content = resp.choices[0].message.content or ""
                return _parse_batch_response(content, len(claims))
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    logger.warning("Batch search failed (attempt %d/%d): %s",
                                   attempt, MAX_RETRIES, exc)
                    time.sleep(RETRY_BACKOFF)

        logger.error("Batch search failed after %d attempts for %d claims",
                      MAX_RETRIES, len(claims))
        return {i: "" for i in range(len(claims))}


def _parse_batch_response(content: str, n_claims: int) -> dict[int, str]:
    """Parse the batched search response into per-claim answers.

    Handles multiple response formats:
      [0] 2023-09-22: Apple released...
      0. 2023-09-22: Apple released...
      **[0]** 2023-09-22 - Apple released...
    """
    import re
    results: dict[int, str] = {}
    lines = content.split("\n")

    current_idx: int | None = None
    current_text: list[str] = []

    # Patterns: [0], **[0]**, 0., 0), - [0]
    idx_pattern = re.compile(
        r"(?:\*{0,2})\[(\d+)\](?:\*{0,2})\s*(.*)"
        r"|^(\d+)[.)]\s*(.*)"
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        m = idx_pattern.match(stripped)
        if m:
            if current_idx is not None:
                results[current_idx] = "\n".join(current_text).strip()
            if m.group(1) is not None:
                current_idx = int(m.group(1))
                current_text = [m.group(2)]
            else:
                current_idx = int(m.group(3))
                current_text = [m.group(4)]
        elif current_idx is not None:
            current_text.append(stripped)

    if current_idx is not None:
        results[current_idx] = "\n".join(current_text).strip()

    for i in range(n_claims):
        if i not in results:
            results[i] = ""

    # Fallback: if parsing yielded <50% results, split content evenly
    non_empty = sum(1 for v in results.values() if v.strip())
    if non_empty < n_claims * 0.5 and content.strip():
        logger.debug("Batch parse fallback: got %d/%d, using full content for all",
                      non_empty, n_claims)
        for i in range(n_claims):
            if not results[i].strip():
                results[i] = content[:800]

    return results


# ---------------------------------------------------------------------------
# Module-level singleton and convenience functions
# ---------------------------------------------------------------------------

_default_client: SearchClient | None = None


def get_client() -> SearchClient:
    """Return (and lazily create) the module-level search client singleton."""
    global _default_client
    if _default_client is None:
        _default_client = SearchClient()
    return _default_client


def search(query: str, *, cutoff_date: str = "", **kwargs: Any) -> dict[str, Any]:
    """Module-level convenience wrapper for single search."""
    return get_client().search(query, cutoff_date=cutoff_date, **kwargs)


def batch_search(
    claims: list[str], *, cutoff_date: str = "", **kwargs: Any,
) -> dict[int, str]:
    """Module-level convenience wrapper for batched search."""
    return get_client().batch_search(claims, cutoff_date=cutoff_date, **kwargs)


# ---------------------------------------------------------------------------
# Disk-based persistent cache (for cross-epoch reuse during training)
# ---------------------------------------------------------------------------


class PersistentSearchCache:
    """JSON-backed search cache that survives process restarts.

    Keyed by (claim_text, cutoff_date) so identical claims across training
    epochs reuse prior search results.
    """

    def __init__(self, cache_path: Path) -> None:
        self.path = cache_path
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info("Loaded %d cached search results from %s", len(self._data), self.path)

    def get(self, claim: str, cutoff_date: str) -> dict[str, Any] | None:
        """Retrieve a cached search result, or None."""
        return self._data.get(_cache_key(claim, cutoff_date))

    def put(self, claim: str, cutoff_date: str, result: dict[str, Any]) -> None:
        """Store a search result."""
        self._data[_cache_key(claim, cutoff_date)] = result

    def save(self) -> None:
        """Flush cache to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)
        logger.info("Saved %d cached search results to %s", len(self._data), self.path)

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _cache_key(query: str, cutoff_date: str) -> str:
    raw = f"{query.strip().lower()}|{cutoff_date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
