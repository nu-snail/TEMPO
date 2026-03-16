"""Shapley value calculator for claim criticality (Phase 2 of evaluation).

Computes Shapley value phi_j for each claim j as the average marginal
contribution across sampled coalitions.  Uses batched LLM calls with a
Claim ID Reference System to minimise token usage.

Faithfully replicates the TimeSPEC evaluation methodology.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.shared import llm_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coalition cache
# ---------------------------------------------------------------------------

class CoalitionCache:
    """Cache prediction results keyed by frozenset of claim indices."""

    def __init__(self) -> None:
        self._cache: dict[frozenset[int], Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, indices: list[int]) -> Any | None:
        key = frozenset(indices)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, indices: list[int], value: Any) -> None:
        self._cache[frozenset(indices)] = value

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._cache),
            "hit_rate": self.hits / total if total else 0.0,
        }


# ---------------------------------------------------------------------------
# Shapley calculator
# ---------------------------------------------------------------------------

class ShapleyCalculator:
    """Compute Shapley values via coalition sampling + batched LLM prediction."""

    SYSTEM_PROMPT = (
        "Output predictions as JSON array. "
        "Use ONLY the provided claims. No explanation."
    )

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        max_samples: int = 100,
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        task: str,
        task_input: dict[str, Any],
        claim_texts: list[str],
        control_prediction: Any,
    ) -> dict[str, Any]:
        """Synchronous entry point for Shapley computation.

        Returns dict with ``shapley_values``, ``control_prediction``,
        ``cache_stats``, ``num_llm_calls``.
        """
        return asyncio.run(
            self._compute_async(task, task_input, claim_texts, control_prediction)
        )

    async def _compute_async(
        self,
        task: str,
        task_input: dict[str, Any],
        claim_texts: list[str],
        control_prediction: Any,
    ) -> dict[str, Any]:
        n = len(claim_texts)
        if n == 0:
            return {
                "shapley_values": [],
                "control_prediction": control_prediction,
                "cache_stats": {},
                "num_llm_calls": 0,
            }

        cache = CoalitionCache()
        cache.put(list(range(n)), control_prediction)

        defaults = _task_default(task, task_input)

        # Phase A: enumerate all needed coalitions
        all_coalitions: set[tuple[int, ...]] = set()
        saved: dict[int, list[tuple[list[int], list[int]]]] = {}

        for j in range(n):
            others = [i for i in range(n) if i != j]
            masks = _sample_coalitions(len(others), self.max_samples, self.rng)
            saved[j] = []
            for mask in masks:
                s_global = [others[i] for i in range(len(others)) if mask[i]] if others else []
                s_union_j = sorted(s_global + [j])
                all_coalitions.add(tuple(s_global))
                all_coalitions.add(tuple(s_union_j))
                saved[j].append((s_global, s_union_j))

        # Phase B: filter cached / empty
        to_predict: list[list[int]] = []
        for ct in all_coalitions:
            indices = list(ct)
            if cache.get(indices) is None:
                if not indices:
                    cache.put(indices, defaults)
                else:
                    to_predict.append(indices)

        # Phase C: batched prediction
        num_calls = 0
        for i in range(0, len(to_predict), self.batch_size):
            batch = to_predict[i : i + self.batch_size]
            preds = await self._batch_predict(task, task_input, claim_texts, batch)
            num_calls += 1
            for indices, pred in zip(batch, preds):
                cache.put(indices, pred)

        # Phase D: compute Shapley values from cache
        shapley_values: list[float] = []
        for j in range(n):
            marginals: list[float] = []
            for s_global, s_union_j in saved[j]:
                pred_s = cache.get(s_global)
                pred_sj = cache.get(s_union_j)
                if pred_s is None or pred_sj is None:
                    continue
                marginals.append(_marginal(task, pred_s, pred_sj))
            shapley_values.append(float(np.mean(marginals)) if marginals else 0.0)

        return {
            "shapley_values": shapley_values,
            "control_prediction": control_prediction,
            "cache_stats": cache.stats(),
            "num_llm_calls": num_calls,
            "num_coalitions_predicted": len(to_predict),
        }

    # ------------------------------------------------------------------
    # Batched LLM prediction (Claim ID Reference System)
    # ------------------------------------------------------------------

    async def _batch_predict(
        self,
        task: str,
        task_input: dict[str, Any],
        all_claims: list[str],
        coalition_list: list[list[int]],
    ) -> list[Any]:
        if not coalition_list:
            return []

        defaults = _task_default(task, task_input)
        prompt = self._build_batch_prompt(task, task_input, all_claims, coalition_list)
        max_tokens = max(500, 15 * len(coalition_list))

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: llm_client.call_llm(
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    model=self.model,
                    temperature=0,
                    max_tokens=max_tokens,
                ),
            )
        except Exception as exc:
            logger.warning("Shapley batch LLM call failed: %s", exc)
            return [defaults] * len(coalition_list)

        content = response.get("content", "")
        if not content:
            return [defaults] * len(coalition_list)

        results = _parse_json_array_from_text(content)
        if not results:
            return [defaults] * len(coalition_list)

        predictions: list[Any] = []
        for i, indices in enumerate(coalition_list):
            result = _find_result(results, i)
            predictions.append(_extract_prediction(result, task, task_input, defaults))

        return predictions

    def _build_batch_prompt(
        self,
        task: str,
        task_input: dict[str, Any],
        all_claims: list[str],
        coalition_list: list[list[int]],
    ) -> str:
        context = _task_context(task, task_input)
        claims_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(all_claims))
        defaults = _task_default(task, task_input)

        sets_lines: list[str] = []
        for i, indices in enumerate(coalition_list):
            if not indices:
                sets_lines.append(f"SET {i+1}: [] (empty - default: {defaults})")
            else:
                ids = [idx + 1 for idx in indices]
                sets_lines.append(f"SET {i+1}: {ids}")

        fmt = _format_example(task, task_input)
        return (
            f"{context}\n\n"
            f"CLAIMS (reference by ID):\n{claims_text}\n\n"
            f"Predict for each SET using ONLY claims with the listed IDs.\n"
            f"Empty set = use default value.\n\n"
            f"{chr(10).join(sets_lines)}\n\n"
            f"OUTPUT: JSON array with exactly {len(coalition_list)} objects.\n"
            f"Format: {fmt}"
        )


# ---------------------------------------------------------------------------
# Coalition sampling (numpy)
# ---------------------------------------------------------------------------

def _sample_coalitions(
    n_other: int, max_samples: int, rng: np.random.Generator,
) -> np.ndarray:
    """Return a 2D boolean array of shape (num_coalitions, n_other)."""
    if n_other == 0:
        return np.array([[]], dtype=bool).reshape(1, 0)

    total = 2**n_other
    bits = np.arange(n_other, dtype=np.uint64)

    if total <= max_samples:
        indices = np.arange(total, dtype=np.uint64)
    else:
        indices = rng.choice(total, size=max_samples, replace=False).astype(np.uint64)

    return ((indices[:, None] >> bits) & 1).astype(bool)


# ---------------------------------------------------------------------------
# Marginal contribution (task-specific)
# ---------------------------------------------------------------------------

def _marginal(task: str, pred_s: Any, pred_sj: Any) -> float:
    """pred(S union {j}) - pred(S), task-appropriate."""
    if task == "legal":
        return float(pred_sj) - float(pred_s)
    if task == "salary":
        return float(pred_sj) - float(pred_s)
    if task == "stock":
        if pred_s == pred_sj:
            return 0.0
        try:
            rank_s = list(range(len(pred_s)))
            rank_sj = [pred_sj.index(t) for t in pred_s]
            rho, _ = spearmanr(rank_s, rank_sj)
            return 1.0 - rho if not np.isnan(rho) else 0.0
        except (ValueError, IndexError):
            return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _task_context(task: str, ti: dict) -> str:
    if task == "legal":
        return (
            f"CASE: {ti.get('case_name', '')} (Term: {ti.get('term', '')})\n"
            f"Petitioner: {ti.get('petitioner', '')}\n"
            f"Respondent: {ti.get('respondent', '')}"
        )
    if task == "salary":
        return (
            f"PLAYER: {ti.get('Player', '')} ({ti.get('Position', '')})\n"
            f"Year: {ti.get('Year', '')}"
        )
    if task == "stock":
        companies = ti.get("companies", [])
        names = ti.get("company_names", {})
        cstr = ", ".join(f"{t} ({names.get(t, t)})" for t in companies)
        return f"SECTOR: {ti.get('sector', '')}\nCOMPANIES: {cstr}"
    return ""


def _task_default(task: str, ti: dict) -> Any:
    if task == "legal":
        return 0.5
    if task == "salary":
        return 15_000_000
    if task == "stock":
        return ti.get("companies", [])
    return 0.5


def _format_example(task: str, ti: dict) -> str:
    if task == "legal":
        return '[{"set": 1, "p": 0.63}, {"set": 2, "p": 0.71}, ...]'
    if task == "salary":
        return '[{"set": 1, "s": 35000000}, {"set": 2, "s": 28000000}, ...]'
    if task == "stock":
        c = ti.get("companies", [])
        return f'[{{"set": 1, "r": {c}}}, ...]'
    return "[]"


def _extract_prediction(
    result: dict | None, task: str, ti: dict, default: Any,
) -> Any:
    if result is None:
        return default
    if task == "legal":
        v = result.get("p", result.get("probability_petitioner", result.get("prob", 0.5)))
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.5
    if task == "salary":
        v = result.get("s", result.get("predicted_salary", result.get("salary", 15_000_000)))
        try:
            return int(v)
        except (TypeError, ValueError):
            return 15_000_000
    if task == "stock":
        v = result.get("r", result.get("ranking", ti.get("companies", [])))
        return list(v) if isinstance(v, list) else default
    return default


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_json_array_from_text(text: str) -> list[dict]:
    text = text.strip()
    try:
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            obj = json.loads(m.group())
            if isinstance(obj, list):
                return obj
    except json.JSONDecodeError:
        pass
    # Fallback: extract individual objects
    results = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            results.append(json.loads(m.group()))
        except json.JSONDecodeError:
            pass
    return results


def _find_result(results: list[dict], idx: int) -> dict | None:
    for r in results:
        if isinstance(r, dict) and r.get("set") == idx + 1:
            return r
    if idx < len(results) and isinstance(results[idx], dict):
        return results[idx]
    return None
