"""End-to-end reward computation for GRPO training.

For a single completion y on prompt x:
  1. Parse the completion to extract prediction + rationale.
  2. Compute R_leakage = 1 - OLR(rationale) via claim extraction + leakage detection.
  3. Compute R_performance = task-specific metric on the prediction.
  4. Combine: R = R_leakage + lambda(t) * R_performance (before MO-GRPO normalisation).

For a GRPO group of G completions on the same prompt, the caller normalises
each reward component independently (MO-GRPO) before computing advantages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.claim_extractor import ClaimExtractor, ExtractionResult
from src.shared.data_loader import TaskInstance
from src.shared.leakage_detector import LeakageDetector, LeakageReport
from src.shared.prompts.builder import parse_response
from src.shared.search_client import PersistentSearchCache

from .performance import compute_performance_reward

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletionReward:
    """Reward breakdown for a single completion."""
    r_leakage: float        # 1 - OLR, in [0, 1]
    r_performance: float    # task-specific, in [0, 1]
    olr: float              # raw OLR for logging
    num_claims: int
    num_leaked: int
    parse_success: bool
    extraction: ExtractionResult | None = None
    leakage_report: LeakageReport | None = None
    parsed_prediction: dict[str, Any] | None = None


class RewardPipeline:
    """Compute rewards for GRPO completions.

    Encapsulates claim extraction, leakage detection, and performance
    scoring behind a single ``score`` method.
    """

    def __init__(
        self,
        extractor_model: str = "google/gemini-2.5-flash",
        detector_model: str = "google/gemini-2.5-flash",
        *,
        use_search: bool = True,
        use_llm_judge: bool = True,
        cache_path: str | None = None,
    ) -> None:
        cache = PersistentSearchCache(Path(cache_path)) if cache_path else None
        self.extractor = ClaimExtractor(model=extractor_model)
        self.detector = LeakageDetector(
            model=detector_model,
            use_search=use_search,
            use_llm_judge=use_llm_judge,
            persistent_cache=cache,
        )
        self._cache = cache

    def score(self, completion: str, instance: TaskInstance) -> CompletionReward:
        """Score a single completion against a TaskInstance.

        Args:
            completion: Raw model-generated text.
            instance: The training instance (contains ground truth + metadata).

        Returns:
            CompletionReward with leakage and performance components.
        """
        # Step 1: parse completion
        parsed = parse_response(completion, instance)
        rationale = _extract_rationale(parsed, completion)

        # Step 2: compute leakage reward
        extraction = self.extractor.extract(
            rationale,
            task_description=instance.question_text,
            event_description=_event_description(instance),
            cutoff_date=instance.cutoff_date,
        )
        report = self.detector.detect(extraction.claims, instance.cutoff_date)
        r_leakage = 1.0 - report.olr

        # Step 3: compute performance reward
        r_perf = compute_performance_reward(parsed, instance)

        return CompletionReward(
            r_leakage=r_leakage,
            r_performance=r_perf,
            olr=report.olr,
            num_claims=report.total_claims,
            num_leaked=report.num_leaked,
            parse_success=parsed is not None,
            extraction=extraction,
            leakage_report=report,
            parsed_prediction=parsed,
        )

    def score_group(
        self,
        completions: list[str],
        instance: TaskInstance,
    ) -> list[CompletionReward]:
        """Score a GRPO group of G completions for the same prompt.

        Returns a list of CompletionReward, one per completion.
        The caller is responsible for MO-GRPO normalisation and
        advantage computation.
        """
        return [self.score(c, instance) for c in completions]

    def score_group_parallel(
        self,
        completions: list[str],
        instance: TaskInstance,
        max_workers: int = 4,
    ) -> list[CompletionReward]:
        """Score a group of completions in parallel using threads.

        Same semantics as ``score_group`` but uses a ThreadPoolExecutor
        to overlap I/O-bound API calls (Claude + Perplexity).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[CompletionReward | None] = [None] * len(completions)

        def _score_one(idx: int) -> tuple[int, CompletionReward]:
            return idx, self.score(completions[idx], instance)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_score_one, i): i for i in range(len(completions))}
            for fut in as_completed(futures):
                try:
                    idx, reward = fut.result()
                    results[idx] = reward
                except Exception as exc:
                    idx = futures[fut]
                    logger.warning("Parallel reward scoring failed for completion %d: %s", idx, exc)
                    results[idx] = CompletionReward(
                        r_leakage=0.5, r_performance=0.0, olr=0.5,
                        num_claims=0, num_leaked=0, parse_success=False,
                    )

        return results  # type: ignore[return-value]

    def score_group_batched(
        self,
        completions: list[str],
        instance: TaskInstance,
        max_workers: int = 4,
    ) -> list[CompletionReward]:
        """Score a group with search- and LLM-optimised batching.

        Reduces both Sonar calls and Gemini tokens by deduplicating
        work across the G completions in a group:

        Phase 1: Extract claims per completion (G Gemini calls, parallel).
        Phase 2: Pre-warm search cache for unique A1-A3 claims (1 Sonar call).
        Phase 3: Group-level leakage detection via ``detect_group`` —
                 query generation, date extraction, and LLM judge run
                 once on deduplicated claims (~3 Gemini calls total
                 instead of G × 3).
        Phase 4: Performance rewards (no API calls).

        Net: G + 3 Gemini calls + 1 Sonar call per group
             (down from G × 3.3 Gemini + G Sonar).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from src.shared import search_client as sc

        G = len(completions)
        _NEEDS_SEARCH = frozenset({"A1", "A2", "A3"})

        # Phase 1: parse + extract claims for all completions in parallel
        def _extract_one(i: int) -> tuple[int, dict | None, ExtractionResult | None]:
            parsed = parse_response(completions[i], instance)
            rationale = _extract_rationale(parsed, completions[i])
            extraction = self.extractor.extract(
                rationale,
                task_description=instance.question_text,
                event_description=_event_description(instance),
                cutoff_date=instance.cutoff_date,
            )
            return i, parsed, extraction

        parsed_arr: list[dict | None] = [None] * G
        extract_arr: list[ExtractionResult | None] = [None] * G

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_extract_one, i): i for i in range(G)}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    _, parsed, ext = fut.result()
                    parsed_arr[idx] = parsed
                    extract_arr[idx] = ext
                except Exception as exc:
                    logger.warning("Extraction failed for completion %d: %s", idx, exc)

        # Phase 2: pre-warm search cache for all unique A1-A3 claims
        all_a123: list[str] = []
        for ext in extract_arr:
            if ext is None:
                continue
            for c in ext.claims:
                if c.category in _NEEDS_SEARCH:
                    all_a123.append(c.text)

        unique_claims = list(dict.fromkeys(all_a123))
        if unique_claims:
            try:
                sc.batch_search(unique_claims, cutoff_date=instance.cutoff_date)
            except Exception as exc:
                logger.warning("Pre-warm batch search failed: %s", exc)

        # Phase 3: group-level leakage detection (deduplicated Gemini calls)
        claims_per_comp = [
            ext.claims if ext is not None else []
            for ext in extract_arr
        ]
        reports = self.detector.detect_group(claims_per_comp, instance.cutoff_date)

        # Phase 4: assemble rewards
        results: list[CompletionReward] = []
        for i in range(G):
            parsed = parsed_arr[i]
            ext = extract_arr[i]
            report = reports[i]
            r_leak = 1.0 - report.olr
            r_perf = compute_performance_reward(parsed, instance)
            results.append(CompletionReward(
                r_leakage=r_leak, r_performance=r_perf, olr=report.olr,
                num_claims=report.total_claims, num_leaked=report.num_leaked,
                parse_success=parsed is not None,
                extraction=ext, leakage_report=report,
                parsed_prediction=parsed,
            ))

        return results

    def score_group_performance_only(
        self,
        completions: list[str],
        instance: TaskInstance,
    ) -> list[CompletionReward]:
        """Score completions using only performance reward (skip leakage).

        Avoids the expensive claim-extraction + search pipeline entirely.
        R_leakage is set to 0 for all completions.
        """
        results: list[CompletionReward] = []
        for c in completions:
            parsed = parse_response(c, instance)
            r_perf = compute_performance_reward(parsed, instance)
            results.append(CompletionReward(
                r_leakage=0.0,
                r_performance=r_perf,
                olr=0.0,
                num_claims=0,
                num_leaked=0,
                parse_success=parsed is not None,
                parsed_prediction=parsed,
            ))
        return results

    def save_cache(self) -> None:
        """Flush the persistent search cache to disk."""
        if self._cache is not None:
            self._cache.save()


# ---------------------------------------------------------------------------
# MO-GRPO normalisation (called by the training loop, not by the pipeline)
# ---------------------------------------------------------------------------

OLR_FLOOR: float = 0.03


def mo_grpo_advantages(
    rewards: list[CompletionReward],
    lambda_t: float,
    eps: float = 1e-6,
) -> list[float]:
    """Compute MO-GRPO advantages with conditioned performance reward.

    Uses group-relative gating (GDPO-style): performance reward is zeroed
    for completions whose OLR >= a dynamic threshold derived from the
    group median, floored at OLR_FLOOR.  This couples the two objectives
    so that the performance gradient also pushes toward lower leakage.

    Each reward component is z-scored independently within the group
    before combining:

        A_i = hat_R_leak_i + lambda * hat_R_perf_cond_i

    Args:
        rewards: List of G CompletionReward objects.
        lambda_t: Current lambda(t) value.
        eps: Small constant for numerical stability.

    Returns:
        List of G advantage values (un-clipped).
    """
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    leaks = [r.r_leakage for r in rewards]
    olrs = [r.olr for r in rewards]
    perfs = [r.r_performance for r in rewards]

    # Group-relative threshold: median OLR, floored at OLR_FLOOR
    sorted_olrs = sorted(olrs)
    median_olr = (sorted_olrs[n // 2 - 1] + sorted_olrs[n // 2]) / 2 if n % 2 == 0 else sorted_olrs[n // 2]
    tau = max(median_olr, OLR_FLOOR)

    # Condition performance reward: zero out for completions above threshold
    perfs_cond = [p if olr < tau else 0.0 for p, olr in zip(perfs, olrs)]

    # Z-score each component independently (GDPO / MO-GRPO)
    mu_l = sum(leaks) / n
    mu_p = sum(perfs_cond) / n
    std_l = (sum((x - mu_l) ** 2 for x in leaks) / n) ** 0.5
    std_p = (sum((x - mu_p) ** 2 for x in perfs_cond) / n) ** 0.5

    advantages: list[float] = []
    for i in range(n):
        hat_l = (leaks[i] - mu_l) / (std_l + eps) if std_l > eps else 0.0
        hat_p = (perfs_cond[i] - mu_p) / (std_p + eps) if std_p > eps else 0.0
        advantages.append(hat_l + lambda_t * hat_p)

    return advantages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_rationale(parsed: dict[str, Any] | None, raw_text: str) -> str:
    """Extract the rationale text for claim analysis."""
    if parsed is not None:
        rationale = parsed.get("rationale", parsed.get("ranking_rationale", ""))
        if rationale:
            return rationale
    return raw_text


def _event_description(instance: TaskInstance) -> str:
    """Build a concise event description for the claim extractor prompt."""
    if instance.task == "stock":
        period = instance.raw.get("past_period", {})
        return (
            f"Stock return ranking for {instance.raw.get('sector', '')} sector "
            f"from {period.get('start_date', '')} to {period.get('end_date', '')}"
        )
    if instance.task == "salary":
        return (
            f"Free-agent contract for {instance.raw.get('Player', '')} "
            f"({instance.raw.get('Sport', 'NBA')}, {instance.raw.get('Year', '')})"
        )
    if instance.task == "legal":
        return f"Outcome of {instance.raw.get('case_name', 'the case')}"
    return instance.question_text
