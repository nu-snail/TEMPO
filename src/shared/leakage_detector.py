"""Temporal leakage detection via search + LLM judge.

Pipeline per rationale:
1. Claims already categorised (A1-A5, B1-B2) by ``claim_extractor``.
2. A4/A5 -> always leaked; B1/B2 -> never leaked (no API calls).
3. A1/A2/A3 -> search for determination date tau(c), compare to cutoff.
   If search is inconclusive, fall back to an LLM judge.

Returns per-claim leak labels and the Overall Leakage Ratio (OLR).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

from . import llm_client, search_client
from .claim_extractor import Claim, temporal_ref_to_date

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimLeakageResult:
    """Leakage verdict for one claim."""
    claim_id: int
    text: str
    category: str
    is_leaked: bool
    reason: str
    method: str  # CATEGORY_RULE | SEARCH | LLM_JUDGE
    determination_date: str | None = None
    cutoff_date: str | None = None


@dataclass
class LeakageReport:
    """Aggregated leakage results for one rationale."""
    results: list[ClaimLeakageResult]
    cutoff_date: str
    total_claims: int = 0
    num_leaked: int = 0

    # Breakdown by method
    cat_leaked: int = 0
    cat_not_leaked: int = 0
    search_leaked: int = 0
    search_not_leaked: int = 0
    judge_leaked: int = 0
    judge_not_leaked: int = 0

    @property
    def olr(self) -> float:
        """Overall Leakage Ratio: # leaked / # total."""
        return self.num_leaked / self.total_claims if self.total_claims > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [asdict(r) for r in self.results],
            "cutoff_date": self.cutoff_date,
            "olr": self.olr,
            "total_claims": self.total_claims,
            "num_leaked": self.num_leaked,
            "breakdown": {
                "cat_leaked": self.cat_leaked,
                "cat_not_leaked": self.cat_not_leaked,
                "search_leaked": self.search_leaked,
                "search_not_leaked": self.search_not_leaked,
                "judge_leaked": self.judge_leaked,
                "judge_not_leaked": self.judge_not_leaked,
            },
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QUERY_GEN_PROMPT = """\
Generate a concise web-search query to find the SPECIFIC DATE when each claim
below became publicly known.  Include year, entity names, and date-related
keywords ("date", "when", "announced", "decided", "signed").

Claims (JSON array):
{claims_json}

Output a JSON array with one object per claim:
[
  {{"index": 0, "query": "<search query>"}},
  ...
]
JSON only, no markdown fences."""


_DATE_EXTRACT_PROMPT = """\
For each claim below, determine the DATE the information first became publicly
available, using the provided search result.

- Events/decisions/signings: use the date the event occurred.
- States ("had X as of Y"): if confirmed, use the start of the period.
- Timeless facts (legal principles, math): use 1900-01-01.
- If no date can be determined, use null.

Claims with search results:
{data_json}

Output a JSON array:
[
  {{"index": 0, "event_date": "YYYY-MM-DD or null", "confidence": "high|medium|low|none", "reasoning": "<brief>"}},
  ...
]
JSON only, no markdown fences."""


_LLM_JUDGE_PROMPT = """\
You are a temporal leakage judge.  Determine whether each claim below could
have been known BEFORE the knowledge cutoff date.

DEFINITION OF LEAKAGE: A claim is "leaked" if the information it asserts
was NOT publicly determinable before the cutoff date.  Focus on when the
fact could first be VERIFIED or DETERMINED, not when the underlying event
happened (e.g. a prediction made before the cutoff is not leaked even if
the predicted event is after the cutoff).

Cutoff date: {cutoff_date}

Claims:
{claims_json}

For each claim, output:
[
  {{"index": 0, "is_leaked": true/false, "reasoning": "<brief explanation>"}},
  ...
]
JSON only, no markdown fences."""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class LeakageDetector:
    """Detect temporal leakage using category rules + search + LLM judge."""

    ALWAYS_LEAKED = frozenset({"A4", "A5"})
    NEVER_LEAKED = frozenset({"B1", "B2"})
    NEEDS_SEARCH = frozenset({"A1", "A2", "A3"})

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        *,
        use_search: bool = True,
        use_llm_judge: bool = True,
        persistent_cache: search_client.PersistentSearchCache | None = None,
    ) -> None:
        self.model = model
        self.use_search = use_search
        self.use_llm_judge = use_llm_judge
        self.cache = persistent_cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, claims: list[Claim], cutoff_date: str) -> LeakageReport:
        """Run the full detection pipeline on a list of claims.

        Args:
            claims: Output of ``ClaimExtractor.extract``.
            cutoff_date: Knowledge cutoff (YYYY-MM-DD).

        Returns:
            LeakageReport with per-claim verdicts and aggregate OLR.
        """
        if not claims:
            return LeakageReport(results=[], cutoff_date=cutoff_date)

        ref = _parse_date(cutoff_date)

        cat_leaked: list[ClaimLeakageResult] = []
        cat_safe: list[ClaimLeakageResult] = []
        to_verify: list[Claim] = []

        for c in claims:
            if c.category in self.ALWAYS_LEAKED:
                cat_leaked.append(ClaimLeakageResult(
                    claim_id=c.claim_id, text=c.text, category=c.category,
                    is_leaked=True,
                    reason=f"Category {c.category} is always leaked in backtesting",
                    method="CATEGORY_RULE", cutoff_date=cutoff_date,
                ))
            elif c.category in self.NEVER_LEAKED:
                cat_safe.append(ClaimLeakageResult(
                    claim_id=c.claim_id, text=c.text, category=c.category,
                    is_leaked=False,
                    reason=f"Category {c.category} is temporally non-applicable",
                    method="CATEGORY_RULE", cutoff_date=cutoff_date,
                ))
            else:
                to_verify.append(c)

        # --- Search + LLM-judge for A1/A2/A3 claims ---
        verified = self._verify_claims(to_verify, cutoff_date, ref)

        all_results = cat_leaked + cat_safe + verified
        all_results.sort(key=lambda r: r.claim_id)

        n_leaked = sum(r.is_leaked for r in all_results)
        return LeakageReport(
            results=all_results,
            cutoff_date=cutoff_date,
            total_claims=len(all_results),
            num_leaked=n_leaked,
            cat_leaked=len(cat_leaked),
            cat_not_leaked=len(cat_safe),
            search_leaked=sum(1 for r in verified if r.is_leaked and r.method == "SEARCH"),
            search_not_leaked=sum(1 for r in verified if not r.is_leaked and r.method == "SEARCH"),
            judge_leaked=sum(1 for r in verified if r.is_leaked and r.method == "LLM_JUDGE"),
            judge_not_leaked=sum(1 for r in verified if not r.is_leaked and r.method == "LLM_JUDGE"),
        )

    # ------------------------------------------------------------------
    # Group-level detection (deduplicated across completions)
    # ------------------------------------------------------------------

    def detect_group(
        self,
        claims_per_completion: list[list[Claim]],
        cutoff_date: str,
    ) -> list[LeakageReport]:
        """Detect leakage for multiple completions sharing the same cutoff.

        Deduplicates the expensive Gemini calls (query generation, date
        extraction, LLM judge) across all completions by operating on
        *unique* A1-A3 claim texts.  Reduces Gemini token usage by
        ~50% compared to calling ``detect()`` independently per completion.

        Args:
            claims_per_completion: List of G claim lists (one per completion).
            cutoff_date: Shared knowledge cutoff (YYYY-MM-DD).

        Returns:
            List of G LeakageReports, one per completion.
        """
        if not claims_per_completion:
            return []

        ref = _parse_date(cutoff_date)

        # --- Step 1: categorise all claims, collect A1-A3 for verification ---
        per_comp: list[tuple[list[ClaimLeakageResult], list[ClaimLeakageResult], list[Claim]]] = []
        unique_map: dict[str, Claim] = {}  # lowered text -> representative Claim

        for claims in claims_per_completion:
            cat_leaked, cat_safe, to_verify = [], [], []
            for c in claims:
                if c.category in self.ALWAYS_LEAKED:
                    cat_leaked.append(ClaimLeakageResult(
                        claim_id=c.claim_id, text=c.text, category=c.category,
                        is_leaked=True,
                        reason=f"Category {c.category} is always leaked in backtesting",
                        method="CATEGORY_RULE", cutoff_date=cutoff_date,
                    ))
                elif c.category in self.NEVER_LEAKED:
                    cat_safe.append(ClaimLeakageResult(
                        claim_id=c.claim_id, text=c.text, category=c.category,
                        is_leaked=False,
                        reason=f"Category {c.category} is temporally non-applicable",
                        method="CATEGORY_RULE", cutoff_date=cutoff_date,
                    ))
                else:
                    to_verify.append(c)
                    key = c.text.lower().strip()
                    if key not in unique_map:
                        unique_map[key] = c
            per_comp.append((cat_leaked, cat_safe, to_verify))

        # --- Step 2: verify unique claims (1 query-gen + 1 search + 1 date-extract + 0-1 judge) ---
        unique_claims = list(unique_map.values())
        verdict_map: dict[str, ClaimLeakageResult] = {}

        if unique_claims:
            verdicts = self._verify_claims(unique_claims, cutoff_date, ref)
            for v in verdicts:
                verdict_map[v.text.lower().strip()] = v

        # --- Step 3: map verdicts back to per-completion reports ---
        reports: list[LeakageReport] = []
        for cat_leaked, cat_safe, to_verify in per_comp:
            verified: list[ClaimLeakageResult] = []
            for c in to_verify:
                key = c.text.lower().strip()
                template = verdict_map.get(key)
                if template is not None:
                    verified.append(ClaimLeakageResult(
                        claim_id=c.claim_id, text=c.text,
                        category=c.category, is_leaked=template.is_leaked,
                        reason=template.reason, method=template.method,
                        determination_date=template.determination_date,
                        cutoff_date=cutoff_date,
                    ))
                else:
                    verified.append(ClaimLeakageResult(
                        claim_id=c.claim_id, text=c.text, category=c.category,
                        is_leaked=False,
                        reason="No verdict available; assumed not leaked",
                        method="SEARCH", cutoff_date=cutoff_date,
                    ))

            all_results = cat_leaked + cat_safe + verified
            all_results.sort(key=lambda r: r.claim_id)
            n_leaked = sum(r.is_leaked for r in all_results)
            reports.append(LeakageReport(
                results=all_results, cutoff_date=cutoff_date,
                total_claims=len(all_results), num_leaked=n_leaked,
                cat_leaked=len(cat_leaked), cat_not_leaked=len(cat_safe),
                search_leaked=sum(1 for r in verified if r.is_leaked and r.method == "SEARCH"),
                search_not_leaked=sum(1 for r in verified if not r.is_leaked and r.method == "SEARCH"),
                judge_leaked=sum(1 for r in verified if r.is_leaked and r.method == "LLM_JUDGE"),
                judge_not_leaked=sum(1 for r in verified if not r.is_leaked and r.method == "LLM_JUDGE"),
            ))

        return reports

    # ------------------------------------------------------------------
    # Search-based verification
    # ------------------------------------------------------------------

    def _verify_claims(
        self,
        claims: list[Claim],
        cutoff_date: str,
        ref: date,
    ) -> list[ClaimLeakageResult]:
        """Verify A1/A2/A3 claims via search, then LLM judge as fallback."""
        if not claims:
            return []

        # Step 1: generate search queries (one LLM call)
        queries = self._generate_queries(claims)

        # Step 2: execute searches (N Perplexity calls)
        search_answers = self._run_searches(claims, queries, cutoff_date)

        # Step 3: extract dates from search answers (one LLM call)
        date_verdicts = self._extract_dates(claims, search_answers)

        # Step 4: resolve each claim; use LLM judge where search is inconclusive
        results: list[ClaimLeakageResult] = []
        unresolved_indices: list[int] = []

        for i, claim in enumerate(claims):
            verdict = date_verdicts[i] if i < len(date_verdicts) else {}
            event_date_str = verdict.get("event_date")
            confidence = verdict.get("confidence", "none")
            reasoning = verdict.get("reasoning", "")

            resolved = self._resolve_by_search(
                claim, event_date_str, confidence, reasoning, cutoff_date, ref,
            )
            if resolved is not None:
                results.append(resolved)
            else:
                unresolved_indices.append(i)
                results.append(None)  # type: ignore[arg-type] — placeholder

        # Step 5: LLM judge fallback for unresolved claims
        if unresolved_indices and self.use_llm_judge:
            unresolved_claims = [claims[i] for i in unresolved_indices]
            judge_verdicts = self._llm_judge(unresolved_claims, cutoff_date)
            for j, idx in enumerate(unresolved_indices):
                jv = judge_verdicts[j] if j < len(judge_verdicts) else {}
                leaked = bool(jv.get("is_leaked", False))
                results[idx] = ClaimLeakageResult(
                    claim_id=claims[idx].claim_id,
                    text=claims[idx].text,
                    category=claims[idx].category,
                    is_leaked=leaked,
                    reason=jv.get("reasoning", "LLM judge fallback"),
                    method="LLM_JUDGE",
                    cutoff_date=cutoff_date,
                )
        elif unresolved_indices:
            # No LLM judge — default to not leaked
            for idx in unresolved_indices:
                results[idx] = ClaimLeakageResult(
                    claim_id=claims[idx].claim_id,
                    text=claims[idx].text,
                    category=claims[idx].category,
                    is_leaked=False,
                    reason="Search inconclusive and LLM judge disabled; assumed not leaked",
                    method="SEARCH",
                    cutoff_date=cutoff_date,
                )

        return [r for r in results if r is not None]

    def _resolve_by_search(
        self,
        claim: Claim,
        event_date_str: str | None,
        confidence: str,
        reasoning: str,
        cutoff_date: str,
        ref: date,
    ) -> ClaimLeakageResult | None:
        """Try to resolve a claim's leakage status from the search date.

        Returns None when the search is inconclusive (triggers LLM judge).
        """
        if not event_date_str or event_date_str.upper() in ("UNKNOWN", "PREDICTION", "NULL", "NONE"):
            # Try temporal_ref from claim extraction as a last resort
            fallback = temporal_ref_to_date(claim.temporal_ref)
            if fallback:
                d = _parse_date(fallback)
                leaked = d > ref
                return ClaimLeakageResult(
                    claim_id=claim.claim_id, text=claim.text, category=claim.category,
                    is_leaked=leaked,
                    reason=(
                        f"Search inconclusive; strict interpretation of "
                        f"'{claim.temporal_ref}' -> {fallback} "
                        f"{'>' if leaked else '<='} cutoff {cutoff_date}"
                    ),
                    method="SEARCH",
                    determination_date=fallback,
                    cutoff_date=cutoff_date,
                )
            return None  # truly inconclusive

        if event_date_str == "1900-01-01":
            return ClaimLeakageResult(
                claim_id=claim.claim_id, text=claim.text, category=claim.category,
                is_leaked=False,
                reason="Timeless/universal knowledge",
                method="SEARCH",
                determination_date="1900-01-01",
                cutoff_date=cutoff_date,
            )

        parsed = _try_parse_date(event_date_str) or _try_parse_date(
            temporal_ref_to_date(event_date_str) or ""
        )
        if parsed is None:
            return None  # unparseable -> fallback to LLM judge

        leaked = parsed > ref
        return ClaimLeakageResult(
            claim_id=claim.claim_id, text=claim.text, category=claim.category,
            is_leaked=leaked,
            reason=f"tau(c)={parsed.isoformat()} {'>' if leaked else '<='} T_ref={cutoff_date}. {reasoning}",
            method="SEARCH",
            determination_date=parsed.isoformat(),
            cutoff_date=cutoff_date,
        )

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _generate_queries(self, claims: list[Claim]) -> list[str]:
        """Generate one search query per claim (single LLM call)."""
        if not self.use_search:
            return ["" for _ in claims]

        items = [{"index": i, "claim": c.text} for i, c in enumerate(claims)]
        prompt = _QUERY_GEN_PROMPT.format(claims_json=json.dumps(items, ensure_ascii=False))
        try:
            raw = llm_client.simple_call(prompt, model=self.model, temperature=0, max_tokens=2048)
            parsed = _parse_json_array(raw)
            by_idx = {r.get("index", i): r.get("query", "") for i, r in enumerate(parsed)}
            return [by_idx.get(i, f"when was this publicly known: {claims[i].text}") for i in range(len(claims))]
        except Exception as exc:
            logger.warning("Query generation failed: %s", exc)
            return [f"when was this publicly known: {c.text}" for c in claims]

    def _run_searches(
        self,
        claims: list[Claim],
        queries: list[str],
        cutoff_date: str,
    ) -> dict[int, str]:
        """Execute batched searches. Returns {index: answer_text}.

        Uses batch_search to combine multiple claims into fewer API calls,
        reducing the per-request search surcharge from $0.005/claim to
        $0.005/batch (~10 claims).
        """
        answers: dict[int, str] = {}
        if not self.use_search:
            return answers

        to_search_idx: list[int] = []
        to_search_claims: list[str] = []

        for i, query in enumerate(queries):
            if not query:
                continue
            if self.cache is not None:
                cached = self.cache.get(claims[i].text, cutoff_date)
                if cached is not None:
                    answers[i] = cached.get("content", "")[:800]
                    continue
            to_search_idx.append(i)
            to_search_claims.append(claims[i].text)

        if not to_search_claims:
            return answers

        try:
            batch_results = search_client.batch_search(
                to_search_claims, cutoff_date=cutoff_date,
            )
            for local_j, orig_i in enumerate(to_search_idx):
                answer = batch_results.get(local_j, "")
                answers[orig_i] = answer[:800]
                if self.cache is not None:
                    self.cache.put(
                        claims[orig_i].text, cutoff_date,
                        {"content": answer, "citations": []},
                    )
        except Exception as exc:
            logger.warning("Batch search failed, falling back to empty: %s", exc)
            for orig_i in to_search_idx:
                answers[orig_i] = ""

        return answers

    def _extract_dates(
        self,
        claims: list[Claim],
        search_answers: dict[int, str],
    ) -> list[dict[str, Any]]:
        """Extract determination dates from search answers (single LLM call)."""
        if not search_answers:
            return [{} for _ in claims]

        data = [
            {"index": i, "claim": c.text, "search_result": search_answers.get(i, "No results")}
            for i, c in enumerate(claims)
        ]
        prompt = _DATE_EXTRACT_PROMPT.format(data_json=json.dumps(data, ensure_ascii=False, indent=2))
        try:
            raw = llm_client.simple_call(prompt, model=self.model, temperature=0, max_tokens=3072)
            parsed = _parse_json_array(raw)
            by_idx: dict[int, dict] = {r.get("index", i): r for i, r in enumerate(parsed)}
            return [by_idx.get(i, {}) for i in range(len(claims))]
        except Exception as exc:
            logger.warning("Date extraction failed: %s", exc)
            return [{} for _ in claims]

    def _llm_judge(self, claims: list[Claim], cutoff_date: str) -> list[dict[str, Any]]:
        """LLM judge fallback for claims where search was inconclusive."""
        items = [{"index": i, "claim": c.text, "category": c.category, "temporal_ref": c.temporal_ref}
                 for i, c in enumerate(claims)]
        prompt = _LLM_JUDGE_PROMPT.format(
            cutoff_date=cutoff_date,
            claims_json=json.dumps(items, ensure_ascii=False, indent=2),
        )
        try:
            raw = llm_client.simple_call(prompt, model=self.model, temperature=0, max_tokens=2048)
            parsed = _parse_json_array(raw)
            by_idx = {r.get("index", i): r for i, r in enumerate(parsed)}
            return [by_idx.get(i, {}) for i in range(len(claims))]
        except Exception as exc:
            logger.warning("LLM judge failed: %s", exc)
            return [{"is_leaked": False, "reasoning": f"Judge error: {exc}"} for _ in claims]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _try_parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM output with fallback strategies."""
    text = text.strip()
    # Strategy 1: direct
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    # Strategy 2: code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    # Strategy 3: find array
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    return []
