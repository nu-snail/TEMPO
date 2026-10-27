"""Prompt templates used by the CoverageGatedExperiment training pipeline."""

from __future__ import annotations

import json
from typing import Any

from .types import TaskInstance

CONFIDENT_LEVELS = frozenset({"high", "medium"})

STOCK_SYSTEM_PROMPT = """You are an expert stock analyst. Your task is to rank a set of companies by
their expected stock return over a specified evaluation period.

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}.
Do NOT use any knowledge of events, earnings reports, market movements,
announcements, or developments from on or after {cutoff_date}.
If uncertain whether a piece of information predates the cutoff, omit it.

EVIDENCE INTEGRITY: Evidence must be factually genuine. Do not fabricate \
numbers, dates, revenue figures, or events. Each evidence item must reflect \
a real, publicly known fact. If you are uncertain about a specific number, \
state the fact qualitatively rather than inventing a precise figure.

You MUST follow this three-step procedure:

STEP 1 — EVIDENCE:
List every verifiable fact you will use. Requirements:
- Each fact must be concrete and specific (a reported number, a completed
  event, a published announcement) — not an opinion or prediction.
- Each fact must have been publicly known before {cutoff_date}.
- For EACH fact, include a source_date field: the YYYY-MM-DD date when this
  information was FIRST PUBLICLY ANNOUNCED (earnings release date, press release
  date, SEC filing date). This must be the ANNOUNCEMENT date, NOT the period end.
  WRONG: source_date "2023-12-31" for Q4 2023 earnings (period end, not yet public)
  RIGHT: source_date "2024-02-15" for Q4 2023 earnings (actual earnings release date)
  If the exact announcement day is uncertain, use the 1st of the announcing month.
- Write each fact as a plain statement. Do NOT include citation markers
  like [1] or [2] in the evidence text.
- Include as many facts as your analysis requires — no fixed number.

STEP 2 — REASONING:
Write a reasoning paragraph that cites evidence by number (e.g. [1], [3]).
Rules:
- You MUST cite every evidence item at least once.
- Cite each piece of evidence INLINE, next to the specific analysis it supports.
  GOOD: "CompanyA's 25% revenue growth [1] outpaced CompanyB's 8% gain [3],
         suggesting stronger momentum."
  BAD:  "CompanyA outpaced CompanyB in revenue growth. [1][2][3][4][5]"
- Do NOT introduce any new facts, statistics, events, or entity names
  that are not listed in your evidence. Only analysis and inference
  from cited evidence are allowed.
- Show a clear logical path from cited evidence to your ranking.

STEP 3 — PREDICTION:
Give your final ranking based solely on the reasoning above.

OUTPUT FORMAT — a single JSON object, no markdown fences:
{{
  "evidence": [
    {{"id": 1, "fact": "<plain verifiable fact>", "source_date": "YYYY-MM-DD"}},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "ranking": ["BEST_TICKER", ..., "WORST_TICKER"]
}}"""

SALARY_SYSTEM_PROMPT = """You are an expert sports salary analyst. Your task is to predict a free
agent's upcoming contract value (Average Annual Value in USD).

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}.
Do NOT use any knowledge of signings, trades, salary-cap changes,
or performance data from on or after {cutoff_date}.
If uncertain whether a piece of information predates the cutoff, omit it.

EVIDENCE INTEGRITY: Evidence must be factually genuine. Do not fabricate \
numbers, dates, contract figures, or events. Each evidence item must reflect \
a real, publicly known fact. If you are uncertain about a specific number, \
state the fact qualitatively rather than inventing a precise figure.

You MUST follow this three-step procedure:

STEP 1 — EVIDENCE:
List every verifiable fact you will use. Requirements:
- Each fact must be concrete and specific (a player stat, a signed contract,
  a reported market condition) — not an opinion or prediction.
- Each fact must have been publicly known before {cutoff_date}.
- For EACH fact, include a source_date field: the YYYY-MM-DD date when this
  information was FIRST PUBLICLY ANNOUNCED (contract signing date, official
  roster announcement date, box score publication date). This must be the
  ANNOUNCEMENT date, NOT the date the event took place.
  WRONG: source_date "2023-07-05" for "Comparable Player signed a 4yr/$180M deal"
    when the cutoff is before July 2023 — contracts signed during or after the
    current free-agency period post-date the cutoff and MUST NOT be used.
  RIGHT: source_date "2022-07-15" for "Comparable Player signed a 3yr/$100M deal"
    — a contract from a prior free-agency cycle, safely before the cutoff.
  If the exact announcement day is uncertain, use the 1st of the announcing month.
- Write each fact as a plain statement. Do NOT include citation markers
  like [1] or [2] in the evidence text.
- Include as many facts as your analysis requires — no fixed number.

STEP 2 — REASONING:
Write a reasoning paragraph that cites evidence by number (e.g. [1], [3]).
Rules:
- You MUST cite every evidence item at least once.
- Cite each piece of evidence INLINE, next to the specific analysis it supports.
  GOOD: "PlayerA's 28.5 PPG [1] and comparable PlayerB's $40M/yr deal [3]
         suggest a contract in the $38-42M range."
  BAD:  "PlayerA deserves a max contract based on performance. [1][2][3][4]"
- Do NOT introduce any new facts, statistics, player details, or contract
  information not listed in your evidence. Only analysis and inference
  from cited evidence are allowed.
- Show how comparable contracts, player performance, and market conditions
  lead to your salary estimate.

STEP 3 — PREDICTION:
Give your predicted AAV based solely on the reasoning above.

OUTPUT FORMAT — a single JSON object, no markdown fences:
{{
  "evidence": [
    {{"id": 1, "fact": "<plain verifiable fact>", "source_date": "YYYY-MM-DD"}},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "predicted_salary": <integer USD>
}}"""

LEGAL_SYSTEM_PROMPT = """You are an expert Supreme Court analyst. Your task is to predict the outcome
of a case based on its facts, the legal questions presented, and relevant
precedent.

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}.
Do NOT use any knowledge of the case's actual decision, subsequent rulings,
commentary, or legislative responses from on or after {cutoff_date}.
If uncertain whether a piece of information predates the cutoff, omit it.

EVIDENCE INTEGRITY: Evidence must be factually genuine. Do not fabricate \
case names, holdings, dates, or legal principles. Each evidence item must \
reflect a real, publicly known fact or decided case. If you are uncertain \
about a specific detail, state the fact qualitatively rather than inventing it.

You MUST follow this three-step procedure:

STEP 1 — EVIDENCE:
List every verifiable fact and legal precedent you will use. Requirements:
- Each item must be a specific case fact, or a decided case with its
  holding — not an opinion or prediction.
- Each item must have been publicly known before {cutoff_date}.
- For EACH item, include a source_date field: the YYYY-MM-DD date when this
  information was FIRST PUBLICLY AVAILABLE (decision date of a precedent case,
  filing date of a brief, lower court opinion date). This must be the date the
  document or decision was publicly released, NOT the date events it describes occurred.
  WRONG: source_date "2022-06-27" for "The Court ruled in Kennedy v. Bremerton
    School District" — this is the very decision you are predicting; the outcome
    and any post-argument information MUST NOT be cited.
  RIGHT: source_date "2020-07-08" for "In Espinoza v. Montana (2020), the Court
    struck down a state Blaine Amendment under the Free Exercise Clause"
    — a prior term's decided case, safely before the cutoff.
  If the exact release day is uncertain, use the 1st of the releasing month.
- Write each item as a plain statement. Do NOT include citation markers
  like [1] or [2] in the evidence text.
- Include as many items as your analysis requires — no fixed number.

STEP 2 — REASONING:
Write a reasoning paragraph that cites evidence by number (e.g. [1], [3]).
Rules:
- You MUST cite every evidence item at least once.
- Cite each piece of evidence INLINE, next to the specific analysis it supports.
  GOOD: "The holding in Chevron [1] conflicts with the broader deference
         framework established in Brand X [3], weakening the respondent's position."
  BAD:  "The petitioner's argument is supported by prior precedent. [1][2][3][4]"
- Do NOT introduce any new facts, case names, legal principles, or holdings
  not listed in your evidence. Only analysis and inference from cited
  evidence are allowed.
- Analyse each side's arguments using only cited precedent and facts.

STEP 3 — PREDICTION:
Give your predicted outcome based solely on the reasoning above.

OUTPUT FORMAT — a single JSON object, no markdown fences:
{{
  "evidence": [
    {{"id": 1, "fact": "<plain verifiable fact or case holding>", "source_date": "YYYY-MM-DD"}},
    ...
  ],
  "reasoning": "<paragraph citing [N], no new facts>",
  "probability_petitioner": <float 0.0 to 1.0>
}}"""

SOURCE_B_PROMPT = """Extract ungrounded factual claims from each completion's reasoning.
A factual claim is an assertion about a specific event, statistic, entity, date, or
measurable state — NOT an opinion, inference, or judgment from the evidence.
For each completion, list claims in the reasoning NOT present in its evidence list.
Each claim must be atomic (one verifiable fact).

---
{completions_block}
---

Output a single JSON array (starting with [ ending with ]):
[{{"completion_idx": 0, "extra_claims": [{{"claim": "<atomic factual claim>"}}, ...]}}, ...]
Output ONLY the JSON array, no other text."""

CATEGORIZE_PROMPT = """Categorize each claim and generate a search query if needed.
Categories: A1=discrete event, A2=state/measurement, A3=publication/announcement,
A4=outcome (always leaked), A5=consequential (always leaked),
B1=background, B2=definitional.
A1/A2/A3: provide a date-oriented search query (entity, event, year).
A4/A5/B1/B2: null.

CLAIMS:
{claims_json}

Output a single JSON array (starting with [ ending with ]):
[{{"index": 0, "category": "A1", "search_query": "query or null"}}, ...]
Output ONLY the JSON array, no other text."""

LLM_DATE_CHECK_PROMPT = """Expert fact-checker. For each claim, use ONLY your own knowledge to determine:
- event_date: YYYY-MM-DD when publicly available, or null if unsure. DO NOT GUESS.
- factual_status: supported / unsupported / contradicted
- confidence: high / medium / low / none
Return null date + "none" confidence rather than guess. Return "unsupported" rather than assume.
Earnings are public on announcement date, not period end.

CLAIMS:
{claims_json}

Output a single JSON array (starting with [ ending with ]):
[{{"index": 0, "event_date": "YYYY-MM-DD", "factual_status": "supported", "confidence": "high"}}, ...]
Output ONLY the JSON array, no other text."""

DATE_VERIFY_PROMPT = """For each claim, use the search result to determine:
- event_date: YYYY-MM-DD when publicly available, or null if unclear.
- factual_status: supported / unsupported / contradicted
- confidence: high / medium / low / none
Minor rounding (e.g. $18.1B vs $18.12B) = supported. Material discrepancy = contradicted.

CLAIMS WITH SEARCH RESULTS:
{data_json}

Output a single JSON array (starting with [ ending with ]):
[{{"index": 0, "event_date": "YYYY-MM-DD", "factual_status": "supported", "confidence": "high"}}, ...]
Output ONLY the JSON array, no other text."""

FALLBACK_PROMPT = """Temporal leakage judge. A claim is "leaked" if its fact was NOT publicly
determinable before the cutoff date. Focus on when the fact was first verifiable.

Cutoff date: {cutoff_date}

Claims:
{claims_json}

Output a single JSON array (starting with [ ending with ]):
[{{"index": 0, "is_leaked": true}}, ...]
Output ONLY the JSON array, no other text."""

LOGIC_PROMPT = """Audit each completion's reasoning for logical flaws.
Flaw types: CONTRADICTION, UNSUPPORTED_LEAP, SELECTIVE_EVIDENCE, PREDICTION_MISMATCH.
Severity: major (undermines conclusion) or minor (weakness, doesn't invalidate).
Count total analytical steps and sound steps.

---
{completions_block}
---

Output a single JSON array (starting with [ ending with ]):
[{{"completion_idx": 0, "flaws": [{{"type": "CONTRADICTION", "severity": "major"}}], "total_reasoning_steps": 5, "sound_steps": 4}}, ...]
Empty flaws array if reasoning is sound. Output ONLY the JSON array, no other text."""


CLAIM_EXTRACTION_PROMPT = """# Factual Claim Extraction for Temporal Leakage Detection

## Task
Extract ALL factual claims from the rationale below. Requirements:
1. Comprehensive: every factual assertion must be captured.
2. Faithful: extract only what is stated; do NOT add external knowledge.
3. Self-contained: each claim must be understandable alone.
4. Atomic: one verifiable fact per claim.

## Context
- Prediction task: {task_description}
- Target event: {event_description}
- Knowledge cutoff (T_ref): {cutoff_date}

## Rationale
{rationale}

## Claim Categories

### Group A -- Temporally Verifiable
- A1 Discrete Event: a specific event occurred (e.g. "Apple announced iPhone 15 on Sep 12 2023").
- A2 State/Measurement: a quantity or state at a point in time (e.g. "NVIDIA Q3 2023 revenue was $18.1B").
- A3 Publication/Announcement: content from a dated document (e.g. "the majority opinion in Dobbs stated ...").
- A4 Outcome: directly states the answer to the prediction question (always leaked).
- A5 Consequential: references events AFTER the target event (always leaked).

### Group B -- Temporally Non-Applicable
- B1 Background: timeless or well-established facts (e.g. "the Supreme Court has nine justices").
- B2 Definitional: definitions, tautologies, logical derivations.

## What NOT to Extract
Exclude predictions, judgments, speculation, confidence statements, and analytical
interpretations -- these are reasoning, not verifiable facts.

## Rules
1. Preserve original meaning; add context to make each claim standalone.
2. Every Group A claim must include a temporal reference.
3. Decompose compound facts into separate claims.
4. Record the original text span for each claim.

## Output -- JSON only, no markdown fences

{{
  "claims": [
    {{
      "claim_id": 1,
      "text": "<self-contained claim>",
      "original_span": "<exact text from rationale>",
      "temporal_ref": "<time reference or null for Group B>",
      "category": "<A1|A2|A3|A4|A5|B1|B2>",
      "category_reason": "<brief justification>"
    }}
  ],
  "notes": "<any extraction issues>"
}}"""

QUERY_GEN_PROMPT = """Generate a concise web-search query to find the SPECIFIC DATE when each claim
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


DATE_PLAUSIBILITY_PROMPT = """You are a temporal audit verifier for financial, sports, and legal analysis.

Cutoff date: {cutoff}

For each evidence item you are given:
- The factual claim text
- The declared_source_date: the date the model claims this information was
  first publicly available (YYYY-MM-DD)

Your task: verify whether the declared_source_date is PLAUSIBLE for this
type of information.

VERIFICATION RULES:
- Full-year financial results for year Y (annual revenue, net income, EPS,
  EBITDA in a 10-K or earnings call) are first published in Jan-Feb of Y+1.
  A date of "YYYY-12-xx" for "Company reported full-year YYYY results" is
  IMPLAUSIBLE — year-end totals require the fiscal year to close first.
- Quarterly results (Q1-Q4) are published 1-2 months after that quarter ends.
  Q3 results (period ending Sep 30) appear in Oct-Nov. Q4/full-year (Dec 31)
  appear in Jan-Feb of the next year.
- Non-calendar fiscal years: if the fact explicitly states the company's
  fiscal quarter or year end date, use that to judge plausibility. E.g.,
  a company whose fiscal year ends July 31 can publish FY results in Sep.
- Stock prices, intraday market data, and exchange-reported closing prices
  are available on the trading date itself — any date on or before the
  stated date is plausible.
- Acquisition / merger announcements: use the press release / 8-K filing date.
- Contract signings (sports, government): use the announced signing date.
- Court decisions: use the decision date or opinion release date.
- SEC 10-K annual reports: typically published 60-90 days after fiscal year end.
- If the declared_source_date seems too early for the event type (model
  under-declared to avoid leakage detection), set plausible=false and provide
  the earliest realistic corrected_date.
- If the declared_source_date is plausible or uncertain, set plausible=true
  and corrected_date=null. Do NOT fabricate corrections.

ITEMS:
{items_json}

Output a JSON array, one entry per item:
[{{"index": 0, "plausible": true, "corrected_date": null, "reason": "<one sentence>"}}]
Output ONLY the JSON array."""

HALLUCINATION_DETECT_PROMPT = """You are a financial fact-checker. For each claim, check if the stated facts/numbers \
are correct based on your knowledge of public financial data.

Verdict rules:
- "accurate": Numbers are correct or within ~5% of the real value AND direction is correct.
- "inaccurate": Numbers are off by >15%, OR the direction is wrong (e.g., claimed increase but \
actually decreased), OR the claimed value is for the wrong entity/time period.
- "uncertain": You genuinely lack knowledge to check this claim. Use sparingly.

IMPORTANT: If you know the real number and it differs from the claimed number by more than 15%, \
that is "inaccurate" even if both are in the same general direction. For example, if the real \
value is $40B but the claim says $34B, that's a ~15% difference — mark "inaccurate".

CLAIMS:
{claims_json}

Output a JSON array:
[{{"index": 0, "verdict": "accurate|inaccurate|uncertain", "known_value": "<what you believe the real value is, or null>", "reason": "<brief>"}}]
Output ONLY the JSON array."""
# DEPRECATED: replaced by HALL_SONAR_VERIFY_PROMPT + HALL_BATCH_JUDGE_PROMPT for validation/evaluation.


HALL_SONAR_VERIFY_PROMPT = """You are a {role}. Verify this specific claim using web search.

Claim: {claim}

Search for direct evidence ({search_hints}).
Return ONLY a JSON object with no extra text:
{{{{
  "verdict": "supported",
  "found_value": "key fact you found (null if nothing found)",
  "source_hint": "source description max 15 words",
  "reason": "one sentence: why this verdict"
}}}}

Verdict rules:
- "supported": search confirms the claim's key numbers/facts
- "unsupported": search found conflicting facts or a clear error
- "undecidable": no relevant evidence found"""


HALL_BATCH_JUDGE_PROMPT = """You are a {role}. Each claim below has a search result attached.
Based ONLY on the provided search evidence, assign a verdict to each claim.

Verdict rules:
- "supported": search evidence clearly confirms the claim
- "unsupported": search evidence clearly contradicts the claim
- "undecidable": search evidence is missing, ambiguous, or insufficient

CLAIMS WITH SEARCH EVIDENCE:
{claims_with_evidence}

Output ONLY a JSON array:
[{{{{"index": 0, "verdict": "supported|unsupported|undecidable", "reason": "max 15 words"}}}}]
Output ONLY the JSON array."""


SONAR_LEAK_VERIFY_PROMPT = """You are a temporal leakage verifier. For each claim, a web search was \
performed to find when the information became publicly available.

Cutoff date: {cutoff}

For each item, determine based on the search result:
1. event_date: The specific date (YYYY-MM-DD) when this fact was first publicly available. \
Use null if the search result does not contain a clear date.
2. verdict: "leaked" if the event_date is AFTER the cutoff date, "not_leaked" if BEFORE or ON \
the cutoff date. Default to "not_leaked" if the date cannot be determined.

DATA:
{data_json}

Output a JSON array:
[{{"index": 0, "event_date": "YYYY-MM-DD or null", "verdict": "leaked|not_leaked"}}]
Output ONLY the JSON array."""


def hall_prompt_params(task: str) -> dict[str, str]:
    """Return role and search_hints for hallucination detection prompts by task."""
    if task == "stock":
        return {
            "role": "financial fact-checker",
            "search_hints": "earnings reports, press releases, SEC filings, news",
        }
    if task == "salary":
        return {
            "role": "sports fact-checker",
            "search_hints": "player statistics, contract records, team transactions, sports news",
        }
    if task == "legal":
        return {
            "role": "legal fact-checker",
            "search_hints": "court opinions, legal filings, case records, news reports",
        }
    return {"role": "fact-checker", "search_hints": "news reports, public records"}


def build_hall_sonar_prompt(task: str, claim: str) -> str:
    """Build a Sonar search prompt for hallucination verification."""
    params = hall_prompt_params(task)
    return HALL_SONAR_VERIFY_PROMPT.format(claim=claim, **params)


def build_hall_batch_prompt(task: str, claims_with_evidence: str) -> str:
    """Build a batch judge prompt for hallucination verdicts."""
    params = hall_prompt_params(task)
    return HALL_BATCH_JUDGE_PROMPT.format(claims_with_evidence=claims_with_evidence, **params)


def build_rationale_text(parsed: dict) -> str:
    """Combine evidence + reasoning into a single rationale string for claim extraction."""

    parts: list[str] = []
    for ev in parsed.get("evidence", []):
        parts.append(f"[{ev.get('id', '?')}] {ev.get('fact', '')}")
    parts.append("")
    parts.append(parsed.get("reasoning", ""))
    return "\n".join(parts)


def task_description(task: str) -> str:
    return {
        "stock": "Stock return ranking prediction",
        "salary": "Free-agent salary prediction",
        "legal": "Supreme Court case outcome prediction",
    }.get(task, "Prediction task")


def event_description(instance: TaskInstance) -> str:
    if instance.task == "stock":
        p = instance.raw.get("past_period", {})
        return f"Stock ranking {p.get('start_date', '')} to {p.get('end_date', '')}"
    if instance.task == "salary":
        return f"Contract for {instance.raw.get('Player', '')} ({instance.raw.get('Year', '')})"
    if instance.task == "legal":
        return f"Outcome of {instance.raw.get('case_name', '')}"
    return ""


def build_generation_prompt(instance: TaskInstance) -> tuple[str, str]:
    """Build `(system, user)` prompt pair for one instance."""

    if instance.task == "stock":
        return STOCK_SYSTEM_PROMPT.format(cutoff_date=instance.cutoff_date), _build_stock_user(instance)
    if instance.task == "salary":
        return SALARY_SYSTEM_PROMPT.format(cutoff_date=instance.cutoff_date), _build_salary_user(instance)
    if instance.task == "legal":
        return LEGAL_SYSTEM_PROMPT.format(cutoff_date=instance.cutoff_date), _build_legal_user(instance)
    raise ValueError(f"Unsupported task: {instance.task}")


def build_completions_block(parsed_outputs: list[dict[str, Any]]) -> str:
    """Serialize parseable completions for batched judge prompts."""

    chunks: list[str] = []
    for idx, parsed in enumerate(parsed_outputs):
        chunk = (
            f"COMPLETION {idx}:\n"
            f"EVIDENCE LIST:\n{json.dumps(parsed.get('evidence', []), ensure_ascii=False)}\n"
            f"REASONING:\n{parsed.get('reasoning', '')}\n"
        )
        if "ranking" in parsed or "predicted_salary" in parsed or "probability_petitioner" in parsed:
            chunk += f"PREDICTION:\n{json.dumps(_prediction_view(parsed), ensure_ascii=False)}\n"
        chunks.append(chunk)
    return "\n".join(chunks)


def _prediction_view(parsed: dict[str, Any]) -> dict[str, Any]:
    if "ranking" in parsed:
        return {"ranking": parsed["ranking"]}
    if "predicted_salary" in parsed:
        return {"predicted_salary": parsed["predicted_salary"]}
    return {
        "probability_petitioner": parsed.get("probability_petitioner"),
        "predicted_winner": parsed.get("predicted_winner"),
    }


def _build_stock_user(instance: TaskInstance) -> str:
    raw = instance.raw
    period = raw.get("past_period", {})
    lines = [
        f"Sector: {raw.get('sector', 'Unknown')}",
        f"Evaluation period: {period.get('start_date', period.get('start', ''))} to {period.get('end_date', period.get('end', ''))}",
        "Companies to rank:",
    ]
    company_names = raw.get("company_names", {})
    for ticker in raw.get("companies", []):
        lines.append(f"  - {ticker} ({company_names.get(ticker, ticker)})")
    return "\n".join(lines)


def _build_salary_user(instance: TaskInstance) -> str:
    raw = instance.raw
    lines = [
        f"Sport: {raw.get('Sport', '')}",
        f"Free-agent year: {raw.get('Year', '')}",
        f"Player: {raw.get('Player', '')}",
    ]
    status = raw.get("Status", "")
    if status:
        lines.append(f"Free-agent type: {status}")
    lines.append("")
    lines.append("Predict this player's Average Annual Value (AAV) in USD for their next contract.")
    return "\n".join(lines)


def _build_legal_user(instance: TaskInstance) -> str:
    raw = instance.raw
    lines = [
        f"Case: {raw.get('case_name', '')}",
        f"Term: {raw.get('term', '')}",
        f"Petitioner: {raw.get('petitioner', '')}",
        f"Respondent: {raw.get('respondent', '')}",
    ]
    summary = raw.get("summary", "")
    if summary:
        lines.append("")
        lines.append(f"Facts: {summary}")
    lines.append("")
    lines.append("Predict the probability that the PETITIONER wins (0.0 to 1.0).")
    return "\n".join(lines)


# ======================================================================
# RAG-specific prompts and builders
# ======================================================================

QUERY_GENERATION_PROMPT = """You are generating web search queries to gather relevant evidence for a task.
All evidence must predate the cutoff date. You will receive the same task
description that an analyst would receive. Generate search queries that would
help you find useful information to answer this task.

CUTOFF DATE: {cutoff_date}

TASK DESCRIPTION:
{user_prompt}

Generate between {min_queries} and {max_queries} search queries.
- Each query should target a DISTINCT aspect of the information you need.
- Include the year in each query to anchor temporal relevance.

Output as JSON array: [{{"query": "..."}}, ...]"""

_RAG_QUERY_RANGES: dict[str, tuple[int, int]] = {
    "stock": (7, 12),
    "salary": (3, 6),
    "legal": (3, 6),
}


def get_rag_query_range(task: str) -> tuple[int, int]:
    """Return (min_queries, max_queries) for a task's RAG query generation."""
    return _RAG_QUERY_RANGES.get(task, (3, 6))


def build_rag_context_block(results: list[dict[str, Any]], cutoff_date: str) -> str:
    """Format search results into a plain-text context block for RAG prompts.

    Following academic convention (Atlas, In-Context RALM, Self-RAG), retrieved
    passages are presented as plain numbered text without temporal metadata or
    validity framing. No dates, no "published before X", no behavioral guidance.
    The model must determine temporal relevance from content alone, the same
    way base/RL variants do from parametric memory.

    Args:
        results: List of dicts with keys 'title', 'snippet', and optionally 'date'.
        cutoff_date: Unused; kept for interface compatibility with retrieval logging.

    Returns:
        Formatted context string ready to prepend to the user prompt.
    """
    if not results:
        return ""
    lines = ["The following material may be relevant to the task:"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        snippet = r.get("snippet") or r.get("content") or ""
        lines.append(f"[{i}] {title}")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


def build_rag_generation_prompt(
    instance: TaskInstance, context_block: str
) -> tuple[str, str]:
    """Build (system, user) prompt pair with RAG context prepended.

    The system prompt is identical to ``build_generation_prompt``.
    The user prompt has the context block prepended before the original user text.
    """
    system, user = build_generation_prompt(instance)
    augmented_user = context_block + user if context_block else user
    return system, augmented_user


# ======================================================================
# TimeSPEC-specific prompts and builders
# ======================================================================

TIMESPEC_CLAIM_VERIFY_PROMPT = """You are a temporal fact verifier. For each claim below, web search
results are provided. Determine when the information was first publicly
available.

Cutoff date: {cutoff}

For each item:
1. event_date: YYYY-MM-DD when first publicly available, or null if
   search results lack a clear date.
2. verdict: "leaked" if event_date is AFTER the cutoff, "not_leaked"
   if BEFORE or ON the cutoff. Default to "not_leaked" if the date
   cannot be determined.

ITEMS:
{items_json}

Output a JSON array:
[{{"index": 0, "event_date": "YYYY-MM-DD or null", "verdict": "leaked|not_leaked"}}]
Output ONLY the JSON array."""


TIMESPEC_AGGREGATOR_CONSTRAINT = """

CLOSED-WORLD CONSTRAINT:
For STEP 1, you MUST select evidence exclusively from the VALIDATED FACTS
list provided in the task input below. Do NOT introduce any facts, numbers,
events, or entities not present in that list. If the list is insufficient,
base your analysis only on what is available."""


def build_timespec_regeneration_prompt(
    instance: TaskInstance,
    context_block: str,
    leaked_claims: list[dict[str, Any]],
) -> tuple[str, str]:
    """Build (system, user) prompt for the TimeSPEC Regenerator phase.

    System prompt is byte-for-byte identical to base/RAG. A constraint block
    listing claims to avoid is appended to the user prompt. No temporal
    information, no explanation of why, no corrected values.
    """
    system, user = build_generation_prompt(instance)
    augmented = context_block + user if context_block else user
    lines = ["\n\nCONSTRAINT — do NOT use the following facts in your analysis:"]
    for i, c in enumerate(leaked_claims, 1):
        lines.append(f"  {i}. {c['text']}")
    lines.append("\nRewrite your analysis without these facts.")
    return system, augmented + "\n".join(lines)


def build_timespec_aggregator_prompt(
    instance: TaskInstance,
    validated_claims: list[dict[str, Any]],
) -> tuple[str, str]:
    """Build (system, user) prompt for the TimeSPEC Aggregator phase.

    System prompt = original task system prompt + closed-world constraint.
    User prompt = plain-text validated facts (no dates, no categories, no
    metadata) followed by the original task details.
    """
    system, user = build_generation_prompt(instance)
    system += TIMESPEC_AGGREGATOR_CONSTRAINT
    facts = "VALIDATED FACTS:\n"
    for i, c in enumerate(validated_claims, 1):
        facts += f"  {i}. {c['text']}\n"
    facts += "\n"
    return system, facts + user

