"""Stock return ranking: prompt template and response parser."""

from __future__ import annotations

import json
from typing import Any

from ..data_loader import TaskInstance

# Temporal-hint variant (default for training and evaluation).
SYSTEM_PROMPT = """\
You are an expert stock analyst. Your task is to rank a set of companies by
their expected stock return over a specified evaluation period.

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}.
Do NOT use any knowledge of events, earnings reports, market movements,
announcements, or developments from on or after {cutoff_date}.
If you are uncertain whether a piece of information was available before
the cutoff, do NOT use it.

ANALYSIS REQUIREMENTS:
- For each company, cite specific, verifiable facts (earnings, contracts,
  industry trends, macro conditions) that were known before the cutoff.
- Explain why you expect each company to outperform or underperform relative
  to its peers over the evaluation period.
- Do NOT recall or reference actual stock performance during the period.

OUTPUT FORMAT -- respond with a single JSON object, no markdown fences:
{{
  "ranking": ["BEST_TICKER", ..., "WORST_TICKER"],
  "rationale": "<detailed analysis with verifiable facts for every company, at least 800 characters>"
}}"""

# Base-model variant: no temporal constraint (zero-shot baseline).
BASE_SYSTEM_PROMPT = """\
You are an expert stock analyst. Your task is to rank a set of companies by
their expected stock return over a specified evaluation period.

ANALYSIS REQUIREMENTS:
- For each company, cite specific, verifiable facts (earnings, contracts,
  industry trends, macro conditions).
- Explain why you expect each company to outperform or underperform relative
  to its peers over the evaluation period.

OUTPUT FORMAT -- respond with a single JSON object, no markdown fences:
{
  "ranking": ["BEST_TICKER", ..., "WORST_TICKER"],
  "rationale": "<detailed analysis with verifiable facts for every company, at least 800 characters>"
}"""


def build_user_prompt(instance: TaskInstance) -> str:
    """Build the user message for a stock ranking question."""
    raw = instance.raw
    companies = raw.get("companies", [])
    names = raw.get("company_names", {})
    sector = raw.get("sector", "Unknown")
    period = raw.get("past_period", {})
    start = period.get("start_date", period.get("start", ""))
    end = period.get("end_date", period.get("end", ""))

    lines = [
        f"Sector: {sector}",
        f"Evaluation period: {start} to {end}",
        f"Companies to rank (best to worst expected return):",
    ]
    for ticker in companies:
        name = names.get(ticker, ticker)
        lines.append(f"  - {ticker} ({name})")

    return "\n".join(lines)


def parse_response(text: str, instance: TaskInstance) -> dict[str, Any] | None:
    """Extract ranking and rationale from model output.

    Returns:
        Dict with ``ranking`` (list[str]) and ``rationale`` (str),
        or None if parsing fails.
    """
    import re

    valid_tickers = set(instance.raw.get("companies", []))

    # Strategy 1: full JSON parse
    parsed = _try_json(text)
    if parsed is not None:
        ranking = parsed.get("ranking")
        rationale = parsed.get("rationale", parsed.get("ranking_rationale", ""))
        if isinstance(ranking, list) and ranking:
            ranking = [t for t in ranking if t in valid_tickers]
            if ranking:
                return {"ranking": ranking, "rationale": rationale}

    # Strategy 2: extract ranking array via regex (handles truncated JSON)
    m = re.search(r'"ranking"\s*:\s*\[([^\]]*)\]', text)
    if m:
        tickers = re.findall(r'"([A-Z]{1,5})"', m.group(1))
        tickers = [t for t in tickers if t in valid_tickers]
        if tickers:
            rationale = _extract_rationale(text)
            return {"ranking": tickers, "rationale": rationale}

    # Strategy 3: look for ordered ticker symbols in the text
    found = []
    seen = set()
    for m in re.finditer(r'\b([A-Z]{1,5})\b', text):
        t = m.group(1)
        if t in valid_tickers and t not in seen:
            found.append(t)
            seen.add(t)
    if len(found) >= 2:
        return {"ranking": found, "rationale": text}

    return None


def _extract_rationale(text: str) -> str:
    """Best-effort rationale extraction from possibly truncated JSON."""
    import re
    m = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)', text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def _try_json(text: str) -> dict | None:
    import re
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try to find outermost JSON object
    m = re.search(r'\{[\s\S]*"ranking"[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None
