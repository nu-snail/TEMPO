"""Salary prediction: prompt template and response parser."""

from __future__ import annotations

import json
import re
from typing import Any

from ..data_loader import TaskInstance

# Temporal-hint variant (default for training and evaluation).
SYSTEM_PROMPT = """\
You are an expert sports salary analyst. Your task is to predict a free
agent's upcoming contract value (Average Annual Value in USD).

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}.
Do NOT use any knowledge of signings, trades, salary-cap changes,
or performance data from on or after {cutoff_date}.
If you are uncertain whether a piece of information was available before
the cutoff, do NOT use it.

ANALYSIS REQUIREMENTS:
- Cite the player's recent performance statistics, accolades, and injury
  history (only information available before the cutoff).
- Reference comparable contracts signed before the cutoff.
- Consider market conditions: salary cap, team needs, positional value.
- Clearly distinguish between established facts and your predictions.

OUTPUT FORMAT -- respond with a single JSON object, no markdown fences:
{{
  "predicted_salary": <integer USD, e.g. 15000000>,
  "rationale": "<detailed analysis with specific facts and comparables, at least 400 characters>"
}}"""

# Base-model variant: no temporal constraint (zero-shot baseline).
BASE_SYSTEM_PROMPT = """\
You are an expert sports salary analyst. Your task is to predict a free
agent's upcoming contract value (Average Annual Value in USD).

ANALYSIS REQUIREMENTS:
- Cite the player's recent performance statistics, accolades, and injury
  history.
- Reference comparable contracts.
- Consider market conditions: salary cap, team needs, positional value.

OUTPUT FORMAT -- respond with a single JSON object, no markdown fences:
{
  "predicted_salary": <integer USD, e.g. 15000000>,
  "rationale": "<detailed analysis with specific facts and comparables, at least 400 characters>"
}"""


def build_user_prompt(instance: TaskInstance) -> str:
    """Build the user message for a salary prediction question."""
    raw = instance.raw
    player = raw.get("Player", "Unknown")
    sport = raw.get("Sport", "NBA")
    year = raw.get("Year", raw.get("Season", ""))
    status = raw.get("Status", "")

    lines = [
        f"Sport: {sport}",
        f"Free-agent year: {year}",
        f"Player: {player}",
    ]
    if status:
        lines.append(f"Free-agent type: {status}")

    lines.append(
        "\nPredict this player's Average Annual Value (AAV) in USD for their "
        "next contract."
    )
    return "\n".join(lines)


def parse_response(text: str, instance: TaskInstance) -> dict[str, Any] | None:
    """Extract predicted salary and rationale from model output.

    Returns:
        Dict with ``predicted_salary`` (float) and ``rationale`` (str),
        or None if parsing fails.
    """
    # Strategy 1: full JSON parse
    parsed = _try_json(text)
    if parsed is not None:
        salary = parsed.get("predicted_salary")
        rationale = parsed.get("rationale", parsed.get("prediction_rationale", ""))
        if salary is not None:
            try:
                return {"predicted_salary": float(salary), "rationale": rationale}
            except (TypeError, ValueError):
                pass

    # Strategy 2: extract salary from truncated JSON via regex
    m = re.search(r'"predicted_salary"\s*:\s*([\d,.]+)', text)
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
            if val > 100_000:
                rationale = _extract_rationale(text)
                return {"predicted_salary": val, "rationale": rationale}
        except ValueError:
            pass

    # Strategy 3: dollar amounts in natural text
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|M)\b", text, re.I)
    if m:
        val = float(m.group(1).replace(",", "")) * 1_000_000
        return {"predicted_salary": val, "rationale": text}

    m = re.search(r"\$\s*([\d,]+)", text)
    if m:
        val = float(m.group(1).replace(",", ""))
        if val > 100_000:
            return {"predicted_salary": val, "rationale": text}

    # Strategy 4: bare large number near "salary" / "AAV" / "contract"
    m = re.search(r'(?:salary|aav|contract|value)[^:]*?:\s*([\d,]+)', text, re.I)
    if m:
        val = float(m.group(1).replace(",", ""))
        if val > 100_000:
            return {"predicted_salary": val, "rationale": text}

    return None


def _extract_rationale(text: str) -> str:
    """Best-effort rationale extraction from possibly truncated JSON."""
    m = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)', text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def _try_json(text: str) -> dict | None:
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
    m = re.search(r'\{[\s\S]*"predicted_salary"[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None
