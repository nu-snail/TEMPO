"""Legal case prediction: prompt template and response parser."""

from __future__ import annotations

import json
import re
from typing import Any

from ..data_loader import TaskInstance

# Temporal-hint variant (default for training and evaluation).
SYSTEM_PROMPT = """\
You are an expert Supreme Court analyst. Your task is to predict the outcome
of a case based on its facts, the legal questions presented, and relevant
precedent.

STRICT TEMPORAL CONSTRAINT:
You may ONLY use information that was publicly available BEFORE {cutoff_date}.
Do NOT use any knowledge of the case's actual decision, subsequent rulings,
commentary, or legislative responses from on or after {cutoff_date}.
If you are uncertain whether a piece of information was available before
the cutoff, do NOT use it.

ANALYSIS REQUIREMENTS:
- Analyse the legal questions and the positions of both parties.
- Cite relevant precedent cases decided before the cutoff.
- Consider the ideological composition of the Court at the time.
- Assess the strength of each side's legal arguments.
- Clearly separate established legal facts from your predictive reasoning.

OUTPUT FORMAT -- respond with a single JSON object, no markdown fences:
{{
  "probability_petitioner": <float from 0.0 to 1.0>,
  "predicted_winner": "<PETITIONER or RESPONDENT>",
  "rationale": "<detailed legal analysis with case citations, at least 400 characters>"
}}"""

# Base-model variant: no temporal constraint (zero-shot baseline).
BASE_SYSTEM_PROMPT = """\
You are an expert Supreme Court analyst. Your task is to predict the outcome
of a case based on its facts, the legal questions presented, and relevant
precedent.

ANALYSIS REQUIREMENTS:
- Analyse the legal questions and the positions of both parties.
- Cite relevant precedent cases.
- Consider the ideological composition of the Court at the time.
- Assess the strength of each side's legal arguments.

OUTPUT FORMAT -- respond with a single JSON object, no markdown fences:
{
  "probability_petitioner": <float from 0.0 to 1.0>,
  "predicted_winner": "<PETITIONER or RESPONDENT>",
  "rationale": "<detailed legal analysis with case citations, at least 400 characters>"
}"""


def build_user_prompt(instance: TaskInstance) -> str:
    """Build the user message for a legal case prediction question."""
    raw = instance.raw
    case_name = raw.get("case_name", "Unknown")
    petitioner = raw.get("petitioner", "")
    respondent = raw.get("respondent", "")
    term = raw.get("term", "")

    lines = [
        f"Case: {case_name}",
        f"Term: {term}",
        f"Petitioner: {petitioner}",
        f"Respondent: {respondent}",
        "\nPredict the probability that the PETITIONER wins (0.0 to 1.0).",
    ]
    return "\n".join(lines)




def parse_response(text: str, instance: TaskInstance) -> dict[str, Any] | None:
    """Extract prediction probability and rationale from model output.

    Returns:
        Dict with ``probability_petitioner`` (float), ``predicted_winner`` (str),
        and ``rationale`` (str), or None if parsing fails.
    """
    # Strategy 1: full JSON parse
    parsed = _try_json(text)
    if parsed is not None:
        prob = parsed.get("probability_petitioner")
        rationale = parsed.get("rationale", parsed.get("prediction_rationale", ""))
        winner = parsed.get("predicted_winner", "")
        if prob is not None:
            try:
                prob = float(prob)
                prob = max(0.0, min(1.0, prob))
                if not winner:
                    winner = "PETITIONER" if prob >= 0.5 else "RESPONDENT"
                return {
                    "probability_petitioner": prob,
                    "predicted_winner": winner,
                    "rationale": rationale,
                }
            except (TypeError, ValueError):
                pass

    # Strategy 2: extract probability from truncated JSON via regex
    m = re.search(r'"probability_petitioner"\s*:\s*([\d.]+)', text)
    if m:
        try:
            prob = max(0.0, min(1.0, float(m.group(1))))
            winner_m = re.search(r'"predicted_winner"\s*:\s*"([^"]*)"', text)
            winner = winner_m.group(1).upper() if winner_m else ""
            if not winner or winner not in ("PETITIONER", "RESPONDENT"):
                winner = "PETITIONER" if prob >= 0.5 else "RESPONDENT"
            rationale = _extract_rationale(text)
            return {
                "probability_petitioner": prob,
                "predicted_winner": winner,
                "rationale": rationale,
            }
        except ValueError:
            pass

    # Strategy 3: natural-language probability mention
    m = re.search(r"probability[^:]*:\s*(0\.\d+|1\.0|0|1)", text, re.I)
    if m:
        prob = float(m.group(1))
        winner = "PETITIONER" if prob >= 0.5 else "RESPONDENT"
        return {
            "probability_petitioner": prob,
            "predicted_winner": winner,
            "rationale": text,
        }

    # Strategy 4: percentage mention
    m = re.search(r"(\d{1,3})%\s*(?:chance|probability|likelihood)", text, re.I)
    if m:
        prob = max(0.0, min(1.0, int(m.group(1)) / 100.0))
        winner = "PETITIONER" if prob >= 0.5 else "RESPONDENT"
        return {
            "probability_petitioner": prob,
            "predicted_winner": winner,
            "rationale": text,
        }

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
    m = re.search(r'\{[\s\S]*"probability_petitioner"[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None
