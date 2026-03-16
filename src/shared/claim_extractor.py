"""Claim extraction from prediction rationales.

Decomposes a free-text rationale into atomic, self-contained factual claims
with temporal references and claim-taxonomy categories (A1-A5, B1-B2).
Used by the reward pipeline to compute OLR and by the evaluation pipeline
to compute Shapley-DCLR.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from . import llm_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

class ClaimCategory(str, Enum):
    """Claim categories for temporal leakage detection."""
    A1_DISCRETE_EVENT = "A1"
    A2_STATE_MEASUREMENT = "A2"
    A3_PUBLICATION = "A3"
    A4_OUTCOME = "A4"
    A5_CONSEQUENTIAL = "A5"
    B1_BACKGROUND = "B1"
    B2_DEFINITIONAL = "B2"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    """A single atomic claim extracted from a rationale."""
    claim_id: int
    text: str
    original_span: str
    temporal_ref: str | None
    category: str
    category_reason: str

    @property
    def is_group_a(self) -> bool:
        return self.category.startswith("A")

    @property
    def is_always_leaked(self) -> bool:
        return self.category in ("A4", "A5")

    @property
    def needs_search(self) -> bool:
        return self.category in ("A1", "A2", "A3")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Result of claim extraction from one rationale."""
    claims: list[Claim]
    notes: str
    raw_response: str

    @property
    def group_a(self) -> list[Claim]:
        return [c for c in self.claims if c.is_group_a]

    @property
    def group_b(self) -> list[Claim]:
        return [c for c in self.claims if not c.is_group_a]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "notes": self.notes,
            "num_claims": len(self.claims),
            "num_group_a": len(self.group_a),
            "num_group_b": len(self.group_b),
        }


# ---------------------------------------------------------------------------
# Prompt  (double braces escaped for str.format)
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
# Factual Claim Extraction for Temporal Leakage Detection

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


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ClaimExtractor:
    """Extract atomic claims from prediction rationales via LLM."""

    def __init__(self, model: str = "google/gemini-2.5-flash") -> None:
        self.model = model

    def extract(
        self,
        rationale: str,
        *,
        task_description: str = "Prediction task",
        event_description: str = "",
        cutoff_date: str = "",
    ) -> ExtractionResult:
        """Extract claims from *rationale*.

        Args:
            rationale: Free-text prediction rationale.
            task_description: Human-readable task name.
            event_description: What is being predicted.
            cutoff_date: Knowledge cutoff date (YYYY-MM-DD).

        Returns:
            ExtractionResult containing parsed claims.
        """
        if not rationale or not rationale.strip():
            return ExtractionResult(claims=[], notes="Empty rationale", raw_response="")

        prompt = _EXTRACTION_PROMPT.format(
            task_description=task_description,
            event_description=event_description,
            cutoff_date=cutoff_date,
            rationale=rationale,
        )

        content = llm_client.simple_call(
            prompt, model=self.model, temperature=0, max_tokens=8192,
        )
        return _parse_extraction_response(content)


# ---------------------------------------------------------------------------
# Parsing helpers (multi-strategy, mirrors TimeSPEC robustness)
# ---------------------------------------------------------------------------

def _parse_extraction_response(content: str) -> ExtractionResult:
    """Parse the LLM JSON response into an ExtractionResult."""
    if not content:
        return ExtractionResult(claims=[], notes="Empty LLM response", raw_response="")

    parsed = _try_parse_json(content)
    if parsed is None or "claims" not in parsed:
        logger.warning("Could not parse claim-extraction response (first 300 chars): %s", content[:300])
        return ExtractionResult(claims=[], notes="JSON parse failure", raw_response=content)

    claims: list[Claim] = []
    for i, raw in enumerate(parsed.get("claims", [])):
        text = raw.get("text", raw.get("claim_text", "")).strip()
        if not text:
            continue
        claims.append(Claim(
            claim_id=raw.get("claim_id", i + 1),
            text=text,
            original_span=raw.get("original_span", raw.get("original_text", "")),
            temporal_ref=raw.get("temporal_ref", raw.get("temporal_reference")),
            category=raw.get("category", raw.get("claim_category", "B1")),
            category_reason=raw.get("category_reason", raw.get("category_reasoning", "")),
        ))

    return ExtractionResult(
        claims=claims,
        notes=parsed.get("notes", parsed.get("extraction_notes", "")),
        raw_response=content,
    )


def _try_parse_json(text: str) -> dict | None:
    """Attempt JSON parse with progressive fallback strategies.

    Handles: clean JSON, markdown-fenced JSON, and truncated JSON responses
    where the model output was cut off before completing the JSON structure.
    """
    text = text.strip()

    # Strip markdown fences unconditionally (LLMs add them despite instructions)
    stripped = text
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        stripped = stripped.strip()

    # Strategy 1: direct parse (works for clean or successfully stripped JSON)
    for candidate in (stripped, text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 2: extract from complete markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find a complete JSON object containing "claims"
    m = re.search(r'\{[\s\S]*"claims"[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: recover from truncated JSON by extracting complete claims
    return _recover_truncated_claims(stripped)


def _recover_truncated_claims(text: str) -> dict | None:
    """Extract complete claim objects from truncated JSON.

    When the LLM response is cut off mid-JSON, the outer structure is
    incomplete but individual claim objects may still be valid.  This
    function finds all fully-formed {...} objects inside the claims array
    and reconstructs a valid result from them.
    """
    if '"claims"' not in text:
        return None

    bracket_pos = text.find("[", text.find('"claims"'))
    if bracket_pos == -1:
        return None

    complete_claims: list[dict] = []
    depth = 0
    obj_start: int | None = None

    for i in range(bracket_pos + 1, len(text)):
        ch = text[i]
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    claim = json.loads(text[obj_start : i + 1])
                    complete_claims.append(claim)
                except json.JSONDecodeError:
                    pass
                obj_start = None

    if not complete_claims:
        return None

    notes_m = re.search(r'"notes"\s*:\s*"([^"]*)"', text)
    notes = notes_m.group(1) if notes_m else "Recovered from truncated response"
    logger.debug(
        "Recovered %d/%d+ claims from truncated JSON",
        len(complete_claims), len(complete_claims),
    )
    return {"claims": complete_claims, "notes": notes}


# ---------------------------------------------------------------------------
# Temporal reference -> date conversion
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, tuple[str, int]] = {
    "january": ("01", 31), "february": ("02", 28), "march": ("03", 31),
    "april": ("04", 30), "may": ("05", 31), "june": ("06", 30),
    "july": ("07", 31), "august": ("08", 31), "september": ("09", 30),
    "october": ("10", 31), "november": ("11", 30), "december": ("12", 31),
    "jan": ("01", 31), "feb": ("02", 28), "mar": ("03", 31),
    "apr": ("04", 30), "jun": ("06", 30), "jul": ("07", 31),
    "aug": ("08", 31), "sep": ("09", 30), "sept": ("09", 30),
    "oct": ("10", 31), "nov": ("11", 30), "dec": ("12", 31),
}


def temporal_ref_to_date(ref: str | None) -> str | None:
    """Convert a temporal reference to a YYYY-MM-DD date (last day of period).

    Examples:
        "2023"        -> "2023-12-31"
        "Q3 2023"     -> "2023-09-30"
        "March 2023"  -> "2023-03-31"
        "2023-03-15"  -> "2023-03-15"
    """
    if not ref:
        return None
    ref = ref.strip()

    # ISO date
    if re.match(r"^\d{4}-\d{2}-\d{2}$", ref):
        return ref

    # "Month Day, Year" or "Day Month Year"
    m = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", ref, re.I)
    if m and m.group(1).lower() in _MONTH_MAP:
        mm, _ = _MONTH_MAP[m.group(1).lower()]
        return f"{m.group(3)}-{mm}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{1,2})\s+(\w+),?\s+(\d{4})", ref, re.I)
    if m and m.group(2).lower() in _MONTH_MAP:
        mm, _ = _MONTH_MAP[m.group(2).lower()]
        return f"{m.group(3)}-{mm}-{int(m.group(1)):02d}"

    # Year only
    if re.match(r"^\d{4}$", ref):
        return f"{ref}-12-31"

    # Quarter
    m = re.match(r"Q(\d)\s*(\d{4})|(\d{4})\s*Q(\d)", ref, re.I)
    if m:
        q = int(m.group(1) or m.group(4))
        y = m.group(2) or m.group(3)
        end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}.get(q)
        return f"{y}-{end}" if end else None

    # "Month Year"
    m = re.match(r"^(\w+)\s+(\d{4})$", ref, re.I)
    if m and m.group(1).lower() in _MONTH_MAP:
        mm, last = _MONTH_MAP[m.group(1).lower()]
        y = int(m.group(2))
        if mm == "02" and y % 4 == 0 and (y % 100 != 0 or y % 400 == 0):
            last = 29
        return f"{y}-{mm}-{last:02d}"

    # Last resort: extract any 4-digit year
    m = re.search(r"(\d{4})", ref)
    if m:
        return f"{m.group(1)}-12-31"

    return None
