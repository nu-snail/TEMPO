"""Pydantic schemas and parsing helpers.

All models use strict validation by default (ConfigDict(strict=True)),
rejecting type coercion (e.g. string "1" for int field).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class _StrictBase(BaseModel):
    """Base with strict validation enforced globally."""

    model_config = ConfigDict(strict=True, extra="forbid")


class EvidenceItem(_StrictBase):
    """Atomic evidence item."""

    id: int
    fact: str = Field(min_length=1)
    source_date: str | None = None


class StockCompletion(_StrictBase):
    """Schema for stock completion output."""

    model_config = ConfigDict(strict=True, extra="ignore")

    evidence: list[EvidenceItem] = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    ranking: list[str] = Field(min_length=2)

    @field_validator("ranking", mode="before")
    @classmethod
    def _coerce_ranking_items(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [str(x) for x in v]
        return v


class SalaryCompletion(_StrictBase):
    """Schema for salary completion output."""

    model_config = ConfigDict(strict=True, extra="ignore")

    evidence: list[EvidenceItem] = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    predicted_salary: float = Field(gt=0)

    @field_validator("predicted_salary", mode="before")
    @classmethod
    def _coerce_salary(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = v.replace(",", "").replace("$", "").strip()
            return float(cleaned)
        return v


class LegalCompletion(_StrictBase):
    """Schema for legal completion output."""

    model_config = ConfigDict(strict=True, extra="ignore")

    evidence: list[EvidenceItem] = Field(min_length=1)
    reasoning: str = Field(min_length=1)
    probability_petitioner: float = Field(ge=0.0, le=1.0)
    predicted_winner: str | None = None

    @field_validator("probability_petitioner", mode="before")
    @classmethod
    def _coerce_prob(cls, v: Any) -> Any:
        if isinstance(v, (int, float)):
            return float(v)
        return v

    @field_validator("predicted_winner", mode="before")
    @classmethod
    def _normalize_winner(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip().upper()
        if s not in {"PETITIONER", "RESPONDENT"}:
            return None
        return s


class SourceBClaim(_StrictBase):
    model_config = ConfigDict(strict=True, extra="ignore")
    id: str = ""
    claim: str = Field(min_length=1)
    span: str = ""


class SourceBResult(_StrictBase):
    model_config = ConfigDict(strict=True, extra="ignore")
    completion_idx: int = Field(ge=0)
    extra_claims: list[SourceBClaim]


class CategorizationResult(_StrictBase):
    model_config = ConfigDict(strict=True, extra="ignore")
    index: int = Field(ge=0)
    category: Literal["A1", "A2", "A3", "A4", "A5", "B1", "B2"]
    category_reason: str = ""
    search_query: str | None


class DateVerifyResult(_StrictBase):
    model_config = ConfigDict(strict=True, extra="ignore")
    index: int = Field(ge=0)
    event_date: str | None
    factual_status: Literal["supported", "unsupported", "contradicted"]
    confidence: Literal["high", "medium", "low", "none"]
    reasoning: str = ""

    @field_validator("event_date", mode="before")
    @classmethod
    def _validate_date_format(cls, v: Any) -> str | None:
        if v is None or v == "null":
            return None
        s = str(v).strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return None
        return s


class PlausibilityResult(_StrictBase):
    """Result from the DATE_PLAUSIBILITY_PROMPT verifier."""

    model_config = ConfigDict(strict=True, extra="ignore")
    index: int = Field(ge=0)
    plausible: bool
    corrected_date: str | None = None
    reason: str = ""

    @field_validator("corrected_date", mode="before")
    @classmethod
    def _validate_corrected_date(cls, v: Any) -> str | None:
        if v is None or v == "null" or v == "":
            return None
        s = str(v).strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return None
        return s


class FallbackResult(_StrictBase):
    model_config = ConfigDict(strict=True, extra="ignore")
    index: int = Field(ge=0)
    is_leaked: bool
    reasoning: str = ""


class LogicFlaw(_StrictBase):
    model_config = ConfigDict(strict=True, extra="ignore")
    type: Literal["CONTRADICTION", "UNSUPPORTED_LEAP", "SELECTIVE_EVIDENCE", "PREDICTION_MISMATCH"]
    description: str = ""
    severity: Literal["major", "minor"]


class LogicResult(_StrictBase):
    model_config = ConfigDict(strict=True, extra="ignore")
    completion_idx: int = Field(ge=0)
    flaws: list[LogicFlaw]
    total_reasoning_steps: int = Field(ge=1)
    sound_steps: int = Field(ge=0)


def parse_completion(task: str, text: str) -> dict[str, Any] | None:
    """Parse one student completion with strict schema validation."""

    # Strip Qwen3 chain-of-thought traces before JSON extraction.
    # The greedy {[\s\S]*} fallback regex breaks if the think section contains
    # any { character (e.g. the model reasoning about JSON schema).
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    payload = _try_extract_json_object(text)
    if payload is None:
        return None

    try:
        if task == "stock":
            return StockCompletion.model_validate(payload).model_dump()
        if task == "salary":
            return SalaryCompletion.model_validate(payload).model_dump()
        if task == "legal":
            legal = LegalCompletion.model_validate(payload)
            winner = legal.predicted_winner or ("PETITIONER" if legal.probability_petitioner >= 0.5 else "RESPONDENT")
            out = legal.model_dump()
            out["predicted_winner"] = winner
            return out
    except ValidationError:
        return None
    return None


def compute_format_score(task: str, parsed: dict[str, Any] | None) -> float:
    """Compute R_format for parseable completion."""

    if parsed is None:
        return 0.0

    required_keys = {
        "stock": {"evidence", "reasoning", "ranking"},
        "salary": {"evidence", "reasoning", "predicted_salary"},
        "legal": {"evidence", "reasoning", "probability_petitioner", "predicted_winner"},
    }[task]
    present = set(parsed.keys())

    r_keys = len(required_keys & present) / len(required_keys)
    evidence = parsed.get("evidence", [])
    if not evidence:
        r_schema = 0.0
    else:
        valid_items = 0
        for item in evidence:
            if isinstance(item, dict) and isinstance(item.get("id"), int) and isinstance(item.get("fact"), str) and item["fact"].strip():
                valid_items += 1
        r_schema = valid_items / len(evidence)

    # Partial reward for source_date coverage (incentivises temporal attribution)
    if evidence:
        dated_items = sum(
            1 for item in evidence
            if isinstance(item, dict) and isinstance(item.get("source_date"), str) and item["source_date"].strip()
        )
        r_source_date = dated_items / len(evidence)
    else:
        r_source_date = 0.0

    r_prediction_type = _prediction_type_score(task, parsed)
    # Weights: keys=0.30, schema=0.25, source_date=0.10, prediction_type=0.35
    return 0.30 * r_keys + 0.25 * r_schema + 0.10 * r_source_date + 0.35 * r_prediction_type


def parse_json_array(text: str, model_cls: type[BaseModel]) -> list[BaseModel]:
    """Parse LLM JSON array response into validated model instances."""

    payload = _try_extract_json_array(text)
    if payload is None:
        return []
    out: list[BaseModel] = []
    for item in payload:
        try:
            out.append(model_cls.model_validate(item))
        except ValidationError:
            continue
    return out


def _prediction_type_score(task: str, parsed: dict[str, Any]) -> float:
    if task == "stock":
        return 1.0 if isinstance(parsed.get("ranking"), list) else 0.0
    if task == "salary":
        val = parsed.get("predicted_salary")
        return 1.0 if isinstance(val, (int, float)) and float(val) > 0 else 0.0
    prob = parsed.get("probability_petitioner")
    winner = parsed.get("predicted_winner")
    if not isinstance(prob, (int, float)):
        return 0.0
    if not (0.0 <= float(prob) <= 1.0):
        return 0.0
    return 1.0 if isinstance(winner, str) and winner in {"PETITIONER", "RESPONDENT"} else 0.0


def _try_extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        candidates.append(fence.group(1).strip())
    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _try_extract_json_array(text: str) -> list[dict[str, Any]] | None:
    text = text.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        candidates.append(fence.group(1).strip())
    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    # NDJSON fallback: one JSON object per line (Flash Lite sometimes outputs this)
    ndjson_out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                ndjson_out.append(obj)
        except json.JSONDecodeError:
            continue
    return ndjson_out if ndjson_out else None
