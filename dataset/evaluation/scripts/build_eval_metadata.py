"""Flatten task_sensitivity_*.jsonl into the metadata schema consumed by
LeakageEvaluationFramework.leakage_eval.

The generation-side JSONL uses the EMNLP 2026 schema (id, question,
temporal_cutoff, answer_type, answer, metadata.*). The leakage_eval pipeline's
legacy validator (legal/salary/stock paths) expects flat top-level fields:
    task, question, cutoff_date, ground_truth, plus task-specific fields
    (case_name/petitioner/respondent for legal; player/year for salary;
    companies/sector for stock).

This script reads the JSONL and emits a JSON mapping
``{instance_id: metadata_dict}`` matching that contract, so the eval CLI can
run unchanged with ``--dataset <emitted file>``.

Usage:
    uv run python -m dataset.evaluation.scripts.build_eval_metadata \
        --input  dataset/evaluation/task_sensitivity_full.jsonl \
        --output dataset/evaluation/task_sensitivity_full_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    instance_id = row["id"]
    task = row["answer_type"]
    meta = row.get("metadata") or {}
    cutoff = row.get("temporal_cutoff") or row.get("cutoff_date") or ""

    base: dict[str, Any] = {
        "instance_id": instance_id,
        "task": task,
        "question": row["question"],
        "cutoff_date": cutoff,
        "resolution_date": row.get("resolution_date", ""),
        "target_description": _target_description(task, meta),
        "ground_truth": dict(meta.get("ground_truth") or {}),
    }

    if task == "legal":
        base.update({
            "case_name": meta.get("case_name", ""),
            "petitioner": meta.get("petitioner", ""),
            "respondent": meta.get("respondent", ""),
            "court": meta.get("court", ""),
            "term": meta.get("term", ""),
            "facts": row.get("background", ""),
        })
        if not base["ground_truth"].get("outcome"):
            outcome = (row.get("answer") or "").strip().upper()
            if outcome:
                base["ground_truth"]["outcome"] = outcome

    elif task == "salary":
        base.update({
            "player": meta.get("player", ""),
            "year": str(meta.get("year", "")),
            "team": meta.get("to_team") or meta.get("team", ""),
            "from_team": meta.get("from_team", ""),
            "sport": "NBA",
            "status": meta.get("status", ""),
        })
        if "aav" not in base["ground_truth"]:
            try:
                base["ground_truth"]["aav"] = float(row.get("answer", 0))
            except (TypeError, ValueError):
                pass

    elif task == "stock":
        base.update({
            "companies": row.get("choices") or [],
            "company_names": meta.get("company_names", {}),
            "sector": meta.get("sector", ""),
            "window_id": meta.get("window_id", ""),
        })
        if "ranking" not in base["ground_truth"]:
            answer = row.get("answer", "")
            if isinstance(answer, str) and answer:
                base["ground_truth"]["ranking"] = [t.strip() for t in answer.split(",") if t.strip()]

    base["raw"] = row
    base["prompt_safe"] = {
        k: v
        for k, v in base.items()
        if k in {
            "instance_id", "task", "question", "cutoff_date", "target_description",
            "case_name", "court", "term", "petitioner", "respondent", "facts",
            "player", "sport", "year", "team", "from_team", "status",
            "companies", "company_names", "sector", "window_id",
        }
    }
    return base


def _target_description(task: str, meta: dict[str, Any]) -> str:
    if task == "legal":
        return f"Whether the petitioner wins in {meta.get('case_name', '')}.".strip()
    if task == "salary":
        return f"{meta.get('player', '')}'s contract AAV.".strip()
    if task == "stock":
        return "Future stock return ranking."
    return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = {row["id"]: _flatten_row(row) for row in rows}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote {len(metadata)} metadata rows to {args.output}")


if __name__ == "__main__":
    main()
