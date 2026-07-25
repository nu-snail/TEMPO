"""Convert dataset/evaluation/{legal_cases.json, nba_salaries.csv, stock_rankings.json}
into a unified JSONL file consumable by LeakageEvaluationFramework.generation.

Cutoff rules (decided 2026-05-20):
  - Legal:  cutoff = oral_argument_date - 1 day if present;
            else cutoff = decision_date - 60 days.
  - Salary: cutoff = signing_date - 7 days.
  - Stock:  cutoff = 2019-12-01 (unchanged; the regime-shift anchor).

Each output row matches the EMNLP 2026 schema (id, question, background,
temporal_cutoff, resolution_date, group, domain, answer_type, choices,
answer, unit, source_url, metadata) so the existing generation/runner.py
can load it without changes.

Usage:
    uv run python -m dataset.evaluation.scripts.build_task_sensitivity_jsonl \
        --eval-root  dataset/evaluation \
        --output     dataset/evaluation/task_sensitivity.jsonl

    # Pilot: 3 instances per task (9 total).
    uv run python -m dataset.evaluation.scripts.build_task_sensitivity_jsonl \
        --eval-root  dataset/evaluation \
        --output     dataset/evaluation/task_sensitivity_pilot.jsonl \
        --per-task   3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("build_task_sensitivity_jsonl")

_LEGAL_FALLBACK_BACKOFF_DAYS = 60
_SALARY_CUTOFF_BACKOFF_DAYS = 7
_STOCK_FIXED_CUTOFF = "2019-12-01"


@dataclass
class _CutoffStats:
    used_oral_arg: int = 0
    used_decision_minus_60: int = 0
    skipped_no_dates: int = 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True,
                        help="Directory containing legal_cases.json, nba_salaries.csv, stock_rankings.json.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL path.")
    parser.add_argument("--per-task", type=int, default=None,
                        help="If set, sample this many instances per task (legal/salary/stock).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default="legal,salary,stock",
                        help="Comma-separated subset of tasks to include.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = random.Random(args.seed)
    requested_tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}

    out_rows: list[dict[str, Any]] = []

    if "legal" in requested_tasks:
        legal_rows, legal_stats = _build_legal(args.eval_root / "legal_cases.json")
        logger.info("Legal: built %d rows (oral_arg=%d, decision-60d=%d, skipped=%d)",
                    len(legal_rows), legal_stats.used_oral_arg,
                    legal_stats.used_decision_minus_60, legal_stats.skipped_no_dates)
        out_rows.extend(_maybe_sample(legal_rows, args.per_task, rng))

    if "salary" in requested_tasks:
        salary_rows = _build_salary(args.eval_root / "nba_salaries.csv")
        logger.info("Salary: built %d rows", len(salary_rows))
        out_rows.extend(_maybe_sample(salary_rows, args.per_task, rng))

    if "stock" in requested_tasks:
        stock_rows = _build_stock(args.eval_root / "stock_rankings.json")
        logger.info("Stock: built %d rows", len(stock_rows))
        out_rows.extend(_maybe_sample(stock_rows, args.per_task, rng))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Wrote %d rows to %s", len(out_rows), args.output)


def _maybe_sample(rows: list[dict[str, Any]], k: int | None, rng: random.Random) -> list[dict[str, Any]]:
    if k is None or k >= len(rows):
        return rows
    return rng.sample(rows, k)


# ---------------------------------------------------------------------------
# Legal
# ---------------------------------------------------------------------------

def _build_legal(path: Path) -> tuple[list[dict[str, Any]], _CutoffStats]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    stats = _CutoffStats()
    for r in raw:
        decision_date = r.get("decision_date")
        if not decision_date:
            stats.skipped_no_dates += 1
            continue
        try:
            decision = date.fromisoformat(decision_date)
        except ValueError:
            stats.skipped_no_dates += 1
            continue

        oral = r.get("oral_argument_date")
        cutoff: date
        if oral:
            try:
                cutoff = date.fromisoformat(oral) - timedelta(days=1)
                stats.used_oral_arg += 1
            except ValueError:
                cutoff = decision - timedelta(days=_LEGAL_FALLBACK_BACKOFF_DAYS)
                stats.used_decision_minus_60 += 1
        else:
            cutoff = decision - timedelta(days=_LEGAL_FALLBACK_BACKOFF_DAYS)
            stats.used_decision_minus_60 += 1

        case_name = r.get("case_name", "") or ""
        petitioner = r.get("petitioner", "") or ""
        respondent = r.get("respondent", "") or ""
        facts = r.get("facts", "") or ""
        court = r.get("court", "") or ""
        term = r.get("term", "") or ""
        outcome = (r.get("outcome", "") or "").strip().upper()
        if outcome not in {"PETITIONER", "RESPONDENT"}:
            continue

        # The decision date is the resolution date for forecasting purposes.
        rows.append({
            "id": f"legal_{r['case_id']}",
            "question": (
                f"In the case '{case_name}' ({court}, {term}), "
                f"will the petitioner ({petitioner}) win against the respondent ({respondent})? "
                "Provide a calibrated probability that the petitioner wins."
            ),
            "background": facts,
            "temporal_cutoff": cutoff.isoformat(),
            "resolution_date": decision.isoformat(),
            "group": "treatment",
            "domain": "legal",
            "answer_type": "legal",
            "choices": None,
            "answer": outcome,
            "unit": None,
            "source_url": "",
            "metadata": {
                "case_name": case_name,
                "petitioner": petitioner,
                "respondent": respondent,
                "court": court,
                "term": term,
                "oral_argument_date": r.get("oral_argument_date"),
                "decision_date": decision_date,
                "vote_split": r.get("vote_split"),
                "ground_truth": {"outcome": outcome},
            },
        })
    return rows, stats


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------

def _build_salary(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            signing_raw = (r.get("Signing_Date") or "").strip()
            try:
                signing = date.fromisoformat(signing_raw)
            except ValueError:
                continue
            cutoff = signing - timedelta(days=_SALARY_CUTOFF_BACKOFF_DAYS)
            aav_raw = (r.get("AAV") or "").strip()
            try:
                aav_value = float(aav_raw.replace("$", "").replace(",", ""))
            except ValueError:
                continue
            if aav_value <= 0:
                continue
            player = (r.get("Player") or "").strip()
            year = (r.get("Year") or "").strip()
            from_team = (r.get("From_Team") or "").strip()
            to_team = (r.get("To_Team") or "").strip()
            position = (r.get("Position") or "").strip()
            years_total = (r.get("Years") or "").strip()
            rows.append({
                "id": f"salary_{player.lower().replace(' ', '_').replace(chr(39), '')}_{year}",
                "question": (
                    f"What will be {player}'s Average Annual Value (AAV) in USD "
                    f"for the {year} NBA contract he is about to sign with the {to_team}? "
                    "Predict the dollar value (integer USD)."
                ),
                "background": (
                    f"{player} is a {position} previously on the {from_team}. "
                    f"He is signing a {years_total}-year contract with the {to_team} "
                    f"during the {year} NBA free agency cycle, around {signing.isoformat()}. "
                    "AAV (Average Annual Value) is the total contract value divided by the number of years."
                ),
                "temporal_cutoff": cutoff.isoformat(),
                "resolution_date": signing.isoformat(),
                "group": "treatment",
                "domain": "sports",
                "answer_type": "salary",
                "choices": None,
                "answer": str(int(aav_value)),
                "unit": "USD",
                "source_url": "",
                "metadata": {
                    "player": player,
                    "position": position,
                    "from_team": from_team,
                    "to_team": to_team,
                    "year": year,
                    "contract_years": years_total,
                    "signing_date": signing_raw,
                    "ground_truth": {"aav": aav_value},
                },
            })
    return rows


# ---------------------------------------------------------------------------
# Stock
# ---------------------------------------------------------------------------

def _build_stock(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", []) if isinstance(payload, dict) else payload
    rows: list[dict[str, Any]] = []
    for q in questions:
        tickers = list(q.get("companies", []))
        if len(tickers) < 2:
            continue
        period = q.get("past_period", {})
        end_date = period.get("end_date", "2020-06-02")
        sector = q.get("sector", "")
        names = q.get("company_names", {})
        sub_industries = q.get("sub_industries", {})
        ranking = list(period.get("ranking", []))
        if len(ranking) != len(tickers):
            continue
        # Background lists tickers + company names + sub-industries (all pre-cutoff fundamentals).
        bg_lines = [
            f"The following are {len(tickers)} companies from the {sector} sector of the S&P 500.",
            "Each is identified by ticker symbol, company name, and sub-industry:",
        ]
        for t in tickers:
            bg_lines.append(f"  - {t}: {names.get(t, t)} ({sub_industries.get(t, 'n/a')})")
        bg_lines.append(
            f"The forecasting window is the 6-month period from {_STOCK_FIXED_CUTOFF} "
            f"(cutoff) to {end_date}. Rank the companies by total return over this window."
        )
        rows.append({
            "id": f"stock_{q.get('question_id', len(rows) + 1)}",
            "question": (
                f"Rank the {len(tickers)} {sector} companies listed below by their total stock return "
                f"over the period {_STOCK_FIXED_CUTOFF} to {end_date}, from highest to lowest. "
                "Output the ticker symbols in ranked order."
            ),
            "background": "\n".join(bg_lines),
            "temporal_cutoff": _STOCK_FIXED_CUTOFF,
            "resolution_date": end_date,
            "group": "treatment",
            "domain": "finance",
            "answer_type": "stock",
            "choices": tickers,   # surfaced to prompt as TICKERS TO RANK
            "answer": ",".join(ranking),
            "unit": None,
            "source_url": "",
            "metadata": {
                "sector": sector,
                "company_names": names,
                "sub_industries": sub_industries,
                "ground_truth": {"ranking": ranking},
            },
        })
    return rows


if __name__ == "__main__":
    main()
