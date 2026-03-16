#!/usr/bin/env python3
"""Analyze date distributions across all dataset files."""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"

MARCH_2025 = "2025-03-01"
FEB_2026 = "2026-02-01"

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def ym(date_str: str) -> str:
    return date_str[:7]


def analyze_dates(dates: list[str], label: str) -> None:
    if not dates:
        print(f"  No dates found for '{label}'")
        return
    dates_sorted = sorted(dates)
    before_mar25 = sum(1 for d in dates if d < MARCH_2025)
    on_or_after_mar25 = len(dates) - before_mar25
    before_feb26 = sum(1 for d in dates if d < FEB_2026)
    on_or_after_feb26 = len(dates) - before_feb26

    print(f"  Field: {label}")
    print(f"    Total entries: {len(dates)}")
    print(f"    Min date: {dates_sorted[0]}")
    print(f"    Max date: {dates_sorted[-1]}")
    print(f"    Before March 2025 (<{MARCH_2025}): {before_mar25}")
    print(f"    On/after March 2025 (>={MARCH_2025}): {on_or_after_mar25}")
    print(f"    Before Feb 2026 (<{FEB_2026}): {before_feb26}")
    print(f"    On/after Feb 2026 (>={FEB_2026}): {on_or_after_feb26}")

    ym_counts = Counter(ym(d) for d in dates)
    print(f"    Year-month distribution:")
    for k in sorted(ym_counts):
        print(f"      {k}: {ym_counts[k]}")
    print()


def analyze_legal_json(path: Path) -> None:
    print(f"\n{'='*80}")
    print(f"FILE: {path.relative_to(DATASET_ROOT.parent)}")
    print(f"{'='*80}")
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("  Not a list, skipping")
        return

    date_fields: dict[str, list[str]] = defaultdict(list)
    for entry in data:
        for key in ("oral_argument_date", "decision_date", "cutoff_date"):
            val = entry.get(key, "")
            if val and DATE_RE.match(val):
                date_fields[key].append(val)

    for field, dates in sorted(date_fields.items()):
        analyze_dates(dates, field)


def analyze_stock_json(path: Path) -> None:
    print(f"\n{'='*80}")
    print(f"FILE: {path.relative_to(DATASET_ROOT.parent)}")
    print(f"{'='*80}")
    with open(path) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    print(f"  Metadata:")
    print(f"    total_questions: {metadata.get('total_questions')}")
    print(f"    generation_date: {metadata.get('generation_date')}")

    if "model_knowledge_cutoff" in metadata:
        print(f"    model_knowledge_cutoff: {metadata['model_knowledge_cutoff']}")

    if "past_period" in metadata:
        pp = metadata["past_period"]
        print(f"    past_period: {pp.get('start')} to {pp.get('end')}")

    if "windows" in metadata:
        for wid, w in metadata["windows"].items():
            print(f"    Window {wid}: {w.get('start')} to {w.get('end')}, cutoff={w.get('cutoff')}")
            print(f"      Description: {w.get('description')}")

    questions = data.get("questions", [])
    cutoff_dates: list[str] = []
    start_dates: list[str] = []
    end_dates: list[str] = []

    for q in questions:
        pp = q.get("past_period", {})
        sd = pp.get("start_date", "")
        ed = pp.get("end_date", "")
        if sd and DATE_RE.match(sd):
            start_dates.append(sd)
        if ed and DATE_RE.match(ed):
            end_dates.append(ed)

        if "model_knowledge_cutoff" in q:
            ck = q["model_knowledge_cutoff"]
            if ck and DATE_RE.match(ck):
                cutoff_dates.append(ck)

    if "model_knowledge_cutoff" in metadata:
        ck = metadata["model_knowledge_cutoff"]
        if ck and DATE_RE.match(ck):
            cutoff_dates.append(ck)

    if "windows" in metadata:
        for w in metadata["windows"].values():
            ck = w.get("cutoff", "")
            if ck and DATE_RE.match(ck):
                cutoff_dates.append(ck)

    if cutoff_dates:
        analyze_dates(cutoff_dates, "model_knowledge_cutoff (metadata-level)")
    if start_dates:
        analyze_dates(start_dates, "past_period.start_date (per question)")
    if end_dates:
        analyze_dates(end_dates, "past_period.end_date (per question)")


def analyze_salary_csv(path: Path) -> None:
    print(f"\n{'='*80}")
    print(f"FILE: {path.relative_to(DATASET_ROOT.parent)}")
    print(f"{'='*80}")
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("  Empty CSV")
        return

    print(f"  Total rows: {len(rows)}")
    print(f"  Columns: {list(rows[0].keys())}")

    years: list[str] = []
    for row in rows:
        yr = row.get("Year", "")
        if yr:
            years.append(yr)

    if years:
        yr_counts = Counter(years)
        print(f"  Year distribution:")
        for k in sorted(yr_counts):
            print(f"    {k}: {yr_counts[k]}")

        year_ints = [int(y) for y in years if y.isdigit()]
        if year_ints:
            print(f"  Min year: {min(year_ints)}")
            print(f"  Max year: {max(year_ints)}")
            before_2025 = sum(1 for y in year_ints if y < 2025)
            on_or_after_2025 = sum(1 for y in year_ints if y >= 2025)
            before_2026 = sum(1 for y in year_ints if y < 2026)
            on_or_after_2026 = sum(1 for y in year_ints if y >= 2026)
            print(f"  Before 2025: {before_2025}")
            print(f"  2025 or later: {on_or_after_2025}")
            print(f"  Before 2026: {before_2026}")
            print(f"  2026 or later: {on_or_after_2026}")

    sports = Counter(row.get("Sport", "N/A") for row in rows)
    if len(sports) > 1 or list(sports.keys()) != ["N/A"]:
        print(f"  Sport distribution: {dict(sports)}")
    print()


def main() -> None:
    print("DATE DISTRIBUTION ANALYSIS")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Thresholds: March 2025 = {MARCH_2025}, Feb 2026 = {FEB_2026}")

    # Legal cases
    for p in sorted(DATASET_ROOT.rglob("legal_cases*.json")):
        analyze_legal_json(p)

    # Stock rankings
    for p in sorted(DATASET_ROOT.rglob("stock_rankings*.json")):
        analyze_stock_json(p)

    # Salary CSVs
    for p in sorted(DATASET_ROOT.rglob("*.csv")):
        analyze_salary_csv(p)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE: Files with dates on/after March 2025")
    print(f"{'='*80}")

    # Re-scan for summary
    all_cutoffs: list[str] = []

    for p in sorted(DATASET_ROOT.rglob("legal_cases*.json")):
        with open(p) as f:
            data = json.load(f)
        for entry in data:
            cd = entry.get("cutoff_date", "")
            if cd:
                all_cutoffs.append(cd)

    for p in sorted(DATASET_ROOT.rglob("stock_rankings*.json")):
        with open(p) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        if "model_knowledge_cutoff" in meta:
            all_cutoffs.append(meta["model_knowledge_cutoff"])
        if "windows" in meta:
            for w in meta["windows"].values():
                if "cutoff" in w:
                    all_cutoffs.append(w["cutoff"])

    print(f"\nAll cutoff dates across legal + stock files:")
    cutoff_counts = Counter(all_cutoffs)
    for k in sorted(cutoff_counts):
        tag = ""
        if k >= MARCH_2025:
            tag = " ** ON/AFTER March 2025 **"
        if k >= FEB_2026:
            tag = " ** ON/AFTER Feb 2026 **"
        print(f"  {k}: {cutoff_counts[k]}{tag}")

    before_mar = sum(v for k, v in cutoff_counts.items() if k < MARCH_2025)
    after_mar = sum(v for k, v in cutoff_counts.items() if k >= MARCH_2025)
    before_feb = sum(v for k, v in cutoff_counts.items() if k < FEB_2026)
    after_feb = sum(v for k, v in cutoff_counts.items() if k >= FEB_2026)
    print(f"\n  Total cutoff dates: {len(all_cutoffs)}")
    print(f"  Before March 2025: {before_mar}")
    print(f"  On/after March 2025: {after_mar}")
    print(f"  Before Feb 2026: {before_feb}")
    print(f"  On/after Feb 2026: {after_feb}")


if __name__ == "__main__":
    main()
