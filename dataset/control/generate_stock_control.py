"""Generate stock ranking control set for Feb-March 2026.

Uses the same logic as the training data generator but for a period
after both models' knowledge cutoffs, producing genuine-prediction
questions where neither model has prior knowledge.
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

SCRIPT_DIR = Path(__file__).parent
TRAIN_DIR = SCRIPT_DIR.parent / "train" / "stock_rankings"
MAPPING_PATH = TRAIN_DIR / "sp500_mapping.json"
SECTOR_SUMMARY_PATH = TRAIN_DIR / "sector_summary_100.json"

CONTROL_WINDOW = {
    "start": "2026-02-02",
    "end": "2026-03-06",
    "cutoff": "2026-02-01",
    "description": "Post-cutoff control window (Feb-Mar 2026)",
}
TOTAL_QUESTIONS = 100
COMPANIES_PER_QUESTION = 5
RANDOM_SEED = 99


def load_mapping() -> dict:
    with open(MAPPING_PATH) as f:
        return json.load(f)


def load_sector_pools() -> dict[str, list[str]]:
    with open(SECTOR_SUMMARY_PATH) as f:
        data = json.load(f)
    return {sector: sorted(set(info["companies"])) for sector, info in data.items()}


def fetch_prices(
    tickers: list[str], start_date: str, end_date: str
) -> tuple[dict, dict, dict]:
    """Fetch adjusted close prices at window boundaries."""
    buf_start = (pd.Timestamp(start_date) - timedelta(days=10)).strftime("%Y-%m-%d")
    buf_end = (pd.Timestamp(end_date) + timedelta(days=5)).strftime("%Y-%m-%d")

    all_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            df = yf.download(
                " ".join(batch),
                start=buf_start,
                end=buf_end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if df.empty:
                continue
            if len(batch) == 1:
                all_data[batch[0]] = df["Close"]
            else:
                if "Close" in df.columns.get_level_values(0):
                    close_df = df["Close"]
                    for t in batch:
                        if t in close_df.columns:
                            all_data[t] = close_df[t]
        except Exception as e:
            print(f"  Warning: batch {batch[0]}: {e}")
        if i + batch_size < len(tickers):
            time.sleep(0.5)

    start_prices, end_prices, actual_dates = {}, {}, {}
    for t, series in all_data.items():
        series = series.dropna()
        if series.empty:
            continue
        sc = series.index[series.index <= pd.Timestamp(start_date)]
        if len(sc) == 0:
            sc = series.index[series.index >= pd.Timestamp(start_date)]
            if len(sc) == 0:
                continue
            actual_start = sc[0]
        else:
            actual_start = sc[-1]
        ec = series.index[series.index <= pd.Timestamp(end_date)]
        if len(ec) == 0:
            continue
        actual_end = ec[-1]
        if (actual_end - actual_start).days < 15:
            continue
        s_price, e_price = float(series.loc[actual_start]), float(series.loc[actual_end])
        if s_price <= 0 or e_price <= 0:
            continue
        start_prices[t] = round(s_price, 2)
        end_prices[t] = round(e_price, 2)
        actual_dates[t] = {
            "start_date": str(actual_start.date()),
            "end_date": str(actual_end.date()),
        }
    return start_prices, end_prices, actual_dates


def main() -> None:
    mapping = load_mapping()
    sector_pools = load_sector_pools()
    rng = random.Random(RANDOM_SEED)
    w = CONTROL_WINDOW

    print(f"Control window: {w['start']} to {w['end']} (cutoff {w['cutoff']})")

    all_tickers = sorted({t for pool in sector_pools.values() for t in pool})
    print(f"Fetching prices for {len(all_tickers)} tickers...")
    start_prices, end_prices, actual_dates = fetch_prices(all_tickers, w["start"], w["end"])
    valid = set(start_prices.keys()) & set(end_prices.keys())
    print(f"Valid tickers: {len(valid)}")

    returns = {
        t: round((end_prices[t] - start_prices[t]) / start_prices[t], 4)
        for t in valid
    }

    all_sectors = sorted(sector_pools.keys())
    per_sector = max(1, TOTAL_QUESTIONS // len(all_sectors))
    remainder = TOTAL_QUESTIONS - per_sector * len(all_sectors)

    questions = []
    qid = 1
    for si, sector in enumerate(all_sectors):
        n = per_sector + (1 if si < remainder else 0)
        sector_valid = sorted(t for t in sector_pools[sector] if t in valid)
        if len(sector_valid) < COMPANIES_PER_QUESTION:
            print(f"  {sector}: only {len(sector_valid)} tickers, skipping")
            continue

        if len(sector_valid) <= 25:
            combos = list(combinations(sector_valid, COMPANIES_PER_QUESTION))
            groups = [list(c) for c in rng.sample(combos, min(n, len(combos)))]
        else:
            seen, groups = set(), []
            for _ in range(n * 20):
                g = tuple(sorted(rng.sample(sector_valid, COMPANIES_PER_QUESTION)))
                if g not in seen:
                    seen.add(g)
                    groups.append(list(g))
                if len(groups) >= n:
                    break

        for group in groups:
            gr = {t: returns[t] for t in group}
            ranking = sorted(gr, key=lambda t: gr[t], reverse=True)
            questions.append({
                "question_id": qid,
                "sector": sector,
                "companies": group,
                "company_names": {t: mapping.get(t, {}).get("name", t) for t in group},
                "sub_industries": {t: mapping.get(t, {}).get("sub_industry", "Unknown") for t in group},
                "past_period": {
                    "start_date": w["start"],
                    "end_date": w["end"],
                    "status": "post_model_knowledge",
                    "returns": gr,
                    "ranking": ranking,
                    "start_prices": {t: start_prices[t] for t in group},
                    "end_prices": {t: end_prices[t] for t in group},
                    "actual_dates": {t: actual_dates[t] for t in group},
                    "cutoff": w["cutoff"],
                },
            })
            qid += 1

    dataset = {
        "metadata": {
            "total_questions": len(questions),
            "companies_per_question": COMPANIES_PER_QUESTION,
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "past_period": {"start": w["start"], "end": w["end"]},
            "model_knowledge_cutoff": w["cutoff"],
            "period_description": w["description"],
            "purpose": "control_set",
        },
        "questions": questions,
    }

    out_path = SCRIPT_DIR / "stock_rankings.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(questions)} control questions to {out_path}")
    sector_dist = {}
    for q in questions:
        sector_dist[q["sector"]] = sector_dist.get(q["sector"], 0) + 1
    for s in sorted(sector_dist):
        print(f"  {s}: {sector_dist[s]}")


if __name__ == "__main__":
    main()
