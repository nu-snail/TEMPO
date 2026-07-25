"""
Generate stock ranking training data for a given time window.

For each window:
- 300 questions total
- ~80% from heavily-affected sectors (~60 per affected sector)
- ~20% from other sectors (~8-9 per other sector)
- Each question: 5 companies from the same sector, ranked by actual return
- All prices fetched from yfinance (adjusted close)

Usage:
    uv run python dataset/train/stock_rankings/generate_stock_rankings.py \
        --window 1  # (or 2, or 3, or all)
"""

import argparse
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
DATASET_DIR = SCRIPT_DIR.parent.parent  # dataset/
MAPPING_PATH = SCRIPT_DIR / "sp500_mapping.json"
SECTOR_SUMMARY_PATH = SCRIPT_DIR / "sector_summary_100.json"
OUTPUT_DIR = SCRIPT_DIR

# ============================================================
# Window Definitions
# ============================================================
WINDOWS = {
    1: {
        "start": "2022-01-03",
        "end": "2022-06-30",
        "cutoff": "2021-12-31",
        "description": "Russia-Ukraine invasion (Jan-Jun 2022)",
        "affected_sectors": ["Energy", "Materials", "Information Technology", "Industrials"],
    },
    2: {
        "start": "2023-03-01",
        "end": "2023-08-31",
        "cutoff": "2023-02-28",
        "description": "SVB collapse + NVIDIA AI surge (Mar-Aug 2023)",
        "affected_sectors": ["Financials", "Information Technology", "Communication Services", "Real Estate"],
    },
    3: {
        "start": "2020-07-01",
        "end": "2020-12-31",
        "cutoff": "2020-06-30",
        "description": "COVID recovery + vaccine rollout (Jul-Dec 2020)",
        "affected_sectors": ["Health Care", "Information Technology", "Consumer Discretionary", "Energy"],
    },
    4: {
        "start": "2024-01-02",
        "end": "2024-06-28",
        "cutoff": "2023-12-31",
        "description": "AI momentum + rate cut expectations (Jan-Jun 2024)",
        "affected_sectors": ["Information Technology", "Communication Services", "Financials", "Health Care"],
    },
}

TOTAL_QUESTIONS = 300
AFFECTED_FRACTION = 0.80
COMPANIES_PER_QUESTION = 5
RANDOM_SEED = 42


def load_mapping() -> dict:
    with open(MAPPING_PATH) as f:
        return json.load(f)


def load_sector_pools() -> dict[str, list[str]]:
    """Load and deduplicate company pools per sector from sector_summary_100.json."""
    with open(SECTOR_SUMMARY_PATH) as f:
        data = json.load(f)
    pools = {}
    for sector, info in data.items():
        unique_tickers = sorted(set(info["companies"]))
        pools[sector] = unique_tickers
    return pools


def get_nearest_trading_day(date_str: str, prices_df: pd.DataFrame) -> str | None:
    """Find the nearest trading day on or before the given date in the price data."""
    target = pd.Timestamp(date_str)
    valid_dates = prices_df.index[prices_df.index <= target]
    if len(valid_dates) == 0:
        return None
    return str(valid_dates[-1].date())


def fetch_prices(
    tickers: list[str], start_date: str, end_date: str
) -> tuple[dict, dict, dict]:
    """
    Fetch adjusted close prices for tickers at start and end of the window.
    Returns: (start_prices, end_prices, actual_dates) for valid tickers only.

    Uses a buffer around start/end dates to handle weekends/holidays.
    """
    # Add buffer for weekends/holidays
    buf_start = (pd.Timestamp(start_date) - timedelta(days=10)).strftime("%Y-%m-%d")
    buf_end = (pd.Timestamp(end_date) + timedelta(days=5)).strftime("%Y-%m-%d")

    # Download data in batch for efficiency
    batch_size = 50
    all_data = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        ticker_str = " ".join(batch)
        try:
            df = yf.download(
                ticker_str,
                start=buf_start,
                end=buf_end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if df.empty:
                continue

            # Handle single vs multi-ticker DataFrame structure
            if len(batch) == 1:
                # Single ticker: columns are just ['Open', 'High', 'Low', 'Close', 'Volume']
                all_data[batch[0]] = df["Close"]
            else:
                # Multi-ticker: columns are MultiIndex (metric, ticker)
                if "Close" in df.columns.get_level_values(0):
                    close_df = df["Close"]
                    for t in batch:
                        if t in close_df.columns:
                            all_data[t] = close_df[t]
        except Exception as e:
            print(f"    Warning: Failed to download batch starting at {batch[0]}: {e}")
            continue

        # Small delay to avoid rate limiting
        if i + batch_size < len(tickers):
            time.sleep(0.5)

    # Find nearest trading days and extract prices
    start_prices = {}
    end_prices = {}
    actual_dates = {}

    for t, series in all_data.items():
        series = series.dropna()
        if series.empty:
            continue

        # Find nearest trading day on or before start_date
        start_candidates = series.index[series.index <= pd.Timestamp(start_date)]
        if len(start_candidates) == 0:
            # Try nearest after
            start_candidates = series.index[series.index >= pd.Timestamp(start_date)]
            if len(start_candidates) == 0:
                continue
            actual_start = start_candidates[0]
        else:
            actual_start = start_candidates[-1]

        # Find nearest trading day on or before end_date
        end_candidates = series.index[series.index <= pd.Timestamp(end_date)]
        if len(end_candidates) == 0:
            continue
        actual_end = end_candidates[-1]

        # Ensure we have at least ~5 months between start and end
        if (actual_end - actual_start).days < 100:
            continue

        s_price = float(series.loc[actual_start])
        e_price = float(series.loc[actual_end])

        if s_price <= 0 or e_price <= 0:
            continue

        start_prices[t] = round(s_price, 2)
        end_prices[t] = round(e_price, 2)
        actual_dates[t] = {
            "start_date": str(actual_start.date()),
            "end_date": str(actual_end.date()),
        }

    return start_prices, end_prices, actual_dates


def compute_returns(
    start_prices: dict, end_prices: dict
) -> dict[str, float]:
    """Compute percentage returns."""
    returns = {}
    for t in start_prices:
        if t in end_prices:
            ret = (end_prices[t] - start_prices[t]) / start_prices[t]
            returns[t] = round(ret, 4)
    return returns


def compute_ranking(returns: dict[str, float]) -> list[str]:
    """Rank tickers by return (highest first)."""
    return sorted(returns.keys(), key=lambda t: returns[t], reverse=True)


def allocate_questions(
    affected_sectors: list[str],
    all_sectors: list[str],
    total: int = TOTAL_QUESTIONS,
    affected_frac: float = AFFECTED_FRACTION,
) -> dict[str, int]:
    """Allocate number of questions per sector."""
    affected_total = int(total * affected_frac)
    other_total = total - affected_total

    other_sectors = [s for s in all_sectors if s not in affected_sectors]

    allocation = {}
    # Distribute affected questions evenly
    per_affected = affected_total // len(affected_sectors)
    remainder = affected_total % len(affected_sectors)
    for i, s in enumerate(affected_sectors):
        allocation[s] = per_affected + (1 if i < remainder else 0)

    # Distribute other questions evenly
    if other_sectors:
        per_other = other_total // len(other_sectors)
        remainder = other_total % len(other_sectors)
        for i, s in enumerate(other_sectors):
            allocation[s] = per_other + (1 if i < remainder else 0)

    return allocation


def sample_groups(
    valid_tickers: list[str],
    n_questions: int,
    companies_per_question: int = COMPANIES_PER_QUESTION,
    rng: random.Random = None,
) -> list[list[str]]:
    """Sample n_questions unique groups of companies_per_question tickers."""
    if len(valid_tickers) < companies_per_question:
        print(f"    Warning: Only {len(valid_tickers)} valid tickers, need {companies_per_question}")
        return []

    # Generate all possible combinations if pool is small enough
    if len(valid_tickers) <= 25:
        all_combos = list(combinations(valid_tickers, companies_per_question))
        if len(all_combos) <= n_questions:
            rng.shuffle(all_combos)
            return [list(c) for c in all_combos]
        else:
            sampled = rng.sample(all_combos, n_questions)
            return [list(c) for c in sampled]
    else:
        # For larger pools, random sampling with dedup
        seen = set()
        groups = []
        max_attempts = n_questions * 20
        attempts = 0
        while len(groups) < n_questions and attempts < max_attempts:
            group = tuple(sorted(rng.sample(valid_tickers, companies_per_question)))
            if group not in seen:
                seen.add(group)
                groups.append(list(group))
            attempts += 1
        return groups


def generate_window_data(window_id: int) -> dict:
    """Generate the full dataset for one window."""
    window = WINDOWS[window_id]
    mapping = load_mapping()
    sector_pools = load_sector_pools()
    rng = random.Random(RANDOM_SEED + window_id)

    all_sectors = sorted(sector_pools.keys())
    allocation = allocate_questions(window["affected_sectors"], all_sectors)

    print(f"\n{'='*60}")
    print(f"Window {window_id}: {window['description']}")
    print(f"Period: {window['start']} to {window['end']}")
    print(f"Cutoff: {window['cutoff']}")
    print(f"{'='*60}")
    print(f"\nQuestion allocation:")
    total_allocated = 0
    for s in all_sectors:
        n = allocation.get(s, 0)
        marker = " [AFFECTED]" if s in window["affected_sectors"] else ""
        print(f"  {s}: {n} questions{marker}")
        total_allocated += n
    print(f"  Total: {total_allocated}")

    # Step 1: Collect all unique tickers across all sectors
    all_tickers = set()
    for sector, pool in sector_pools.items():
        all_tickers.update(pool)
    all_tickers = sorted(all_tickers)
    print(f"\nTotal unique tickers to validate: {len(all_tickers)}")

    # Step 2: Fetch prices for all tickers
    print(f"Fetching prices from yfinance ({window['start']} to {window['end']})...")
    start_prices, end_prices, actual_dates = fetch_prices(
        all_tickers, window["start"], window["end"]
    )
    valid_tickers = set(start_prices.keys()) & set(end_prices.keys())
    print(f"  Valid tickers with price data: {len(valid_tickers)}")

    # Step 3: Compute returns for all valid tickers
    returns = compute_returns(start_prices, end_prices)

    # Step 4: Generate questions per sector
    questions = []
    question_id = 1

    for sector in all_sectors:
        n_questions = allocation.get(sector, 0)
        if n_questions == 0:
            continue

        # Filter to valid tickers in this sector
        sector_valid = [t for t in sector_pools[sector] if t in valid_tickers]
        sector_valid = sorted(set(sector_valid))  # deduplicate

        if len(sector_valid) < COMPANIES_PER_QUESTION:
            print(f"  WARNING: {sector} has only {len(sector_valid)} valid tickers, skipping")
            continue

        print(f"\n  {sector}: {len(sector_valid)} valid tickers, generating {n_questions} questions")

        groups = sample_groups(sector_valid, n_questions, rng=rng)
        print(f"    Sampled {len(groups)} unique groups")

        for group in groups:
            group_returns = {t: returns[t] for t in group}
            ranking = compute_ranking(group_returns)

            # Build company names and sub-industries from mapping
            company_names = {}
            sub_industries = {}
            for t in group:
                if t in mapping:
                    company_names[t] = mapping[t]["name"]
                    sub_industries[t] = mapping[t]["sub_industry"]
                else:
                    company_names[t] = t  # fallback
                    sub_industries[t] = "Unknown"

            question = {
                "question_id": question_id,
                "sector": sector,
                "companies": group,
                "company_names": company_names,
                "sub_industries": sub_industries,
                "past_period": {
                    "start_date": window["start"],
                    "end_date": window["end"],
                    "status": "within_model_knowledge",
                    "returns": group_returns,
                    "ranking": ranking,
                    "start_prices": {t: start_prices[t] for t in group},
                    "end_prices": {t: end_prices[t] for t in group},
                    "actual_dates": {t: actual_dates[t] for t in group},
                    "cutoff": window["cutoff"],
                },
            }
            questions.append(question)
            question_id += 1

    # Build sector distribution
    sector_dist = {}
    for q in questions:
        s = q["sector"]
        sector_dist[s] = sector_dist.get(s, 0) + 1

    dataset = {
        "metadata": {
            "total_questions": len(questions),
            "companies_per_question": COMPANIES_PER_QUESTION,
            "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "past_period": {
                "start": window["start"],
                "end": window["end"],
            },
            "model_knowledge_cutoff": window["cutoff"],
            "period_description": window["description"],
            "sector_distribution": sector_dist,
            "affected_sectors": window["affected_sectors"],
            "window_id": window_id,
            "test_mode": False,
        },
        "questions": questions,
    }

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate stock ranking training data")
    parser.add_argument(
        "--window",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Which window to generate (1, 2, 3, 4, or all)",
    )
    args = parser.parse_args()

    all_window_ids = sorted(WINDOWS.keys())
    windows_to_generate = all_window_ids if args.window == "all" else [int(args.window)]

    for wid in windows_to_generate:
        dataset = generate_window_data(wid)
        output_path = OUTPUT_DIR / f"stock_rankings_window{wid}.json"
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved {dataset['metadata']['total_questions']} questions to {output_path}")

    if args.window == "all":
        combined_questions = []
        qid = 1
        for wid in all_window_ids:
            path = OUTPUT_DIR / f"stock_rankings_window{wid}.json"
            with open(path) as f:
                data = json.load(f)
            for q in data["questions"]:
                q["question_id"] = qid
                q["window_id"] = wid
                combined_questions.append(q)
                qid += 1

        combined = {
            "metadata": {
                "total_questions": len(combined_questions),
                "companies_per_question": COMPANIES_PER_QUESTION,
                "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "windows": {
                    str(wid): {
                        "start": WINDOWS[wid]["start"],
                        "end": WINDOWS[wid]["end"],
                        "cutoff": WINDOWS[wid]["cutoff"],
                        "description": WINDOWS[wid]["description"],
                    }
                    for wid in all_window_ids
                },
                "test_mode": False,
            },
            "questions": combined_questions,
        }
        combined_path = OUTPUT_DIR / "stock_rankings_train_combined.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"Combined dataset: {len(combined_questions)} total questions -> {combined_path}")


if __name__ == "__main__":
    main()
