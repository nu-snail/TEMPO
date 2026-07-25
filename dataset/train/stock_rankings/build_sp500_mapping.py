"""
Build S&P 500 ticker-to-company mapping from Wikipedia.
Outputs: sp500_mapping.json with ticker -> {name, sector, sub_industry}

Cross-references with sector_summary_100.json to ensure all our tickers are covered.
"""

import json
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR.parent.parent  # dataset/
OUTPUT_PATH = SCRIPT_DIR / "sp500_mapping.json"
SECTOR_SUMMARY_PATH = DATASET_DIR / "sector_summary_100.json"


def fetch_sp500_from_wikipedia() -> pd.DataFrame:
    """Fetch current S&P 500 constituent list from Wikipedia."""
    import requests
    from io import StringIO

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    df = tables[0]  # First table is the current constituents
    # Standardize column names
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    df = df[["ticker", "name", "sector", "sub_industry"]].copy()
    # Clean tickers (some have dots like BRK.B -> BRK.B, keep as-is)
    df["ticker"] = df["ticker"].str.strip()
    return df


def get_our_tickers(sector_summary_path: Path) -> set:
    """Get all unique tickers from our sector_summary_100.json."""
    with open(sector_summary_path) as f:
        data = json.load(f)
    tickers = set()
    for sector, info in data.items():
        for t in info["companies"]:
            tickers.add(t.strip())
    return tickers


def build_mapping(df: pd.DataFrame) -> dict:
    """Build ticker -> {name, sector, sub_industry} mapping."""
    mapping = {}
    for _, row in df.iterrows():
        ticker = row["ticker"]
        mapping[ticker] = {
            "name": row["name"],
            "sector": row["sector"],
            "sub_industry": row["sub_industry"],
        }
    return mapping


def main():
    print("Fetching S&P 500 data from Wikipedia...")
    df = fetch_sp500_from_wikipedia()
    print(f"  Found {len(df)} entries in Wikipedia table")

    mapping = build_mapping(df)

    # Load our tickers
    our_tickers = get_our_tickers(SECTOR_SUMMARY_PATH)
    print(f"\nOur sector_summary has {len(our_tickers)} unique tickers")

    # Check coverage
    covered = our_tickers & set(mapping.keys())
    missing = our_tickers - set(mapping.keys())
    print(f"  Covered by Wikipedia S&P 500: {len(covered)}")
    if missing:
        print(f"  Missing from Wikipedia S&P 500: {sorted(missing)}")
        print("  These may be former S&P 500 members or ticker changes.")
        print("  Will need manual additions.")

    # Save the full mapping
    with open(OUTPUT_PATH, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"\nSaved mapping to {OUTPUT_PATH}")
    print(f"  Total tickers in mapping: {len(mapping)}")

    # Report missing tickers for manual review
    if missing:
        print(f"\n=== MISSING TICKERS (need manual addition) ===")
        for t in sorted(missing):
            print(f"  {t}")


if __name__ == "__main__":
    main()
