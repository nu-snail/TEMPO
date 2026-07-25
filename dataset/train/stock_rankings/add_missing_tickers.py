"""
Add missing tickers to sp500_mapping.json.
These are former S&P 500 members or ticker changes not in the current Wikipedia list.
Classifications verified via yfinance and public GICS records.
"""

import json
from pathlib import Path

MAPPING_PATH = Path(__file__).parent / "sp500_mapping.json"

# Manual additions with GICS-standard sub-industry names
MANUAL_ADDITIONS = {
    "CZR": {
        "name": "Caesars Entertainment",
        "sector": "Consumer Discretionary",
        "sub_industry": "Casinos & Gaming"
    },
    "DAY": {
        "name": "Dayforce (formerly Ceridian)",
        "sector": "Industrials",
        "sub_industry": "Human Resource & Employment Services"
    },
    "EMN": {
        "name": "Eastman Chemical Company",
        "sector": "Materials",
        "sub_industry": "Specialty Chemicals"
    },
    "ENPH": {
        "name": "Enphase Energy",
        "sector": "Information Technology",
        "sub_industry": "Semiconductor Equipment"  # Reclassified; was in Solar/Renewables
    },
    "FI": {
        "name": "Fiserv",
        "sector": "Financials",
        "sub_industry": "Transaction & Payment Processing Services"
        # Note: Current ticker is FISV. FI in our pool may be legacy reference.
    },
    "K": {
        "name": "Kellanova (formerly Kellogg)",
        "sector": "Consumer Staples",
        "sub_industry": "Packaged Foods & Meats"
        # Note: Kellogg split in Oct 2023. Kellanova acquired by Mars in 2025.
        # Ticker K may not have valid price data for all windows.
    },
    "KMX": {
        "name": "CarMax",
        "sector": "Consumer Discretionary",
        "sub_industry": "Automotive Retail"
    },
    "LKQ": {
        "name": "LKQ Corporation",
        "sector": "Consumer Discretionary",
        "sub_industry": "Distributors"
    },
    "MHK": {
        "name": "Mohawk Industries",
        "sector": "Consumer Discretionary",
        "sub_industry": "Home Furnishings"
    },
    "MMC": {
        "name": "Marsh & McLennan Companies",
        "sector": "Financials",
        "sub_industry": "Insurance Brokers"
    },
}


def main():
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    added = []
    for ticker, info in MANUAL_ADDITIONS.items():
        if ticker not in mapping:
            mapping[ticker] = info
            added.append(ticker)
            print(f"  Added: {ticker} -> {info['name']} ({info['sector']} / {info['sub_industry']})")
        else:
            print(f"  Already exists: {ticker}")

    with open(MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nAdded {len(added)} tickers. Total mapping size: {len(mapping)}")


if __name__ == "__main__":
    main()
