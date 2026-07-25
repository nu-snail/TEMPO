# Stock Return Ranking Training Dataset

## Motivation

This dataset is constructed for training LLMs to **avoid using temporally leaked (post-cutoff) knowledge** when predicting stock return rankings. The core task asks the model to rank 5 companies within the same sector by their stock return over a future period, given a knowledge cutoff date before the period begins. A model relying on hindsight knowledge (e.g., recalling that energy stocks surged after Russia's invasion of Ukraine) would achieve artificially high accuracy, while a model that has learned proper temporal discipline would reason only from pre-cutoff information.

### Why Black-Swan Windows?

Each time window is deliberately chosen to span a **major, hard-to-predict event** (a "black swan") that caused dramatic, sector-specific return divergences:

- If the model has memorized post-cutoff outcomes, it is strongly incentivized to use that knowledge — the prediction gains from leakage are large and obvious (e.g., knowing that oil prices spiked after the Ukraine invasion trivially determines energy stock rankings).
- If the model has learned temporal discipline, it must rely solely on pre-cutoff fundamentals, making the task genuinely difficult.

This design maximizes the **separability between a leaking model and a non-leaking model**, providing a strong training signal for the RL reward (OLR: Overall Leakage Ratio).

### Why Within-Sector Ranking?

All 5 companies in each question belong to the same GICS sector. Within-sector ranking is harder than cross-sector ranking because companies share macro exposures — differentiating them requires granular, company-specific reasoning rather than broad sector-level knowledge. This forces the model to demonstrate genuine analytical ability, not just recall which sectors performed well.

### Why 80/20 Sector Allocation?

For each window, approximately 80% of questions are drawn from the sectors most disrupted by the landmark event, while 20% come from other sectors. This ensures the model sees a high density of leakage-prone scenarios (where the RL reward signal is strongest) while still learning general temporal discipline across all sectors.

## Dataset Summary

| File | Questions | Window Period | Knowledge Cutoff | Landmark Event |
|------|-----------|---------------|------------------|----------------|
| `stock_rankings_window1.json` | 300 | 2022-01-03 to 2022-06-30 | 2021-12-31 | Russian invasion of Ukraine |
| `stock_rankings_window2.json` | 300 | 2023-03-01 to 2023-08-31 | 2023-02-28 | SVB banking collapse & NVIDIA AI surge |
| `stock_rankings_window3.json` | 300 | 2025-04-01 to 2025-10-31 | 2025-03-31 | Liberation Day tariffs, Iran-Israel conflict & Fed rate cuts |
| `stock_rankings_train_combined.json` | 900 | All three windows | — | Combined training set |

### Evaluation Set (Separate)

The evaluation set is stored at `dataset/evaluation/stock_rankings.json` (100 questions, Dec 2019 – Jun 2020, COVID-19 period). The evaluation set is held out entirely from training and uses the same format, enabling direct comparison with the TimeSPEC baselines from our prior work.

## Window Details

### Window 1: The Great Macro Reset (Jan – Jun 2022)

- **Landmark event:** Russian invasion of Ukraine (Feb 24, 2022)
- **Affected sectors (80% of questions):** Energy, Materials, Information Technology, Industrials
- **Key dynamics:** Oil prices surged above $130/barrel; rising interest rates and inflation compressed growth stock valuations; defense spending increased; fertilizer and commodity supply chains destabilized.

**Sector return statistics (Window 1):**

| Sector | Avg Return | N | Min | Max |
|--------|-----------|---|-----|-----|
| Energy | +0.278 | 300 | -0.047 | +0.904 |
| Industrials | -0.179 | 300 | -0.551 | +0.251 |
| Information Technology | -0.251 | 300 | -0.541 | +0.063 |
| Materials | -0.107 | 300 | -0.337 | +0.224 |
| Communication Services | -0.216 | 45 | -0.535 | +0.258 |
| Consumer Discretionary | -0.315 | 45 | -0.590 | -0.067 |
| Consumer Staples | -0.031 | 45 | -0.386 | +0.168 |
| Financials | -0.200 | 45 | -0.642 | +0.142 |
| Health Care | -0.137 | 40 | -0.444 | +0.273 |
| Real Estate | -0.176 | 40 | -0.391 | +0.006 |
| Utilities | +0.015 | 40 | -0.185 | +0.147 |

### Window 2: The Systemic Crisis & AI Rebirth (Mar – Aug 2023)

- **Landmark events:** SVB banking collapse (Mar 10, 2023); NVIDIA AI earnings "lift-off" (May 24, 2023)
- **Affected sectors (80% of questions):** Financials, Information Technology, Communication Services, Real Estate
- **Key dynamics:** Regional banks faced systemic sell-offs; NVIDIA-led semiconductor rally (individual returns up to +170%); AI-adjacent companies in Communication Services rallied; tightened lending standards impacted Real Estate.

**Sector return statistics (Window 2):**

| Sector | Avg Return | N | Min | Max |
|--------|-----------|---|-----|-----|
| Communication Services | +0.166 | 300 | -0.295 | +0.706 |
| Financials | -0.039 | 300 | -0.361 | +0.273 |
| Information Technology | +0.193 | 300 | -0.406 | +1.697 |
| Real Estate | +0.008 | 300 | -0.200 | +0.337 |
| Consumer Discretionary | +0.089 | 45 | -0.219 | +0.497 |
| Consumer Staples | +0.008 | 45 | -0.348 | +0.222 |
| Energy | +0.100 | 45 | -0.036 | +0.309 |
| Health Care | +0.014 | 45 | -0.171 | +0.772 |
| Industrials | +0.127 | 40 | -0.051 | +0.316 |
| Materials | +0.002 | 40 | -0.215 | +0.247 |
| Utilities | +0.026 | 40 | -0.261 | +0.470 |

### Window 3: The Global Policy & Geopolitical Shock (Apr – Oct 2025)

- **Landmark events:** "Liberation Day" tariffs (Apr 2, 2025); Iran-Israel military conflict (Jun 13–24, 2025); Federal Reserve rate cuts (Sep–Oct 2025)
- **Affected sectors (80% of questions):** Consumer Discretionary, Consumer Staples, Energy, Utilities
- **Key dynamics:** Sweeping tariffs on Mexico, Canada, and China hit Automotive and Manufacturing; inflation fears roiled Retail and Consumer Goods; energy price spikes during Iran-Israel conflict; defensive sectors rallied on Fed rate cuts.

**Sector return statistics (Window 3):**

| Sector | Avg Return | N | Min | Max |
|--------|-----------|---|-----|-----|
| Consumer Discretionary | +0.112 | 300 | -0.469 | +0.701 |
| Consumer Staples | -0.056 | 300 | -0.319 | +0.437 |
| Energy | -0.036 | 300 | -0.309 | +0.344 |
| Utilities | +0.102 | 300 | -0.116 | +0.756 |
| Communication Services | +0.197 | 45 | -0.370 | +1.199 |
| Financials | +0.058 | 45 | -0.413 | +0.456 |
| Health Care | +0.040 | 45 | -0.533 | +0.535 |
| Industrials | +0.163 | 45 | -0.248 | +0.766 |
| Information Technology | +0.445 | 40 | -0.511 | +2.685 |
| Materials | -0.038 | 40 | -0.309 | +0.690 |
| Real Estate | -0.049 | 40 | -0.229 | +0.204 |

## Question Format

Each question follows the same JSON schema as the evaluation set:

```json
{
  "question_id": 1,
  "sector": "Energy",
  "companies": ["APA", "DVN", "EOG", "FANG", "TPL"],
  "company_names": {
    "APA": "APA Corporation",
    "DVN": "Devon Energy",
    "EOG": "EOG Resources",
    "FANG": "Diamondback Energy",
    "TPL": "Texas Pacific Land Corporation"
  },
  "sub_industries": {
    "APA": "Oil & Gas Exploration & Production",
    "DVN": "Oil & Gas Exploration & Production",
    "EOG": "Oil & Gas Exploration & Production",
    "FANG": "Oil & Gas Exploration & Production",
    "TPL": "Oil & Gas Royalty Trusts"
  },
  "past_period": {
    "start_date": "2022-01-03",
    "end_date": "2022-06-30",
    "status": "within_model_knowledge",
    "returns": {"APA": 0.2517, "DVN": 0.2501, "EOG": 0.2554, "FANG": 0.1141, "TPL": 0.1868},
    "ranking": ["EOG", "APA", "DVN", "TPL", "FANG"],
    "start_prices": {"APA": 24.51, "DVN": 37.35, "EOG": 75.13, "FANG": 92.45, "TPL": 134.25},
    "end_prices": {"APA": 30.68, "DVN": 46.69, "EOG": 94.32, "FANG": 103.0, "TPL": 159.33},
    "actual_dates": {
      "APA": {"start_date": "2022-01-03", "end_date": "2022-06-30"},
      "...": "..."
    },
    "cutoff": "2021-12-31"
  }
}
```

**Field descriptions:**

- `companies`: 5 ticker symbols, all from the same GICS sector
- `company_names` / `sub_industries`: Human-readable names and GICS sub-industry classification
- `returns`: Actual percentage return over the window, computed as `(end_price - start_price) / start_price`
- `ranking`: Companies sorted by return (highest first) — this is the ground truth for evaluation
- `start_prices` / `end_prices`: Adjusted close prices at the start and end of the window
- `actual_dates`: The actual trading dates used (snapped from the nominal window dates to the nearest trading day)
- `cutoff`: The knowledge cutoff date; the model should only use information available before this date

## Data Sources

- **Stock prices:** Retrieved from [Yahoo Finance](https://finance.yahoo.com/) via the [`yfinance`](https://github.com/ranaroussi/yfinance) Python library (v1.1.0). The `yfinance` library accesses Yahoo Finance's public API endpoints (no authentication required). All prices are **adjusted close prices** (accounting for stock splits and dividends). Prices are snapped to the nearest prior trading day when the nominal window start/end falls on a weekend or holiday.
  - Yahoo Finance URL pattern for individual stocks: `https://finance.yahoo.com/quote/{TICKER}/history/`
  - yfinance PyPI: `https://pypi.org/project/yfinance/`
  - yfinance GitHub: `https://github.com/ranaroussi/yfinance`
- **Company metadata:** GICS (Global Industry Classification Standard) sector and sub-industry classifications sourced from the [Wikipedia List of S&P 500 companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) (table parsed via `pandas.read_html`), supplemented with manual additions for recently removed constituents (verified via `yfinance` Ticker info API).
  - Wikipedia source URL: `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`
  - GICS classification standard: maintained by [S&P Dow Jones Indices](https://www.spglobal.com/spdji/) and [MSCI](https://www.msci.com/gics)
- **Company pools:** Tickers are drawn from `dataset/sector_summary_100.json`, which contains S&P 500 constituents organized by GICS sector. After deduplication, the pool contains approximately 15–45 unique tickers per sector.
- **Reproducibility:** The generation script (`generate_stock_rankings.py`) uses a fixed random seed (`RANDOM_SEED = 42 + window_id`) for deterministic question sampling. Running the script again with the same seed and the same version of `yfinance` should produce identical results, subject to Yahoo Finance data revisions.

## Generation Scripts

| Script | Purpose |
|--------|---------|
| `build_sp500_mapping.py` | Fetches the S&P 500 constituent list from Wikipedia and builds `sp500_mapping.json` (ticker → company name, sector, sub-industry) |
| `add_missing_tickers.py` | Adds manually verified entries for tickers removed from the S&P 500 or with changed symbols |
| `generate_stock_rankings.py` | Main generation script — validates tickers, fetches prices, samples groups, computes returns/rankings, outputs JSON |

**Usage:**

```bash
# Generate a single window
uv run python dataset/train/stock_rankings/generate_stock_rankings.py --window 1

# Generate all windows + combined file
uv run python dataset/train/stock_rankings/generate_stock_rankings.py --window all
```

## Data Integrity

All generated data has been verified:

1. **Return calculations** are mathematically correct: `return = (end_price - start_price) / start_price`, rounded to 4 decimal places.
2. **Rankings** are sorted correctly (highest return first).
3. **Sector return patterns** are consistent with known market events (e.g., Energy avg +27.8% in Window 1 during the Ukraine war oil spike; IT avg +19.3% in Window 2 during the NVIDIA AI rally).
4. **No hallucinated data** — all prices are programmatically fetched from Yahoo Finance; no values are manually entered or estimated.
