"""Enrich salary CSVs with precise per-player signing dates.

Uses an LLM with web access to look up exact contract announcement dates
for each free-agent signing, then adds a Signing_Date (YYYY-MM-DD) column.

Processes players in batches and caches results to allow resumption.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[3] / ".env")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
SEARCH_MODEL = "perplexity/sonar"
OUT_DIR = Path(__file__).parent
CACHE_PATH = OUT_DIR / ".signing_dates_cache.json"
BATCH_SIZE = 15
REQUEST_DELAY = 2.0

NFL_FA_OPENS = {
    2018: "2018-03-14", 2019: "2019-03-13", 2020: "2020-03-18",
    2021: "2021-03-17", 2022: "2022-03-16", 2023: "2023-03-15",
    2024: "2024-03-13", 2025: "2025-03-12",
}
NHL_FA_OPENS = {y: f"{y}-07-01" for y in range(2018, 2026)}

# NBA eval data: FA moratorium lifts ~June 30 / July 1
NBA_FA_OPENS = {
    2019: "2019-06-30", 2020: "2020-11-20", 2021: "2021-08-02",
    2022: "2022-06-30", 2023: "2023-06-30", 2024: "2024-06-30",
    2025: "2025-06-30",
}

LOOKUP_PROMPT = """\
Look up the exact date each of these {sport} free-agent contracts was publicly \
announced or officially signed. Return ONLY valid JSON (no markdown fences): \
a list of objects with "player" and "signing_date" (YYYY-MM-DD format).

If you cannot find the exact date, use your best estimate based on the \
free agency period for that sport and year, but always provide a specific date.

Players ({sport} {year} free agency, signed with new teams):
{players}
"""


def load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict[str, str]) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def cache_key(player: str, sport: str, year: int) -> str:
    return f"{sport}|{year}|{player}"


def fallback_date(sport: str, year: int) -> str:
    """Conservative fallback: first day of the free agency window."""
    if sport == "NFL":
        return NFL_FA_OPENS.get(year, f"{year}-03-15")
    if sport == "NHL":
        return NHL_FA_OPENS.get(year, f"{year}-07-01")
    if sport == "MLB":
        return f"{year - 1}-12-01"
    if sport == "NBA":
        return NBA_FA_OPENS.get(year, f"{year}-07-01")
    return f"{year}-01-01"


def lookup_batch(
    players: list[str], sport: str, year: int
) -> dict[str, str]:
    """Ask the search LLM for exact signing dates of a batch of players."""
    player_list = "\n".join(f"- {p}" for p in players)
    prompt = LOOKUP_PROMPT.format(
        sport=sport, year=year, players=player_list
    )

    try:
        resp = requests.post(
            LLM_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": SEARCH_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2048,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        results = json.loads(content)
        date_map: dict[str, str] = {}
        for item in results:
            name = item.get("player", "")
            date = item.get("signing_date", "")
            if name and re.match(r"\d{4}-\d{2}-\d{2}", date):
                date_map[name] = date
        return date_map
    except Exception as e:
        print(f"  LLM lookup failed: {e}")
        return {}


def match_player_name(lookup_name: str, csv_names: list[str]) -> str | None:
    """Fuzzy-match a name from LLM response to a name in our CSV."""
    lookup_lower = lookup_name.lower().strip()
    for csv_name in csv_names:
        csv_lower = csv_name.lower().strip()
        if csv_lower == lookup_lower:
            return csv_name
        if csv_lower.split()[-1] == lookup_lower.split()[-1]:
            if csv_lower.split()[0][0] == lookup_lower.split()[0][0]:
                return csv_name
    return None


def enrich_csv(csv_path: Path, cache: dict[str, str]) -> int:
    """Add Signing_Date column to a salary CSV. Returns count of enriched rows."""
    df = pd.read_csv(csv_path)
    if "Signing_Date" not in df.columns:
        df["Signing_Date"] = ""

    # Infer sport from filename if Sport column missing
    if "Sport" not in df.columns:
        stem = csv_path.stem.lower()
        if "nba" in stem:
            df["Sport"] = "NBA"
        elif "nfl" in stem:
            df["Sport"] = "NFL"
        elif "mlb" in stem:
            df["Sport"] = "MLB"
        elif "nhl" in stem:
            df["Sport"] = "NHL"
        else:
            df["Sport"] = "UNKNOWN"

    groups = df.groupby(["Sport", "Year"])
    uncached_batches: list[tuple[str, int, list[str]]] = []

    for (sport, year), group in groups:
        uncached = []
        for _, row in group.iterrows():
            key = cache_key(row["Player"], sport, year)
            if key not in cache:
                uncached.append(row["Player"])
        if uncached:
            uncached_batches.append((sport, year, uncached))

    total_looked_up = 0
    for sport, year, players in uncached_batches:
        for i in range(0, len(players), BATCH_SIZE):
            batch = players[i : i + BATCH_SIZE]
            print(f"  Looking up {sport} {year}: {len(batch)} players...")
            date_map = lookup_batch(batch, sport, year)

            for p in batch:
                key = cache_key(p, sport, year)
                matched = match_player_name(p, list(date_map.keys()))
                if matched and matched in date_map:
                    cache[key] = date_map[matched]
                else:
                    for lname, ldate in date_map.items():
                        if p.split()[-1].lower() in lname.lower():
                            cache[key] = ldate
                            break
                    else:
                        cache[key] = fallback_date(sport, year)
                total_looked_up += 1

            save_cache(cache)
            time.sleep(REQUEST_DELAY)

    enriched = 0
    for idx, row in df.iterrows():
        key = cache_key(row["Player"], row["Sport"], row["Year"])
        if key in cache:
            df.at[idx, "Signing_Date"] = cache[key]
            enriched += 1

    # Read original columns to preserve format (don't add Sport if it wasn't there)
    orig_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if "Signing_Date" not in orig_cols:
        orig_cols.append("Signing_Date")
    save_cols = [c for c in df.columns if c in orig_cols]
    df[save_cols].to_csv(csv_path, index=False)
    return enriched


def main() -> None:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    cache = load_cache()
    print(f"Loaded {len(cache)} cached signing dates")

    csv_files = [
        OUT_DIR / "salary_prediction_train_combined.csv",
        OUT_DIR / "nfl_salaries.csv",
        OUT_DIR / "mlb_salaries.csv",
        OUT_DIR / "nhl_salaries.csv",
    ]

    eval_csv = OUT_DIR.parent.parent / "evaluation" / "nba_salaries.csv"
    if eval_csv.exists():
        csv_files.append(eval_csv)

    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"Skipping {csv_path.name}: not found")
            continue
        print(f"\nEnriching {csv_path.name}...")
        n = enrich_csv(csv_path, cache)
        print(f"  Enriched {n} rows")

    print(f"\nTotal cached dates: {len(cache)}")
    print("Done.")


if __name__ == "__main__":
    main()
