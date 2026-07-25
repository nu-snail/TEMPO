"""Scrape free agent signing data from Spotrac for NFL and MLB.

Generates training data for the salary prediction task.
Only includes unrestricted free agents who changed teams (From != To)
with AAV >= $5M.

Data source: Spotrac (https://www.spotrac.com)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

YEARS = range(2018, 2026)
MIN_AAV = {"NFL": 5_000_000, "MLB": 5_000_000, "NHL": 3_000_000}
REQUEST_DELAY = 2.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
SPOTRAC_URL = "https://www.spotrac.com/{sport}/free-agents/_/year/{year}"
OUT_DIR = Path(__file__).parent


SPORT_SLUGS = {"NFL": "nfl", "MLB": "mlb", "NHL": "nhl"}


def load_team_mapping(sport: str) -> dict:
    """Load team abbreviation -> name mapping."""
    filename = f"{sport.lower()}_teams.json"
    path = OUT_DIR / "team_mappings" / filename
    with open(path) as f:
        return json.load(f)


def clean_player_name(raw: str) -> str:
    """Remove Spotrac tags (QO, CC, ND, etc.) from player names."""
    cleaned = re.sub(r"(QO|CC|ND|TS|NRI|V|MUT)\s*$", "", raw).strip()
    cleaned = re.sub(r"(QO|CC|ND|TS|NRI|V|MUT)(\s)", r"\2", cleaned).strip()
    return cleaned


def clean_position(pos: str, sport: str) -> str:
    """Simplify Spotrac position labels to standard abbreviations."""
    if sport == "MLB":
        if re.match(r"^SP/SP\d$", pos):
            return "SP"
        parts = pos.split("/")
        if len(parts) == 2 and parts[0] == parts[1]:
            return parts[0]
        mapping = {
            "RP/CL": "CL",
            "RP/SU7": "RP",
            "RP/SU8": "RP",
        }
        if pos in mapping:
            return mapping[pos]
        if pos.count("/") >= 2:
            tokens = pos.split("/")
            seen = []
            for t in tokens:
                if t not in seen:
                    seen.append(t)
            return "/".join(seen[:2])
    elif sport == "NFL":
        mapping = {
            "RT/T": "RT",
            "LT/T": "LT",
            "T/RT": "RT",
            "FS/S": "FS",
            "SS/S": "SS",
            "S/FS": "FS",
        }
        if pos in mapping:
            return mapping[pos]
    elif sport == "NHL":
        mapping = {
            "LW/RW": "W",
            "RW/LW": "W",
            "C/LW": "C",
            "C/RW": "C",
        }
        if pos in mapping:
            return mapping[pos]
    return pos


def parse_money(value: str) -> int | None:
    """Parse '$180,000,000' -> 180000000. Returns None on failure."""
    try:
        return int(value.replace("$", "").replace(",", ""))
    except (ValueError, AttributeError):
        return None


def scrape_year(sport: str, year: int) -> list[dict]:
    """Scrape signed free agents from Spotrac for a given sport/year."""
    sport_slug = SPORT_SLUGS.get(sport, sport.lower())
    url = SPOTRAC_URL.format(sport=sport_slug, year=year)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")

    if not tables:
        print(f"  WARNING: No tables found for {sport} {year}")
        return []

    # Table 0 = signed free agents
    table = tables[0]
    rows = table.find_all("tr")

    entries = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 8:
            continue

        from_team = cells[0].get_text(strip=True)
        # cells[1] is the arrow icon (empty)
        to_team = cells[2].get_text(strip=True)
        player_raw = cells[3].get_text(strip=True)
        position = cells[4].get_text(strip=True)
        years_str = cells[5].get_text(strip=True)
        value_str = cells[6].get_text(strip=True)
        aav_str = cells[7].get_text(strip=True)

        if not all([from_team, to_team, player_raw, years_str, value_str, aav_str]):
            continue

        player = clean_player_name(player_raw)
        try:
            contract_years = int(years_str)
        except ValueError:
            continue

        total_value = parse_money(value_str)
        aav = parse_money(aav_str)
        if total_value is None or aav is None:
            continue

        position = clean_position(position, sport)

        entries.append(
            {
                "Sport": sport,
                "Year": year,
                "From_Team": from_team,
                "To_Team": to_team,
                "Player": player,
                "Position": position,
                "Years": contract_years,
                "Total_Value": total_value,
                "AAV": aav,
            }
        )

    return entries


def validate_teams(df: pd.DataFrame, sport: str, team_map: dict) -> None:
    """Print warnings for any team abbreviations not in our mapping."""
    all_teams = set(df["From_Team"].unique()) | set(df["To_Team"].unique())
    unknown = all_teams - set(team_map.keys())
    if unknown:
        print(f"  WARNING: Unknown {sport} team abbreviations: {unknown}")
        print("  → Add these to the team mapping file before finalizing.")


def verify_aav(df: pd.DataFrame) -> tuple[int, int]:
    """Verify AAV ≈ Total_Value / Years. Returns (correct, total)."""
    correct = 0
    total = 0
    for _, row in df.iterrows():
        expected_aav = row["Total_Value"] // row["Years"]
        # Allow $1 rounding tolerance
        if abs(row["AAV"] - expected_aav) <= 1:
            correct += 1
        total += 1
    return correct, total


def format_dollar(value: int) -> str:
    """Format integer as '$XX,XXX,XXX'."""
    return f"${value:,}"


def main() -> None:
    all_entries: list[dict] = []

    for sport in ["NFL", "MLB", "NHL"]:
        team_map = load_team_mapping(sport)
        sport_entries: list[dict] = []

        for year in YEARS:
            print(f"Scraping {sport} {year}...", end=" ", flush=True)
            entries = scrape_year(sport, year)
            print(f"raw={len(entries)}", end=" ", flush=True)

            min_aav = MIN_AAV[sport]
            entries = [e for e in entries if e["AAV"] >= min_aav]
            print(f"aav>={min_aav // 1_000_000}M: {len(entries)}", end=" ", flush=True)

            # Filter: team changers only
            entries = [e for e in entries if e["From_Team"] != e["To_Team"]]
            print(f"team_changers: {len(entries)}")

            sport_entries.extend(entries)
            time.sleep(REQUEST_DELAY)

        print(f"\n{sport} total: {len(sport_entries)} signings")

        # Validate team abbreviations
        if sport_entries:
            sport_df = pd.DataFrame(sport_entries)
            validate_teams(sport_df, sport, team_map)

            # Verify AAV calculations
            correct, total = verify_aav(sport_df)
            print(f"{sport} AAV verification: {correct}/{total} correct")
            if correct < total:
                bad = sport_df[
                    sport_df.apply(
                        lambda r: abs(r["AAV"] - r["Total_Value"] // r["Years"]) > 1,
                        axis=1,
                    )
                ]
                for _, r in bad.head(10).iterrows():
                    expected = r["Total_Value"] // r["Years"]
                    print(
                        f"  MISMATCH: {r['Player']} ({r['Year']}): "
                        f"AAV={r['AAV']:,} vs expected={expected:,} "
                        f"(total={r['Total_Value']:,} / {r['Years']}yr)"
                    )

        all_entries.extend(sport_entries)
        print()

    # Build combined DataFrame
    df = pd.DataFrame(all_entries)
    df = df.sort_values(["Sport", "Year", "AAV"], ascending=[True, True, False])
    df = df.reset_index(drop=True)

    print(f"Combined total: {len(df)} signings")
    for s in ["NFL", "MLB", "NHL"]:
        print(f"  {s}: {len(df[df['Sport'] == s])}")
    print(f"  Years: {df['Year'].min()}-{df['Year'].max()}")
    print(f"  AAV range: ${df['AAV'].min():,} - ${df['AAV'].max():,}")

    # Format monetary values for CSV output
    df["Total_Value"] = df["Total_Value"].apply(format_dollar)
    df["AAV"] = df["AAV"].apply(format_dollar)
    df["Status"] = "OFFICIAL"

    columns = [
        "Sport",
        "Year",
        "From_Team",
        "To_Team",
        "Player",
        "Position",
        "Years",
        "Total_Value",
        "AAV",
        "Status",
    ]
    df = df[columns]

    # Save per-sport CSVs
    for sport in ["NFL", "MLB", "NHL"]:
        sport_df = df[df["Sport"] == sport].copy()
        out_path = OUT_DIR / f"{sport.lower()}_salaries.csv"
        sport_df.to_csv(out_path, index=False)
        print(f"Saved {len(sport_df)} {sport} entries → {out_path.name}")

    # Save combined CSV
    combined_path = OUT_DIR / "salary_prediction_train_combined.csv"
    df.to_csv(combined_path, index=False)
    print(f"Saved {len(df)} total entries → {combined_path.name}")

    # Print per-year summary
    print("\n=== Per-Year Summary ===")
    for sport in ["NFL", "MLB", "NHL"]:
        print(f"\n{sport}:")
        sport_df = df[df["Sport"] == sport]
        for year in YEARS:
            year_df = sport_df[sport_df["Year"] == year]
            print(f"  {year}: {len(year_df)} signings")


if __name__ == "__main__":
    main()
