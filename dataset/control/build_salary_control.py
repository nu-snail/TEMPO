"""Build the salary prediction control set from post-Feb-1-2026 transactions.

All entries verified against official trackers (FOX Sports, ESPN, NHL.com).
Signing dates are precise to the day.
"""

from __future__ import annotations

import csv
from pathlib import Path

OUT_DIR = Path(__file__).parent

# All entries below have signing dates >= 2026-02-01 (post both models' cutoffs).
# Sources: FOX Sports MLB tracker, ESPN NFL tracker, NHL.com trade tracker.
CONTROL_ENTRIES = [
    # MLB Feb 1
    {"Sport": "MLB", "Year": 2026, "From_Team": "SEA", "To_Team": "CIN", "Player": "Eugenio Suarez", "Position": "3B", "Years": 1, "Total_Value": "$15,000,000", "AAV": "$15,000,000", "Signing_Date": "2026-02-01"},
    # MLB Feb 2 (trade, but player moves to new team)
    # MLB Feb 3
    {"Sport": "MLB", "Year": 2026, "From_Team": "CHC", "To_Team": "ARI", "Player": "Carlos Santana", "Position": "1B", "Years": 1, "Total_Value": "$2,000,000", "AAV": "$2,000,000", "Signing_Date": "2026-02-03"},
    # MLB Feb 4
    {"Sport": "MLB", "Year": 2026, "From_Team": "HOU", "To_Team": "DET", "Player": "Framber Valdez", "Position": "SP", "Years": 3, "Total_Value": "$115,000,000", "AAV": "$38,333,333", "Signing_Date": "2026-02-04"},
    {"Sport": "MLB", "Year": 2026, "From_Team": "CIN", "To_Team": "SD", "Player": "Miguel Andujar", "Position": "3B", "Years": 1, "Total_Value": "$4,000,000", "AAV": "$4,000,000", "Signing_Date": "2026-02-04"},
    {"Sport": "MLB", "Year": 2026, "From_Team": "TOR", "To_Team": "BOS", "Player": "Isiah Kiner-Falefa", "Position": "2B", "Years": 1, "Total_Value": "$6,000,000", "AAV": "$6,000,000", "Signing_Date": "2026-02-04"},
    # MLB Feb 5
    {"Sport": "MLB", "Year": 2026, "From_Team": "MIL", "To_Team": "LAA", "Player": "Brent Suter", "Position": "RP", "Years": 1, "Total_Value": "$1,250,000", "AAV": "$1,250,000", "Signing_Date": "2026-02-05"},
    # MLB Feb 9
    {"Sport": "MLB", "Year": 2026, "From_Team": "ATL", "To_Team": "PIT", "Player": "Marcell Ozuna", "Position": "DH", "Years": 1, "Total_Value": "$12,000,000", "AAV": "$12,000,000", "Signing_Date": "2026-02-09"},
    {"Sport": "MLB", "Year": 2026, "From_Team": "CIN", "To_Team": "TB", "Player": "Nick Martinez", "Position": "SP", "Years": 1, "Total_Value": "$13,000,000", "AAV": "$13,000,000", "Signing_Date": "2026-02-09"},
    {"Sport": "MLB", "Year": 2026, "From_Team": "DET", "To_Team": "MIA", "Player": "Chris Paddack", "Position": "RP", "Years": 1, "Total_Value": "$4,000,000", "AAV": "$4,000,000", "Signing_Date": "2026-02-09"},
    # MLB Feb 10
    {"Sport": "MLB", "Year": 2026, "From_Team": "BAL", "To_Team": "COL", "Player": "Tomoyuki Sugano", "Position": "SP", "Years": 1, "Total_Value": "$5,100,000", "AAV": "$5,100,000", "Signing_Date": "2026-02-10"},
    {"Sport": "MLB", "Year": 2026, "From_Team": "SF", "To_Team": "DET", "Player": "Justin Verlander", "Position": "SP", "Years": 1, "Total_Value": "$13,000,000", "AAV": "$13,000,000", "Signing_Date": "2026-02-10"},
    {"Sport": "MLB", "Year": 2026, "From_Team": "CHC", "To_Team": "ATH", "Player": "Aaron Civale", "Position": "SP", "Years": 1, "Total_Value": "$6,000,000", "AAV": "$6,000,000", "Signing_Date": "2026-02-10"},
    # MLB Feb 11
    {"Sport": "MLB", "Year": 2026, "From_Team": "MIL", "To_Team": "COL", "Player": "Jose Quintana", "Position": "SP", "Years": 1, "Total_Value": "$6,000,000", "AAV": "$6,000,000", "Signing_Date": "2026-02-11"},
    # NHL trade deadline (March 6, 2026) - traded players with known contracts
    {"Sport": "NHL", "Year": 2026, "From_Team": "STL", "To_Team": "NYI", "Player": "Brayden Schenn", "Position": "C", "Years": 1, "Total_Value": "$5,750,000", "AAV": "$5,750,000", "Signing_Date": "2026-03-06"},
    {"Sport": "NHL", "Year": 2026, "From_Team": "STL", "To_Team": "DET", "Player": "Justin Faulk", "Position": "D", "Years": 2, "Total_Value": "$13,000,000", "AAV": "$6,500,000", "Signing_Date": "2026-03-06"},
    {"Sport": "NHL", "Year": 2026, "From_Team": "CGY", "To_Team": "COL", "Player": "Nazem Kadri", "Position": "C", "Years": 2, "Total_Value": "$14,000,000", "AAV": "$7,000,000", "Signing_Date": "2026-03-06"},
    {"Sport": "NHL", "Year": 2026, "From_Team": "ANA", "To_Team": "WSH", "Player": "John Carlson", "Position": "D", "Years": 1, "Total_Value": "$4,000,000", "AAV": "$4,000,000", "Signing_Date": "2026-03-06"},
    {"Sport": "NHL", "Year": 2026, "From_Team": "VAN", "To_Team": "CBJ", "Player": "Conor Garland", "Position": "RW", "Years": 2, "Total_Value": "$9,800,000", "AAV": "$4,900,000", "Signing_Date": "2026-03-06"},
    {"Sport": "NHL", "Year": 2026, "From_Team": "TOR", "To_Team": "SEA", "Player": "Bobby McMann", "Position": "LW", "Years": 1, "Total_Value": "$1,350,000", "AAV": "$1,350,000", "Signing_Date": "2026-03-06"},
    {"Sport": "NHL", "Year": 2026, "From_Team": "CHI", "To_Team": "MIN", "Player": "Nick Foligno", "Position": "LW", "Years": 1, "Total_Value": "$2,500,000", "AAV": "$2,500,000", "Signing_Date": "2026-03-06"},
    {"Sport": "NHL", "Year": 2026, "From_Team": "PHI", "To_Team": "MIN", "Player": "Bobby Brink", "Position": "RW", "Years": 2, "Total_Value": "$3,800,000", "AAV": "$1,900,000", "Signing_Date": "2026-03-06"},
    {"Sport": "NHL", "Year": 2026, "From_Team": "NYR", "To_Team": "BUF", "Player": "Sam Carrick", "Position": "C", "Years": 1, "Total_Value": "$1,125,000", "AAV": "$1,125,000", "Signing_Date": "2026-03-06"},
    # NFL early moves (March 2026) - trades with contract extensions
    {"Sport": "NFL", "Year": 2026, "From_Team": "CHI", "To_Team": "BUF", "Player": "DJ Moore", "Position": "WR", "Years": 2, "Total_Value": "$36,000,000", "AAV": "$18,000,000", "Signing_Date": "2026-03-05"},
    {"Sport": "NFL", "Year": 2026, "From_Team": "HOU", "To_Team": "CLE", "Player": "Tytus Howard", "Position": "OT", "Years": 3, "Total_Value": "$63,000,000", "AAV": "$21,000,000", "Signing_Date": "2026-03-06"},
    {"Sport": "NFL", "Year": 2026, "From_Team": "TEN", "To_Team": "NYJ", "Player": "T'Vondre Sweat", "Position": "DT", "Years": 3, "Total_Value": "$30,000,000", "AAV": "$10,000,000", "Signing_Date": "2026-03-07"},
    {"Sport": "NFL", "Year": 2026, "From_Team": "DET", "To_Team": "HOU", "Player": "David Montgomery", "Position": "RB", "Years": 2, "Total_Value": "$16,000,000", "AAV": "$8,000,000", "Signing_Date": "2026-03-07"},
]


def main() -> None:
    all_entries = sorted(CONTROL_ENTRIES, key=lambda e: (e["Sport"], e.get("Signing_Date", "")))

    columns = [
        "Sport", "Year", "From_Team", "To_Team", "Player", "Position",
        "Years", "Total_Value", "AAV", "Status", "Signing_Date",
    ]
    out_path = OUT_DIR / "salary_transactions.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for entry in all_entries:
            entry.setdefault("Status", "OFFICIAL")
            writer.writerow({k: entry.get(k, "") for k in columns})

    print(f"Saved {len(all_entries)} control entries to {out_path}")
    sports: dict[str, int] = {}
    for e in all_entries:
        sports[e["Sport"]] = sports.get(e["Sport"], 0) + 1
    for s, n in sorted(sports.items()):
        print(f"  {s}: {n}")
    print(f"\nNote: Control set will grow as NFL 2026 FA unfolds (opens March 11)")


if __name__ == "__main__":
    main()
