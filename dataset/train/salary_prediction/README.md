# Free Agent Salary Prediction Training Dataset

## Motivation

This dataset is constructed for training LLMs to **avoid using temporally leaked (post-cutoff) knowledge** when predicting free agent contract values in professional sports. The core task asks the model to predict the Average Annual Value (AAV) and contract details for a player entering free agency, given only information available before that year's free agency period.

A model relying on hindsight knowledge (e.g., recalling that Shohei Ohtani signed a $700M deal with the Dodgers, or that Tom Brady left the Patriots for Tampa Bay) would achieve artificially precise predictions, while a model that has learned proper temporal discipline would reason from pre-cutoff information only — the player's performance history, age, position value, and market context.

### Why NFL + MLB?

We selected the NFL and MLB for their **data transparency and reliability**:

- Both leagues officially disclose all contract details (total value, years, guaranteed money). There is no estimation or rumor involved — every entry in this dataset is an officially confirmed signing.
- Both sports have robust free agency markets with 40–90+ qualifying signings per year, providing sufficient training volume.
- Both sports feature high-profile "blockbuster" signings that are deeply embedded in LLM training data, creating strong leakage signals (e.g., the model may recall that Juan Soto signed with the Mets for $765M, or that Kirk Cousins went to Atlanta for $180M).

Other sports were considered but excluded: European soccer wages are almost never officially disclosed (journalist estimates only), and NHL was deemed unnecessary given the combined NFL+MLB volume already approaches 1,000 data points.

### Why Team-Changers Only?

Only players who signed with a **different team** (From_Team ≠ To_Team) are included. Re-signings are excluded because:

1. They are more predictable (incumbent teams have negotiation advantages and more private information).
2. The leakage signal is weaker — knowing a player re-signed tells the model less about market dynamics than knowing they moved to a specific new team at a specific price.

## Dataset Summary

| File | Entries | Sport | Years |
|------|---------|-------|-------|
| `nfl_salaries.csv` | 493 | NFL | 2019–2025 |
| `mlb_salaries.csv` | 386 | MLB | 2019–2025 |
| `salary_prediction_train_combined.csv` | 879 | Both | 2019–2025 |

### Evaluation Set (Separate)

The evaluation set is stored at `dataset/evaluation/nba_salaries.csv` (154 NBA free agent signings, 2019–2025). The evaluation set is held out entirely from training and uses the same CSV format, enabling direct comparison with the TimeSPEC baselines from our prior work.

## Per-Year Statistics

### NFL (493 signings)

| Year | Signings | AAV Range | Median AAV |
|------|----------|-----------|------------|
| 2019 | 64 | $5,000,000 – $22,000,000 | $8,450,000 |
| 2020 | 66 | $5,000,000 – $25,000,000 | $9,000,000 |
| 2021 | 57 | $5,000,000 – $18,000,000 | $7,800,000 |
| 2022 | 67 | $5,000,000 – $20,000,000 | $8,000,000 |
| 2023 | 71 | $5,000,000 – $37,500,000 | $7,166,667 |
| 2024 | 82 | $5,000,000 – $45,000,000 | $8,000,000 |
| 2025 | 86 | $5,000,000 – $33,500,000 | $10,000,000 |

### MLB (386 signings)

| Year | Signings | AAV Range | Median AAV |
|------|----------|-----------|------------|
| 2019 | 40 | $5,000,000 – $30,000,000 | $10,000,000 |
| 2020 | 43 | $5,000,000 – $36,000,000 | $9,000,000 |
| 2021 | 39 | $5,000,000 – $34,000,000 | $8,000,000 |
| 2022 | 70 | $5,000,000 – $43,333,333 | $8,500,000 |
| 2023 | 75 | $5,000,000 – $43,333,333 | $10,000,000 |
| 2024 | 58 | $5,000,000 – $70,000,000 | $12,000,000 |
| 2025 | 61 | $5,000,000 – $51,000,000 | $12,500,000 |

## Data Format

Each entry follows the same CSV schema as the NBA evaluation set, with an additional `Sport` column:

```csv
Sport,Year,From_Team,To_Team,Player,Position,Years,Total_Value,AAV,Status
NFL,2024,MIN,ATL,Kirk Cousins,QB,4,$180000000,$45000000,OFFICIAL
MLB,2024,LAA,LAD,Shohei Ohtani,SP/DH,10,$700000000,$70000000,OFFICIAL
```

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `Sport` | League identifier: `NFL` or `MLB` |
| `Year` | Free agency year (the season the contract starts; e.g., 2024 = March 2024 for NFL, 2023–24 offseason for MLB) |
| `From_Team` | Team the player was on before free agency (standard abbreviation; `OS` = overseas/international for MLB) |
| `To_Team` | Team the player signed with |
| `Player` | Full player name |
| `Position` | Sport-specific position: NFL uses `QB`, `WR`, `RB`, `TE`, `DL`, `ED`, `LB`, `CB`, `S`, `G`, `C`, `RT`, `LT`, etc.; MLB uses `SP`, `RP`, `CL`, `C`, `1B`, `2B`, `3B`, `SS`, `LF`, `CF`, `RF`, `DH`, etc. |
| `Years` | Contract duration in years |
| `Total_Value` | Total contract value (dollar-formatted, e.g., `$180,000,000`) |
| `AAV` | Average Annual Value = Total_Value / Years (dollar-formatted) |
| `Status` | `OFFICIAL` for all entries (only confirmed signings are included) |

### Filters Applied

- **AAV ≥ $5,000,000**: Only meaningful signings are included (minimum-salary depth signings are excluded).
- **Team-changers only**: `From_Team ≠ To_Team` (re-signings excluded).
- **Unrestricted free agents only**: Only players whose prior contracts fully expired. Franchise-tagged players (NFL) and restricted free agents are excluded by the team-changer filter (they almost never change teams above the AAV threshold).
- **Official signings only**: No rumored, unsigned, or pending deals.

## Data Sources

- **Contract data:** [Spotrac](https://www.spotrac.com/) — the industry-standard sports contract database, used by front offices and media across all major North American leagues. Spotrac sources its data from official league filings, certified agent disclosures, and NFLPA/MLBPA records.
  - NFL free agency pages: `https://www.spotrac.com/nfl/free-agents/_/year/{year}` (e.g., [2024 NFL Free Agents](https://www.spotrac.com/nfl/free-agents/_/year/2024))
  - MLB free agency pages: `https://www.spotrac.com/mlb/free-agents/_/year/{year}` (e.g., [2024 MLB Free Agents](https://www.spotrac.com/mlb/free-agents/_/year/2024))
  - Spotrac website: `https://www.spotrac.com/`
- **Team abbreviation mappings:** Compiled from standard league conventions. See `team_mappings/nfl_teams.json` and `team_mappings/mlb_teams.json` for full mappings.
  - NFL: 32 teams + historical abbreviation `OAK` (Oakland Raiders, now `LV` Las Vegas Raiders)
  - MLB: 30 teams + international origins (`JPN`, `KOR`, `OS` for overseas players) + `ATH` (Athletics, formerly Oakland `OAK`)

## High-Leakage Targets for Case Studies

The following signings are especially strong candidates for temporal leakage detection, as they were major news events deeply embedded in LLM training data. A model that has memorized these outcomes would be strongly tempted to use hindsight knowledge when predicting:

### NFL

| Year | Player | Move | Contract | Why High-Leakage |
|------|--------|------|----------|------------------|
| 2020 | Tom Brady | NE→TB | 2yr/$50M | Most iconic free agent move in NFL history; Brady leaving New England after 20 years was global news |
| 2024 | Kirk Cousins | MIN→ATL | 4yr/$180M | Largest NFL free agent QB contract of 2024; surprise destination |
| 2024 | Saquon Barkley | NYG→PHI | 3yr/$37.75M | Star RB joining a division rival; heavily covered |
| 2025 | Sam Darnold | MIN→SEA | 3yr/$100.5M | After breakout 2024 season with Vikings, moved to Seattle |
| 2025 | Davante Adams | NYJ→LAR | 2yr/$44M | Multi-team saga (GB→LV→NYJ→LAR) widely reported |
| 2025 | Stefon Diggs | HOU→NE | 3yr/$63.5M | Veteran WR changing teams amid injury concerns |
| 2025 | Justin Fields | PIT→NYJ | 2yr/$40M | Former top draft pick finding new home |
| 2024 | Derrick Henry | TEN→BAL | 2yr/$16M | Iconic Titans RB joining Lamar Jackson; heavily debated value |

### MLB

| Year | Player | Move | Contract | Why High-Leakage |
|------|--------|------|----------|------------------|
| 2024 | Shohei Ohtani | LAA→LAD | 10yr/$700M | Largest contract in sports history at the time; global news event |
| 2025 | Juan Soto | NYY→NYM | 15yr/$765M | Broke Ohtani's record; the defining move of the 2024–25 offseason |
| 2019 | Bryce Harper | WSH→PHI | 13yr/$330M | Landmark free agent deal; dominated 2018–19 offseason coverage |
| 2019 | Manny Machado | LAD→SD | 10yr/$300M | Paired with Harper as the two marquee free agents of 2019 |
| 2020 | Gerrit Cole | HOU→NYY | 9yr/$324M | Record pitching contract at the time |
| 2022 | Freddie Freeman | ATL→LAD | 6yr/$162M | Reigning World Series MVP leaving his championship team |
| 2023 | Trea Turner | LAD→PHI | 11yr/$300M | Headline signing of the 2022–23 offseason |
| 2022 | Carlos Correa | HOU→MIN | 3yr/$105.3M | Controversial signing saga that spanned three teams (MIN, SF, NYM) before landing in MIN |

## Generation Scripts

| Script | Purpose |
|--------|---------|
| `build_salary_data.py` | Main script: scrapes Spotrac for NFL and MLB free agency data (2019–2025), filters, cleans, verifies, and outputs CSVs |
| `team_mappings/nfl_teams.json` | NFL team abbreviation → full name, conference, division |
| `team_mappings/mlb_teams.json` | MLB team abbreviation → full name, league, division (includes international origins) |

**Usage:**

```bash
# Generate all salary data (NFL + MLB, 2019-2025)
uv run python dataset/train/salary_prediction/build_salary_data.py
```

The script includes built-in verification:
- AAV = Total_Value / Years (verified for 100% of entries)
- No duplicate entries (same Player + Year)
- All team abbreviations validated against mappings
- No re-signings (From_Team ≠ To_Team for all entries)

## Data Integrity

All generated data has been verified:

1. **AAV calculations** are mathematically correct: AAV = Total_Value / Years, verified for all 879 entries (493 NFL + 386 MLB).
2. **No duplicates**: Zero duplicate entries across the entire dataset.
3. **No re-signings**: All 879 entries have From_Team ≠ To_Team.
4. **No missing values**: All fields are populated for every entry.
5. **High-profile spot-checks** against known facts:
   - Tom Brady (2020): NE→TB, 2yr/$50M ✓
   - Bryce Harper (2019): WSH→PHI, 13yr/$330M ✓
   - Shohei Ohtani (2024): LAA→LAD, 10yr/$700M ✓
   - Juan Soto (2025): NYY→NYM, 15yr/$765M ✓
   - Kirk Cousins (2024): MIN→ATL, 4yr/$180M ✓
   - Saquon Barkley (2024): NYG→PHI, 3yr/$37.75M ✓
6. **No hallucinated data**: All entries are programmatically scraped from Spotrac's official free agency records; no values are manually entered or estimated.
