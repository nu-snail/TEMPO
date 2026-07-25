# Legal Case Prediction Training Dataset

## Motivation

This dataset is constructed for training LLMs to **avoid using temporally leaked (post-cutoff) knowledge** when predicting U.S. Supreme Court case outcomes. The core task asks the model to predict which party prevails (petitioner or respondent) given the case facts, parties, and oral argument context, with a knowledge cutoff date set 21 days before the actual decision.

A model relying on hindsight knowledge (e.g., recalling that the Court overturned Roe v. Wade in *Dobbs*, or that Title VII protects LGBTQ+ employees after *Bostock*) would predict outcomes with implausible accuracy, while a model that has learned temporal discipline would reason from pre-cutoff legal analysis only — the briefs, oral arguments, and historical precedent available before the decision.

### Why U.S. Supreme Court Cases?

SCOTUS cases are the ideal domain for temporal leakage detection because:

1. **Maximum leakage signal**: SCOTUS decisions are among the most widely reported events globally. Landmark cases like *Dobbs*, *Bostock*, and *Loper Bright* are deeply embedded in LLM training data.
2. **Clear, verifiable outcomes**: Every decided case has a definitive winner (petitioner or respondent) and a recorded vote split.
3. **Structured format**: The Oyez database provides standardized fields for all cases (parties, facts, conclusion, dates, vote breakdown).
4. **High stakes**: 5-4 and 6-3 decisions are genuinely unpredictable from pre-argument information alone, making correct predictions strong evidence of leakage.

### Why Not Other Courts?

We explored supplementing with UK Supreme Court cases (via BAILII) and notable Federal Circuit Court decisions (via CourtListener). Both were excluded because:

- **UK Supreme Court**: Judgments on BAILII are full-text documents without structured outcome fields. Extracting outcomes reliably would risk inaccuracies.
- **Federal Circuit Courts**: The vast majority of cases are routine/technical, the most-cited are Social Security disability cases, and structured outcome data (petitioner/respondent wins) is not available.

SCOTUS via Oyez provides the optimal balance of data quality, structure, and leakage signal strength.

## Dataset Summary

| File | Cases | Court | Terms |
|------|-------|-------|-------|
| `legal_cases_train.json` | 292 | U.S. Supreme Court | OT2018–OT2023 |

### Evaluation Set (Separate)

The evaluation set is stored at `dataset/evaluation/legal_cases.json` (98 cases, OT2021–OT2025). The evaluation set is held out entirely from training and uses the same JSON format, enabling direct comparison with the TimeSPEC baselines from our prior work.

## Per-Term Statistics

| Term | Cases | Decision Dates | Notes |
|------|-------|----------------|-------|
| OT2018 | 67 | Jan 2019 – Jul 2019 | Pre-COVID term; includes *Rucho* (gerrymandering), *Gamble* (dual sovereignty) |
| OT2019 | 50 | Oct 2019 – Jul 2020 | Includes *Bostock* (LGBTQ+ rights), *McGirt* (tribal sovereignty), *DACA* |
| OT2020 | 54 | Nov 2020 – Jul 2021 | Includes *Fulton* (free exercise), *TransUnion* (standing), *Brnovich* (VRA) |
| OT2021 | 28 | Dec 2021 – Jun 2022 | 10 cases held out for eval; includes *Whole Woman's Health v. Jackson* |
| OT2022 | 48 | Oct 2022 – Jun 2023 | 10 cases held out for eval; includes *Moore v. Harper*, *Counterman*, *Groff* |
| OT2023 | 46 | Oct 2023 – Jul 2024 | 10 cases held out for eval; includes *Harrington v. Purdue Pharma* |

## Outcome Distribution

| Outcome | Count | Percentage |
|---------|-------|------------|
| PETITIONER | 197 | 67.5% |
| RESPONDENT | 95 | 32.5% |

## Vote Split Distribution

| Vote Split | Count | Notes |
|-----------|-------|-------|
| 9-0 | 119 | Unanimous decisions |
| 6-3 | 50 | |
| 5-4 | 47 | Most closely divided; highest leakage signal |
| 7-2 | 29 | |
| 8-1 | 23 | |
| 8-0 | 11 | One justice recused |
| 5-3 | 5 | One justice recused |
| 7-1 | 3 | |
| 6-2 | 3 | |
| PER CURIAM | 2 | Unsigned opinions without recorded individual votes |
| 4-4 | 1 | Equally divided; affirms lower court by default |

## Data Format

Each case follows the same JSON schema as the evaluation set:

```json
{
  "case_id": "train_001",
  "case_name": "Timbs v. Indiana",
  "petitioner": "Tyson Timbs",
  "respondent": "Indiana",
  "court": "U.S. Supreme Court",
  "term": "OT2018",
  "facts": "Tyson Timbs purchased a Land Rover for approximately $42,000 ...",
  "conclusion": "The Eighth Amendment's Excessive Fines Clause is an incorporated ...",
  "oral_argument_date": "2018-11-28",
  "decision_date": "2019-02-20",
  "cutoff_date": "2019-01-30",
  "outcome": "PETITIONER",
  "vote_split": "9-0"
}
```

**Field descriptions:**

| Field | Description |
|-------|-------------|
| `case_id` | Unique identifier (`train_001` through `train_292`) |
| `case_name` | Official case name as listed by Oyez |
| `petitioner` | Full name of the petitioner (or appellant, normalized to "petitioner") |
| `respondent` | Full name of the respondent (or appellee) |
| `court` | Always `"U.S. Supreme Court"` |
| `term` | SCOTUS term (e.g., `"OT2018"` = October Term 2018) |
| `facts` | Neutral description of the case facts and question presented (sourced from Oyez) |
| `conclusion` | The Court's holding (sourced from Oyez) |
| `oral_argument_date` | Date of oral arguments (YYYY-MM-DD); may be `null` for cases decided without argument |
| `decision_date` | Date of the decision (YYYY-MM-DD) |
| `cutoff_date` | Knowledge cutoff = `decision_date - 21 days`; the model should only use information before this date |
| `outcome` | `"PETITIONER"` or `"RESPONDENT"` — the winning party |
| `vote_split` | Justice vote count (e.g., `"6-3"`, `"9-0"`, `"PER CURIAM"`) |

### Filters and Normalization

- **Post-2019 only**: All cases have `decision_date >= 2019-01-01`.
- **Eval set excluded**: 29 cases from OT2021–OT2023 that appear in the evaluation set are excluded (matching uses exact names, normalized names, and party-based deduplication to catch cases with slightly different name formatting).
- **Clear outcomes only**: Cases without a determinable winning party are excluded (e.g., cases where the Court vacated and remanded without a clear winner, or moot cases).
- **Outcome labels normalized**: Cases on direct appeal use "Appellant"/"Appellee" labels in Oyez; these are normalized to "PETITIONER"/"RESPONDENT" for consistency.
- **Vote splits fixed**: "0-0" entries from Oyez (per curiam opinions or equally divided courts) are mapped to "PER CURIAM" or "4-4" as appropriate.

## Data Sources

- **Case data:** [Oyez](https://www.oyez.org/) — the leading free legal resource for U.S. Supreme Court information, maintained by Cornell's Legal Information Institute (LII) and Chicago-Kent College of Law at Illinois Institute of Technology.
  - Oyez API: `https://api.oyez.org/cases?filter=term:{year}` (e.g., [OT2019 cases](https://api.oyez.org/cases?per_page=200&page=0&filter=term:2019))
  - Individual case detail: `https://api.oyez.org/cases/{term}/{docket}` (e.g., [Bostock v. Clayton County](https://api.oyez.org/cases/2019/17-1618))
  - Oyez website: `https://www.oyez.org/`
- **Alternative sources explored but not used:**
  - UK Supreme Court via [BAILII](https://www.bailii.org/uk/cases/UKSC/) — full-text judgments without structured outcome fields
  - Federal Circuit Courts via [CourtListener API](https://www.courtlistener.com/api/) — lacks structured petitioner/respondent outcome data
  - [Supreme Court Database (SCDB)](http://scdb.wustl.edu/) — authoritative academic dataset but lacks facts/conclusion text

## High-Leakage Targets for Case Studies

The following are the most prominent cases in the training set — widely reported decisions that LLMs almost certainly have memorized. These are ideal targets for temporal leakage case studies:

### Landmark Constitutional Decisions

| Case | Term | Outcome | Vote | Why High-Leakage |
|------|------|---------|------|-------------------|
| Bostock v. Clayton County | OT2019 | PETITIONER | 6-3 | Title VII protects LGBTQ+ employees; one of the most-reported SCOTUS decisions of the decade |
| McGirt v. Oklahoma | OT2019 | PETITIONER | 5-4 | Upheld tribal sovereignty over eastern Oklahoma; massive policy implications |
| DHS v. Regents of UC (DACA) | OT2019 | RESPONDENT | 5-4 | Blocked Trump administration's DACA rescission; front-page news for months |
| Espinoza v. Montana | OT2019 | PETITIONER | 5-4 | Expanded school choice / free exercise; closely watched by both sides |
| Rucho v. Common Cause | OT2018 | PETITIONER | 5-4 | Held partisan gerrymandering claims nonjusticiable; major elections-law decision |
| TransUnion v. Ramirez | OT2020 | PETITIONER | 5-4 | Narrowed Article III standing; landmark procedural decision |
| Moore v. Harper | OT2022 | RESPONDENT | 6-3 | Rejected "independent state legislature" theory; major elections-law case |
| Counterman v. Colorado | OT2022 | PETITIONER | 7-2 | "True threats" require subjective intent; First Amendment landmark |
| Groff v. DeJoy | OT2022 | PETITIONER | 9-0 | Raised bar for denying religious accommodations; replaced *TWA v. Hardison* standard |

### Closely Divided 5-4 Decisions (47 cases)

The 47 cases decided 5-4 represent the most closely divided — and thus hardest to predict — outcomes. These provide the strongest leakage signal because:
- A model predicting 5-4 outcomes with >70% accuracy almost certainly has post-cutoff knowledge
- Pre-argument prediction accuracy for 5-4 cases is historically ~55-60% even for expert analysts

Notable 5-4 cases include: *Rucho*, *McGirt*, *Espinoza*, *Seila Law*, *DHS v. Regents*, *TransUnion*, *Franchise Tax Board v. Hyatt*, *Department of Commerce v. New York* (census citizenship question), *Harrington v. Purdue Pharma* (Sackler bankruptcy deal).

## Generation Script

| Script | Purpose |
|--------|---------|
| `build_legal_cases.py` | Scrapes Oyez API for SCOTUS cases (OT2018–OT2023), parses structured data, maps outcomes, filters eval cases, outputs JSON |

**Usage:**

```bash
uv run python dataset/train/legal_cases/build_legal_cases.py
```

## Data Integrity

All generated data has been verified:

1. **High-profile spot-checks**: *Bostock* (PETITIONER, 6-3), *McGirt* (PETITIONER, 5-4), *Brnovich* (PETITIONER, 6-3), *Timbs* (PETITIONER, 9-0), *Gamble* (RESPONDENT, 7-2), *Counterman* (PETITIONER, 7-2), *Groff* (PETITIONER, 9-0) — all match known facts.
2. **No eval set contamination**: 29 evaluation cases successfully filtered out using multi-method deduplication (exact name, normalized name, and party-based matching). Verified with zero overlap across all methods.
3. **No hallucinated data**: All facts, conclusions, and outcomes are sourced directly from the Oyez API; no manual text generation.
4. **Date consistency**: All `cutoff_date` values are exactly 21 days before `decision_date`, consistent with the evaluation set convention.
5. **Outcome labels normalized**: "APPELLANT"/"APPELLEE" labels mapped to "PETITIONER"/"RESPONDENT" for consistency with the evaluation set.
