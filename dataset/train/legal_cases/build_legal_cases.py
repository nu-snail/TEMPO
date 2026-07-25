"""Scrape U.S. Supreme Court case data from the Oyez API.

Generates training data for the legal case prediction task.
Only includes cases with clear outcomes (PETITIONER/RESPONDENT wins).
All data is sourced from Oyez (https://www.oyez.org), maintained by
Cornell's Legal Information Institute and Chicago-Kent College of Law.

Data source: Oyez API (https://api.oyez.org)
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from html import unescape
from pathlib import Path

import requests

TERMS = list(range(2014, 2024))  # OT2014 through OT2023
REQUEST_DELAY = 0.5  # seconds between API requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
OYEZ_LIST_URL = "https://api.oyez.org/cases?per_page=200&page=0&filter=term:{term}"
OUT_DIR = Path(__file__).parent
EVAL_PATH = OUT_DIR.parent.parent / "evaluation" / "legal_cases.json"


def strip_html(html_str: str | None) -> str:
    """Strip HTML tags from Oyez API text fields and clean up whitespace."""
    if not html_str:
        return ""
    text = re.sub(r"<[^>]+>", "", unescape(html_str))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def timestamp_to_date(ts: int) -> str:
    """Convert Unix timestamp to YYYY-MM-DD string."""
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def compute_cutoff(argued_date_str: str | None, decision_date_str: str) -> str:
    """cutoff_date = decision_date (when the outcome is published)."""
    return decision_date_str


def make_acronym(name: str) -> str:
    """Generate acronym from a multi-word name (e.g., 'Department of Homeland Security' -> 'DHS')."""
    stop = {"of", "the", "and", "for", "in", "on", "to", "a", "an", "et", "al", "inc", "llc"}
    words = re.findall(r"[A-Za-z]+", name)
    letters = [w[0].upper() for w in words if w.lower() not in stop and len(w) > 1]
    return "".join(letters)


def determine_outcome(
    winning_party: str | None,
    first_party: str,
    second_party: str,
    first_party_label: str,
    second_party_label: str,
) -> str | None:
    """Map Oyez 'winning_party' to PETITIONER or RESPONDENT."""
    if not winning_party:
        return None

    wp = winning_party.strip()
    wp_lower = wp.lower()
    fp_lower = first_party.lower().strip()
    sp_lower = second_party.lower().strip()

    # 1. Direct substring match
    if wp_lower in fp_lower or fp_lower in wp_lower:
        return first_party_label.upper()
    if wp_lower in sp_lower or sp_lower in wp_lower:
        return second_party_label.upper()

    # 2. Acronym match (DHS, CFPB, EEOC, etc.)
    fp_acro = make_acronym(first_party)
    sp_acro = make_acronym(second_party)
    if wp.upper() == fp_acro:
        return first_party_label.upper()
    if wp.upper() == sp_acro:
        return second_party_label.upper()

    # 3. Word-overlap match
    wp_words = set(re.findall(r"\w+", wp_lower))
    fp_words = set(re.findall(r"\w+", fp_lower))
    sp_words = set(re.findall(r"\w+", sp_lower))

    fp_overlap = len(wp_words & fp_words)
    sp_overlap = len(wp_words & sp_words)

    if fp_overlap > sp_overlap and fp_overlap > 0:
        return first_party_label.upper()
    if sp_overlap > fp_overlap and sp_overlap > 0:
        return second_party_label.upper()

    # 4. Single-word last name match (e.g., "Liu" in "Charles C. Lui")
    if len(wp_words) == 1:
        wp_word = list(wp_words)[0]
        for fp_word in fp_words:
            if fp_word.startswith(wp_word[:3]) or wp_word.startswith(fp_word[:3]):
                return first_party_label.upper()
        for sp_word in sp_words:
            if sp_word.startswith(wp_word[:3]) or wp_word.startswith(sp_word[:3]):
                return second_party_label.upper()

    return None


def normalize_case_name(name: str) -> str:
    """Normalize a case name for fuzzy matching."""
    n = name.lower().strip()
    n = re.sub(r"\s+v\.?\s+", " v ", n)
    n = re.sub(r",?\s*(inc|llc|llp|corp|ltd|et al|et ux)\.?", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def load_eval_case_names() -> set[str]:
    """Load evaluation set case names (both exact and normalized) for deduplication."""
    if not EVAL_PATH.exists():
        return set()
    with open(EVAL_PATH) as f:
        eval_cases = json.load(f)
    names = set()
    for c in eval_cases:
        names.add(c["case_name"].lower().strip())
        names.add(normalize_case_name(c["case_name"]))
    return names


def fetch_case_list(term: int) -> list[dict]:
    """Fetch the list of cases for a SCOTUS term from Oyez."""
    url = OYEZ_LIST_URL.format(term=term)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_case_detail(href: str) -> dict | None:
    """Fetch detailed case data from Oyez."""
    try:
        resp = requests.get(href, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def parse_case(detail: dict, term: int) -> dict | None:
    """Parse an Oyez case detail into our training format. Returns None if invalid."""
    name = detail.get("name", "")
    first_party = detail.get("first_party", "")
    second_party = detail.get("second_party", "")
    first_party_label = detail.get("first_party_label", "Petitioner")
    second_party_label = detail.get("second_party_label", "Respondent")

    facts = strip_html(detail.get("facts_of_the_case"))
    conclusion = strip_html(detail.get("conclusion"))

    if not facts or not conclusion:
        return None

    # Parse timeline for dates
    timeline = detail.get("timeline", [])
    argued_date = None
    decided_date = None
    for event in (timeline or []):
        if not event or not isinstance(event, dict):
            continue
        event_name = event.get("event", "").lower()
        dates = event.get("dates", [])
        if not dates:
            continue
        if "argued" in event_name and not argued_date:
            argued_date = timestamp_to_date(dates[0])
        if "decided" in event_name:
            decided_date = timestamp_to_date(dates[0])

    if not decided_date:
        return None

    # Parse decision (outcome and vote split)
    decisions = detail.get("decisions", [])
    if not decisions:
        return None

    # Find the main decision (usually the last one, or the only one)
    decision = None
    for dec in decisions:
        if isinstance(dec, dict):
            decision = dec
    if not decision:
        return None

    majority = decision.get("majority_vote")
    minority = decision.get("minority_vote")
    winning_party = decision.get("winning_party")

    if majority is None or minority is None:
        return None

    vote_split = f"{majority}-{minority}"

    outcome = determine_outcome(
        winning_party, first_party, second_party,
        first_party_label, second_party_label,
    )

    if not outcome:
        return None

    cutoff_date = compute_cutoff(argued_date, decided_date)

    # Normalize outcome labels: APPELLANT → PETITIONER, APPELLEE → RESPONDENT
    outcome_map = {
        "PETITIONER": "PETITIONER",
        "RESPONDENT": "RESPONDENT",
        "APPELLANT": "PETITIONER",
        "APPELLEE": "RESPONDENT",
    }
    outcome = outcome_map.get(outcome, outcome)

    # Fix 0-0 vote splits
    decision_type = decision.get("decision_type", "")
    if vote_split == "0-0":
        if "equally divided" in conclusion.lower():
            vote_split = "4-4"
        elif "per curiam" in decision_type.lower() or "per curiam" in conclusion.lower():
            vote_split = "PER CURIAM"
        else:
            vote_split = "PER CURIAM"

    # Determine petitioner and respondent based on labels
    label_lower = first_party_label.lower()
    if label_lower in ("petitioner", "appellant"):
        petitioner = first_party
        respondent = second_party
    else:
        petitioner = second_party
        respondent = first_party

    return {
        "case_name": name,
        "petitioner": petitioner,
        "respondent": respondent,
        "court": "U.S. Supreme Court",
        "term": f"OT{term}",
        "facts": facts,
        "conclusion": conclusion,
        "oral_argument_date": argued_date,
        "decision_date": decided_date,
        "cutoff_date": cutoff_date,
        "outcome": outcome,
        "vote_split": vote_split,
    }


def main() -> None:
    eval_names = load_eval_case_names()
    print(f"Loaded {len(eval_names)} evaluation case names for deduplication")

    all_cases: list[dict] = []
    skipped_reasons: dict[str, int] = {}

    for term in TERMS:
        print(f"\n=== OT{term} ===")
        case_list = fetch_case_list(term)
        print(f"Cases listed: {len(case_list)}")

        term_cases = []
        for i, case_summary in enumerate(case_list):
            href = case_summary.get("href")
            name = case_summary.get("name", "???")
            if not href:
                skipped_reasons["no_href"] = skipped_reasons.get("no_href", 0) + 1
                continue

            # Check against eval set (exact and normalized)
            if name.lower().strip() in eval_names or normalize_case_name(name) in eval_names:
                skipped_reasons["in_eval_set"] = skipped_reasons.get("in_eval_set", 0) + 1
                continue

            detail = fetch_case_detail(href)
            if not detail:
                skipped_reasons["fetch_failed"] = skipped_reasons.get("fetch_failed", 0) + 1
                print(f"  WARN: Failed to fetch {name}")
                continue

            parsed = parse_case(detail, term)
            if not parsed:
                # Determine skip reason
                if not strip_html(detail.get("facts_of_the_case")):
                    reason = "no_facts"
                elif not strip_html(detail.get("conclusion")):
                    reason = "no_conclusion"
                elif not detail.get("decisions"):
                    reason = "no_decisions"
                else:
                    reason = "no_outcome_mapped"
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue

            term_cases.append(parsed)
            time.sleep(REQUEST_DELAY)

        print(f"  Parsed: {len(term_cases)} valid cases")
        all_cases.extend(term_cases)

    # Assign case IDs
    for i, case in enumerate(all_cases, start=1):
        case["case_id"] = f"train_{i:04d}"

    # Reorder fields for readability
    ordered_cases = []
    field_order = [
        "case_id", "case_name", "petitioner", "respondent", "court",
        "term", "facts", "conclusion", "oral_argument_date",
        "decision_date", "cutoff_date", "outcome", "vote_split",
    ]
    for case in all_cases:
        ordered = {k: case[k] for k in field_order if k in case}
        ordered_cases.append(ordered)

    # Save
    out_path = OUT_DIR / "legal_cases_train.json"
    with open(out_path, "w") as f:
        json.dump(ordered_cases, f, indent=2, ensure_ascii=False)

    print(f"\n=== SUMMARY ===")
    print(f"Total training cases: {len(ordered_cases)}")
    print(f"Output: {out_path.name}")
    print(f"\nSkipped reasons:")
    for reason, count in sorted(skipped_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nPer-term breakdown:")
    for term in TERMS:
        count = sum(1 for c in ordered_cases if c["term"] == f"OT{term}")
        print(f"  OT{term}: {count}")
    print(f"\nOutcome distribution:")
    pet = sum(1 for c in ordered_cases if c["outcome"] == "PETITIONER")
    resp = sum(1 for c in ordered_cases if c["outcome"] == "RESPONDENT")
    print(f"  PETITIONER: {pet}")
    print(f"  RESPONDENT: {resp}")
    print(f"\nVote split distribution:")
    splits = {}
    for c in ordered_cases:
        splits[c["vote_split"]] = splits.get(c["vote_split"], 0) + 1
    for split, count in sorted(splits.items(), key=lambda x: -x[1]):
        print(f"  {split}: {count}")


if __name__ == "__main__":
    main()
