"""Fetch notable federal appellate cases from CourtListener and extract structured data.

Downloads PDF opinions from court CDNs, extracts text, and uses an LLM to produce
structured training data matching the existing legal_cases_train.json schema.

Data source: CourtListener (https://www.courtlistener.com)
PDF opinions: official U.S. Courts CDN
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import pdfplumber
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from shared.config import OPENROUTER_API_KEY  # noqa: E402

COURTS = [f"ca{i}" for i in range(1, 12)] + ["cadc", "cafc"]
COURT_NAMES = {
    "ca1": "U.S. Court of Appeals for the First Circuit",
    "ca2": "U.S. Court of Appeals for the Second Circuit",
    "ca3": "U.S. Court of Appeals for the Third Circuit",
    "ca4": "U.S. Court of Appeals for the Fourth Circuit",
    "ca5": "U.S. Court of Appeals for the Fifth Circuit",
    "ca6": "U.S. Court of Appeals for the Sixth Circuit",
    "ca7": "U.S. Court of Appeals for the Seventh Circuit",
    "ca8": "U.S. Court of Appeals for the Eighth Circuit",
    "ca9": "U.S. Court of Appeals for the Ninth Circuit",
    "ca10": "U.S. Court of Appeals for the Tenth Circuit",
    "ca11": "U.S. Court of Appeals for the Eleventh Circuit",
    "cadc": "U.S. Court of Appeals for the D.C. Circuit",
    "cafc": "U.S. Court of Appeals for the Federal Circuit",
}
SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "google/gemini-2.0-flash-001"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.0
PDF_TIMEOUT = 15
PDF_MAX_PAGES = 12
TEXT_MAX_CHARS = 8000
BLOCKED_HOSTS = {"pacer.ca4.uscourts.gov", "media.ca8.uscourts.gov"}
YEARS = range(2017, 2025)
CASES_PER_YEAR = 80
OUT_DIR = Path(__file__).parent
CACHE_PATH = OUT_DIR / ".appellate_cache.json"

EXTRACTION_PROMPT = """\
You are a legal research assistant. Extract structured information from this \
federal appellate court opinion text.

OPINION TEXT:
{text}

CASE METADATA:
- Case name: {case_name}
- Court: {court}
- Date filed: {date_filed}

Extract the following fields and return ONLY valid JSON (no markdown fences):
{{
  "petitioner": "name of the appellant/petitioner who brought the appeal",
  "respondent": "name of the appellee/respondent",
  "facts": "concise factual background of the case (3-5 sentences)",
  "conclusion": "the court's legal holding and reasoning (3-5 sentences)",
  "outcome": "PETITIONER if lower court reversed/vacated, RESPONDENT if affirmed",
  "vote_split": "panel vote, e.g. 3-0 or 2-1"
}}

Rules:
- For outcome: REVERSED or VACATED = PETITIONER wins. AFFIRMED = RESPONDENT wins.
- If partly affirmed and partly reversed, choose the dominant disposition.
- If you cannot determine a field with confidence, set it to null.
- Do NOT include markdown code fences in your response.
"""


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {"search_results": {}, "extracted": {}}


def save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False)


def search_year(year: int) -> list[dict]:
    """Fetch top cases by citation count for a single year."""
    results = []
    courts_str = ",".join(COURTS)
    url = SEARCH_URL
    params = {
        "type": "o",
        "court": courts_str,
        "filed_after": f"{year}-01-01",
        "filed_before": f"{year + 1}-01-01",
        "order_by": "citeCount desc",
        "page_size": 20,
        "stat_Published": "on",
    }

    while len(results) < CASES_PER_YEAR:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        for r in data.get("results", []):
            if len(results) >= CASES_PER_YEAR:
                break
            download_url = None
            opinions = r.get("opinions", [])
            if opinions:
                download_url = opinions[0].get("download_url")
            if not download_url:
                continue
            results.append({
                "cluster_id": r["cluster_id"],
                "case_name": r.get("caseName", ""),
                "court_id": r.get("court_id", ""),
                "court": r.get("court", ""),
                "date_filed": r.get("dateFiled", ""),
                "cite_count": r.get("citeCount", 0),
                "download_url": download_url,
                "docket_number": r.get("docketNumber", ""),
            })

        next_url = data.get("next")
        if not next_url:
            break
        url = next_url
        params = {}
        time.sleep(REQUEST_DELAY)

    return results


def download_and_extract_text(url: str) -> str | None:
    """Download PDF from court CDN and extract text from first pages."""
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    if host in BLOCKED_HOSTS:
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=PDF_TIMEOUT, stream=True)
        if resp.status_code != 200:
            return None

        pdf_bytes = resp.content
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = pdf.pages[:PDF_MAX_PAGES]
            text = "\n".join(p.extract_text() or "" for p in pages)

        text = text.strip()
        if len(text) < 200:
            return None
        return text[:TEXT_MAX_CHARS]
    except Exception as e:
        print(f"  PDF failed ({host}): {e}")
        return None


def call_llm(text: str, case_name: str, court: str, date_filed: str) -> dict | None:
    """Use LLM to extract structured case data from opinion text."""
    prompt = EXTRACTION_PROMPT.format(
        text=text, case_name=case_name, court=court, date_filed=date_filed
    )
    try:
        resp = requests.post(
            LLM_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1024,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        return json.loads(content)
    except (json.JSONDecodeError, KeyError, requests.RequestException) as e:
        print(f"  LLM extraction failed for {case_name}: {e}")
        return None


def build_case_record(
    meta: dict, extracted: dict, case_id: str
) -> dict | None:
    """Assemble a case record matching the legal_cases_train.json schema."""
    outcome = extracted.get("outcome")
    if outcome not in ("PETITIONER", "RESPONDENT"):
        return None

    petitioner = extracted.get("petitioner") or meta["case_name"].split(" v. ")[0].strip()
    parts = meta["case_name"].split(" v. ")
    respondent = extracted.get("respondent") or (parts[1].strip() if len(parts) > 1 else "")

    facts = extracted.get("facts")
    conclusion = extracted.get("conclusion")
    if not facts or not conclusion:
        return None

    court_name = COURT_NAMES.get(meta["court_id"], meta["court"])
    date_filed = meta["date_filed"]

    return {
        "case_id": case_id,
        "case_name": meta["case_name"],
        "petitioner": petitioner,
        "respondent": respondent,
        "court": court_name,
        "term": str(int(date_filed[:4])) if date_filed else "",
        "facts": facts,
        "conclusion": conclusion,
        "oral_argument_date": None,
        "decision_date": date_filed,
        "cutoff_date": date_filed,
        "outcome": outcome,
        "vote_split": extracted.get("vote_split", "3-0"),
    }


def main() -> None:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in .env")

    cache = load_cache()
    all_search = cache["search_results"]
    all_extracted = cache["extracted"]

    # Phase 1: search
    for year in YEARS:
        key = str(year)
        if key in all_search and len(all_search[key]) >= CASES_PER_YEAR:
            print(f"[search] {year}: {len(all_search[key])} cached")
            continue
        print(f"[search] {year}: fetching...")
        results = search_year(year)
        all_search[key] = results
        print(f"  got {len(results)} cases")
        save_cache(cache)
        time.sleep(REQUEST_DELAY)

    # Deduplicate by cluster_id
    seen_ids: set[int] = set()
    candidates: list[dict] = []
    for year in YEARS:
        for meta in all_search.get(str(year), []):
            cid = meta["cluster_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                candidates.append(meta)
    print(f"\nTotal unique candidates: {len(candidates)}")

    # Phase 2: download + extract
    records: list[dict] = []
    idx = 0
    for i, meta in enumerate(candidates):
        cid = str(meta["cluster_id"])

        if cid in all_extracted:
            extracted = all_extracted[cid]
        else:
            print(f"[extract] ({i + 1}/{len(candidates)}) {meta['case_name'][:60]}...")
            text = download_and_extract_text(meta["download_url"])
            if not text:
                all_extracted[cid] = {"_skip": True}
                save_cache(cache)
                continue

            extracted = call_llm(text, meta["case_name"], meta["court"], meta["date_filed"])
            if not extracted:
                all_extracted[cid] = {"_skip": True}
                save_cache(cache)
                continue

            all_extracted[cid] = extracted
            if (i + 1) % 10 == 0:
                save_cache(cache)
            time.sleep(0.3)

        if extracted.get("_skip"):
            continue

        idx += 1
        case_id = f"appellate_{idx:04d}"
        record = build_case_record(meta, extracted, case_id)
        if record:
            records.append(record)

    save_cache(cache)

    # Save output
    out_path = OUT_DIR / "appellate_cases.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n=== SUMMARY ===")
    print(f"Total appellate cases: {len(records)}")
    print(f"Output: {out_path}")

    # Stats
    by_year: dict[str, int] = {}
    by_court: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    for r in records:
        y = r["decision_date"][:4]
        by_year[y] = by_year.get(y, 0) + 1
        by_court[r["court"]] = by_court.get(r["court"], 0) + 1
        by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1

    print("\nPer-year:")
    for y in sorted(by_year):
        print(f"  {y}: {by_year[y]}")
    print("\nPer-court:")
    for c, n in sorted(by_court.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")
    print("\nOutcome:")
    for o, n in sorted(by_outcome.items()):
        print(f"  {o}: {n}")


if __name__ == "__main__":
    main()
