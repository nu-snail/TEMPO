"""Build the legal case prediction control set from post-Feb-2026 federal appellate decisions.

Fetches recent appellate opinions filed after Feb 1, 2026 from CourtListener,
extracts structured data using the same pipeline as the training appellate builder.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pdfplumber
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
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
BLOCKED_HOSTS = {"pacer.ca4.uscourts.gov", "media.ca8.uscourts.gov"}
PDF_TIMEOUT = 15
PDF_MAX_PAGES = 12
TEXT_MAX_CHARS = 8000
TARGET_CASES = 120
OUT_DIR = Path(__file__).parent

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


def search_recent_cases() -> list[dict]:
    """Search for published federal appellate opinions filed after Feb 1, 2026."""
    results = []
    courts_str = ",".join(COURTS)
    url = SEARCH_URL
    params = {
        "type": "o",
        "court": courts_str,
        "filed_after": "2026-02-01",
        "order_by": "dateFiled desc",
        "page_size": 20,
        "stat_Published": "on",
    }

    while len(results) < TARGET_CASES:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        for r in data.get("results", []):
            if len(results) >= TARGET_CASES:
                break
            download_url = None
            opinions = r.get("opinions", [])
            if opinions:
                download_url = opinions[0].get("download_url")
            if not download_url:
                continue
            host = urlparse(download_url).hostname or ""
            if host in BLOCKED_HOSTS:
                continue
            results.append({
                "cluster_id": r["cluster_id"],
                "case_name": r.get("caseName", ""),
                "court_id": r.get("court_id", ""),
                "court": r.get("court", ""),
                "date_filed": r.get("dateFiled", ""),
                "download_url": download_url,
            })

        next_url = data.get("next")
        if not next_url:
            break
        url = next_url
        params = {}
        time.sleep(1)

    return results


def download_and_extract_text(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=PDF_TIMEOUT, stream=True)
        if resp.status_code != 200:
            return None
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages[:PDF_MAX_PAGES])
        text = text.strip()
        return text[:TEXT_MAX_CHARS] if len(text) >= 200 else None
    except Exception:
        return None


def call_llm(text: str, case_name: str, court: str, date_filed: str) -> dict | None:
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
    except Exception:
        return None


def main() -> None:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    print("Searching for recent federal appellate opinions (after 2026-02-01)...")
    candidates = search_recent_cases()
    print(f"Found {len(candidates)} candidates")

    records: list[dict] = []
    for i, meta in enumerate(candidates):
        print(f"  ({i + 1}/{len(candidates)}) {meta['case_name'][:60]}...")
        text = download_and_extract_text(meta["download_url"])
        if not text:
            continue

        extracted = call_llm(text, meta["case_name"], meta["court"], meta["date_filed"])
        if not extracted:
            continue

        outcome = extracted.get("outcome")
        if outcome not in ("PETITIONER", "RESPONDENT"):
            continue

        facts = extracted.get("facts")
        conclusion = extracted.get("conclusion")
        if not facts or not conclusion:
            continue

        court_name = COURT_NAMES.get(meta["court_id"], meta["court"])
        petitioner = extracted.get("petitioner") or meta["case_name"].split(" v. ")[0].strip()
        parts = meta["case_name"].split(" v. ")
        respondent = extracted.get("respondent") or (parts[1].strip() if len(parts) > 1 else "")

        records.append({
            "case_id": f"control_{len(records) + 1:04d}",
            "case_name": meta["case_name"],
            "petitioner": petitioner,
            "respondent": respondent,
            "court": court_name,
            "term": meta["date_filed"][:4],
            "facts": facts,
            "conclusion": conclusion,
            "oral_argument_date": None,
            "decision_date": meta["date_filed"],
            "cutoff_date": meta["date_filed"],
            "outcome": outcome,
            "vote_split": extracted.get("vote_split", "3-0"),
        })

        if len(records) >= 100:
            break
        time.sleep(0.3)

    out_path = OUT_DIR / "legal_cases.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(records)} control cases to {out_path}")


if __name__ == "__main__":
    main()
