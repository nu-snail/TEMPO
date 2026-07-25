"""Count undated evidence claims and verifier date-corrections in saved eval outputs.

Answers X7Nt W1/Q1: how often does the conservative "undated -> not leaked"
default actually fire, and how often does the verifier correct an implausible
declared date?

Usage (from QualityGatedExperiment/):
    uv run python scripts/rebuttal/undated_claims_count.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

RESULTS = Path("runtime/results")
MODELS = ["qwen3_30b", "qwen35_35b"]
TASKS = ["stock", "salary", "legal"]
METHODS = {"base": "Temporal Hint", "rag": "RAG", "timespec": "TimeSPEC", "best": "TEMPO"}

FACT_RE = re.compile(r'"fact"\s*:')
DATE_RE = re.compile(r'"source_date"\s*:\s*(null|"(?P<val>[^"]*)")')


def scan_dir(d: Path) -> dict | None:
    log = d / "eval_step_log.jsonl"
    if not log.exists():
        return None
    n_facts = n_dated = n_claims = n_corrected = 0
    verdicts: Counter = Counter()
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completion = row.get("completion", "")
        n_facts += len(FACT_RE.findall(completion))
        for m in DATE_RE.finditer(completion):
            val = m.group("val")
            if val is not None and re.match(r"\d{4}-\d{2}-\d{2}", val.strip()):
                n_dated += 1
        for c in row.get("claims", []):
            n_claims += 1
            verdicts[c.get("leak_verdict")] += 1
            if c.get("leak_plausible") is False:
                n_corrected += 1
    return {
        "evidence_items": n_facts,
        "valid_dates": n_dated,
        "undated": n_facts - n_dated,
        "undated_pct": 100.0 * (n_facts - n_dated) / n_facts if n_facts else None,
        "scored_claims": n_claims,
        "date_corrected": n_corrected,
        "corrected_pct": 100.0 * n_corrected / n_claims if n_claims else None,
        "verdicts": dict(verdicts),
    }


def main() -> None:
    out = {}
    for model in MODELS:
        for task in TASKS:
            for suffix, label in METHODS.items():
                d = RESULTS / model / f"eval_{task}_{suffix}"
                stats = scan_dir(d)
                if stats is None:
                    continue
                out[f"{model}/{task}/{label}"] = stats
                print(
                    f"{model:12s} {task:7s} {label:14s} "
                    f"items={stats['evidence_items']:5d} undated={stats['undated']:4d} "
                    f"({stats['undated_pct']:.2f}%)  corrected={stats['date_corrected']:4d} "
                    f"({stats['corrected_pct']:.2f}%)"
                )
    path = RESULTS / "rebuttal" / "undated_claims_count.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")

    grand_items = sum(s["evidence_items"] for s in out.values())
    grand_undated = sum(s["undated"] for s in out.values())
    grand_claims = sum(s["scored_claims"] for s in out.values())
    grand_corr = sum(s["date_corrected"] for s in out.values())
    print(
        f"TOTAL: items={grand_items} undated={grand_undated} "
        f"({100.0 * grand_undated / grand_items:.2f}%) "
        f"corrected={grand_corr}/{grand_claims} ({100.0 * grand_corr / grand_claims:.2f}%)"
    )
    all_verdicts: Counter = Counter()
    for s in out.values():
        all_verdicts.update(s["verdicts"])
    print("verdict distribution:", dict(all_verdicts))


if __name__ == "__main__":
    main()
