"""Test TEMPO's date-plausibility verifier against 120 human gold-labeled claims.

Verifier gold-label test (paper Appendix H.1). Reuses the exact
DATE_PLAUSIBILITY_PROMPT + judge model + decision rule from the paper's
pipeline. Two conditions:

  A (honest):      declared date = gold effective date (or an old safe date for
                   timeless claims). Verdict should match gold_is_leaked.
  B (adversarial): gold-leaked claims declared with a fake pre-cutoff date.
                   Catch rate = share the plausibility correction still flags.

Usage (from the repository root):
    uv run python scripts/analysis/verifier_gold_test.py
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os
import sys

sys.path.insert(0, "src")
from quality_gated_experiment.llm import OpenRouterClient
from quality_gated_experiment.prompts import DATE_PLAUSIBILITY_PROMPT

JUDGE_MODEL = "google/gemini-3-flash-preview"  # same as paper (saved eval config)
CHUNK = 80
TIMELESS_DECLARED = "2010-01-01"  # safe pre-cutoff date for claims with no gold date
FAKE_EARLY_DAYS = 180  # adversarial: declare 6 months before cutoff

GOLD_PATH = Path("dataset/verifier_gold/gold_claims.jsonl")
OUT_PATH = Path("results/analysis/verifier_gold_test.json")


def load_gold() -> list[dict]:
    gold = []
    for line in open(GOLD_PATH):
        r = json.loads(line)
        gold.append({
            "id": r["id"],
            "text": r["fact"],
            "cutoff": r["cutoff_date"],
            "gold_date": r.get("gold_effective_source_date"),
            "gold_leaked": bool(r["gold_is_leaked"]),
            "gold_category": r.get("gold_category"),
        })
    return gold


def run_verifier(client: OpenRouterClient, items: list[dict], cutoff: str) -> dict[int, dict]:
    """Replicates reward_pipeline._detect_leakage for a standalone item list."""
    out: dict[int, dict] = {}
    for start in range(0, len(items), CHUNK):
        chunk = items[start:start + CHUNK]
        payload = [
            {"index": it["index"], "fact": it["text"], "declared_source_date": it["declared"]}
            for it in chunk
        ]
        text, _ = client.call_text(
            JUDGE_MODEL,
            DATE_PLAUSIBILITY_PROMPT.format(
                cutoff=cutoff, items_json=json.dumps(payload, indent=2, ensure_ascii=False)),
            temperature=0.0,
            max_tokens=8000,
        )
        m = re.search(r"\[.*\]", text, re.DOTALL)
        rows = json.loads(m.group(0)) if m else []
        by_idx = {r["index"]: r for r in rows if "index" in r}
        for it in chunk:
            vr = by_idx.get(it["index"], {})
            plausible = bool(vr.get("plausible", True))
            corrected = vr.get("corrected_date")
            effective = corrected if (not plausible and corrected) else it["declared"]
            leaked = bool(effective and effective > cutoff)
            out[it["index"]] = {"leaked": leaked, "effective": effective, "plausible": plausible}
    return out


def fake_early(cutoff: str) -> str:
    y, m, d = map(int, cutoff.split("-"))
    return (date(y, m, d) - timedelta(days=FAKE_EARLY_DAYS)).isoformat()


def main() -> None:
    client = OpenRouterClient(api_key=os.environ["OPENROUTER_API_KEY"])
    gold = load_gold()
    print(f"gold claims: {len(gold)}  leaked: {sum(g['gold_leaked'] for g in gold)}")

    # Group by cutoff since the prompt takes one cutoff per call.
    results = {"A": [], "B": []}
    cutoffs = sorted({g["cutoff"] for g in gold})
    for cutoff in cutoffs:
        subset = [g for g in gold if g["cutoff"] == cutoff]
        # Condition A: honest declared dates.
        items_a = [
            {"index": i, "text": g["text"], "declared": g["gold_date"] or TIMELESS_DECLARED}
            for i, g in enumerate(subset)
        ]
        verd_a = run_verifier(client, items_a, cutoff)
        for i, g in enumerate(subset):
            results["A"].append({**g, "pred_leaked": verd_a[i]["leaked"],
                                 "declared": items_a[i]["declared"],
                                 "effective": verd_a[i]["effective"]})
        # Condition B: adversarial early dates on gold-leaked claims.
        leaked_sub = [g for g in subset if g["gold_leaked"]]
        if leaked_sub:
            items_b = [
                {"index": i, "text": g["text"], "declared": fake_early(cutoff)}
                for i, g in enumerate(leaked_sub)
            ]
            verd_b = run_verifier(client, items_b, cutoff)
            for i, g in enumerate(leaked_sub):
                results["B"].append({**g, "pred_leaked": verd_b[i]["leaked"],
                                     "declared": items_b[i]["declared"],
                                     "effective": verd_b[i]["effective"]})

    # Condition A metrics (leaked = positive class).
    a = results["A"]
    tp = sum(1 for r in a if r["gold_leaked"] and r["pred_leaked"])
    fp = sum(1 for r in a if not r["gold_leaked"] and r["pred_leaked"])
    fn = sum(1 for r in a if r["gold_leaked"] and not r["pred_leaked"])
    tn = sum(1 for r in a if not r["gold_leaked"] and not r["pred_leaked"])
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    catch = sum(1 for r in results["B"] if r["pred_leaked"])
    summary = {
        "condition_A": {"n": len(a), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "accuracy": (tp + tn) / len(a), "precision": prec,
                        "recall": rec, "f1": f1},
        "condition_B": {"n": len(results["B"]), "caught": catch,
                        "catch_rate": catch / len(results["B"]) if results["B"] else None},
    }
    print(json.dumps(summary, indent=2))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"summary": summary, "rows": results}, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
