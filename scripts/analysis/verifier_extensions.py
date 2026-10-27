"""Verifier checks for undated evidence and later company names (paper Sec. 3, Appendix H.1).

Undated: evidence items without a declared source_date default to
not-leaked. Count them exactly on the outputs behind Table 1, then date each one
from its content with the paper's judge and report how many are leaked.

Entity names: the human audit found clean-judged claims that use a company's
later name or ticker (e.g., "Meta Platforms" for Facebook in 2019). Check every
35B stock evidence claim for such names, and report the effect on OLR if they
were counted as leaked. The three audit cases serve as a recall check.

Usage (from the repository root):
    uv run python scripts/analysis/verifier_extensions.py {undated,entity} [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, "src")
from quality_gated_experiment.llm import OpenRouterClient
from quality_gated_experiment.schemas import parse_completion

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

JUDGE = "google/gemini-3-flash-preview"
PRICE_IN, PRICE_OUT = 0.50e-6, 3.00e-6  # USD per token, judge pricing on OpenRouter
CHUNK = 40
WORKERS = 8
RESULTS = Path("results")
OUT_DIR = RESULTS / "analysis" / "verifier_extensions"
METHODS = {"base": "Temporal Hint", "rag": "RAG", "timespec": "TimeSPEC"}

# Saved evaluation outputs behind Table 2 (results/<model>/eval_<task>_<method>).
TABLE1_CELLS = [
    (model, task, suffix, label)
    for model in ["qwen3_30b", "qwen35_35b"]
    for task in ["stock", "salary", "legal"]
    for suffix, label in {**METHODS, "tempo": "TEMPO"}.items()
]
ENTITY_CELLS = [c for c in TABLE1_CELLS if c[0] == "qwen35_35b" and c[1] == "stock"]

AUDIT_FLAGGED = {"stock/tempo/60/18", "stock/tempo/2/7", "stock/tempo/73/8"}

UNDATED_PROMPT = """Each item is a factual claim used as evidence for a prediction, with the cutoff date of that prediction.
For each claim, estimate from its content alone the earliest date its information became publicly available.
The claim is leaked if that date is after the cutoff date.
A claim without time-sensitive information (for example, a stable definition or long-standing fact) is not leaked.

Items:
{items_json}

Output a JSON array with one entry per item:
[{{"index": 0, "event_date": "YYYY-MM-DD or null", "leaked": false, "reason": "<one sentence>"}}]
Output ONLY the JSON array."""

ENTITY_PROMPT = """Each item is a factual claim with a cutoff date.
Check only the names and tickers of companies, organizations, or products in the claim.
Flag an item if it uses a name or ticker that was adopted after the cutoff date, for example because the company was renamed, merged, or changed its ticker later.
Do not judge whether the stated facts are true or correctly dated.

Items:
{items_json}

Output a JSON array with one entry per item:
[{{"index": 0, "anachronistic": false, "name_used": null, "name_at_cutoff": null}}]
Output ONLY the JSON array."""


def load_rows(model: str, task: str, suffix: str) -> list[tuple[int, dict]]:
    log = RESULTS / model / f"eval_{task}_{suffix}" / "eval_step_log.jsonl"
    rows = {}
    for li, line in enumerate(log.read_text().splitlines()):
        if line.strip():
            row = json.loads(line)
            rows[row["instance_id"]] = (li, row)  # last run wins (append-mode log)
    return list(rows.values())


def instance_olr(row: dict, extra_leaked: set[int] | None = None) -> float:
    claims = row.get("claims", [])
    if not claims:
        return 0.0
    extra = extra_leaked or set()
    leaked = sum(c.get("leak_verdict") == "leaked" or c.get("index") in extra for c in claims)
    return leaked / len(claims)


def parse_array(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [x for x in out if isinstance(x, dict)]


def judge_batches(client: OpenRouterClient, prompt: str, items: list[dict]) -> tuple[dict[int, dict], float]:
    """items: [{"key":..., ...payload}] -> {global position: verdict}, cost in USD."""
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]

    def run(chunk: list[dict]):
        payload = [{k: v for k, v in it.items() if k != "key"} | {"index": j} for j, it in enumerate(chunk)]
        text, usage = client.call_text(
            JUDGE, prompt.format(items_json=json.dumps(payload, indent=1, ensure_ascii=False)),
            temperature=0.0, max_tokens=8192,
        )
        return chunk, parse_array(text), usage

    verdicts, cost = {}, 0.0
    with ThreadPoolExecutor(WORKERS) as pool:
        for chunk, parsed, usage in pool.map(run, chunks):
            by_idx = {p.get("index"): p for p in parsed}
            for j, it in enumerate(chunk):
                if j in by_idx:
                    verdicts[it["key"]] = by_idx[j]
            if usage:
                cost += usage.prompt_tokens * PRICE_IN + usage.completion_tokens * PRICE_OUT
    return verdicts, cost


def run_undated(client: OpenRouterClient, limit: int | None, cells: list[tuple] | None = None) -> dict:
    per_cell, undated = {}, []
    for model, task, suffix, label in cells or TABLE1_CELLS:
        n_items = n_missing = n_malformed = 0
        for li, row in load_rows(model, task, suffix):
            parsed = parse_completion(task, row.get("completion", "")) or {}
            for item in parsed.get("evidence", []):
                if not (isinstance(item, dict) and str(item.get("fact") or "").strip()):
                    continue
                n_items += 1
                sd = str(item.get("source_date") or "").strip()
                if not sd:
                    n_missing += 1
                    undated.append({
                        "key": f"{model}/{task}/{suffix}/{li}/{len(undated)}",
                        "cell": f"{model}/{task}/{label}",
                        "claim": item["fact"].strip(),
                        "cutoff": row["cutoff_date"],
                    })
                elif not re.match(r"\d{4}-\d{2}-\d{2}$", sd):
                    n_malformed += 1
        per_cell[f"{model}/{task}/{label}"] = {
            "evidence_items": n_items, "undated": n_missing, "malformed_date": n_malformed,
        }
    targets = undated[:limit] if limit else undated
    verdicts, cost = judge_batches(
        client, UNDATED_PROMPT, [{k: u[k] for k in ("key", "claim", "cutoff")} for u in targets]
    )
    for u in targets:
        v = verdicts.get(u["key"], {})
        u["leaked"], u["event_date"], u["reason"] = v.get("leaked"), v.get("event_date"), v.get("reason")
    total_items = sum(c["evidence_items"] for c in per_cell.values())
    return {
        "per_cell": per_cell,
        "total_evidence_items": total_items,
        "total_undated": len(undated),
        "judged": len(targets),
        "judged_leaked": sum(u.get("leaked") is True for u in targets),
        "unparsed": sum(u.get("leaked") is None for u in targets),
        "items": targets,
        "cost_usd": round(cost, 4),
    }


def run_entity(client: OpenRouterClient, limit: int | None) -> dict:
    records, rows_by_cell = [], {}
    for model, task, suffix, label in ENTITY_CELLS:
        rows = load_rows(model, task, suffix)
        rows_by_cell[label] = rows
        for li, row in rows:
            for c in row.get("claims", []):
                if c.get("source") == "evidence":
                    records.append({
                        "key": f"{task}/{suffix}/{li}/{c['index']}",
                        "label": label, "li": li, "index": c["index"],
                        "claim": c["text"], "cutoff": row["cutoff_date"],
                        "gemini_leaked": c.get("leak_verdict") == "leaked",
                    })
    targets = records[:limit] if limit else records
    verdicts, cost = judge_batches(
        client, ENTITY_PROMPT, [{k: r[k] for k in ("key", "claim", "cutoff")} for r in targets]
    )
    flagged = {r["key"] for r in targets if verdicts.get(r["key"], {}).get("anachronistic") is True}

    summary = {}
    for label, rows in rows_by_cell.items():
        cell = [r for r in targets if r["label"] == label]
        if not cell:
            continue
        new_leaks = [r for r in cell if r["key"] in flagged and not r["gemini_leaked"]]
        extra_by_li: dict[int, set[int]] = {}
        for r in new_leaks:
            extra_by_li.setdefault(r["li"], set()).add(r["index"])
        olr = sum(instance_olr(row) for _, row in rows) / len(rows)
        olr_new = sum(instance_olr(row, extra_by_li.get(li)) for li, row in rows) / len(rows)
        summary[label] = {
            "claims_checked": len(cell),
            "flagged": sum(r["key"] in flagged for r in cell),
            "flagged_pct": round(100 * sum(r["key"] in flagged for r in cell) / len(cell), 2),
            "flagged_not_already_leaked": len(new_leaks),
            "olr_pct": round(100 * olr, 2),
            "olr_if_flagged_counted_pct": round(100 * olr_new, 2),
        }
    return {
        "summary": summary,
        "audit_recall": {k: (k in flagged) for k in AUDIT_FLAGGED if any(r["key"] == k for r in targets)},
        "unparsed": sum(r["key"] not in verdicts for r in targets),
        "flagged_items": [
            r | {"verdict": verdicts.get(r["key"])} for r in targets if r["key"] in flagged
        ],
        "cost_usd": round(cost, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=["undated", "entity"])
    parser.add_argument("--limit", type=int, default=None, help="Judge only the first N items (smoke test).")
    parser.add_argument(
        "--cells", default=None,
        help="Override cells for 'undated', e.g. qwen3_30b/salary/tempo/TEMPO",
    )
    args = parser.parse_args()
    cells = [tuple(c.split("/")) for c in args.cells.split(",")] if args.cells else None

    client = OpenRouterClient(os.environ["OPENROUTER_API_KEY"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_undated(client, args.limit, cells) if args.check == "undated" else run_entity(client, args.limit)
    suffix = f"_smoke{args.limit}" if args.limit else ("_custom" if cells else "")
    out = OUT_DIR / f"{args.check}{suffix}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    printable = {k: v for k, v in result.items() if k not in ("items", "flagged_items")}
    print(json.dumps(printable, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
