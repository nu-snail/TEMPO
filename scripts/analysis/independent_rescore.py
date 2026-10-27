"""Re-judge saved evaluation claims with independent verifiers (Claude, Kimi).

Paper Appendix H.1: rule out training-judge circularity by re-scoring
every evidence claim in the saved 35B eval outputs with judges that share no
provider with the Gemini training verifier. Same DATE_PLAUSIBILITY_PROMPT and
decision rule as the paper.

Usage (from the repository root):
    uv run python scripts/analysis/independent_rescore.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os
import sys

sys.path.insert(0, "src")
from quality_gated_experiment.llm import OpenRouterClient
from quality_gated_experiment.prompts import DATE_PLAUSIBILITY_PROMPT
from quality_gated_experiment.schemas import parse_completion

JUDGES = {
    "claude": "anthropic/claude-sonnet-4",
    "kimi": "moonshotai/kimi-k2.5",
}
CHUNK = 80
WORKERS = 8
MODEL_DIR = "qwen35_35b"
# The paper reports this check on the Qwen3.5-35B stock outputs.
TASKS = ["stock"]
METHODS = {"base": "Temporal Hint", "rag": "RAG", "timespec": "TimeSPEC", "tempo": "TEMPO"}

RESULTS = Path("results")
OUT_DIR = RESULTS / "analysis" / "independent_rescore"


def collect_claims() -> list[dict]:
    """One record per evidence claim with declared date, cutoff, Gemini verdict."""
    records = []
    for task in TASKS:
        for suffix, label in METHODS.items():
            log = RESULTS / MODEL_DIR / f"eval_{task}_{suffix}" / "eval_step_log.jsonl"
            for li, line in enumerate(log.read_text().splitlines()):
                if not line.strip():
                    continue
                row = json.loads(line)
                parsed = parse_completion(task, row.get("completion", ""))
                date_by_fact = {}
                if parsed:
                    for item in parsed.get("evidence", []):
                        if isinstance(item, dict) and item.get("fact"):
                            date_by_fact.setdefault(item["fact"].strip(), item.get("source_date"))
                for c in row.get("claims", []):
                    if c.get("source") != "evidence":
                        continue
                    records.append({
                        "uid": f"{task}/{suffix}/{li}/{c['index']}",
                        "task": task,
                        "method": label,
                        "text": c["text"],
                        "declared": date_by_fact.get(c["text"].strip()),
                        "cutoff": row["cutoff_date"],
                        "gemini_leaked": c.get("leak_verdict") == "leaked",
                    })
    return records


def judge_chunk(client: OpenRouterClient, model: str, chunk: list[dict], cutoff: str) -> dict[str, bool]:
    payload = [
        {"index": i, "fact": r["text"], "declared_source_date": r["declared"]}
        for i, r in enumerate(chunk)
    ]
    text, _ = client.call_text(
        model,
        DATE_PLAUSIBILITY_PROMPT.format(
            cutoff=cutoff, items_json=json.dumps(payload, indent=2, ensure_ascii=False)),
        temperature=0.0,
        max_tokens=8000,
    )
    m = re.search(r"\[.*\]", text, re.DOTALL)
    rows = json.loads(m.group(0)) if m else []
    by_idx = {r["index"]: r for r in rows if "index" in r}
    out = {}
    for i, r in enumerate(chunk):
        vr = by_idx.get(i, {})
        plausible = bool(vr.get("plausible", True))
        corrected = vr.get("corrected_date")
        effective = corrected if (not plausible and corrected) else r["declared"]
        out[r["uid"]] = bool(effective and effective > cutoff)
    return out


def main() -> None:
    client = OpenRouterClient(api_key=os.environ["OPENROUTER_API_KEY"])
    records = collect_claims()
    dated = [r for r in records if r["declared"]]
    print(f"claims: {len(records)} total, {len(dated)} with declared date (undated => clean)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_cutoff: dict[str, list[dict]] = defaultdict(list)
    for r in dated:
        by_cutoff[r["cutoff"]].append(r)

    verdicts: dict[str, dict[str, bool]] = {}
    for judge, model in JUDGES.items():
        jobs = []
        for cutoff, rs in by_cutoff.items():
            for start in range(0, len(rs), CHUNK):
                jobs.append((rs[start:start + CHUNK], cutoff))
        print(f"{judge}: {len(jobs)} calls")
        judge_verdicts: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(judge_chunk, client, model, chunk, cutoff)
                       for chunk, cutoff in jobs]
            for n, fut in enumerate(futures, 1):
                judge_verdicts.update(fut.result())
                if n % 20 == 0:
                    print(f"  {judge}: {n}/{len(jobs)} done")
        verdicts[judge] = judge_verdicts
        (OUT_DIR / f"verdicts_{judge}.json").write_text(json.dumps(judge_verdicts))

    # Metrics: OLR per method/task per judge; agreement with Gemini on dated claims.
    summary: dict[str, dict] = {}
    for task in TASKS:
        for label in METHODS.values():
            cell = [r for r in records if r["task"] == task and r["method"] == label]
            key = f"{task}/{label}"
            entry = {"n_claims": len(cell),
                     "olr_gemini": sum(r["gemini_leaked"] for r in cell) / len(cell)}
            for judge in JUDGES:
                jl = [verdicts[judge].get(r["uid"], False) for r in cell]
                entry[f"olr_{judge}"] = sum(jl) / len(cell)
                dated_cell = [(r, v) for r, v in zip(cell, jl) if r["declared"]]
                agree = sum(1 for r, v in dated_cell if v == r["gemini_leaked"])
                entry[f"agree_{judge}"] = agree / len(dated_cell) if dated_cell else None
            summary[key] = entry
            print(key, json.dumps(entry))

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records))
    print(f"Wrote {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
