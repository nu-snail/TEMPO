"""
Test: VERIFY-style retrieval-grounded hallucination detection.

Compares two approaches on real claims from val_log.jsonl:
  A) Current: parametric Gemini judge (stored verdicts in log)
  B) New: perplexity/sonar per-claim retrieval + batch Gemini judge
     (HalluMeasure all-claims-eval + VERIFY retrieval methodology)

Three test instances are chosen to stress-test both reliability tiers:
  - stock_33  (cutoff 2021-12-31): mix of clear errors + correct + uncertain
  - stock_107 (cutoff 2021-12-31): heavily hallucinated revenue/acquisition claims
  - stock_716 (cutoff 2025-03-31): 2025 earnings — parametric Gemini has NO reliable knowledge

Usage:
    cd /home/pcv1327/RL-Based-Temporal-Prevention/FullExperiment
    OPENROUTER_API_KEY=<key> uv run python scripts/test_search_hallucination.py
"""

from __future__ import annotations

import json
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from full_experiment.llm import OpenRouterClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VAL_LOG = Path(__file__).parent.parent / "runtime/results/train_stock/val_log.jsonl"

# Instances to test — chosen for diversity of time period and hallucination type
TEST_INSTANCE_IDS = ["stock_33", "stock_107", "stock_716"]

# Models
SEARCH_MODEL = "perplexity/sonar"           # cheapest search-grounded Perplexity model
JUDGE_MODEL  = "google/gemini-3-flash-preview"  # same as current judge, for batch verdict

# Pricing reference (for cost estimate)
SONAR_INPUT_PRICE_PER_M  = 1.0   # $/M tokens
SONAR_OUTPUT_PRICE_PER_M = 1.0   # $/M tokens
SONAR_REQUEST_FEE        = 0.005  # $/request (search fee)
GEMINI_INPUT_PRICE_PER_M = 0.15  # $/M tokens (gemini-flash estimate)
GEMINI_OUTPUT_PRICE_PER_M = 0.60 # $/M tokens

MAX_WORKERS = 8  # parallel Sonar calls

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Per-claim Sonar search prompt — focused and concise to minimise tokens
SONAR_VERIFY_PROMPT = """You are a financial fact-checker. Verify this specific claim using web search.

Claim: {claim}

Search for direct evidence (earnings reports, press releases, SEC filings, news).
Return ONLY a JSON object with no extra text:
{{
  "verdict": "supported" | "unsupported" | "undecidable",
  "found_value": "key fact you found (null if nothing found)",
  "source_hint": "source description max 15 words",
  "reason": "one sentence: why this verdict"
}}

Rules:
- "supported": search confirms the claim's key numbers/facts
- "unsupported": search found conflicting facts or clear error
- "undecidable": no relevant evidence found"""

# After collecting Sonar evidence, batch-judge all claims for one completion
# (HalluMeasure all-claims-eval: one LLM call per completion instead of per claim)
BATCH_JUDGE_PROMPT = """You are a financial fact-checker. Each claim below has a search result attached.
Based ONLY on the provided search evidence, assign a verdict to each claim.

Verdicts:
- "supported": search evidence clearly confirms the claim
- "unsupported": search evidence clearly contradicts the claim
- "undecidable": search evidence is missing, ambiguous, or insufficient

CLAIMS WITH SEARCH EVIDENCE:
{claims_with_evidence}

Output ONLY a JSON array:
[{{"index": N, "verdict": "supported|unsupported|undecidable", "reason": "max 15 words"}}]"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_val_instances(path: Path, ids: list[str]) -> dict[str, dict]:
    instances: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("instance_id") in ids:
                instances[obj["instance_id"]] = obj
    missing = set(ids) - set(instances)
    if missing:
        print(f"[WARN] Instances not found in val_log: {missing}")
    return instances


# ---------------------------------------------------------------------------
# Approach B: per-claim Sonar search + batch Gemini judge
# ---------------------------------------------------------------------------

@dataclass
class SonarResult:
    index: int
    claim_text: str
    verdict: str          # supported / unsupported / undecidable
    found_value: str | None
    source_hint: str | None
    reason: str
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def _parse_sonar_json(raw: str) -> dict[str, Any]:
    """Extract JSON object from Sonar response, which may contain markdown fences."""
    text = raw.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        inner = []
        for ln in lines:
            if ln.startswith("```"):
                continue
            inner.append(ln)
        text = "\n".join(inner).strip()
    # Find first { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in: {text[:200]}")
    return json.loads(text[start:end])


def search_one_claim(
    client: OpenRouterClient,
    index: int,
    claim_text: str,
) -> SonarResult:
    """Run one Sonar call to verify a single claim."""
    prompt = SONAR_VERIFY_PROMPT.format(claim=claim_text)
    try:
        raw, usage = client.call_search_single(SEARCH_MODEL, prompt)
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        data = _parse_sonar_json(raw)
        return SonarResult(
            index=index,
            claim_text=claim_text,
            verdict=data.get("verdict", "undecidable"),
            found_value=data.get("found_value"),
            source_hint=data.get("source_hint"),
            reason=data.get("reason", ""),
            raw_response=raw,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
    except Exception as exc:
        return SonarResult(
            index=index,
            claim_text=claim_text,
            verdict="undecidable",
            found_value=None,
            source_hint=None,
            reason="",
            raw_response="",
            error=str(exc),
        )


def run_sonar_per_claim(
    client: OpenRouterClient,
    claims: list[dict],
    instance_id: str,
) -> list[SonarResult]:
    """Parallel Sonar calls for all claims in one instance."""
    print(f"\n  [Sonar] Launching {len(claims)} parallel calls for {instance_id} ...")
    t0 = time.time()
    results: dict[int, SonarResult] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(search_one_claim, client, c["index"], c["text"]): c["index"]
            for c in claims
        }
        for future in as_completed(futures):
            r = future.result()
            results[r.index] = r

    elapsed = time.time() - t0
    print(f"  [Sonar] Done in {elapsed:.1f}s")
    return [results[c["index"]] for c in claims]


def _parse_batch_judge(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner = [ln for ln in lines if not ln.startswith("```")]
        text = "\n".join(inner).strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array in: {text[:200]}")
    return json.loads(text[start:end])


def run_batch_judge(
    client: OpenRouterClient,
    claims: list[dict],
    sonar_results: list[SonarResult],
) -> dict[int, str]:
    """One Gemini call to judge all claims given Sonar evidence (HalluMeasure all-claims-eval)."""
    sonar_by_idx = {r.index: r for r in sonar_results}

    items = []
    for c in claims:
        sr = sonar_by_idx.get(c["index"])
        evidence = ""
        if sr and not sr.error:
            found = f" Found: {sr.found_value}." if sr.found_value else ""
            hint  = f" Source: {sr.source_hint}." if sr.source_hint else ""
            evidence = f"{sr.reason}.{found}{hint}"
        else:
            evidence = "Search failed or returned no evidence."
        items.append({
            "index": c["index"],
            "claim": c["text"],
            "search_evidence": evidence,
        })

    prompt = BATCH_JUDGE_PROMPT.format(
        claims_with_evidence=json.dumps(items, indent=2, ensure_ascii=False),
    )
    raw, usage = client.call_text(JUDGE_MODEL, prompt, temperature=0.0, max_tokens=1024)
    verdicts_list = _parse_batch_judge(raw)
    return {item["index"]: item["verdict"] for item in verdicts_list}


# ---------------------------------------------------------------------------
# Approach A: extract stored parametric Gemini verdicts from val_log
# ---------------------------------------------------------------------------

OLD_VERDICT_MAP = {
    "accurate": "supported",
    "inaccurate": "unsupported",
    "uncertain": "undecidable",
}

def get_old_verdicts(claims: list[dict]) -> dict[int, str]:
    """Map stored hall_verdict → Supported/Unsupported/Undecidable."""
    out = {}
    for c in claims:
        raw = c.get("hall_verdict", "uncertain")
        out[c["index"]] = OLD_VERDICT_MAP.get(raw, "undecidable")
    return out


# ---------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------

def compare_instance(
    inst: dict,
    client: OpenRouterClient,
) -> dict[str, Any]:
    iid         = inst["instance_id"]
    cutoff      = inst["cutoff_date"]
    claims      = [c for c in inst["claims"] if c.get("source") == "evidence"]  # focus on evidence claims

    print(f"\n{'='*70}")
    print(f"Instance: {iid}  cutoff={cutoff}  evidence_claims={len(claims)}")

    # --- Approach A (old) ---
    old_verdicts = get_old_verdicts(claims)

    # --- Approach B: Sonar per-claim ---
    sonar_results = run_sonar_per_claim(client, claims, iid)
    sonar_by_idx  = {r.index: r for r in sonar_results}

    # --- Approach B: batch Gemini judge ---
    print(f"  [BatchJudge] Calling {JUDGE_MODEL} with all claims + evidence ...")
    t0 = time.time()
    batch_verdicts = run_batch_judge(client, claims, sonar_results)
    print(f"  [BatchJudge] Done in {time.time()-t0:.1f}s")

    # --- Comparison table ---
    print(f"\n  {'IDX':>3}  {'CLAIM (first 65 chars)':<65}  {'OLD':>12}  {'SONAR':>12}  {'BATCH':>12}  AGREE?")
    print(f"  {'-'*3}  {'-'*65}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*6}")

    agreements = 0
    disagreements = 0
    rows = []
    for c in claims:
        idx        = c["index"]
        clip       = c["text"][:65].ljust(65)
        old_v      = old_verdicts.get(idx, "?")
        sr         = sonar_by_idx.get(idx)
        sonar_v    = sr.verdict if sr and not sr.error else "ERROR"
        batch_v    = batch_verdicts.get(idx, "?")
        old_known  = c.get("hall_known_value", None)
        agree = "✓" if sonar_v == batch_v else "~"
        agreements += (sonar_v == batch_v)
        disagreements += (sonar_v != batch_v)

        flag = ""
        # Flag cases where old had no evidence but new has a verdict
        if old_v == "unsupported" and old_known is None and sonar_v != "undecidable":
            flag = " ← NEW"

        print(f"  {idx:>3}  {clip}  {old_v:>12}  {sonar_v:>12}  {batch_v:>12}  {agree}{flag}")
        rows.append({
            "index": idx,
            "claim": c["text"],
            "old_verdict": old_v,
            "old_known_value": old_known,
            "sonar_verdict": sonar_v,
            "sonar_found_value": sr.found_value if sr else None,
            "sonar_reason": sr.reason if sr else None,
            "batch_verdict": batch_v,
            "sonar_error": sr.error if sr else None,
        })

    # Cost estimate for this instance
    total_pt = sum(r.prompt_tokens for r in sonar_results)
    total_ct = sum(r.completion_tokens for r in sonar_results)
    n_calls  = len(sonar_results)
    sonar_cost = (
        n_calls * SONAR_REQUEST_FEE
        + total_pt  / 1e6 * SONAR_INPUT_PRICE_PER_M
        + total_ct  / 1e6 * SONAR_OUTPUT_PRICE_PER_M
    )
    print(f"\n  Sonar calls: {n_calls}  tokens: {total_pt}in/{total_ct}out  "
          f"est. sonar cost: ${sonar_cost:.4f}  sonar/batch agree: {agreements}/{len(claims)}")

    return {
        "instance_id": iid,
        "cutoff": cutoff,
        "claims": rows,
        "sonar_cost_est": sonar_cost,
        "sonar_batch_agreement": agreements / len(claims) if claims else 0,
    }


# ---------------------------------------------------------------------------
# Deep analysis section — manually assess selected cases
# ---------------------------------------------------------------------------

def print_deep_analysis(results: list[dict]) -> None:
    print(f"\n\n{'='*70}")
    print("DEEP ANALYSIS")
    print("="*70)

    for inst_result in results:
        iid    = inst_result["instance_id"]
        cutoff = inst_result["cutoff"]
        print(f"\n--- {iid} (cutoff {cutoff}) ---")

        for row in inst_result["claims"]:
            idx        = row["index"]
            old_v      = row["old_verdict"]
            sonar_v    = row["sonar_verdict"]
            batch_v    = row["batch_verdict"]
            old_kv     = row["old_known_value"] or "(no correction)"
            sonar_fv   = row["sonar_found_value"] or "(none)"
            sonar_r    = row["sonar_reason"] or ""

            # Highlight interesting / discrepant cases
            interesting = (
                old_v == "unsupported" and row["old_known_value"] is None  # old flagged without evidence
                or old_v != sonar_v      # disagreement between approaches
                or old_v != batch_v      # further disagreement with batch judge
            )
            marker = "  *** " if interesting else "      "
            print(f"{marker}[{idx}] {row['claim'][:90]}")
            print(f"        Old:   {old_v}  known_value={old_kv}")
            print(f"        Sonar: {sonar_v}  found={sonar_fv}")
            print(f"        Batch: {batch_v}  reason={sonar_r}")

    # Summary across all instances
    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print("="*70)
    total_claims = sum(len(r["claims"]) for r in results)
    old_no_correction = sum(
        1 for r in results for c in r["claims"]
        if c["old_verdict"] == "unsupported" and not c["old_known_value"]
    )
    sonar_grounded = sum(
        1 for r in results for c in r["claims"]
        if c["sonar_verdict"] != "undecidable" and not c.get("sonar_error")
    )
    old_sonar_agree = sum(
        1 for r in results for c in r["claims"]
        if c["old_verdict"] == c["sonar_verdict"]
    )
    total_sonar_cost = sum(r["sonar_cost_est"] for r in results)

    print(f"Total evidence claims tested:     {total_claims}")
    print(f"Old approach - no correction:     {old_no_correction} "
          f"({100*old_no_correction/total_claims:.0f}%) — these were epistemically unreliable")
    print(f"Sonar grounded verdicts:          {sonar_grounded} "
          f"({100*sonar_grounded/total_claims:.0f}%) — these are evidence-backed")
    print(f"Old vs Sonar agreement:           {old_sonar_agree}/{total_claims} "
          f"({100*old_sonar_agree/total_claims:.0f}%)")
    print(f"Total Sonar API cost (test):      ${total_sonar_cost:.4f}")

    # Extrapolate to full validation run
    n_val_instances = 45
    avg_claims_per = total_claims / len(results)
    proj_val_cost = (total_sonar_cost / len(results)) * n_val_instances
    print(f"\nProjected cost for 45-instance validation run:  ${proj_val_cost:.2f}")
    print(f"Projected cost for 10 validation runs:           ${proj_val_cost*10:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY not set.")
        sys.exit(1)

    print(f"Loading val_log from {VAL_LOG}")
    instances = load_val_instances(VAL_LOG, TEST_INSTANCE_IDS)
    if not instances:
        print("[ERROR] No matching instances found in val_log.")
        sys.exit(1)

    client = OpenRouterClient(api_key)

    print(f"\nSearch model:  {SEARCH_MODEL}")
    print(f"Judge model:   {JUDGE_MODEL}")
    print(f"Test instances: {list(instances.keys())}")
    print(f"Approach: per-claim Sonar search → batch Gemini judgment")
    print(f"          (VERIFY retrieval + HalluMeasure all-claims-eval)")

    all_results = []
    for iid in TEST_INSTANCE_IDS:
        if iid not in instances:
            print(f"[SKIP] {iid} not in val_log")
            continue
        result = compare_instance(instances[iid], client)
        all_results.append(result)

    # Save raw results
    out_path = Path(__file__).parent.parent / "runtime/results/test_search_hallucination.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved raw results → {out_path}]")

    # Deep analysis
    print_deep_analysis(all_results)


if __name__ == "__main__":
    main()
