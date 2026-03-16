"""Re-run the e2e pipeline but save all data for detailed audit."""
import os, sys, json, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
from cov_gated_experiment.llm import OpenRouterClient
from cov_gated_experiment.prompts import STOCK_SYSTEM_PROMPT

client = OpenRouterClient(api_key=os.environ["OPENROUTER_API_KEY"])

CACHE = Path("/tmp/e2e_pipeline_cache.json")

if CACHE.exists():
    print("Loading cached pipeline results...")
    with open(CACHE) as f:
        cache = json.load(f)
    completions = cache["completions"]
    claim_pool = cache["claim_pool"]
    claim_to_completions = cache["claim_to_completions"]
    leak_results = {int(k): v for k, v in cache["leak_results"].items()}
    hall_results = {int(k): v for k, v in cache["hall_results"].items()}
    merged = {int(k): v for k, v in cache["merged"].items()}
    cutoff = cache["cutoff"]
else:
    # Full pipeline (same as test_e2e_pipeline.py)
    with open(ROOT.parent / "dataset/train/stock_rankings/stock_rankings_train_combined.json") as f:
        dataset = json.load(f)
    instance_raw = dataset["questions"][0]
    cutoff = instance_raw["past_period"]["cutoff"]
    companies = instance_raw["companies"]
    company_names = instance_raw["company_names"]
    sector = instance_raw["sector"]
    period = instance_raw["past_period"]

    system_prompt = STOCK_SYSTEM_PROMPT.format(cutoff_date=cutoff)
    user_lines = [f"Sector: {sector}", f"Evaluation period: {period['start_date']} to {period['end_date']}", "Companies to rank:"]
    for t in companies:
        user_lines.append(f"  - {t} ({company_names[t]})")
    user_prompt = "\n".join(user_lines)

    MODEL = "qwen/qwen3-30b-a3b"

    def generate_one(idx):
        text, usage = client._execute({
            "model": MODEL,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.8, "max_tokens": 4096,
        })
        return idx, text, usage

    print("Generating 5 completions...")
    completions = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(generate_one, i) for i in range(5)]
        raw_outputs = []
        for f in as_completed(futures):
            idx, text, usage = f.result()
            raw_outputs.append((idx, text))
            print(f"  Completion {idx} done")
    raw_outputs.sort(key=lambda x: x[0])

    for idx, text in raw_outputs:
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>")[-1].strip()
        m = re.search(r"\{[\s\S]*\}", cleaned)
        parsed = json.loads(m.group()) if m else {"evidence": [], "reasoning": ""}
        completions.append({"index": idx, "parsed": parsed})

    # Claim extraction
    claim_pool = []
    claim_to_completions = {}
    for comp in completions:
        ci = comp["index"]
        for ev in comp["parsed"].get("evidence", []):
            fact = ev.get("fact", "").strip()
            if not fact: continue
            key = fact.lower()
            if key not in claim_to_completions:
                claim_to_completions[key] = []
                claim_pool.append({"text": fact, "key": key, "source": "evidence"})
            claim_to_completions[key].append(ci)

    # Reasoning extraction
    batch_input = [{"completion_index": c["index"], "evidence": [ev["fact"] for ev in c["parsed"].get("evidence", [])], "reasoning": c["parsed"].get("reasoning", "")} for c in completions]
    EXTRACT_PROMPT = """You are a grounding auditor. For each completion below, check whether the REASONING paragraph introduces any NEW factual claims (specific numbers, dates, events, entity names, statistics) that are NOT already stated in its EVIDENCE list.
RULES:
- Analysis, opinions, predictions, and logical inferences from evidence are NOT new facts. Skip them.
- Only flag CONCRETE factual assertions: specific numbers, reported events, dates, statistics, entity names, or data points that do not appear in the evidence list.
- If the reasoning merely paraphrases or summarizes evidence items, that is NOT a new fact.
COMPLETIONS:
{completions_json}
Output a JSON array with one entry per completion:
[{{"completion_index": 0, "new_facts": [{{"fact": "<the new factual claim>", "why_new": "<brief>"}}]}}]
If a completion has NO new facts, set "new_facts" to [].
Output ONLY the JSON array."""
    text, _ = client.call_text("google/gemini-3-flash-preview", EXTRACT_PROMPT.format(completions_json=json.dumps(batch_input, indent=2, ensure_ascii=False)), temperature=0.0, max_tokens=4096)
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
    m = re.search(r"\[[\s\S]*\]", cleaned)
    for entry in (json.loads(m.group()) if m else []):
        ci = entry["completion_index"]
        for nf in entry.get("new_facts", []):
            key = nf["fact"].strip().lower()
            if key not in claim_to_completions:
                claim_to_completions[key] = []
                claim_pool.append({"text": nf["fact"], "key": key, "source": "reasoning"})
            claim_to_completions[key].append(ci)

    for i, c in enumerate(claim_pool):
        c["index"] = i

    # Leakage
    LEAK_PROMPT = """You are a temporal leakage detector for financial analysis.
Cutoff date: {cutoff}
For each claim, determine: could an analyst have KNOWN ABOUT this type of event/information before the cutoff date? Focus ONLY on TIMING.
CRITICAL RULES:
- IGNORE whether the specific numbers are accurate.
- "leaked" means the EVENT ITSELF had not yet occurred or been announced before the cutoff.
- "not_leaked" means the EVENT TYPE was publicly known before the cutoff.
- Financial quarterly results (e.g., Q4 2024) are announced ~1-2 months after quarter end.
- Full-year results for year Y are announced in Jan-Feb of Y+1.
CLAIMS:
{claims_json}
Output a JSON array:
[{{"index": 0, "verdict": "leaked|not_leaked", "event_date": "YYYY-MM or null", "confidence": "high|medium|low"}}]
Output ONLY the JSON array."""
    leak_payload = [{"index": c["index"], "claim": c["text"]} for c in claim_pool]
    text, _ = client.call_text("google/gemini-3-flash-preview", LEAK_PROMPT.format(cutoff=cutoff, claims_json=json.dumps(leak_payload, indent=2, ensure_ascii=False)), temperature=0.0, max_tokens=8192)
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
    m = re.search(r"\[[\s\S]*\]", cleaned)
    leak_results = {}
    for r in (json.loads(m.group()) if m else []):
        if isinstance(r, dict) and "index" in r:
            leak_results[r["index"]] = r

    # Hallucination
    HALL_PROMPT = """You are a factual accuracy auditor. Your ONLY job is to check whether each claim below is factually accurate based on your knowledge.
DO NOT assess timing, leakage, or temporal availability. ONLY check factual correctness.
For each claim, respond:
- "accurate": The specific facts/numbers stated are correct or very close to correct.
- "inaccurate": The specific facts/numbers are materially wrong.
- "unverifiable": You don't have enough knowledge to verify this claim. Default to this if unsure.
IMPORTANT: If you are not confident about the exact numbers, say "unverifiable" rather than guessing. Only mark "inaccurate" when you are CERTAIN the claim is factually wrong.
CLAIMS:
{claims_json}
Output a JSON array:
[{{"index": 0, "verdict": "accurate|inaccurate|unverifiable", "reason": "<brief>"}}, ...]
Output ONLY the JSON array, no other text."""
    hall_payload = [{"index": c["index"], "claim": c["text"]} for c in claim_pool]
    text, _ = client.call_text("anthropic/claude-sonnet-4.6", HALL_PROMPT.format(claims_json=json.dumps(hall_payload, indent=2, ensure_ascii=False)), temperature=0.0, max_tokens=8192)
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
    m = re.search(r"\[[\s\S]*\]", cleaned)
    hall_results = {}
    for r in (json.loads(m.group()) if m else []):
        if isinstance(r, dict) and "index" in r:
            hall_results[r["index"]] = r

    # Merge
    merged = {}
    for c in claim_pool:
        lr = leak_results.get(c["index"], {})
        hr = hall_results.get(c["index"], {})
        leak_v = lr.get("verdict", "not_leaked")
        hall_v = hr.get("verdict", "unverifiable")
        if leak_v == "leaked": merged[c["index"]] = "leaked"
        elif hall_v == "inaccurate": merged[c["index"]] = "hallucinated"
        else: merged[c["index"]] = "valid"

    # Cache
    with open(CACHE, "w") as f:
        json.dump({"completions": completions, "claim_pool": claim_pool, "claim_to_completions": claim_to_completions, "leak_results": leak_results, "hall_results": hall_results, "merged": merged, "cutoff": cutoff}, f)

# ── AUDIT OUTPUT ──────────────────────────────────────────────────────────────
print(f"\nInstance: Communication Services, cutoff={cutoff}")
print(f"Companies: CHTR, DIS, GOOG, META, WBD\n")

for comp in completions:
    ci = comp["index"]
    parsed = comp["parsed"]
    
    # Collect claims for this completion
    comp_claims = []
    for c in claim_pool:
        if ci in claim_to_completions[c["key"]]:
            comp_claims.append(c)
    
    n_valid = sum(1 for c in comp_claims if merged[c["index"]] == "valid")
    n_leaked = sum(1 for c in comp_claims if merged[c["index"]] == "leaked")
    n_hall = sum(1 for c in comp_claims if merged[c["index"]] == "hallucinated")
    denom = n_valid + n_leaked + n_hall
    r_valid = n_valid / (denom + 1e-8) if denom > 0 else 0.0
    
    print("=" * 80)
    print(f"COMPLETION {ci}  |  R_valid={r_valid:.3f}  |  valid={n_valid}, leaked={n_leaked}, hall={n_hall}")
    print("=" * 80)
    
    print(f"\nRanking: {parsed.get('ranking', '?')}")
    print(f"\nReasoning:\n  {parsed.get('reasoning', 'N/A')[:500]}")
    
    print(f"\nClaims ({len(comp_claims)}):")
    for c in comp_claims:
        lr = leak_results.get(c["index"], {})
        hr = hall_results.get(c["index"], {})
        label = merged[c["index"]]
        leak_v = lr.get("verdict", "?")
        leak_date = lr.get("event_date", "?")
        leak_conf = lr.get("confidence", "?")
        hall_v = hr.get("verdict", "?")
        hall_reason = hr.get("reason", "")
        
        marker = {"leaked": "LEAK", "hallucinated": "HALL", "valid": " OK "}.get(label, " ?  ")
        src = "[R]" if c["source"] == "reasoning" else "[E]"
        
        print(f"\n  [{marker}] {src} Claim #{c['index']}:")
        print(f"    TEXT: {c['text']}")
        print(f"    LEAK: {leak_v} (event_date={leak_date}, conf={leak_conf})")
        print(f"    HALL: {hall_v} — {hall_reason}")
        print(f"    MERGED: {label}")
    print()
