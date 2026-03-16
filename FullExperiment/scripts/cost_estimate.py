"""Cost estimation for full training pipeline across all three tasks."""
import math

# ─── Dataset sizes ────────────────────────────────────────────────────────────
datasets = {
    "stock":  {"total": 900},
    "legal":  {"total": 292},
    "salary": {"total": 879},
}

# ─── Training parameters ─────────────────────────────────────────────────────
BATCH_SIZE = 8
GROUP_SIZE = 16
NUM_EPOCHS = 3
VAL_FRACTION = 0.05
NUM_VALIDATIONS = 10  # per task

# ─── Token estimates per instance (based on our e2e test, scaled to group_size=16) ─
# Our test: 5 completions → ~38 unique claims, 0% dedup (different instances)
# Real GRPO: 16 completions, same instance → estimate ~70 unique claims, ~50% dedup
CLAIMS_PER_INSTANCE = 70
UNVERIFIABLE_FRACTION = 0.25  # ~25% of claims get "unverifiable" from Claude
UNVERIFIABLE_PER_INSTANCE = int(CLAIMS_PER_INSTANCE * UNVERIFIABLE_FRACTION)

# Per-instance token estimates (from actual measurements, scaled)
tokens_per_instance = {
    "reasoning_extraction": {"input": 8000, "output": 500},
    "leakage_detection":    {"input": 4300, "output": 2660},
    "hallucination_claude": {"input": 4400, "output": 4375},
    "fallback_gemini":      {"input": 1700, "output": 1700},
}

# For validation: 1 completion → ~10 claims, no dedup needed
val_tokens_per_instance = {
    "reasoning_extraction": {"input": 800, "output": 100},
    "leakage_detection":    {"input": 700, "output": 400},
    "hallucination_claude": {"input": 700, "output": 600},
    "fallback_gemini":      {"input": 300, "output": 200},
}

# ─── Model pricing ($/M tokens) ──────────────────────────────────────────────
pricing = {
    "gemini-3-flash":     {"input": 0.50, "output": 3.00},
    "gemini-2.5-flash":   {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "claude-sonnet-4.6":  {"input": 3.00, "output": 15.00},
}

def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = pricing[model]
    return input_tokens * p["input"] / 1e6 + output_tokens * p["output"] / 1e6


# ─── Compute steps ────────────────────────────────────────────────────────────
print("=" * 80)
print("DATASET & STEP BREAKDOWN")
print("=" * 80)

total_train_steps = 0
total_val_instances = 0
for task, d in datasets.items():
    n_train = int(d["total"] * (1 - VAL_FRACTION))
    n_val = d["total"] - n_train
    steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
    total_steps = steps_per_epoch * NUM_EPOCHS
    val_evals = n_val * NUM_VALIDATIONS
    d["n_train"] = n_train
    d["n_val"] = n_val
    d["steps"] = total_steps
    d["val_evals"] = val_evals
    total_train_steps += total_steps
    total_val_instances += val_evals
    print(f"  {task:>8}: {n_train} train, {n_val} val → {total_steps} steps, {val_evals} val evaluations")

print(f"\n  TOTAL: {total_train_steps} training steps, {total_val_instances} validation evaluations")
print(f"  Per step: {BATCH_SIZE} instances × {GROUP_SIZE} completions = {BATCH_SIZE * GROUP_SIZE} completions")

# ─── Pipeline configurations ──────────────────────────────────────────────────
configs = {
    "A: Gemini 3 Flash + Claude 4.6 (current)": {
        "reasoning": "gemini-3-flash",
        "leakage": "gemini-3-flash",
        "hallucination": "claude-sonnet-4.6",
        "fallback": "gemini-3-flash",
    },
    "B: Gemini 3 Flash only (no Claude)": {
        "reasoning": "gemini-3-flash",
        "leakage": "gemini-3-flash",
        "hallucination": "gemini-3-flash",
        "fallback": None,
    },
    "C: Gemini 2.5 Flash + Claude 4.6": {
        "reasoning": "gemini-2.5-flash",
        "leakage": "gemini-3-flash",  # keep 3 Flash for leakage (best temporal reasoning)
        "hallucination": "claude-sonnet-4.6",
        "fallback": "gemini-3-flash",
    },
    "D: Gemini 3 Flash (leakage) + 2.5 Flash Lite (rest)": {
        "reasoning": "gemini-2.5-flash-lite",
        "leakage": "gemini-3-flash",
        "hallucination": "gemini-2.5-flash-lite",
        "fallback": None,
    },
}

print("\n" + "=" * 80)
print("COST ESTIMATES PER CONFIGURATION")
print("=" * 80)

for config_name, models in configs.items():
    print(f"\n{'─' * 80}")
    print(f"  {config_name}")
    print(f"{'─' * 80}")

    # Per-step cost
    step_cost = 0.0
    step_breakdown = {}

    for stage, tok in tokens_per_instance.items():
        if stage == "fallback_gemini" and models.get("fallback") is None:
            continue
        if stage == "hallucination_claude":
            model = models["hallucination"]
        elif stage == "fallback_gemini":
            model = models["fallback"]
        elif stage == "reasoning_extraction":
            model = models["reasoning"]
        else:
            model = models["leakage"]
        
        instance_cost = cost(model, tok["input"], tok["output"])
        batch_cost = instance_cost * BATCH_SIZE
        step_cost += batch_cost
        step_breakdown[stage] = {"model": model, "cost_per_step": batch_cost}

    # Per-validation-instance cost
    val_cost_per_instance = 0.0
    for stage, tok in val_tokens_per_instance.items():
        if stage == "fallback_gemini" and models.get("fallback") is None:
            continue
        if stage == "hallucination_claude":
            model = models["hallucination"]
        elif stage == "fallback_gemini":
            model = models["fallback"]
        elif stage == "reasoning_extraction":
            model = models["reasoning"]
        else:
            model = models["leakage"]
        val_cost_per_instance += cost(model, tok["input"], tok["output"])

    total_train_cost = step_cost * total_train_steps
    total_val_cost = val_cost_per_instance * total_val_instances
    total_cost = total_train_cost + total_val_cost

    print(f"\n  Per-step breakdown (×{BATCH_SIZE} instances):")
    for stage, info in step_breakdown.items():
        print(f"    {stage:25s}  {info['model']:25s}  ${info['cost_per_step']:.4f}")
    print(f"    {'TOTAL per step':25s}  {'':25s}  ${step_cost:.4f}")

    print(f"\n  Training:   {total_train_steps} steps × ${step_cost:.4f} = ${total_train_cost:.2f}")
    print(f"  Validation: {total_val_instances} evals × ${val_cost_per_instance:.5f} = ${total_val_cost:.2f}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  GRAND TOTAL: ${total_cost:.2f}")

    # Time estimate
    # From our test: ~6s per Gemini call, ~10s per Claude call
    # With concurrent_workers=4, 8 instances processed in 2 batches
    gemini_calls = sum(1 for s in ["reasoning", "leakage", "fallback"] if models.get(s if s != "reasoning" else "reasoning"))
    claude_calls = 1 if "claude" in models.get("hallucination", "") else 0
    
    if claude_calls:
        # Leakage (Gemini) and Hallucination (Claude) run in parallel within score_group
        per_instance_time = 5 + max(8, 12) + 6  # extraction + max(leakage, hall) + fallback
    else:
        per_instance_time = 5 + 8 + 8  # extraction + leakage + hallucination (sequential Gemini)
    
    # 4 workers, 8 instances per step → 2 rounds
    per_step_time = 2 * per_instance_time + 30  # + 30s Tinker generation
    total_hours = total_train_steps * per_step_time / 3600
    val_hours = total_val_instances * (per_instance_time / 2) / 3600  # val is lighter
    
    print(f"\n  Est. time: ~{per_step_time}s/step → training ~{total_hours:.1f}h, validation ~{val_hours:.1f}h")
    print(f"  Total wall time: ~{total_hours + val_hours:.1f}h")
