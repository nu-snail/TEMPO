"""Layer 3: Mini evaluation (~$3-5).

Runs the FULL 4-phase evaluation pipeline on 1 instance per task (3 total),
using the OpenRouter API (not Tinker).  This is the closest thing to the
real experiment without spending $100+.

Tests:
  - Prompt → LLM completion → claim extraction → Shapley → leakage detection → metrics
  - Output file structure (completions.json, summary.json, phase1-4 dirs)
  - Aggregate metrics computation

Usage:
    uv run python tests/test_layer3_mini_eval.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

PASS = 0
FAIL = 0
OUT_DIR = Path("results/_test_mini_eval")


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  — {detail}")


def main() -> None:
    global PASS, FAIL

    for key in ("OPENROUTER_API_KEY", "PERPLEXITY_API_KEY"):
        if not os.getenv(key):
            print(f"ERROR: {key} not set.")
            sys.exit(1)

    print("=" * 60)
    print("  Layer 3: Mini Evaluation (1 instance per task)")
    print("=" * 60)

    # Clean output
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    from src.evaluation.evaluate import EvalConfig, run_evaluation

    cfg = EvalConfig(
        model_name="google/gemini-2.5-flash",
        use_tinker=False,
        prompt_mode="temporal_hint",
        temperature=0.0,
        max_tokens=2048,
        detector_model="google/gemini-2.5-flash",
        shapley_model="google/gemini-2.5-flash",
        shapley_max_samples=10,
        shapley_batch_size=8,
        output_dir=str(OUT_DIR),
        tasks="stock,salary,legal",
        search_cache_path=str(OUT_DIR / "search_cache.json"),
    )

    # Monkey-patch load_all_eval to return only 1 instance per task
    import src.evaluation.evaluate as eval_mod
    original_load = eval_mod.load_all_eval

    def load_mini(tasks=None):
        all_inst = original_load(tasks)
        by_task: dict[str, list] = {}
        for inst in all_inst:
            by_task.setdefault(inst.task, []).append(inst)
        mini = []
        for task, insts in by_task.items():
            mini.append(insts[0])
        print(f"\n  [mini] Using {len(mini)} instances: "
              f"{[f'{i.task}:{i.instance_id}' for i in mini]}\n")
        return mini

    eval_mod.load_all_eval = load_mini

    try:
        t0 = time.time()
        run_evaluation(cfg)
        dt = time.time() - t0
        print(f"\n  Evaluation completed in {dt:.1f}s")
    finally:
        eval_mod.load_all_eval = original_load

    # ── Verify output structure ──
    print("\n── Output Verification ──")

    check("config.json exists", (OUT_DIR / "config.json").exists())
    check("completions.json exists", (OUT_DIR / "completions.json").exists())
    check("instance_results.json exists", (OUT_DIR / "instance_results.json").exists())
    check("summary.json exists", (OUT_DIR / "summary.json").exists())

    for phase in ("phase1_claims", "phase2_shapley", "phase3_leakage", "phase4_metrics"):
        phase_dir = OUT_DIR / phase
        n_files = len(list(phase_dir.glob("*.json"))) if phase_dir.exists() else 0
        check(f"{phase}/ has files", n_files > 0, f"found {n_files}")

    # ── Verify summary content ──
    print("\n── Summary Verification ──")

    summary = json.loads((OUT_DIR / "summary.json").read_text())
    check("summary has 'config' key", "config" in summary)
    check("summary has 'overall' key", "overall" in summary)

    for task in ("stock", "salary", "legal"):
        if task in summary:
            t = summary[task]
            olr = t.get("olr", {})
            dclr = t.get("shapley_dclr", {})
            perf = t.get("performance", {})
            print(f"\n  {task}:")
            print(f"    OLR:  {olr.get('mean', '?'):.4f} (±{olr.get('std', 0):.4f})")
            print(f"    DCLR: {dclr.get('mean', '?'):.4f} (±{dclr.get('std', 0):.4f})")
            print(f"    Perf: {perf.get('mean', '?'):.4f}")
            check(f"{task}: OLR in [0,1]",
                  0 <= olr.get("mean", -1) <= 1)
            check(f"{task}: DCLR in [0,1]",
                  0 <= dclr.get("mean", -1) <= 1)

    overall = summary.get("overall", {})
    check("overall OLR computed", "olr" in overall)

    # ── Verify completions ──
    completions = json.loads((OUT_DIR / "completions.json").read_text())
    check("completions logged", len(completions) >= 1, f"got {len(completions)}")
    for c in completions:
        check(f"completion for {c['task']}:{c['instance_id'][:20]} is non-empty",
              len(c.get("completion", "")) > 50)

    # ── Verify instance results ──
    results = json.loads((OUT_DIR / "instance_results.json").read_text())
    for r in results:
        check(f"instance {r['id'][:20]} has metrics",
              r.get("olr") is not None and r.get("r_performance") is not None)

    print("\n" + "=" * 60)
    if FAIL == 0:
        print(f"  ALL {PASS} CHECKS PASSED")
    else:
        print(f"  {PASS} passed, {FAIL} FAILED")
    print(f"\n  Output preserved at: {OUT_DIR}/")
    print(f"  Inspect summary:     cat {OUT_DIR}/summary.json")
    print("=" * 60)

    # Clean up
    # shutil.rmtree(OUT_DIR)  # uncomment to auto-clean
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
