"""Bootstrap CI for (TEMPO eval memorization gap) - (control-set null gap).

The null gap is Temporal Hint on the control set (Qwen3-30B), where
memorization is impossible, split by the 30B control probe. A difference
CI containing zero supports "TEMPO's gap is explained by familiarity".

Usage (from QualityGatedExperiment/):
    uv run python scripts/rebuttal/gap_vs_null_diff.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from analyze_recall_probe import is_memorized, load_instance_performance, load_probe_scores, mean

BOOTSTRAP_SAMPLES = 10_000
SEED = 42
RHO_THRESHOLD = 0.6

PROBE_DIR = Path("runtime/results/rebuttal/recall_probe")
RESULTS = Path("runtime/results")


def split_perf(probe: dict, perf: dict, task: str) -> tuple[list[float], list[float]]:
    mem, non = [], []
    for iid, row in probe.items():
        if row["task"] != task or iid not in perf:
            continue
        flag = is_memorized(row, task, RHO_THRESHOLD)
        if flag is None:
            continue
        (mem if flag else non).append(perf[iid])
    return mem, non


def gap(mem: list[float], non: list[float], rng: random.Random | None = None) -> float:
    if rng is not None:
        mem = [rng.choice(mem) for _ in mem]
        non = [rng.choice(non) for _ in non]
    return mean(mem) - mean(non)


def main() -> None:
    rng = random.Random(SEED)

    null_probe = load_probe_scores(PROBE_DIR / "qwen3_30b_a3b_control.jsonl")
    null_perf = load_instance_performance(RESULTS / "qwen3_30b" / "control_stock" / "instance_results.json")
    null_mem, null_non = split_perf(null_probe, null_perf, "stock")
    print(f"null: n_mem={len(null_mem)} n_non={len(null_non)} gap={gap(null_mem, null_non):.3f}")

    for model_dir, probe_tag in [("qwen3_30b", "qwen3_30b_a3b"), ("qwen35_35b", "qwen35_35b_a3b")]:
        probe = load_probe_scores(PROBE_DIR / f"{probe_tag}.jsonl")
        for method, res_dir in [("TEMPO", "eval_stock_best"), ("Hint", "eval_stock_base")]:
            perf = load_instance_performance(RESULTS / model_dir / res_dir / "instance_results.json")
            mem, non = split_perf(probe, perf, "stock")
            point = gap(mem, non) - gap(null_mem, null_non)
            diffs = sorted(
                gap(mem, non, rng) - gap(null_mem, null_non, rng)
                for _ in range(BOOTSTRAP_SAMPLES)
            )
            lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
            print(
                f"{method} {model_dir}: gap={gap(mem, non):.3f} "
                f"diff_vs_null={point:.3f} 95% CI [{lo:.3f}, {hi:.3f}]"
            )


if __name__ == "__main__":
    main()
