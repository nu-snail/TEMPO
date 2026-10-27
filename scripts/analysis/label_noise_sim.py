"""Label-noise sensitivity: do the paper's leakage conclusions survive verifier errors?

Paper Appendix H.1 (sensitivity to verifier errors). Flip each claim's leakage label
independently with probability p (symmetric noise), recompute the paper's
macro OLR (per-instance mean), and check survival of the claims the paper
actually makes on stock:

  (1) TEMPO < Temporal Hint  (training reduces leakage vs. the untrained base)
  (2) TEMPO < RAG            (and vs. retrieval)

The paper claims leakage *parity* between TEMPO and TimeSPEC, not superiority,
so TEMPO-vs-TimeSPEC ordering is intentionally not a survival criterion.

Stock only: the local outputs reproduce Table 1 exactly (the salary TEMPO run
behind Table 1 lives on another machine).

Usage (from the repository root):
    uv run python scripts/analysis/label_noise_sim.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

RESULTS = Path("results")
MODEL_DIR = "qwen35_35b"
TASK = "stock"
METHODS = {"base": "Temporal Hint", "rag": "RAG", "timespec": "TimeSPEC", "tempo": "TEMPO"}

NOISE_LEVELS = [0.05, 0.10, 0.20]
REPS = 1000
SEED = 42


def load_instances() -> dict[str, list[list[bool]]]:
    """Per method: list of instances, each a list of per-claim leak labels."""
    out: dict[str, list[list[bool]]] = {}
    for suffix, name in METHODS.items():
        log = RESULTS / MODEL_DIR / f"eval_{TASK}_{suffix}" / "eval_step_log.jsonl"
        instances = {}
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            labels = [c.get("leak_verdict") == "leaked"
                      for c in row.get("claims", []) if c.get("source") == "evidence"]
            instances[row["instance_id"]] = labels  # last run wins
        out[name] = list(instances.values())
    return out


def macro_olr(instances: list[list[bool]], p: float, rng: random.Random) -> float:
    rates = []
    for labels in instances:
        if not labels:
            rates.append(0.0)
            continue
        flipped = sum((not l) if rng.random() < p else l for l in labels)
        rates.append(flipped / len(labels))
    return sum(rates) / len(rates)


def main() -> None:
    rng = random.Random(SEED)
    data = load_instances()

    print("clean macro OLR:")
    zero = random.Random(0)
    for name, insts in data.items():
        print(f"  {name:14s} {100 * macro_olr(insts, 0.0, zero):6.2f}%  (n_inst={len(insts)})")

    summary = {}
    for p in NOISE_LEVELS:
        wins_hint = wins_rag = 0
        for _ in range(REPS):
            olr = {name: macro_olr(insts, p, rng) for name, insts in data.items()}
            wins_hint += olr["TEMPO"] < olr["Temporal Hint"]
            wins_rag += olr["TEMPO"] < olr["RAG"]
        summary[p] = {"tempo_below_hint": wins_hint / REPS, "tempo_below_rag": wins_rag / REPS}
        print(f"noise {p:.0%}: TEMPO<Hint {wins_hint / REPS:.1%}, TEMPO<RAG {wins_rag / REPS:.1%}")

    out = RESULTS / "analysis" / "label_noise_sim.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
