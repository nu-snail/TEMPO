"""Reconstruction gap: can stated evidence alone reproduce the prediction?

For each saved evaluation completion, a closed-world model receives ONLY
the task framing and the evidence list stated in that completion, and must
re-derive the prediction. If the original prediction relied on unstated
(possibly post-cutoff) internal knowledge, the rebuilt accuracy drops; a
small gap shows the audited evidence chain is sufficient to support the
prediction (NeurIPS rebuttal, reviewer YbBe W1/W2).

The closed-world evaluator is the corresponding base model (original
trained checkpoints are no longer stored on Tinker); a shared evaluator
across methods keeps the comparison fair.

Usage (from QualityGatedExperiment/):
    uv run python scripts/rebuttal/reconstruction_gap.py \
        --model_dir qwen35_35b --model_name qwen/qwen3.5-35b-a3b \
        --backend openrouter --task stock --variant best
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quality_gated_experiment.config import load_env
from quality_gated_experiment.data import load_eval_instances
from quality_gated_experiment.prompts import _build_salary_user, _build_stock_user
from quality_gated_experiment.reward_pipeline import _performance_score
from quality_gated_experiment.types import TaskInstance

from recall_probe import make_sampler, parse_json_object

logger = logging.getLogger("reconstruction_gap")

RECON_SYSTEM = (
    "You are an expert analyst operating in a strict closed-world setting. "
    "You are given a prediction task and a fixed list of evidence facts. "
    "Base your prediction EXCLUSIVELY on the provided evidence items. Do not "
    "use any background knowledge, memory of actual outcomes, or facts not "
    "present in the evidence list. If the evidence is thin, still give your "
    "best prediction supported only by the evidence."
)

OUTPUT_SPECS = {
    "stock": '{"ranking": ["BEST_TICKER", "...", "...", "...", "WORST_TICKER"]}',
    "salary": '{"predicted_salary": <integer USD>}',
}


def build_recon_messages(inst: TaskInstance, evidence: list[dict]) -> list[dict[str, str]]:
    """Build closed-world rebuild messages from an instance and stated evidence."""
    task_block = _build_stock_user(inst) if inst.task == "stock" else _build_salary_user(inst)
    user = (
        f"{task_block}\n\n"
        f"EVIDENCE (the only information you may use):\n"
        f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"Output a single JSON object and nothing else, no markdown fences:\n"
        f"{OUTPUT_SPECS[inst.task]}"
    )
    return [
        {"role": "system", "content": RECON_SYSTEM},
        {"role": "user", "content": user},
    ]


def extract_original(completion: str, task: str) -> tuple[list[dict], dict] | None:
    """Extract (evidence, parsed_output) from a saved completion, or None."""
    parsed = parse_json_object(completion)
    if parsed is None:
        return None
    evidence = parsed.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return None
    pred_key = "ranking" if task == "stock" else "predicted_salary"
    if pred_key not in parsed:
        return None
    return evidence, parsed


def run(
    model_dir: str,
    model_name: str,
    backend: str,
    task: str,
    variant: str,
    results_root: Path,
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    limit: int | None,
) -> None:
    """Rebuild predictions for one method run and write per-instance JSONL."""
    method_dir = results_root / model_dir / f"eval_{task}_{variant}"
    completions = json.loads((method_dir / "completions.json").read_text())
    instances = {i.instance_id: i for i in load_eval_instances(data_root, [task])}

    jobs = []
    n_skipped = 0
    for row in completions:
        inst = instances.get(row["instance_id"])
        extracted = extract_original(row["completion"], task) if inst else None
        if extracted is None:
            n_skipped += 1
            continue
        jobs.append((inst, *extracted))
    if limit:
        jobs = jobs[:limit]
    logger.info("Rebuilding %d completions (%d skipped as unparseable)", len(jobs), n_skipped)

    submit, collect = make_sampler(model_name, backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_dir}_{task}_{variant}.jsonl"

    n_done = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            futures = [
                (inst, orig, submit(build_recon_messages(inst, ev)))
                for inst, ev, orig in batch
            ]
            for inst, orig, future in futures:
                completion = collect(future)
                rebuilt = parse_json_object(completion)
                row = {
                    "instance_id": inst.instance_id,
                    "task": task,
                    "variant": variant,
                    "n_evidence": len(orig.get("evidence", [])),
                    "perf_original": _performance_score(task, orig, inst.ground_truth),
                    "perf_rebuilt": (
                        _performance_score(task, rebuilt, inst.ground_truth)
                        if rebuilt is not None
                        else None
                    ),
                    "rebuilt_output": rebuilt,
                }
                f.write(json.dumps(row) + "\n")
                n_done += 1
            logger.info("Rebuilt %d / %d", n_done, len(jobs))

    _summarize(out_path)


def _summarize(out_path: Path) -> None:
    rows = [json.loads(l) for l in out_path.read_text().splitlines()]
    scored = [r for r in rows if r["perf_rebuilt"] is not None]
    if not scored:
        logger.warning("No scored rebuilds in %s", out_path)
        return
    orig = sum(r["perf_original"] for r in scored) / len(scored)
    rebuilt = sum(r["perf_rebuilt"] for r in scored) / len(scored)
    logger.info(
        "%s: n=%d, perf_original=%.3f, perf_rebuilt=%.3f, gap=%.3f",
        out_path.name, len(scored), orig, rebuilt, orig - rebuilt,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, choices=["qwen3_30b", "qwen35_35b"])
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--backend", default="tinker", choices=["tinker", "openrouter"])
    parser.add_argument("--task", required=True, choices=["stock", "salary"])
    parser.add_argument("--variant", required=True, choices=["base", "best", "rag", "timespec"])
    parser.add_argument("--results_root", default="runtime/results")
    parser.add_argument("--data_root", default="runtime/data")
    parser.add_argument("--output_dir", default="runtime/results/rebuttal/reconstruction_gap")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_env(Path(__file__).resolve().parents[2])
    run(
        model_dir=args.model_dir,
        model_name=args.model_name,
        backend=args.backend,
        task=args.task,
        variant=args.variant,
        results_root=Path(args.results_root),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
