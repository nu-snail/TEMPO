"""Outcome-recall probe: measure base-model memorization of eval outcomes.

For each stock/salary evaluation instance, the *base* model is asked to
recall the realized outcome directly, with no temporal constraint. The
per-instance memorization score is later joined with saved per-instance
evaluation results to test whether method accuracy concentrates on
memorized instances (NeurIPS rebuttal, reviewer YbBe W1).

Usage (from QualityGatedExperiment/):
    uv run python scripts/rebuttal/recall_probe.py \
        --model_name Qwen/Qwen3.5-35B-A3B \
        --output_dir runtime/results/rebuttal/recall_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from quality_gated_experiment.config import load_env
from quality_gated_experiment.data import load_control_instances, load_eval_instances
from quality_gated_experiment.types import TaskInstance

logger = logging.getLogger("recall_probe")

MAX_TOKENS = 768
RENDERER_NAME = "qwen3_disable_thinking"

STOCK_PROBE_SYSTEM = (
    "You are a financial data assistant recalling historical market data. "
    "This is a factual recall question about what actually happened, not a "
    "prediction task. Answer from your knowledge of historical stock returns. "
    "If you do not remember exact figures, give your best recollection."
)

STOCK_PROBE_USER = """Rank the following five companies by their ACTUAL realized total stock return over the period {start_date} to {end_date}, from highest return to lowest.

Sector: {sector}
Companies:
{company_lines}

Output a single JSON object and nothing else, no markdown fences:
{{"ranking": ["BEST_TICKER", "...", "...", "...", "WORST_TICKER"], "confidence": "high|medium|low"}}"""

SALARY_PROBE_SYSTEM = (
    "You are a sports contracts data assistant recalling historical contract "
    "data. This is a factual recall question about what actually happened, "
    "not a prediction task. Answer from your knowledge of NBA contracts. "
    "If you do not remember the exact figure, give your best recollection."
)

SALARY_PROBE_USER = """In {year} NBA free agency, {player} ({position}) changed teams from {from_team} to {to_team}. What was the ACTUAL average annual value (AAV) of the contract he signed, in US dollars?

Output a single JSON object and nothing else, no markdown fences:
{{"aav_usd": <number>, "confidence": "high|medium|low"}}"""


def build_probe_messages(inst: TaskInstance) -> list[dict[str, str]]:
    """Build the recall-probe chat messages for one instance."""
    raw = inst.raw
    if inst.task == "stock":
        period = raw.get("past_period", {})
        names = raw.get("company_names", {})
        company_lines = "\n".join(
            f"- {t} ({names.get(t, t)})" for t in raw.get("companies", [])
        )
        user = STOCK_PROBE_USER.format(
            start_date=period.get("start_date", ""),
            end_date=period.get("end_date", ""),
            sector=raw.get("sector", ""),
            company_lines=company_lines,
        )
        return [
            {"role": "system", "content": STOCK_PROBE_SYSTEM},
            {"role": "user", "content": user},
        ]
    if inst.task == "salary":
        user = SALARY_PROBE_USER.format(
            year=raw.get("Year", ""),
            player=raw.get("Player", ""),
            position=raw.get("Position", ""),
            from_team=raw.get("From_Team", ""),
            to_team=raw.get("To_Team", ""),
        )
        return [
            {"role": "system", "content": SALARY_PROBE_SYSTEM},
            {"role": "user", "content": user},
        ]
    raise ValueError(f"Recall probe does not support task: {inst.task}")


def parse_json_object(text: str) -> dict | None:
    """Extract the first JSON object from a completion, tolerating fences."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def spearman_rho(pred: list[str], truth: list[str]) -> float | None:
    """Spearman rank correlation between two orderings of the same items."""
    if sorted(pred) != sorted(truth) or len(truth) < 2:
        return None
    n = len(truth)
    truth_rank = {item: i for i, item in enumerate(truth)}
    d2 = sum((i - truth_rank[item]) ** 2 for i, item in enumerate(pred))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def score_stock(parsed: dict | None, inst: TaskInstance) -> dict:
    """Score a stock recall: Spearman rho of recalled vs. actual ranking."""
    gt_ranking = inst.ground_truth.get("ranking", [])
    result = {"recall_valid": False, "recall_rho": None, "recalled_ranking": None}
    if parsed is None:
        return result
    ranking = parsed.get("ranking")
    if not isinstance(ranking, list):
        return result
    tickers = [str(t).strip().upper() for t in ranking]
    rho = spearman_rho(tickers, [t.upper() for t in gt_ranking])
    if rho is None:
        return result
    result.update(
        recall_valid=True,
        recall_rho=rho,
        recalled_ranking=tickers,
        confidence=str(parsed.get("confidence", "")),
    )
    return result


def score_salary(parsed: dict | None, inst: TaskInstance) -> dict:
    """Score a salary recall: relative error of recalled vs. actual AAV."""
    gt_aav = float(inst.ground_truth.get("aav", 0.0))
    result = {"recall_valid": False, "recall_rel_err": None, "recalled_aav": None}
    if parsed is None or gt_aav <= 0:
        return result
    aav = parsed.get("aav_usd")
    if not isinstance(aav, (int, float)) or aav <= 0:
        return result
    result.update(
        recall_valid=True,
        recall_rel_err=abs(float(aav) - gt_aav) / gt_aav,
        recalled_aav=float(aav),
        confidence=str(parsed.get("confidence", "")),
    )
    return result


def make_sampler(model_name: str, backend: str):
    """Create a greedy sampler (submit, collect) for the requested backend.

    The Tinker backend samples the base model directly; the OpenRouter
    backend is the fallback for models Tinker no longer serves (e.g.
    Qwen3.5-35B-A3B instruct).
    """
    if backend == "tinker":
        return _make_tinker_sampler(model_name)
    if backend == "openrouter":
        return _make_openrouter_sampler(model_name)
    raise ValueError(f"Unknown backend: {backend}")


def _make_tinker_sampler(model_name: str):
    import tinker
    from tinker import types as tinker_types
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    renderer = renderers.get_renderer(RENDERER_NAME, get_tokenizer(model_name))
    sampling_params = tinker_types.SamplingParams(
        max_tokens=MAX_TOKENS, temperature=0.0, stop=renderer.get_stop_sequences()
    )

    def submit(messages: list[dict[str, str]]):
        prompt = renderer.build_generation_prompt(messages)
        return sampling_client.sample(
            prompt=prompt, num_samples=1, sampling_params=sampling_params
        )

    def collect(future) -> str:
        parsed_msg, _ = renderer.parse_response(future.result().sequences[0].tokens)
        return renderers.get_text_content(parsed_msg)

    return submit, collect


def _make_openrouter_sampler(model_name: str):
    import os
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    pool = ThreadPoolExecutor(max_workers=16)

    def _call(messages: list[dict[str, str]]) -> str:
        last_error: Exception | None = None
        for _ in range(3):
            try:
                # Reasoning disabled to match the paper's inference setup
                # (qwen3_disable_thinking renderer on Tinker).
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                    extra_body={"reasoning": {"enabled": False}},
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # provider/network errors: retry
                last_error = exc
        raise RuntimeError(f"OpenRouter call failed after 3 attempts: {last_error}")

    def submit(messages: list[dict[str, str]]):
        return pool.submit(_call, messages)

    def collect(future) -> str:
        return future.result()

    return submit, collect


def run_probe(
    model_name: str,
    backend: str,
    split: str,
    tasks: list[str],
    data_root: Path,
    output_dir: Path,
    batch_size: int,
    limit: int | None,
) -> None:
    """Run the recall probe for one base model and write per-task JSONL."""
    loader = load_eval_instances if split == "eval" else load_control_instances
    instances = loader(data_root, tasks)
    if limit:
        by_task: dict[str, list[TaskInstance]] = {}
        for inst in instances:
            by_task.setdefault(inst.task, []).append(inst)
        instances = [i for task_insts in by_task.values() for i in task_insts[:limit]]

    submit, collect = make_sampler(model_name, backend)
    model_tag = model_name.split("/")[-1].lower().replace(".", "").replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if split == "eval" else f"_{split}"
    out_path = output_dir / f"{model_tag}{suffix}.jsonl"

    scorers = {"stock": score_stock, "salary": score_salary}
    n_done, n_valid = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, len(instances), batch_size):
            batch = instances[start : start + batch_size]
            futures = [(inst, submit(build_probe_messages(inst))) for inst in batch]
            for inst, future in futures:
                completion = collect(future)
                scores = scorers[inst.task](parse_json_object(completion), inst)
                row = {
                    "instance_id": inst.instance_id,
                    "task": inst.task,
                    "model": model_name,
                    "completion": completion,
                    **scores,
                }
                f.write(json.dumps(row) + "\n")
                n_done += 1
                n_valid += int(bool(scores.get("recall_valid")))
            logger.info("Probed %d / %d (valid: %d)", n_done, len(instances), n_valid)

    logger.info("Done. %d instances, %d valid recalls -> %s", n_done, n_valid, out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--backend", default="tinker", choices=["tinker", "openrouter"])
    parser.add_argument("--split", default="eval", choices=["eval", "control"])
    parser.add_argument("--tasks", default="stock,salary")
    parser.add_argument("--data_root", default="runtime/data")
    parser.add_argument("--output_dir", default="runtime/results/rebuttal/recall_probe")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Instances per task (smoke test)")
    args = parser.parse_args()

    load_env(Path(__file__).resolve().parents[2])
    run_probe(
        model_name=args.model_name,
        backend=args.backend,
        split=args.split,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
