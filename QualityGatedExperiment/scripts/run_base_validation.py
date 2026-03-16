"""Evaluate the base (untrained) model on the training validation split.

Produces step-0 baseline data for the training dynamics figure.
Reproduces the exact validation split used during training (seed=42, val_fraction=0.10)
and evaluates the base model on those instances using the same scoring pipeline.

Usage:
    cd QualityGatedExperiment && uv run python scripts/run_base_validation.py \
        --model qwen3_30b --task stock

Outputs:
    runtime/results/{model}/base_val_{task}/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quality_gated_experiment.config import load_config, load_env, require_api_keys
from quality_gated_experiment.data import load_train_instances, split_train_val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["qwen3_30b", "qwen35_35b"])
    parser.add_argument("--task", required=True, choices=["stock", "salary", "legal"])
    parser.add_argument("--config", default=None, help="Training config YAML (auto-detected if omitted)")
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_name = "train_qwen3_30b.yaml" if "30b" in args.model else "train_qwen35_35b.yaml"
        config_path = str(PROJECT_ROOT / "configs" / config_name)

    cfg = load_config(config_path)
    load_env(PROJECT_ROOT)
    _, tinker_key = require_api_keys()
    os.environ["TINKER_API_KEY"] = tinker_key

    from quality_gated_experiment.llm import OpenRouterClient
    from quality_gated_experiment.prompts import build_generation_prompt
    from quality_gated_experiment.reward_pipeline import RewardPipeline

    import tinker
    from tinker import types as tinker_types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    data_root = PROJECT_ROOT / cfg.data_root
    tasks = [args.task]
    all_train = load_train_instances(data_root, tasks)
    _, val_set = split_train_val(all_train, cfg.val_fraction, cfg.data_seed)
    logger.info("Validation set: %d instances for task=%s", len(val_set), args.task)

    router_key = os.getenv("OPENROUTER_API_KEY", "")
    judge_client = OpenRouterClient(router_key)
    reward_pipe = RewardPipeline(cfg, judge_client)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)

    tokenizer = get_tokenizer(cfg.model_name)
    if cfg.renderer_name:
        renderer_name = cfg.renderer_name
    else:
        renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    stop_sequences = renderer.get_stop_sequences()
    greedy_params = tinker_types.SamplingParams(
        max_tokens=cfg.max_tokens, temperature=0.0, stop=stop_sequences,
    )

    # Submit all greedy samples in parallel via futures
    pending = []
    for inst in val_set:
        system, user = build_generation_prompt(inst)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        model_input = renderer.build_generation_prompt(messages)
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=greedy_params,
        )
        pending.append((inst, future))

    logger.info("Submitted %d inference requests, collecting results...", len(pending))

    leaks, covs, perfs = [], [], []
    n_parse_failed = 0

    for i, (inst, future) in enumerate(pending):
        result = future.result()
        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        text = renderers.get_text_content(parsed_msg)
        rewards, _ = reward_pipe.score_group([text], inst, skip_hallucination=True)
        r = rewards[0]
        if r.parse_success:
            leaks.append(r.leakage_rate)
            covs.append(r.coverage)
            perfs.append(r.r_performance)
        else:
            n_parse_failed += 1
        if (i + 1) % 5 == 0 or (i + 1) == len(pending):
            logger.info("Scored %d/%d (failed=%d)", i + 1, len(pending), n_parse_failed)

    import numpy as np
    summary = {
        args.task: {
            "leakage_rate": {"mean": float(np.mean(leaks)), "std": float(np.std(leaks))},
            "r_performance": {"mean": float(np.mean(perfs)), "std": float(np.std(perfs))},
            "coverage": {"mean": float(np.mean(covs)), "std": float(np.std(covs))},
            "n_instances": len(leaks),
            "n_parse_failed": n_parse_failed,
        }
    }

    out_dir = PROJECT_ROOT / "runtime" / "results" / args.model / f"base_val_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved to %s", out_dir / "summary.json")
    logger.info("OLR=%.1f%% Perf=%.3f Cov=%.1f%% (n=%d, failed=%d)",
                summary[args.task]["leakage_rate"]["mean"] * 100,
                summary[args.task]["r_performance"]["mean"],
                summary[args.task]["coverage"]["mean"] * 100,
                len(leaks), n_parse_failed)


if __name__ == "__main__":
    main()
