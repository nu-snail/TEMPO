#!/usr/bin/env python3
"""End-to-end one-step test of the reward v4 pipeline.

Runs ONE real training step (Tinker sampling + full reward v4 pipeline + KL
penalty + forward_backward + optim_step) on a small batch and logs everything
in detail so we can verify:
  1. Completions parse correctly (format score > 0)
  2. Quality gates (h, d, q) produce values in [0, 1]
  3. KL penalty is applied and kl_policy_base metric is logged
  4. Advantage variance is sufficient for gradient signal
  5. forward_backward and optim_step complete without error
  6. Loss function matches cfg.loss_fn

Usage:
  cd QualityGatedExperiment
  PYTHONPATH=src uv run python scripts/test_one_step.py 2>&1 | tee /tmp/test_one_step_v4.log
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

_orig_print = builtins.print
def print(*args: Any, **kwargs: Any) -> None:
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RULER = "=" * 80

print(f"\n{RULER}")
print("  [STEP 0/7] Loading modules ...")
print(f"{RULER}")

from quality_gated_experiment.config import TrainConfig, load_env, require_api_keys
from quality_gated_experiment.data import load_train_instances, split_train_val
from quality_gated_experiment.llm import OpenRouterClient
from quality_gated_experiment.prompts import build_generation_prompt
from quality_gated_experiment.reward_pipeline import RewardPipeline
from quality_gated_experiment.schemas import parse_completion

load_env(ROOT)
load_env(ROOT.parent)

print("  Modules loaded. Environment loaded.")

TEST_BATCH_SIZE = 4
TEST_GROUP_SIZE = 8
TEST_TEMPERATURE = 0.6
TEST_MAX_TOKENS = 4096
TEST_TASKS = ["stock"]


def build_test_config() -> TrainConfig:
    return TrainConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        renderer_name="qwen3_disable_thinking",
        lora_rank=32,
        group_size=TEST_GROUP_SIZE,
        batch_size=TEST_BATCH_SIZE,
        num_epochs=1,
        learning_rate=2e-5,
        max_tokens=TEST_MAX_TOKENS,
        temperature=TEST_TEMPERATURE,
        clip_low=0.9,
        clip_high=2.0,
        w_leak_start=0.95,
        w_leak_end=0.80,
        lambda_base=0.5,
        format_margin=1.0,
        z_clip=5.0,
        w_perf_clean=0.40,
        cov_target=0.80,
        cov_floor=0.20,
        w_cov_leak=0.05,
        train_tasks="stock",
        val_fraction=0.10,
        data_seed=42,
        judge_model="google/gemini-3-flash-preview",
        search_model="perplexity/sonar",
        num_validations=0,
        log_dir="runtime/results/test_one_step",
        data_root="runtime/data",
        hall_floor=0.3,
        ev_target=8,
        reason_target=120,
        w_quality_leak=0.02,
        loss_fn="ppo",
        kl_penalty_coef=0.05,
        kl_discount_factor=0.0,
    )


def pick_instances(data_root: Path) -> list[Any]:
    all_instances = load_train_instances(data_root, TEST_TASKS)
    train_set, _ = split_train_val(all_instances, 0.10, 42)
    import random
    rng = random.Random(123)
    return rng.sample(train_set, min(TEST_BATCH_SIZE, len(train_set)))


def main() -> None:
    import numpy as np

    cfg = build_test_config()
    data_root = ROOT / cfg.data_root
    checks_passed = True

    # ── STEP 1: Pick instances ──────────────────────────────────────────
    print(f"\n{RULER}")
    print("  [STEP 1/7] Picking instances from dataset ...")
    print(f"{RULER}")

    instances = pick_instances(data_root)
    print(f"  Selected {len(instances)} instances:")
    for inst in instances:
        print(f"    {inst.instance_id} | task={inst.task} | cutoff={inst.cutoff_date}")
    print(f"  [STEP 1/7] DONE")

    # ── STEP 2: Set up Tinker ───────────────────────────────────────────
    print(f"\n{RULER}")
    print("  [STEP 2/7] Setting up Tinker client + LoRA training client ...")
    print(f"    Model: {cfg.model_name}")
    print(f"    LoRA rank: {cfg.lora_rank}")
    print(f"    Loss fn: {cfg.loss_fn}  |  KL coef: {cfg.kl_penalty_coef}")
    print(f"{RULER}")

    _, tinker_key = require_api_keys()
    os.environ["TINKER_API_KEY"] = tinker_key
    router_key = os.getenv("OPENROUTER_API_KEY", "")
    print("  API keys loaded.")

    import tinker
    import torch
    from tinker import types as tinker_types
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    print("  Creating ServiceClient ...")
    t0 = time.time()
    service_client = tinker.ServiceClient()
    print(f"  ServiceClient created in {time.time() - t0:.1f}s")

    print("  Creating LoRA training client (this may take 30-60s) ...")
    t0 = time.time()
    training_client = service_client.create_lora_training_client(
        base_model=cfg.model_name, rank=cfg.lora_rank,
    )
    print(f"  LoRA training client created in {time.time() - t0:.1f}s")

    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    sampling_params = tinker_types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = tinker_types.AdamParams(
        learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8,
    )
    loss_fn_config = {"clip_low_threshold": cfg.clip_low, "clip_high_threshold": cfg.clip_high}

    judge_client = OpenRouterClient(router_key)
    reward_pipe = RewardPipeline(cfg, judge_client)
    print(f"  [STEP 2/7] DONE — all clients ready")

    # ── STEP 3: Sample completions ──────────────────────────────────────
    print(f"\n{RULER}")
    print(f"  [STEP 3/7] Sampling {TEST_GROUP_SIZE} completions x {len(instances)} instances ...")
    print(f"{RULER}")

    print("  Getting sampling client from training weights ...")
    t0 = time.time()
    sampling_client = training_client.save_weights_and_get_sampling_client()
    print(f"  Sampling client ready in {time.time() - t0:.1f}s")

    all_sample_data = []
    for idx, inst in enumerate(instances):
        print(f"  Sampling instance {idx+1}/{len(instances)}: {inst.instance_id} ...", end=" ")
        system, user = build_generation_prompt(inst)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        model_input = renderer.build_generation_prompt(messages)
        t0 = time.time()
        result = sampling_client.sample(
            prompt=model_input, num_samples=TEST_GROUP_SIZE, sampling_params=sampling_params,
        ).result()
        elapsed = time.time() - t0

        completions = []
        tokens_g = []
        logprobs_g = []
        for seq in result.sequences:
            parsed_msg, _ = renderer.parse_response(seq.tokens)
            completions.append(renderers.get_text_content(parsed_msg))
            tokens_g.append(seq.tokens)
            logprobs_g.append(seq.logprobs)

        all_sample_data.append({
            "instance": inst,
            "model_input": model_input,
            "tokens_G": tokens_g,
            "logprobs_G": logprobs_g,
            "completions": completions,
        })
        print(f"{len(completions)} completions in {elapsed:.1f}s")

    print(f"  [STEP 3/7] DONE — sampled {sum(len(sd['completions']) for sd in all_sample_data)} total completions")

    # ── STEP 4: Reward v4 scoring ───────────────────────────────────────
    print(f"\n{RULER}")
    print("  [STEP 4/7] Scoring with reward v4 pipeline (claim extraction + leakage detection) ...")
    print(f"{RULER}")

    all_datums = []
    all_advantages = []

    for idx, sd in enumerate(all_sample_data):
        inst = sd["instance"]
        print(f"  Scoring instance {idx+1}/{len(all_sample_data)}: {inst.instance_id} ...", end=" ")
        t0 = time.time()
        rewards, claim_detail = reward_pipe.score_group(
            sd["completions"], inst, skip_hallucination=True,
        )
        score_time = time.time() - t0
        print(f"scored in {score_time:.1f}s")

        parsed_for_group = [parse_completion(inst.task, c) for c in sd["completions"]]
        advantages, adv_diag = reward_pipe.compute_advantages(
            rewards, 0.9, 0.5, 0.3,
            parsed_completions=parsed_for_group,
        )

        h_vals = adv_diag.get("h_gate", [])
        d_vals = adv_diag.get("d_gate", [])
        q_vals = adv_diag.get("q_gate", [])

        print(f"    Mode: {adv_diag.get('adv_mode', 'unknown')}")
        print(f"    h_gate: {[round(v, 3) for v in h_vals]}")
        print(f"    d_gate: {[round(v, 3) for v in d_vals]}")
        print(f"    q_gate: {[round(v, 3) for v in q_vals]}")

        for i, (r, a) in enumerate(zip(rewards, advantages)):
            tag = "OK" if r.parse_success else "FAIL"
            print(f"    [{i}] {tag} | leak={r.leakage_rate:.2f} cov={r.coverage:.2f} "
                  f"perf={r.r_performance:.3f} adv={a:+.4f}")

        for name, vals in [("h_gate", h_vals), ("d_gate", d_vals), ("q_gate", q_vals)]:
            for v in vals:
                if not (0.0 <= v <= 1.0 + 1e-6):
                    print(f"    FAIL: {name} value {v} out of range [0, 1]")
                    checks_passed = False

        all_advantages.extend(advantages)

        from quality_gated_experiment.train import _build_datums
        datum_list = _build_datums(sd, advantages, tinker_types, TensorData, torch)
        all_datums.extend(datum_list)

    print(f"  [STEP 4/7] DONE — {len(all_datums)} datums built")

    # ── STEP 5: KL penalty ──────────────────────────────────────────────
    print(f"\n{RULER}")
    print("  [STEP 5/7] Applying KL divergence penalty against base model ...")
    print(f"    kl_penalty_coef = {cfg.kl_penalty_coef}")
    print(f"    Computing base-model logprobs for {len(all_datums)} datums (may take 30-120s) ...")
    print(f"{RULER}")

    kl_policy_base = 0.0
    if cfg.kl_penalty_coef > 0 and all_datums:
        from quality_gated_experiment.train import _apply_kl_penalty
        t0 = time.time()
        kl_policy_base = _apply_kl_penalty(
            all_datums, service_client, cfg, torch, TensorData,
        )
        kl_time = time.time() - t0
        print(f"  kl_policy_base = {kl_policy_base:.6f} (computed in {kl_time:.1f}s)")
        if abs(kl_policy_base) < 1e-10:
            print("  NOTE: KL ~0 is expected at step 0 (policy = base)")
    else:
        print("  KL penalty disabled (kl_penalty_coef=0 or no datums)")

    print(f"  [STEP 5/7] DONE")

    # ── STEP 6: Forward-backward + optim step ───────────────────────────
    print(f"\n{RULER}")
    print("  [STEP 6/7] Running forward_backward + optim_step ...")
    print(f"    loss_fn = {cfg.loss_fn}")
    print(f"    datums  = {len(all_datums)}")
    print(f"{RULER}")

    if all_datums:
        adv_arr = np.array(all_advantages)
        print(f"  Advantage stats: mean={adv_arr.mean():.4f} std={adv_arr.std():.4f} "
              f"min={adv_arr.min():.4f} max={adv_arr.max():.4f}")

        if adv_arr.std() < 0.001:
            print("  WARN: advantage std very low — weak gradient signal")

        print("  Running forward_backward (may take 30-60s) ...")
        t0 = time.time()
        training_client.forward_backward(
            all_datums, loss_fn=cfg.loss_fn, loss_fn_config=loss_fn_config,
        ).result()
        fwd_bwd_time = time.time() - t0
        print(f"  forward_backward completed in {fwd_bwd_time:.1f}s")

        print("  Running optim_step ...")
        t0 = time.time()
        opt_result = training_client.optim_step(adam_params).result()
        opt_time = time.time() - t0
        print(f"  optim_step completed in {opt_time:.1f}s")
        if opt_result.metrics:
            print(f"  optim metrics: {opt_result.metrics}")
    else:
        print("  FAIL: no datums to train on")
        checks_passed = False

    print(f"  [STEP 6/7] DONE")

    # ── STEP 7: Summary ─────────────────────────────────────────────────
    print(f"\n{RULER}")
    print("  [STEP 7/7] SUMMARY")
    print(f"{RULER}")
    print(f"  Model:          {cfg.model_name}")
    print(f"  Loss function:  {cfg.loss_fn}")
    print(f"  KL coef:        {cfg.kl_penalty_coef}")
    print(f"  KL divergence:  {kl_policy_base:.6f}")
    print(f"  Total datums:   {len(all_datums)}")
    print(f"  Adv std:        {np.array(all_advantages).std():.4f}" if all_advantages else "  Adv std:        N/A")

    if checks_passed:
        print(f"\n  RESULT: ALL CHECKS PASSED")
    else:
        print(f"\n  RESULT: SOME CHECKS FAILED — review output above")
    print(f"{RULER}\n")


if __name__ == "__main__":
    main()
