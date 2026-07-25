"""Reward pipeline v4: Quality-gated performance mode.

Training pipeline (2-stage, hallucination skipped):
  1. Claim extraction: evidence facts
  2. Leakage detection: temporal availability check per unique claim

Validation/evaluation pipeline (3-stage, full):
  1-2. Same as training
  3. Hallucination detection: factual accuracy check per unique claim

Advantage (two-mode, quality-gated, v4 — see reward_design_v4.md):

  Quality gates (applied in performance mode and as bonus in leakage mode):
    h(rate)  = max(1 - hall_rate, hall_floor)       hallucination gate
    d(count) = min(evidence_count / ev_target, 1)   evidence diversity gate
    q(words) = min(reasoning_words / reason_target, 1)  reasoning quality gate

  Coverage gate (unchanged from v3):
    g(cov)   = cov_floor + (1 - cov_floor) * min(1, cov / cov_target)

  Mode selection (per group):
    all_clean := min(r_leak) >= 0.99

  PERFORMANCE MODE (all_clean = True):
    r_eff      = r_performance * g(cov) * h(hall) * d(ev) * q(reason)
    z_eff      = zscore(r_eff) within group
    perf_base  = lambda_base * (r_eff_i - batch_mean_r_eff_clean)
    adv_i      = z_eff_i + perf_base_i

  LEAKAGE MODE (all_clean = False):
    r_leak_i   = exp(-0.5 * num_leaked_i)
    z_leak     = zscore(r_leak) within group
    leak_base  = lambda_base * (r_leak_i - batch_mean_r_leak)
    base_adv_i = z_leak_i + leak_base_i

    Coverage bonus (clean completions only):
      cov_bonus_i = w_cov_leak * g(cov_i) * is_clean_i

    Quality bonus (clean completions only):
      quality_bonus_i = w_quality_leak * h(hall_i) * d(ev_i) * is_clean_i

    adv_i = base_adv_i + cov_bonus_i + quality_bonus_i

Format failure penalty (ratio-capped, unchanged from v3).
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .config import TrainConfig
from .llm import OpenRouterClient
from .prompts import (
    DATE_PLAUSIBILITY_PROMPT,
    build_hall_batch_prompt,
    build_hall_sonar_prompt,
)
from .schemas import compute_format_score, parse_completion
from .types import CompletionReward, TaskInstance

logger = logging.getLogger(__name__)

_JUDGE_CHUNK_SIZE = 80
_MAX_TOKENS_JUDGE = 8192


class RewardPipeline:
    """Compute format, validity, and performance rewards."""

    def __init__(self, cfg: TrainConfig, client: OpenRouterClient) -> None:
        self.cfg = cfg
        self.client = client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_group(
        self,
        completions: list[str],
        instance: TaskInstance,
        *,
        skip_hallucination: bool = False,
    ) -> tuple[list[CompletionReward], dict[str, Any]]:
        """Score one GRPO group.

        When *skip_hallucination* is True (training mode), only leakage
        detection runs, saving ~33% of Gemini cost.  Validation and
        evaluation call with skip_hallucination=False (the default).

        Returns:
            Tuple of (rewards, detail) where detail contains claim-level
            information for structured logging.
        """

        parsed = [parse_completion(instance.task, c) for c in completions]
        r_format = [compute_format_score(instance.task, p) for p in parsed]

        parseable_idx = [i for i, p in enumerate(parsed) if p is not None]
        if not parseable_idx:
            return [self._parse_fail_reward() for _ in completions], {"claims": []}

        parseable = [parsed[i] for i in parseable_idx if parsed[i] is not None]

        claim_pool, claim_to_completions = self._build_claim_pool(parseable)

        if not claim_pool:
            return self._rewards_no_claims(
                completions, parseable, parseable_idx, r_format, instance,
            ), {"claims": []}

        if skip_hallucination:
            leak_results = self._detect_leakage(claim_pool, instance.cutoff_date)
            hall_results: dict[int, dict[str, Any]] = {}
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_leak = pool.submit(
                    self._detect_leakage, claim_pool, instance.cutoff_date,
                )
                future_hall = pool.submit(self._detect_hallucination, claim_pool, instance.task)
                leak_results = future_leak.result()
                hall_results = future_hall.result()

        detail = _build_claim_detail(
            claim_pool, claim_to_completions, leak_results, hall_results,
        )

        # --- Per-completion reward assembly ---
        rewards: list[CompletionReward] = [
            self._parse_fail_reward() for _ in completions
        ]
        for local_idx, global_idx in enumerate(parseable_idx):
            parsed_item = parseable[local_idx]
            my_indices = [
                c["index"]
                for c in claim_pool
                if local_idx in claim_to_completions[c["key"]]
            ]
            leak_rate, hall_rate, r_valid, n_leaked, n_unsupported, n_undecidable = (
                _compute_rates(my_indices, leak_results, hall_results)
            )
            total = len(my_indices)

            coverage = _coverage(
                parsed_item.get("reasoning", ""),
                parsed_item.get("evidence", []),
            )
            r_perf = _performance_score(
                instance.task, parsed_item, instance.ground_truth,
            )

            rewards[global_idx] = CompletionReward(
                r_format=r_format[global_idx],
                r_valid=float(r_valid),
                r_performance=float(np.clip(r_perf, 0.0, 1.0)),
                leakage_rate=leak_rate,
                hallucination_rate=hall_rate,
                coverage=coverage,
                parse_success=True,
                num_claims=total,
                num_leaked=n_leaked,
                num_unsupported=n_unsupported,
                num_undecidable=n_undecidable,
            )

        return rewards, detail

    def compute_advantages(
        self,
        rewards: list[CompletionReward],
        w_leak: float,
        batch_mean_r_leak: float,
        batch_mean_r_eff_clean: float,
        *,
        parsed_completions: list[dict[str, Any] | None] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        """Two-mode quality-gated advantage computation (v4).

        Mode selection: all_clean := min(r_leak) >= 0.99
        Performance Mode: r_eff = r_perf * g(cov) * h(hall) * d(ev) * q(reason)
        Leakage Mode: base + cov_bonus + quality_bonus

        Args:
            parsed_completions: Optional list aligned with rewards; each entry is
                the parsed dict (or None for parse failures). Used to compute
                evidence count and reasoning word count for quality gates.
                If None, quality gates d() and q() default to 1.0.

        Returns (advantages, diagnostics).
        """

        parseable = [r for r in rewards if r.parse_success]

        if len(parseable) == 0:
            return (
                [0.0 for _ in rewards],
                {"r_leak_std": 0.0, "r_perf_std": 0.0, "skipped_all_fail": True},
            )

        if len(parseable) == 1:
            floor = -(self.cfg.z_clip * w_leak + self.cfg.format_margin)
            advs = [floor if not r.parse_success else 0.0 for r in rewards]
            return advs, {"r_leak_std": 0.0, "r_perf_std": 0.0, "format_penalty": floor}

        # Extract per-parseable quality signals
        hall_rates, ev_counts, reason_words = _extract_quality_signals(
            rewards, parsed_completions,
        )
        h_gate = _hallucination_gate(hall_rates, self.cfg.hall_floor)
        if self.cfg.gate_shape == "trapezoidal":
            d_gate = _evidence_gate_trapezoidal(
                ev_counts, self.cfg.ev_target, self.cfg.ev_cap,
                self.cfg.ev_decay, self.cfg.ev_floor,
            )
            q_gate = _reasoning_gate_trapezoidal(
                reason_words, self.cfg.reason_target, self.cfg.reason_cap,
                self.cfg.reason_decay, self.cfg.reason_floor,
            )
        else:
            d_gate = _evidence_gate(ev_counts, self.cfg.ev_target)
            q_gate = _reasoning_gate(reason_words, self.cfg.reason_target)

        r_leak = np.array(
            [np.exp(-0.5 * r.num_leaked) for r in parseable], dtype=np.float64,
        )

        # Reward-design ablations (NeurIPS rebuttal): override the group-level
        # mode gate. two_mode keeps the paper's design; performance_only /
        # leakage_only force one branch; scalar_mix collapses both signals
        # into a single scalar reward with no mode separation.
        variant = self.cfg.reward_variant
        if variant == "scalar_mix":
            return self._scalar_mix_advantages(
                rewards, parseable, r_leak, h_gate, d_gate, q_gate,
                batch_mean_r_leak, batch_mean_r_eff_clean,
            )
        if variant == "performance_only":
            all_clean = True
        elif variant == "leakage_only":
            all_clean = False
        else:
            all_clean = bool(r_leak.min() >= 0.99)

        if all_clean:
            # -- PERFORMANCE MODE (quality-gated) --
            r_perf = np.array(
                [r.r_performance for r in parseable], dtype=np.float64,
            )
            cov = np.array([r.coverage for r in parseable], dtype=np.float64)
            cov_gate = np.minimum(1.0, cov / self.cfg.cov_target)
            g_cov = self.cfg.cov_floor + (1 - self.cfg.cov_floor) * cov_gate

            r_eff = r_perf * g_cov * h_gate * d_gate * q_gate

            z_eff = _zscore(r_eff, self.cfg.z_clip)
            perf_baseline = self.cfg.lambda_base * (r_eff - batch_mean_r_eff_clean)
            adv_raw = z_eff + perf_baseline

            baseline_values = perf_baseline
            diag_extra: dict[str, Any] = {
                "r_leak_std": 0.0,
                "r_perf_std": float(r_perf.std()),
                "r_eff_std": float(r_eff.std()),
                "z_leak": [],
                "z_perf": z_eff.tolist(),
                "r_eff_values": r_eff.tolist(),
                "cov_values": cov.tolist(),
                "h_gate": h_gate.tolist(),
                "d_gate": d_gate.tolist(),
                "q_gate": q_gate.tolist(),
                "all_clean": True,
                "adv_mode": "performance",
            }
        else:
            # -- LEAKAGE MODE (with coverage + quality bonus) --
            z_leak = _zscore(r_leak, self.cfg.z_clip)
            leak_baseline = self.cfg.lambda_base * (r_leak - batch_mean_r_leak)
            base_adv = z_leak + leak_baseline

            cov = np.array([r.coverage for r in parseable], dtype=np.float64)
            is_clean = (r_leak >= 0.99).astype(np.float64)
            cov_gate = np.minimum(1.0, cov / self.cfg.cov_target)
            cov_bonus = self.cfg.w_cov_leak * (
                self.cfg.cov_floor + (1 - self.cfg.cov_floor) * cov_gate
            ) * is_clean

            quality_bonus = self.cfg.w_quality_leak * h_gate * d_gate * is_clean

            adv_raw = base_adv + cov_bonus + quality_bonus
            baseline_values = leak_baseline
            diag_extra = {
                "r_leak_std": float(r_leak.std()),
                "r_perf_std": 0.0,
                "r_eff_std": 0.0,
                "z_leak": z_leak.tolist(),
                "z_perf": [],
                "r_eff_values": [],
                "cov_values": cov.tolist(),
                "cov_bonus": cov_bonus.tolist(),
                "quality_bonus": quality_bonus.tolist(),
                "h_gate": h_gate.tolist(),
                "d_gate": d_gate.tolist(),
                "q_gate": q_gate.tolist(),
                "all_clean": False,
                "adv_mode": "leakage",
            }

        fp_base = float(adv_raw.min()) - self.cfg.format_margin
        max_pos = float(adv_raw.max())
        if max_pos > 0.01:
            ratio_cap = -self.cfg.max_neg_ratio * max_pos
            format_penalty = max(fp_base, ratio_cap)
        else:
            format_penalty = fp_base

        out: list[float] = []
        baseline_list: list[float] = []
        j = 0
        for r in rewards:
            if not r.parse_success:
                out.append(format_penalty)
            else:
                out.append(float(adv_raw[j]))
                baseline_list.append(float(baseline_values[j]))
                j += 1

        diag: dict[str, Any] = {
            "baseline_bonus": baseline_list,
            "r_leak_values": r_leak.tolist(),
            "format_penalty": format_penalty,
            **diag_extra,
        }
        return out, diag

    def _scalar_mix_advantages(
        self,
        rewards: list[CompletionReward],
        parseable: list[CompletionReward],
        r_leak: np.ndarray,
        h_gate: np.ndarray,
        d_gate: np.ndarray,
        q_gate: np.ndarray,
        batch_mean_r_leak: float,
        batch_mean_r_eff_clean: float,
    ) -> tuple[list[float], dict[str, Any]]:
        """Scalar-mixture ablation: no mode separation.

        r_scalar = alpha * r_leak + (1 - alpha) * r_eff, applied to every
        completion regardless of group cleanliness. Uses the same quality
        gates as the two-mode design so the ablation isolates only the
        mode-separation factor.
        """
        alpha = self.cfg.scalar_alpha
        r_perf = np.array([r.r_performance for r in parseable], dtype=np.float64)
        cov = np.array([r.coverage for r in parseable], dtype=np.float64)
        cov_gate = np.minimum(1.0, cov / self.cfg.cov_target)
        g_cov = self.cfg.cov_floor + (1 - self.cfg.cov_floor) * cov_gate
        r_eff = r_perf * g_cov * h_gate * d_gate * q_gate

        r_scalar = alpha * r_leak + (1 - alpha) * r_eff
        batch_mean_scalar = (
            alpha * batch_mean_r_leak + (1 - alpha) * batch_mean_r_eff_clean
        )
        z_scalar = _zscore(r_scalar, self.cfg.z_clip)
        baseline = self.cfg.lambda_base * (r_scalar - batch_mean_scalar)
        adv_raw = z_scalar + baseline

        fp_base = float(adv_raw.min()) - self.cfg.format_margin
        max_pos = float(adv_raw.max())
        if max_pos > 0.01:
            format_penalty = max(fp_base, -self.cfg.max_neg_ratio * max_pos)
        else:
            format_penalty = fp_base

        out: list[float] = []
        baseline_list: list[float] = []
        j = 0
        for r in rewards:
            if not r.parse_success:
                out.append(format_penalty)
            else:
                out.append(float(adv_raw[j]))
                baseline_list.append(float(baseline[j]))
                j += 1

        diag: dict[str, Any] = {
            "baseline_bonus": baseline_list,
            "r_leak_values": r_leak.tolist(),
            "format_penalty": format_penalty,
            "r_leak_std": float(r_leak.std()),
            "r_perf_std": float(r_perf.std()),
            "r_eff_std": float(r_eff.std()),
            "z_leak": [],
            "z_perf": z_scalar.tolist(),
            "r_eff_values": r_eff.tolist(),
            "cov_values": cov.tolist(),
            "h_gate": h_gate.tolist(),
            "d_gate": d_gate.tolist(),
            "q_gate": q_gate.tolist(),
            "all_clean": bool(r_leak.min() >= 0.99),
            "adv_mode": "scalar_mix",
        }
        return out, diag

    # ------------------------------------------------------------------
    # Stage 1: Claim extraction
    # ------------------------------------------------------------------

    def _build_claim_pool(
        self,
        parseable: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
        """Build deduplicated claim pool from evidence items only."""

        claim_pool: list[dict[str, Any]] = []
        claim_to_completions: dict[str, list[int]] = {}

        for i, p in enumerate(parseable):
            for ev in p.get("evidence", []):
                fact = str(ev.get("fact", "")).strip()
                if not fact:
                    continue
                key = fact.lower()
                if key not in claim_to_completions:
                    claim_to_completions[key] = []
                    sd = ev.get("source_date")
                    source_date = str(sd).strip() if sd else None
                    claim_pool.append(
                        {"text": fact, "key": key, "source": "evidence", "source_date": source_date}
                    )
                if i not in claim_to_completions[key]:
                    claim_to_completions[key].append(i)

        for idx, c in enumerate(claim_pool):
            c["index"] = idx

        return claim_pool, claim_to_completions

    # ------------------------------------------------------------------
    # Stage 2 & 3: Leakage + Hallucination detection (run in parallel)
    # ------------------------------------------------------------------

    def _detect_leakage(
        self, claim_pool: list[dict[str, Any]], cutoff_date: str
    ) -> dict[int, dict[str, Any]]:
        """Leakage detection via DATE_PLAUSIBILITY_PROMPT + deterministic date comparison.

        Only evidence claims with a declared source_date are checked. Claims
        without source_date are treated as clean (not checked).
        """
        anchored = [c for c in claim_pool if c.get("source") == "evidence" and c.get("source_date")]

        results: dict[int, dict[str, Any]] = {}

        # Stage 1: plausibility verifier + deterministic date comparison
        if anchored:
            for chunk_start in range(0, len(anchored), _JUDGE_CHUNK_SIZE):
                chunk = anchored[chunk_start : chunk_start + _JUDGE_CHUNK_SIZE]
                items_payload = [
                    {
                        "index": c["index"],
                        "fact": c["text"],
                        "declared_source_date": c["source_date"],
                    }
                    for c in chunk
                ]
                text, _ = self.client.call_text(
                    self.cfg.judge_model,
                    DATE_PLAUSIBILITY_PROMPT.format(
                        cutoff=cutoff_date,
                        items_json=json.dumps(items_payload, indent=2, ensure_ascii=False),
                    ),
                    temperature=0.0,
                    max_tokens=_MAX_TOKENS_JUDGE,
                )
                verifier_by_idx = {r["index"]: r for r in _parse_json_array_raw(text) if "index" in r}

                for c in chunk:
                    idx = c["index"]
                    declared = c["source_date"]
                    vr = verifier_by_idx.get(idx, {})
                    plausible = bool(vr.get("plausible", True))
                    corrected = vr.get("corrected_date")
                    effective = corrected if (not plausible and corrected) else declared
                    try:
                        leaked = bool(effective > cutoff_date)
                    except TypeError:
                        leaked = False
                    results[idx] = {
                        "index": idx,
                        "verdict": "leaked" if leaked else "not_leaked",
                        "event_date": effective,
                        "confidence": "high",
                        "method": "anchored",
                        "plausible": plausible,
                    }
                    if not plausible:
                        logger.debug(
                            "claim %d: source_date %s corrected to %s → %s",
                            idx, declared, corrected, results[idx]["verdict"],
                        )

        return results

    def _detect_hallucination(
        self, claim_pool: list[dict[str, Any]], task: str
    ) -> dict[int, dict[str, Any]]:
        """VERIFY-style hallucination detection: per-claim Sonar retrieval + batch Gemini judgment.

        Step 1: Parallel Sonar search per claim to retrieve external evidence.
        Step 2: Single Gemini batch-judge call with all claims + evidence.
        Verdicts: supported / unsupported / undecidable
          - "unsupported" → hallucinated
          - "undecidable" → zero penalty (epistemic honesty; follows VERIFY §4.6)
        """
        if not claim_pool:
            return {}

        # Step 1: parallel Sonar search, one query per claim
        queries = [build_hall_sonar_prompt(task, c["text"]) for c in claim_pool]
        raw_sonar, _ = self.client.call_search_batch(
            self.cfg.search_model, queries, cutoff_date="",
        )
        sonar_parsed: dict[int, dict[str, Any]] = {}
        for i, raw in raw_sonar.items():
            sonar_parsed[i] = _parse_sonar_json(raw)

        # Step 2: build evidence payload; one Gemini call per _JUDGE_CHUNK_SIZE claims
        results: dict[int, dict[str, Any]] = {}
        for chunk_start in range(0, len(claim_pool), _JUDGE_CHUNK_SIZE):
            chunk = claim_pool[chunk_start: chunk_start + _JUDGE_CHUNK_SIZE]
            items = []
            for local_i, c in enumerate(chunk):
                global_i = chunk_start + local_i
                sr = sonar_parsed.get(global_i, {})
                found = sr.get("found_value") or ""
                hint = sr.get("source_hint") or ""
                reason = sr.get("reason") or ""
                evidence = (
                    f"{reason}. Found: {found}. Source: {hint}."
                    if reason else "No evidence retrieved."
                )
                items.append({"index": c["index"], "claim": c["text"], "search_evidence": evidence})
            text, _ = self.client.call_text(
                self.cfg.judge_model,
                build_hall_batch_prompt(task, json.dumps(items, indent=2, ensure_ascii=False)),
                temperature=0.0,
                max_tokens=_MAX_TOKENS_JUDGE,
            )
            for entry in _parse_json_array_raw(text):
                if isinstance(entry, dict) and "index" in entry:
                    idx = entry["index"]
                    pos = next(
                        (chunk_start + j for j, c in enumerate(chunk) if c["index"] == idx),
                        None,
                    )
                    sr = sonar_parsed.get(pos, {}) if pos is not None else {}
                    results[idx] = {
                        "verdict": entry.get("verdict", "undecidable"),
                        "known_value": sr.get("found_value"),
                        "reason": entry.get("reason", ""),
                    }
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rewards_no_claims(
        self,
        completions: list[str],
        parseable: list[dict[str, Any]],
        parseable_idx: list[int],
        r_format: list[float],
        instance: TaskInstance,
    ) -> list[CompletionReward]:
        """Build rewards when the claim pool is empty."""

        rewards: list[CompletionReward] = [
            self._parse_fail_reward() for _ in completions
        ]
        for local_idx, global_idx in enumerate(parseable_idx):
            pi = parseable[local_idx]
            cov = _coverage(pi.get("reasoning", ""), pi.get("evidence", []))
            r_perf = _performance_score(
                instance.task, pi, instance.ground_truth,
            )
            rewards[global_idx] = CompletionReward(
                r_format=r_format[global_idx],
                r_valid=1.0,
                r_performance=float(np.clip(r_perf, 0.0, 1.0)),
                leakage_rate=0.0,
                hallucination_rate=0.0,
                coverage=cov,
                parse_success=True,
                num_claims=0,
                num_leaked=0,
                num_unsupported=0,
                num_undecidable=0,
            )
        return rewards

    def _parse_fail_reward(self) -> CompletionReward:
        return CompletionReward(
            r_format=0.0,
            r_valid=0.0,
            r_performance=0.0,
            leakage_rate=1.0,
            hallucination_rate=1.0,
            coverage=0.0,
            parse_success=False,
            num_claims=0,
            num_leaked=0,
            num_unsupported=0,
            num_undecidable=0,
        )


# ======================================================================
# Pure helper functions (no state)
# ======================================================================

def _parse_sonar_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a Sonar response (may be markdown-wrapped)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return {}


def _parse_json_array_raw(text: str) -> list[dict[str, Any]]:
    """Parse JSON array from LLM response, handling markdown fences and NDJSON."""

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _build_claim_detail(
    claim_pool: list[dict[str, Any]],
    claim_to_completions: dict[str, list[int]],
    leak_results: dict[int, dict[str, Any]],
    hall_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Build structured claim-level detail for logging.

    Leakage and hallucination verdicts are logged as independent fields.
    No merged label — each attribute stands on its own.
    """
    claims = []
    for c in claim_pool:
        idx = c["index"]
        lr = leak_results.get(idx, {})
        hr = hall_results.get(idx, {})
        claims.append({
            "index": idx,
            "text": c["text"],
            "source": c["source"],
            "leak_verdict": lr.get("verdict", "unknown"),
            "leak_confidence": lr.get("confidence", "unknown"),
            "leak_event_date": lr.get("event_date"),
            "leak_method": lr.get("method"),
            "leak_plausible": lr.get("plausible"),
            "hall_verdict": hr.get("verdict", "unknown"),
            "hall_known_value": hr.get("known_value"),
            "owners": claim_to_completions.get(c["key"], []),
        })
    return {"claims": claims}


def _compute_rates(
    claim_indices: list[int],
    leak_results: dict[int, dict[str, Any]],
    hall_results: dict[int, dict[str, Any]],
) -> tuple[float, float, float, int, int, int]:
    """Compute leak_rate, hall_rate, r_valid from two independent verdict dicts.

    Leakage and hallucination are fully independent — a claim may be leaked,
    hallucinated, both, or neither, and each metric is computed without
    conflating the two.

    leak_rate  = n_leaked / total
    hall_rate  = (n_unsupported + 0.5 × n_undecidable) / total   [VERIFY §4.6, α=0.5]
    r_valid    = (n_clean + 0.5 × n_half_clean) / total
                 where n_clean      = not leaked AND supported
                       n_half_clean = not leaked AND undecidable

    Returns:
        (leak_rate, hall_rate, r_valid, n_leaked, n_unsupported, n_undecidable)
    """
    total = len(claim_indices)
    if total == 0:
        return 0.0, 0.0, 1.0, 0, 0, 0

    n_leaked = sum(
        1 for i in claim_indices
        if leak_results.get(i, {}).get("verdict") == "leaked"
    )
    n_unsupported = sum(
        1 for i in claim_indices
        if hall_results.get(i, {}).get("verdict") == "unsupported"
    )
    n_undecidable = sum(
        1 for i in claim_indices
        if hall_results.get(i, {}).get("verdict") == "undecidable"
    )
    # r_valid: claims that are not leaked AND not hallucinated
    # undecidable-but-not-leaked counts as half-valid (α=0.5)
    n_clean = sum(
        1 for i in claim_indices
        if leak_results.get(i, {}).get("verdict") != "leaked"
        and hall_results.get(i, {}).get("verdict") not in ("unsupported", "undecidable")
    )
    n_half_clean = sum(
        1 for i in claim_indices
        if leak_results.get(i, {}).get("verdict") != "leaked"
        and hall_results.get(i, {}).get("verdict") == "undecidable"
    )

    leak_rate = n_leaked / total
    hall_rate = (n_unsupported + 0.5 * n_undecidable) / total
    r_valid = (n_clean + 0.5 * n_half_clean) / total
    return leak_rate, hall_rate, r_valid, n_leaked, n_unsupported, n_undecidable


def _extract_quality_signals(
    rewards: list[CompletionReward],
    parsed_completions: list[dict[str, Any] | None] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-parseable hallucination rate, evidence count, and reasoning word count."""
    hall_rates: list[float] = []
    ev_counts: list[float] = []
    reason_words: list[float] = []
    p_idx = 0
    for r in rewards:
        if not r.parse_success:
            continue
        hall_rates.append(r.hallucination_rate)
        if parsed_completions and p_idx < len(parsed_completions):
            p = parsed_completions[p_idx]
        else:
            p = None
        if p is not None:
            ev_counts.append(float(len(p.get("evidence", []))))
            reason_words.append(float(len(p.get("reasoning", "").split())))
        else:
            ev_counts.append(float(r.num_claims))
            reason_words.append(120.0)  # default to target when unavailable
        p_idx += 1
    return (
        np.array(hall_rates, dtype=np.float64),
        np.array(ev_counts, dtype=np.float64),
        np.array(reason_words, dtype=np.float64),
    )


def _hallucination_gate(hall_rates: np.ndarray, hall_floor: float) -> np.ndarray:
    """h(rate) = max(1 - rate, hall_floor). Range: [hall_floor, 1.0]."""
    return np.maximum(1.0 - hall_rates, hall_floor)


def _evidence_gate(ev_counts: np.ndarray, ev_target: int) -> np.ndarray:
    """d(count) = min(count / ev_target, 1.0). Range: [0, 1.0]."""
    return np.minimum(ev_counts / max(ev_target, 1), 1.0)


def _reasoning_gate(reason_words: np.ndarray, reason_target: int) -> np.ndarray:
    """q(words) = min(words / reason_target, 1.0). Range: [0, 1.0]."""
    return np.minimum(reason_words / max(reason_target, 1), 1.0)


def _evidence_gate_trapezoidal(
    ev_counts: np.ndarray,
    ev_target: int,
    ev_cap: int,
    ev_decay: float,
    ev_floor: float,
) -> np.ndarray:
    """Trapezoidal: ramp to ev_target, plateau to ev_cap, linear decay after."""
    ramp = np.minimum(ev_counts / max(ev_target, 1), 1.0)
    overflow = np.maximum(ev_counts - ev_cap, 0)
    decay = np.maximum(1.0 - ev_decay * overflow, ev_floor)
    return np.where(ev_counts <= ev_cap, ramp, decay)


def _reasoning_gate_trapezoidal(
    reason_words: np.ndarray,
    reason_target: int,
    reason_cap: int,
    reason_decay: float,
    reason_floor: float,
) -> np.ndarray:
    """Trapezoidal: ramp to reason_target, plateau to reason_cap, linear decay after."""
    ramp = np.minimum(reason_words / max(reason_target, 1), 1.0)
    overflow = np.maximum(reason_words - reason_cap, 0)
    decay = np.maximum(1.0 - reason_decay * overflow, reason_floor)
    return np.where(reason_words <= reason_cap, ramp, decay)


def _zscore(arr: np.ndarray, clip: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-6:
        return np.zeros_like(arr)
    z = (arr - mean) / std
    return np.clip(z, -clip, clip)


def _coverage(reasoning: str, evidence: list[dict[str, Any]]) -> float:
    cited_ids = {int(x) for x in re.findall(r"\[(\d+)\]", reasoning)}
    valid_ids = set(range(1, len(evidence) + 1))
    return len(cited_ids & valid_ids) / max(len(evidence), 1)


def _performance_score(
    task: str, parsed: dict[str, Any], gt: dict[str, Any]
) -> float:
    if task == "stock":
        pred = parsed.get("ranking", [])
        true = gt.get("ranking", [])
        return _rank_score(pred, true)
    if task == "salary":
        pred = float(parsed.get("predicted_salary", 0.0))
        true = float(gt.get("aav", 0.0) or 0.0)
        if true <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(pred - true) / true)
    p_hat = float(parsed.get("probability_petitioner", 0.5))
    outcome = str(gt.get("outcome", "")).upper()
    p_true = 1.0 if "PETITIONER" in outcome else 0.0
    return 1.0 - (p_hat - p_true) ** 2


def _rank_score(pred: list[str], true: list[str]) -> float:
    if not pred or not true:
        return 0.0
    pos_true = {t: i for i, t in enumerate(true)}
    common = [t for t in pred if t in pos_true]
    if len(common) < 2:
        return 0.0
    pred_order = np.array([pred.index(t) for t in common], dtype=np.float64)
    true_order = np.array([pos_true[t] for t in common], dtype=np.float64)
    pred_center = pred_order - pred_order.mean()
    true_center = true_order - true_order.mean()
    denom = np.sqrt((pred_center**2).sum() * (true_center**2).sum())
    if denom < 1e-9:
        return 0.5
    rho = float((pred_center * true_center).sum() / denom)
    return (rho + 1.0) / 2.0
