"""Layer 1: Local logic tests (no API calls, no cost).

Validates all pure-logic components:
  - Data loading (all 3 tasks × train + eval)
  - Prompt building (temporal_hint + base modes)
  - Response parsing (valid, truncated, garbage inputs)
  - Performance reward computation
  - MO-GRPO advantage calculation
  - Lambda schedule
  - Overlong filtering + soft length penalty logic

Usage:
    uv run python tests/test_layer1_local.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  — {detail}")


# ===================================================================
# 1. Data Loading
# ===================================================================

def test_data_loading() -> None:
    print("\n── 1. Data Loading ──")
    from src.shared.data_loader import (
        load_all_eval,
        load_all_train,
        load_legal_eval,
        load_legal_train,
        load_salary_eval,
        load_salary_train,
        load_stock_eval,
        load_stock_train,
        make_train_val_split,
    )

    stock_train = load_stock_train()
    check("stock_train loads", len(stock_train) > 0, f"got {len(stock_train)}")
    check("stock_train has cutoff_date", all(i.cutoff_date for i in stock_train),
          f"missing: {[i.instance_id for i in stock_train if not i.cutoff_date][:3]}")
    check("stock_train has ground_truth.true_ranking",
          all(i.ground_truth.get("true_ranking") for i in stock_train[:10]))

    salary_train = load_salary_train()
    check("salary_train loads", len(salary_train) > 0, f"got {len(salary_train)}")
    check("salary_train has cutoff_date", all(i.cutoff_date for i in salary_train))
    s0 = salary_train[0]
    raw_aav = s0.ground_truth.get("aav", "")
    try:
        float(str(raw_aav).replace(",", "").replace("$", ""))
        aav_ok = True
    except (TypeError, ValueError):
        aav_ok = False
    check("salary_train AAV is parseable to float", aav_ok, f"aav='{raw_aav}'")

    legal_train = load_legal_train()
    check("legal_train loads", len(legal_train) > 0, f"got {len(legal_train)}")
    check("legal_train has outcome",
          all(i.ground_truth.get("outcome") for i in legal_train))

    stock_eval = load_stock_eval()
    salary_eval = load_salary_eval()
    legal_eval = load_legal_eval()
    check("stock_eval loads", len(stock_eval) > 0)
    check("salary_eval loads", len(salary_eval) > 0)
    check("legal_eval loads", len(legal_eval) > 0)

    all_train = load_all_train()
    check("load_all_train returns all tasks",
          len(all_train) == len(stock_train) + len(salary_train) + len(legal_train))
    all_eval = load_all_eval()
    check("load_all_eval returns all tasks",
          len(all_eval) == len(stock_eval) + len(salary_eval) + len(legal_eval))

    split = make_train_val_split(all_train, val_fraction=0.1, seed=42)
    check("train/val split sums to total",
          len(split.train) + len(split.val) == len(all_train))
    check("val set is non-empty", len(split.val) > 0)

    print(f"\n  Summary: stock={len(stock_train)}, salary={len(salary_train)}, "
          f"legal={len(legal_train)} | eval: stock={len(stock_eval)}, "
          f"salary={len(salary_eval)}, legal={len(legal_eval)}")


# ===================================================================
# 2. Prompt Building
# ===================================================================

def test_prompts() -> None:
    print("\n── 2. Prompt Building ──")
    from src.shared.data_loader import load_stock_train, load_salary_train, load_legal_train
    from src.shared.prompts.builder import build_prompt, build_messages

    stock = load_stock_train()[0]
    salary = load_salary_train()[0]
    legal = load_legal_train()[0]

    for mode in ("temporal_hint", "base"):
        for task_name, inst in [("stock", stock), ("salary", salary), ("legal", legal)]:
            sys_p, user_p = build_prompt(inst, prompt_mode=mode)
            check(f"{task_name}/{mode} system prompt non-empty", len(sys_p) > 50)
            check(f"{task_name}/{mode} user prompt non-empty", len(user_p) > 20)

            if mode == "temporal_hint":
                check(f"{task_name}/{mode} has cutoff date in system",
                      inst.cutoff_date in sys_p,
                      f"cutoff={inst.cutoff_date}")
            if mode == "base":
                check(f"{task_name}/{mode} no {{ braces (not a template)",
                      "{cutoff_date}" not in sys_p)

    # Verify excluded fields
    sys_p, user_p = build_prompt(stock, prompt_mode="temporal_hint")
    check("stock user prompt has no question_id", "question_id" not in user_p)
    check("stock user prompt has no window_id", "window_id" not in user_p)

    sys_p, user_p = build_prompt(salary, prompt_mode="temporal_hint")
    check("salary user prompt has no Previous_Team/From_Team",
          "Previous" not in user_p and "From_Team" not in user_p)
    check("salary user prompt has no Position field",
          "Position:" not in user_p)

    sys_p, user_p = build_prompt(legal, prompt_mode="temporal_hint")
    check("legal user prompt has no facts", "facts" not in user_p.lower() or "Facts:" not in user_p)
    check("legal user prompt has no oral_argument_date", "oral_argument" not in user_p)

    msgs = build_messages(stock)
    check("build_messages returns 2 messages", len(msgs) == 2)
    check("build_messages has system+user roles",
          msgs[0]["role"] == "system" and msgs[1]["role"] == "user")


# ===================================================================
# 3. Response Parsing
# ===================================================================

def test_parsing() -> None:
    print("\n── 3. Response Parsing ──")
    from src.shared.data_loader import load_stock_train, load_salary_train, load_legal_train
    from src.shared.prompts.builder import parse_response

    stock = load_stock_train()[0]
    salary = load_salary_train()[0]
    legal = load_legal_train()[0]

    companies = stock.raw.get("companies", [])
    if len(companies) >= 2:
        good_stock = json.dumps({
            "ranking": companies,
            "rationale": "Based on analysis of Q3 earnings..."
        })
        parsed = parse_response(good_stock, stock)
        check("stock: valid JSON parses", parsed is not None)
        check("stock: ranking preserved", parsed and parsed.get("ranking") == companies)

        truncated = '{"ranking": ["' + '", "'.join(companies[:3])
        parsed = parse_response(truncated, stock)
        check("stock: truncated JSON fallback works", parsed is not None)

    good_salary = json.dumps({
        "predicted_salary": 25000000,
        "rationale": "Based on comparable contracts..."
    })
    parsed = parse_response(good_salary, salary)
    check("salary: valid JSON parses", parsed is not None)
    check("salary: predicted_salary is float",
          parsed is not None and isinstance(parsed.get("predicted_salary"), float))

    truncated_salary = '{"predicted_salary": 30000000, "rationale": "The player has shown'
    parsed = parse_response(truncated_salary, salary)
    check("salary: truncated JSON fallback works", parsed is not None)

    natural_salary = "I predict his AAV will be approximately $35 million per year."
    parsed = parse_response(natural_salary, salary)
    check("salary: natural-language dollar amount parsed", parsed is not None)

    good_legal = json.dumps({
        "probability_petitioner": 0.65,
        "predicted_winner": "PETITIONER",
        "rationale": "Based on precedent in..."
    })
    parsed = parse_response(good_legal, legal)
    check("legal: valid JSON parses", parsed is not None)
    check("legal: probability in [0,1]",
          parsed is not None and 0 <= parsed.get("probability_petitioner", -1) <= 1)

    garbage = "This is not JSON at all, just a text response with no numbers."
    for name, inst in [("stock", stock), ("salary", salary), ("legal", legal)]:
        parsed = parse_response(garbage, inst)
        check(f"{name}: garbage returns None", parsed is None)


# ===================================================================
# 4. Performance Rewards
# ===================================================================

def test_performance_rewards() -> None:
    print("\n── 4. Performance Rewards ──")
    from src.shared.data_loader import TaskInstance
    from src.training.reward.performance import compute_performance_reward

    # Stock: Spearman
    stock_inst = TaskInstance(
        task="stock", instance_id="test",
        cutoff_date="2023-01-01",
        ground_truth={"true_ranking": ["A", "B", "C", "D", "E"]},
        raw={"companies": ["A", "B", "C", "D", "E"]},
    )
    perfect = {"ranking": ["A", "B", "C", "D", "E"]}
    r = compute_performance_reward(perfect, stock_inst)
    check("stock: perfect ranking → reward=1.0", abs(r - 1.0) < 1e-6, f"got {r}")

    reversed_r = {"ranking": ["E", "D", "C", "B", "A"]}
    r = compute_performance_reward(reversed_r, stock_inst)
    check("stock: reversed ranking → reward=0.0", abs(r - 0.0) < 1e-6, f"got {r}")

    r = compute_performance_reward(None, stock_inst)
    check("stock: None parsed → reward=0.0", r == 0.0)

    # Salary: 1 - relative error
    salary_inst = TaskInstance(
        task="salary", instance_id="test",
        cutoff_date="2023-01-01",
        ground_truth={"aav": "$30,000,000"},
        raw={},
    )
    exact = {"predicted_salary": 30_000_000.0}
    r = compute_performance_reward(exact, salary_inst)
    check("salary: exact prediction → reward=1.0", abs(r - 1.0) < 1e-6, f"got {r}")

    off_50pct = {"predicted_salary": 45_000_000.0}
    r = compute_performance_reward(off_50pct, salary_inst)
    check("salary: 50% off → reward=0.5", abs(r - 0.5) < 1e-6, f"got {r}")

    off_200pct = {"predicted_salary": 90_000_000.0}
    r = compute_performance_reward(off_200pct, salary_inst)
    check("salary: 200% off → reward=0.0 (clamped)", r == 0.0, f"got {r}")

    # Salary with string AAV (as loaded from CSV)
    salary_inst2 = TaskInstance(
        task="salary", instance_id="test2",
        cutoff_date="2023-01-01",
        ground_truth={"aav": "25000000"},
        raw={},
    )
    r = compute_performance_reward({"predicted_salary": 25_000_000.0}, salary_inst2)
    check("salary: string AAV without $ → reward=1.0", abs(r - 1.0) < 1e-6, f"got {r}")

    # Legal: 1 - Brier score
    legal_inst = TaskInstance(
        task="legal", instance_id="test",
        cutoff_date="2023-01-01",
        ground_truth={"outcome": "PETITIONER"},
        raw={},
    )
    perfect_legal = {"probability_petitioner": 1.0}
    r = compute_performance_reward(perfect_legal, legal_inst)
    check("legal: perfect prediction → reward=1.0", abs(r - 1.0) < 1e-6, f"got {r}")

    wrong_legal = {"probability_petitioner": 0.0}
    r = compute_performance_reward(wrong_legal, legal_inst)
    check("legal: completely wrong → reward=0.0", abs(r - 0.0) < 1e-6, f"got {r}")

    coin_flip = {"probability_petitioner": 0.5}
    r = compute_performance_reward(coin_flip, legal_inst)
    check("legal: 0.5 → reward=0.75", abs(r - 0.75) < 1e-6, f"got {r}")


# ===================================================================
# 5. MO-GRPO Advantages
# ===================================================================

def test_mo_grpo() -> None:
    print("\n── 5. MO-GRPO Advantages ──")
    from src.training.reward.reward_pipeline import CompletionReward, mo_grpo_advantages

    rewards = [
        CompletionReward(r_leakage=0.9, r_performance=0.3, olr=0.1, num_claims=5, num_leaked=0, parse_success=True),
        CompletionReward(r_leakage=0.5, r_performance=0.7, olr=0.5, num_claims=5, num_leaked=2, parse_success=True),
        CompletionReward(r_leakage=0.7, r_performance=0.5, olr=0.3, num_claims=5, num_leaked=1, parse_success=True),
        CompletionReward(r_leakage=0.3, r_performance=0.9, olr=0.7, num_claims=5, num_leaked=3, parse_success=True),
    ]

    advs = mo_grpo_advantages(rewards, lambda_t=0.2)
    check("MO-GRPO: returns correct length", len(advs) == 4)
    check("MO-GRPO: advantages sum near zero",
          abs(sum(advs)) < 0.01, f"sum={sum(advs):.6f}")

    # leakage_only: lambda=0 → performance component ignored
    advs_leak = mo_grpo_advantages(rewards, lambda_t=0.0)
    check("MO-GRPO lambda=0: perf ignored (highest leak → highest adv)",
          advs_leak[0] == max(advs_leak))

    # When leak and perf are positively correlated, lambda=1 amplifies both
    corr_rewards = [
        CompletionReward(r_leakage=0.9, r_performance=0.8, olr=0.1, num_claims=5, num_leaked=0, parse_success=True),
        CompletionReward(r_leakage=0.5, r_performance=0.4, olr=0.5, num_claims=5, num_leaked=2, parse_success=True),
        CompletionReward(r_leakage=0.3, r_performance=0.2, olr=0.7, num_claims=5, num_leaked=3, parse_success=True),
        CompletionReward(r_leakage=0.7, r_performance=0.6, olr=0.3, num_claims=5, num_leaked=1, parse_success=True),
    ]
    advs_perf = mo_grpo_advantages(corr_rewards, lambda_t=1.0)
    check("MO-GRPO lambda=1: best combined gets highest advantage",
          advs_perf[0] == max(advs_perf))

    # Identical rewards → all advantages zero
    same = [CompletionReward(0.5, 0.5, 0.5, 3, 1, True)] * 4
    advs_same = mo_grpo_advantages(same, lambda_t=0.2)
    check("MO-GRPO: identical rewards → all zero",
          all(abs(a) < 1e-6 for a in advs_same))

    # Single completion → zero advantage
    advs_one = mo_grpo_advantages([rewards[0]], lambda_t=0.2)
    check("MO-GRPO: single completion → [0.0]", advs_one == [0.0])


# ===================================================================
# 6. Lambda Schedule
# ===================================================================

def test_lambda_schedule() -> None:
    print("\n── 6. Lambda Schedule ──")
    from src.training.reward.lambda_schedule import LambdaSchedule

    sched = LambdaSchedule(total_steps=100, lambda_min=0.1, lambda_max=0.3)
    l0 = sched.get_lambda(0)
    check("lambda(0) = lambda_min", abs(l0 - 0.1) < 1e-6, f"got {l0}")
    l_end = sched.get_lambda(100)
    check("lambda(T) = lambda_max", abs(l_end - 0.3) < 1e-6, f"got {l_end}")
    l_mid = sched.get_lambda(50)
    check("lambda(T/2) between min and max", 0.1 < l_mid < 0.3, f"got {l_mid}")
    check("lambda monotonically increases (cosine)",
          sched.get_lambda(25) <= sched.get_lambda(50) <= sched.get_lambda(75))

    # Adaptive component (use separate schedules to avoid EMA carry-over)
    sched_low = LambdaSchedule(total_steps=100, lambda_min=0.1, lambda_max=0.3,
                               theta_high=0.3, theta_low=0.1)
    l_low_olr = sched_low.get_lambda(0, val_olr=0.05)
    check("adaptive: very low OLR → lambda_max", abs(l_low_olr - 0.3) < 1e-6, f"got {l_low_olr}")

    sched_high = LambdaSchedule(total_steps=100, lambda_min=0.1, lambda_max=0.3,
                                theta_high=0.3, theta_low=0.1)
    l_high_olr = sched_high.get_lambda(0, val_olr=0.5)
    check("adaptive: high OLR → lambda_min", abs(l_high_olr - 0.1) < 1e-6, f"got {l_high_olr}")


# ===================================================================
# 7. Overlong Filtering + Soft Penalty (logic only)
# ===================================================================

def test_overlong_and_soft_penalty() -> None:
    print("\n── 7. Overlong Filtering + Soft Penalty ──")
    from src.training.reward.reward_pipeline import CompletionReward, mo_grpo_advantages

    max_tokens = 4096
    soft_start = 3072

    class FakeSeq:
        def __init__(self, n: int):
            self.tokens = list(range(n))

    rewards = [
        CompletionReward(0.8, 0.6, 0.2, 5, 1, True),   # 2000 tok
        CompletionReward(0.9, 0.7, 0.1, 4, 0, True),   # 3500 tok (soft zone)
        CompletionReward(0.5, 0.3, 0.5, 8, 4, True),   # 4096 tok (OVERLONG)
        CompletionReward(0.7, 0.5, 0.3, 6, 2, True),   # 1500 tok
    ]
    sequences = [FakeSeq(2000), FakeSeq(3500), FakeSeq(4096), FakeSeq(1500)]

    valid_idx = [i for i, seq in enumerate(sequences) if len(seq.tokens) < max_tokens]
    check("overlong: filters index 2 (4096 tokens)", valid_idx == [0, 1, 3])

    valid_rewards = [rewards[i] for i in valid_idx]
    advantages = mo_grpo_advantages(valid_rewards, lambda_t=0.2)
    check("overlong: advantages computed for 3 valid only", len(advantages) == 3)

    # Apply soft penalty
    penalty_range = max_tokens - soft_start
    for k, vi in enumerate(valid_idx):
        seq_len = len(sequences[vi].tokens)
        if seq_len > soft_start and penalty_range > 0:
            frac = (seq_len - soft_start) / penalty_range
            advantages[k] *= max(0.0, 1.0 - frac)

    check("soft penalty: idx=0 (2000 tok) unpenalised", advantages[0] != 0)
    # idx=1 maps to original index 1 (3500 tokens) — should be penalised
    expected_frac = (3500 - 3072) / (4096 - 3072)
    check(f"soft penalty: idx=1 (3500 tok) frac≈{expected_frac:.3f}",
          0.3 < expected_frac < 0.5)
    check("soft penalty: idx=2 (1500 tok) unpenalised", True)

    # All overlong → skip group
    all_overlong = [FakeSeq(5000)] * 4
    valid_all = [i for i, seq in enumerate(all_overlong) if len(seq.tokens) < max_tokens]
    check("all overlong → empty valid list → skip", len(valid_all) == 0)


# ===================================================================
# 8. Metrics Calculation
# ===================================================================

def test_metrics() -> None:
    print("\n── 8. Metrics Calculation ──")
    from src.evaluation.metrics_calculator import calculate_shapley_metrics, aggregate_metrics

    sv = [0.3, -0.1, 0.5, 0.2]
    leaked = [True, False, True, False]
    m = calculate_shapley_metrics(sv, leaked)
    check("OLR = 2/4 = 0.5", abs(m["olr"] - 0.5) < 1e-6)
    # DCLR = (|0.3| + |0.5|) / (|0.3| + |0.1| + |0.5| + |0.2|) = 0.8 / 1.1
    expected_dclr = 0.8 / 1.1
    check(f"DCLR = 0.8/1.1 ≈ {expected_dclr:.4f}",
          abs(m["shapley_dclr"] - expected_dclr) < 1e-4, f"got {m['shapley_dclr']}")

    # Empty claims
    m0 = calculate_shapley_metrics([], [])
    check("empty claims → OLR=0, DCLR=0", m0["olr"] == 0 and m0["shapley_dclr"] == 0)

    # Aggregation
    agg = aggregate_metrics([m, m0])
    check("aggregate: n_instances=2", agg["n_instances"] == 2)
    check("aggregate: mean OLR correct",
          abs(agg["olr"]["mean"] - 0.25) < 1e-6)


# ===================================================================
# 9. Config & CLI Override Logic
# ===================================================================

def test_config() -> None:
    print("\n── 9. Config & Model Resolution ──")
    from src.shared.config import get_model_name

    check("gemini-2.5-flash resolves correctly",
          get_model_name("google/gemini-2.5-flash") == "google/gemini-2.5-flash")
    check("slash model returned as-is",
          get_model_name("qwen/qwen3-30b-a3b") == "qwen/qwen3-30b-a3b")
    check("logical name resolves",
          get_model_name("gemini-2.5-flash", "openrouter") == "google/gemini-2.5-flash")


# ===================================================================
# 10. TimeSPEC Field Exclusion
# ===================================================================

def test_timespec_exclusions() -> None:
    print("\n── 10. TimeSPEC Field Exclusion ──")
    from src.evaluation.timespec_agent import _filter_ground_truth
    from src.shared.data_loader import TaskInstance

    stock_raw = {
        "question_id": "q1", "window_id": "w1", "sector": "Tech",
        "companies": ["AAPL"], "true_ranking": ["AAPL"], "returns": {},
        "past_period": {"start_date": "2023-01-01", "end_date": "2023-06-30"},
    }
    inst = TaskInstance(task="stock", instance_id="t1", cutoff_date="2023-01-01",
                        ground_truth={}, raw=stock_raw)
    filtered = _filter_ground_truth(inst)
    check("TimeSPEC: question_id excluded", "question_id" not in filtered)
    check("TimeSPEC: window_id excluded", "window_id" not in filtered)
    check("TimeSPEC: true_ranking excluded", "true_ranking" not in filtered)
    check("TimeSPEC: sector preserved", "sector" in filtered)

    legal_raw = {
        "case_name": "Test v. Test", "petitioner": "A", "respondent": "B",
        "facts": "Some facts...", "oral_argument_date": "2023-01-01",
        "outcome": "PETITIONER", "decision_date": "2023-06-01",
    }
    inst2 = TaskInstance(task="legal", instance_id="t2", cutoff_date="2023-01-01",
                         ground_truth={}, raw=legal_raw)
    filtered2 = _filter_ground_truth(inst2)
    check("TimeSPEC: facts excluded", "facts" not in filtered2)
    check("TimeSPEC: oral_argument_date excluded", "oral_argument_date" not in filtered2)
    check("TimeSPEC: winning_party excluded", "winning_party" not in filtered2)
    check("TimeSPEC: case_name preserved", "case_name" in filtered2)


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    global PASS, FAIL
    tests = [
        test_data_loading,
        test_prompts,
        test_parsing,
        test_performance_rewards,
        test_mo_grpo,
        test_lambda_schedule,
        test_overlong_and_soft_penalty,
        test_metrics,
        test_config,
        test_timespec_exclusions,
    ]

    print("=" * 60)
    print("  Layer 1: Local Logic Tests (no API, no cost)")
    print("=" * 60)

    for test_fn in tests:
        try:
            test_fn()
        except Exception:
            FAIL += 1
            print(f"\n  ✗ {test_fn.__name__} CRASHED:")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if FAIL == 0:
        print(f"  ALL {PASS} CHECKS PASSED")
    else:
        print(f"  {PASS} passed, {FAIL} FAILED")
    print("=" * 60)
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
