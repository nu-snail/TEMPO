"""End-to-end test of the step-stamped checkpoint renewal strategy.

The test simulates three successive "new best" saves (steps 31, 63, 95) on a
real Tinker training session and verifies:

  ROUND 1 (step 31, first best):
    - best_s0031 checkpoint is saved successfully.
    - checkpoints.jsonl has 1 "best_s*" entry.
    - Remote checkpoint exists on Tinker.

  ROUND 2 (step 63, improved best):
    - best_s0063 checkpoint is saved successfully (different GCS path → no conflict).
    - _delete_prev_best_checkpoints removes best_s0031 from Tinker.
    - checkpoints.jsonl has 2 entries total (stale step-31 + live step-63).
    - Step-31 remote checkpoint is gone; step-63 is alive.

  ROUND 3 (step 95, improved again):
    - best_s0095 saved successfully.
    - best_s0063 deleted.
    - Step-63 remote checkpoint is gone; step-95 is alive.

  SESSION HEALTH:
    - After 3 renewals, training session is still usable (not poisoned).
      Confirmed by successfully saving a fourth checkpoint.

  get_best_sampler_path:
    - Returns the sampler_path from the entry with the lowest val_leakage_rate.

  CLEANUP: deletes surviving remote checkpoints.

Run with:
    uv run python scripts/test_checkpoint_renewal.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── helpers mirroring train.py and checkpoint_io.py exactly ──────────────────

def _read_checkpoints_jsonl(log_dir: str | Path) -> list[dict]:
    path = Path(log_dir) / "checkpoints.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _delete_checkpoint_by_name(name: str, log_path: str) -> None:
    import tinker
    rows = _read_checkpoints_jsonl(log_path)
    existing = [r for r in rows if r.get("name") == name]
    if not existing:
        logger.warning("  No entry named '%s' in checkpoints.jsonl — nothing to delete", name)
        return
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    for key in ("state_path", "sampler_path"):
        tp = existing[-1].get(key)
        if tp:
            try:
                rest_client.delete_checkpoint_from_tinker_path(tp).result()
                logger.info("  Deleted %s", tp)
            except Exception:
                logger.warning("  Failed to delete %s (may already be gone)", tp, exc_info=True)


def _delete_prev_best_checkpoints(log_path: str, current_name: str) -> None:
    rows = _read_checkpoints_jsonl(log_path)
    prev_names = {
        r["name"] for r in rows
        if r.get("name", "").startswith("best_s") and r["name"] != current_name
    }
    for old_name in prev_names:
        logger.info("  Deleting old best checkpoint '%s'", old_name)
        _delete_checkpoint_by_name(old_name, log_path)


def _save_checkpoint_safe(checkpoint_utils, training_client, name: str,
                           log_path: str, loop_state: dict) -> None:
    """Exact mirror of train.py's _save_checkpoint_safe."""
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=name,
        log_path=log_path,
        kind="both",
        loop_state=loop_state,
    )


def _get_best_sampler_path(log_dir: str | Path) -> str:
    """Mirror of checkpoint_io.get_best_sampler_path."""
    rows = _read_checkpoints_jsonl(log_dir)
    candidates = [
        r for r in rows
        if r.get("name", "").startswith("best_s")
        and "sampler_path" in r
        and "val_leakage_rate" in r
    ]
    if not candidates:
        return ""
    best = min(candidates, key=lambda r: r["val_leakage_rate"])
    return best["sampler_path"]


# ── assertion helpers ─────────────────────────────────────────────────────────

def _tinker_checkpoint_exists(rest_client, tinker_path: str) -> bool:
    try:
        result = rest_client.list_user_checkpoints().result()
        live_paths = {ckpt.tinker_path for ckpt in result.checkpoints}
        return tinker_path in live_paths
    except Exception as e:
        logger.warning("  list_user_checkpoints failed: %s", e)
        return False


def _assert_exists(rest_client, tinker_path: str, label: str) -> None:
    assert _tinker_checkpoint_exists(rest_client, tinker_path), (
        f"[{label}] FAIL — expected {tinker_path} to be on Tinker but it is gone"
    )
    logger.info("  ASSERT OK [%s] — %s exists on Tinker", label, tinker_path)


def _assert_gone(rest_client, tinker_path: str, label: str) -> None:
    assert not _tinker_checkpoint_exists(rest_client, tinker_path), (
        f"[{label}] FAIL — expected {tinker_path} to be deleted but it still exists"
    )
    logger.info("  ASSERT OK [%s] — %s deleted from Tinker", label, tinker_path)


def _assert_entry_count(log_dir, prefix: str, expected: int, label: str) -> None:
    rows = _read_checkpoints_jsonl(log_dir)
    matches = [r for r in rows if r.get("name", "").startswith(prefix)]
    assert len(matches) == expected, (
        f"[{label}] FAIL — expected {expected} entries starting with '{prefix}', "
        f"got {len(matches)}: {[r['name'] for r in matches]}"
    )
    logger.info("  ASSERT OK [%s] — %d '%s*' entries in checkpoints.jsonl", label, expected, prefix)


# ── main test ─────────────────────────────────────────────────────────────────

def run_test() -> None:
    import tinker
    from tinker_cookbook import checkpoint_utils

    logger.info("=" * 64)
    logger.info("  Step-stamped checkpoint renewal — end-to-end test")
    logger.info("=" * 64)

    logger.info("\n[SETUP] Creating Tinker LoRA training session...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model="Qwen/Qwen3-30B-A3B", rank=32
    )
    rest_client = service_client.create_rest_client()
    logger.info("  Training session ready")

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_dir = tmp_dir
        saved_paths: dict[str, dict] = {}  # name → {state_path, sampler_path}

        # ── ROUND 1: first best at step 31 ──────────────────────────────────
        logger.info("\n[ROUND 1] First best — step 31 (val_leakage_rate=0.014)...")
        name_r1 = "best_s0031"
        _save_checkpoint_safe(
            checkpoint_utils, training_client, name=name_r1,
            log_path=log_dir, loop_state={"step": 31, "val_leakage_rate": 0.014},
        )
        _delete_prev_best_checkpoints(log_dir, current_name=name_r1)

        rows = _read_checkpoints_jsonl(log_dir)
        r1 = next(r for r in rows if r.get("name") == name_r1)
        saved_paths[name_r1] = r1

        _assert_entry_count(log_dir, "best_s", 1, "R1")
        _assert_exists(rest_client, r1["state_path"], "R1 state_path alive")
        _assert_exists(rest_client, r1["sampler_path"], "R1 sampler_path alive")

        best = _get_best_sampler_path(log_dir)
        assert best == r1["sampler_path"], f"R1: get_best_sampler_path returned '{best}', expected '{r1['sampler_path']}'"
        logger.info("  get_best_sampler_path → correct")
        logger.info("  ROUND 1 PASSED\n")

        # ── ROUND 2: improved best at step 63 ───────────────────────────────
        logger.info("[ROUND 2] New best — step 63 (val_leakage_rate=0.009)...")
        name_r2 = "best_s0063"
        _save_checkpoint_safe(
            checkpoint_utils, training_client, name=name_r2,
            log_path=log_dir, loop_state={"step": 63, "val_leakage_rate": 0.009},
        )
        _delete_prev_best_checkpoints(log_dir, current_name=name_r2)

        rows = _read_checkpoints_jsonl(log_dir)
        r2 = next(r for r in rows if r.get("name") == name_r2)
        saved_paths[name_r2] = r2

        _assert_entry_count(log_dir, "best_s", 2, "R2")   # stale r1 entry stays in file
        _assert_gone(rest_client,   r1["state_path"],   "R2 r1 state_path gone")
        _assert_gone(rest_client,   r1["sampler_path"], "R2 r1 sampler_path gone")
        _assert_exists(rest_client, r2["state_path"],   "R2 r2 state_path alive")
        _assert_exists(rest_client, r2["sampler_path"], "R2 r2 sampler_path alive")

        best = _get_best_sampler_path(log_dir)
        assert best == r2["sampler_path"], f"R2: get_best_sampler_path wrong: '{best}'"
        logger.info("  get_best_sampler_path → correct")
        logger.info("  ROUND 2 PASSED\n")

        # ── ROUND 3: improved best at step 95 ───────────────────────────────
        logger.info("[ROUND 3] New best — step 95 (val_leakage_rate=0.005)...")
        name_r3 = "best_s0095"
        _save_checkpoint_safe(
            checkpoint_utils, training_client, name=name_r3,
            log_path=log_dir, loop_state={"step": 95, "val_leakage_rate": 0.005},
        )
        _delete_prev_best_checkpoints(log_dir, current_name=name_r3)

        rows = _read_checkpoints_jsonl(log_dir)
        r3 = next(r for r in rows if r.get("name") == name_r3)

        _assert_entry_count(log_dir, "best_s", 3, "R3")
        _assert_gone(rest_client,   r2["state_path"],   "R3 r2 state_path gone")
        _assert_gone(rest_client,   r2["sampler_path"], "R3 r2 sampler_path gone")
        _assert_exists(rest_client, r3["state_path"],   "R3 r3 state_path alive")
        _assert_exists(rest_client, r3["sampler_path"], "R3 r3 sampler_path alive")

        best = _get_best_sampler_path(log_dir)
        assert best == r3["sampler_path"], f"R3: get_best_sampler_path wrong: '{best}'"
        logger.info("  get_best_sampler_path → correct")
        logger.info("  ROUND 3 PASSED\n")

        # ── SESSION HEALTH: a 4th save must still succeed (not poisoned) ────
        logger.info("[HEALTH] Confirming training session is NOT poisoned (4th save)...")
        name_r4 = "best_s0127"
        _save_checkpoint_safe(
            checkpoint_utils, training_client, name=name_r4,
            log_path=log_dir, loop_state={"step": 127, "val_leakage_rate": 0.003},
        )
        _delete_prev_best_checkpoints(log_dir, current_name=name_r4)

        rows = _read_checkpoints_jsonl(log_dir)
        r4 = next(r for r in rows if r.get("name") == name_r4)
        _assert_gone(rest_client,   r3["state_path"],   "Health r3 gone")
        _assert_exists(rest_client, r4["state_path"],   "Health r4 alive")
        logger.info("  SESSION HEALTH PASSED — 4 saves, no poisoning\n")

        # ── CLEANUP ──────────────────────────────────────────────────────────
        logger.info("[CLEANUP] Deleting surviving remote checkpoints...")
        for key in ("state_path", "sampler_path"):
            tp = r4.get(key)
            if tp:
                try:
                    rest_client.delete_checkpoint_from_tinker_path(tp).result()
                    logger.info("  Deleted %s", tp)
                except Exception:
                    logger.warning("  Could not delete %s", tp, exc_info=True)

    logger.info("=" * 64)
    logger.info("  ALL TESTS PASSED — step-stamped renewal is safe")
    logger.info("=" * 64)


if __name__ == "__main__":
    import os
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
    os.environ.setdefault("TINKER_API_KEY", os.getenv("TINKER_API_KEY", ""))
    run_test()
