"""Checkpoint I/O utilities for Tinker remote checkpoints.

The ``tinker_cookbook.checkpoint_utils.save_checkpoint`` helper writes
checkpoint metadata (including ``sampler_path`` and ``state_path``) to a
``checkpoints.jsonl`` file under the log directory.  Paths look like
``tinker://<run_id>/sampler_weights/<name>`` — they are **remote URIs**,
not local filesystem paths.

This module provides helpers to:
1. Read back the ``sampler_path`` for a named checkpoint (used by
   the evaluation pipeline and the orchestration shell script).
2. Delete all remote checkpoints via the Tinker REST API (used by
   the orchestration script after evaluation is complete).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_checkpoints_jsonl(log_dir: str | Path) -> list[dict[str, Any]]:
    """Read all entries from ``checkpoints.jsonl`` under *log_dir*."""
    path = Path(log_dir) / "checkpoints.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_sampler_path(log_dir: str | Path, name: str) -> str:
    """Return the ``sampler_path`` for checkpoint *name*, or empty string."""
    rows = read_checkpoints_jsonl(log_dir)
    matches = [r for r in rows if r.get("name") == name and "sampler_path" in r]
    if not matches:
        return ""
    return matches[-1]["sampler_path"]


def get_best_sampler_path(log_dir: str | Path) -> str:
    """Return the ``sampler_path`` of the checkpoint with the lowest val_leakage_rate.

    Best checkpoints are saved with step-stamped names (e.g. ``best_s0031``).
    This function finds the entry across all such names that has the minimum
    recorded ``val_leakage_rate`` and returns its ``sampler_path``.

    Returns empty string if no qualifying checkpoint exists.
    """
    rows = read_checkpoints_jsonl(log_dir)
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


def get_best_state_path(log_dir: str | Path) -> str:
    """Return the ``state_path`` of the checkpoint with the lowest val_leakage_rate."""
    rows = read_checkpoints_jsonl(log_dir)
    candidates = [
        r for r in rows
        if r.get("name", "").startswith("best_s")
        and "state_path" in r
        and "val_leakage_rate" in r
    ]
    if not candidates:
        return ""
    best = min(candidates, key=lambda r: r["val_leakage_rate"])
    return best["state_path"]


def delete_remote_checkpoints(log_dir: str | Path) -> list[str]:
    """Delete every checkpoint referenced in *log_dir*/checkpoints.jsonl.

    Uses ``rest_client.delete_checkpoint_from_tinker_path`` for each
    unique ``sampler_path`` and ``state_path`` found.

    Returns:
        List of deleted tinker paths.
    """
    import tinker

    rows = read_checkpoints_jsonl(log_dir)
    paths_to_delete: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in ("sampler_path", "state_path"):
            p = row.get(key, "")
            if p and p not in seen:
                seen.add(p)
                paths_to_delete.append(p)

    if not paths_to_delete:
        logger.info("No remote checkpoints to delete in %s", log_dir)
        return []

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    deleted: list[str] = []
    for tp in paths_to_delete:
        try:
            rest_client.delete_checkpoint_from_tinker_path(tp).result()
            deleted.append(tp)
            logger.info("Deleted %s", tp)
        except Exception:
            logger.warning("Failed to delete %s (may already be gone)", tp, exc_info=True)
    return deleted


def main() -> None:
    """CLI: read a checkpoint path or delete all checkpoints."""
    parser = argparse.ArgumentParser(description="Tinker checkpoint I/O helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_read = sub.add_parser("read", help="Print sampler_path for a named checkpoint.")
    p_read.add_argument("--log_dir", required=True)
    p_read.add_argument("--name", required=True, help='Checkpoint name, e.g. "final".')

    p_best = sub.add_parser(
        "read_best",
        help="Print sampler_path of the checkpoint with the lowest val_leakage_rate.",
    )
    p_best.add_argument("--log_dir", required=True)

    p_del = sub.add_parser("delete", help="Delete all remote checkpoints for a run.")
    p_del.add_argument("--log_dir", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "read":
        path = get_sampler_path(args.log_dir, args.name)
        print(path)
    elif args.cmd == "read_best":
        path = get_best_sampler_path(args.log_dir)
        print(path)
    elif args.cmd == "delete":
        from .config import load_env
        project_root = Path(__file__).resolve().parents[2]
        load_env(project_root)
        os.environ.setdefault("TINKER_API_KEY", os.getenv("TINKER_API_KEY", ""))
        deleted = delete_remote_checkpoints(args.log_dir)
        print(f"Deleted {len(deleted)} checkpoint(s)")


if __name__ == "__main__":
    main()
