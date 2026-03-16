"""Local PEFT adapter checkpoint I/O utilities.

Replaces Tinker remote GCS checkpoint management with local filesystem
operations. Checkpoint metadata is stored in ``checkpoints.jsonl`` under
the log directory, matching the CoverageGatedExperiment format.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_checkpoints_jsonl(log_dir: str | Path) -> list[dict[str, Any]]:
    """Read all entries from checkpoints.jsonl under *log_dir*."""
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


def save_checkpoint(
    trainer: Any,
    name: str,
    log_dir: str | Path,
    val_metrics: dict[str, Any] | None = None,
    loop_state: dict[str, Any] | None = None,
) -> Path:
    """Save PEFT adapter checkpoint and record metadata.

    Args:
        trainer: TRL GRPOTrainer instance.
        name: Checkpoint name (e.g. "best_s0031", "final").
        log_dir: Directory containing checkpoints.jsonl.
        val_metrics: Optional validation metrics to record.
        loop_state: Optional training state (step, epoch).

    Returns:
        Path to the saved adapter directory.
    """
    log_dir = Path(log_dir)
    adapter_path = log_dir / name
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_path))

    entry: dict[str, Any] = {
        "name": name,
        "adapter_path": str(adapter_path),
    }
    if val_metrics:
        entry.update(val_metrics)
    if loop_state:
        entry.update(loop_state)

    with open(log_dir / "checkpoints.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Saved checkpoint '%s' to %s", name, adapter_path)
    return adapter_path


def get_best_checkpoint_path(log_dir: str | Path) -> str:
    """Return adapter_path of the checkpoint with the lowest val_leakage_rate.

    Returns empty string if no qualifying checkpoint exists.
    """
    rows = read_checkpoints_jsonl(log_dir)
    candidates = [
        r for r in rows
        if r.get("name", "").startswith("best_s")
        and "adapter_path" in r
        and "val_leakage_rate" in r
    ]
    if not candidates:
        return ""
    best = min(candidates, key=lambda r: r["val_leakage_rate"])
    return best["adapter_path"]


def get_checkpoint_path(log_dir: str | Path, name: str) -> str:
    """Return adapter_path for a named checkpoint, or empty string."""
    rows = read_checkpoints_jsonl(log_dir)
    matches = [r for r in rows if r.get("name") == name and "adapter_path" in r]
    if not matches:
        return ""
    return matches[-1]["adapter_path"]


def delete_checkpoints(log_dir: str | Path) -> list[str]:
    """Delete all local adapter directories referenced in checkpoints.jsonl.

    Returns:
        List of deleted paths.
    """
    rows = read_checkpoints_jsonl(log_dir)
    deleted: list[str] = []
    seen: set[str] = set()
    for row in rows:
        p = row.get("adapter_path", "")
        if p and p not in seen:
            seen.add(p)
            path = Path(p)
            if path.exists():
                shutil.rmtree(path)
                deleted.append(p)
                logger.info("Deleted %s", p)
            else:
                logger.warning("Already gone: %s", p)
    return deleted


def main() -> None:
    """CLI: read a checkpoint path or delete all checkpoints."""
    parser = argparse.ArgumentParser(description="Local PEFT checkpoint I/O.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_read = sub.add_parser("read", help="Print adapter_path for a named checkpoint.")
    p_read.add_argument("--log_dir", required=True)
    p_read.add_argument("--name", required=True)

    p_best = sub.add_parser("read_best", help="Print adapter_path of best checkpoint.")
    p_best.add_argument("--log_dir", required=True)

    p_del = sub.add_parser("delete", help="Delete all local checkpoints for a run.")
    p_del.add_argument("--log_dir", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "read":
        print(get_checkpoint_path(args.log_dir, args.name))
    elif args.cmd == "read_best":
        print(get_best_checkpoint_path(args.log_dir))
    elif args.cmd == "delete":
        deleted = delete_checkpoints(args.log_dir)
        print(f"Deleted {len(deleted)} checkpoint(s)")


if __name__ == "__main__":
    main()
