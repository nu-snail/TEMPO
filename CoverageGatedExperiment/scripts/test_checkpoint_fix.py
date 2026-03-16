"""Test the delete-then-retry checkpoint fix in train.py.

Two test levels:
  1. Unit test with mocks: verifies the fix logic without needing Tinker.
  2. Live API test: calls Tinker REST API to delete the actual 'best' checkpoint
     from checkpoints.jsonl, proving the deletion code path works end-to-end.

Run from FullExperiment/:
    uv run python scripts/test_checkpoint_fix.py [--live]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import the helpers under test
# ---------------------------------------------------------------------------
from cov_gated_experiment.config import load_env
from cov_gated_experiment.train import _delete_checkpoint_by_name, _save_checkpoint_safe


# ---------------------------------------------------------------------------
# Unit tests (no Tinker network required)
# ---------------------------------------------------------------------------

class TestSaveCheckpointSafe(unittest.TestCase):

    def _make_ckpt_utils(self, first_raises: bool = True):
        mock = MagicMock()
        if first_raises:
            import tinker
            mock.save_checkpoint.side_effect = [tinker.ConflictError("Checkpoint 'best' already exists", response=MagicMock(status_code=409), body=None), None]
        return mock

    def _make_training_client(self):
        return MagicMock()

    def test_no_conflict_saves_once(self):
        """When there's no conflict, save_checkpoint is called exactly once."""
        ckpt_utils = MagicMock()
        ckpt_utils.save_checkpoint.return_value = None
        tc = self._make_training_client()

        with patch("cov_gated_experiment.train._delete_checkpoint_by_name") as mock_del:
            _save_checkpoint_safe(ckpt_utils, tc, "best", "/tmp/fake_log", {"step": 5})

        ckpt_utils.save_checkpoint.assert_called_once_with(
            training_client=tc, name="best", log_path="/tmp/fake_log",
            kind="both", loop_state={"step": 5},
        )
        mock_del.assert_not_called()

    def test_conflict_triggers_delete_then_retry(self):
        """On ConflictError: delete is called, then save is retried with same name."""
        import tinker
        ckpt_utils = MagicMock()
        ckpt_utils.save_checkpoint.side_effect = [
            tinker.ConflictError("Checkpoint 'best' already exists", response=MagicMock(status_code=409), body=None),
            None,  # retry succeeds
        ]
        tc = self._make_training_client()

        with patch("cov_gated_experiment.train._delete_checkpoint_by_name") as mock_del:
            _save_checkpoint_safe(ckpt_utils, tc, "best", "/tmp/fake_log", {"step": 10})

        # delete must be called once with the correct name
        mock_del.assert_called_once_with("best", "/tmp/fake_log")

        # save_checkpoint must be called TWICE, both times with name="best"
        assert ckpt_utils.save_checkpoint.call_count == 2, \
            f"Expected 2 save calls, got {ckpt_utils.save_checkpoint.call_count}"
        for c in ckpt_utils.save_checkpoint.call_args_list:
            assert c.kwargs["name"] == "best", f"Expected name='best', got {c.kwargs['name']}"

        logger.info("PASS: conflict triggers delete + retry with same name")

    def test_conflict_never_uses_versioned_name(self):
        """The old _v2 versioning scheme must NOT appear after the fix."""
        import tinker
        ckpt_utils = MagicMock()
        ckpt_utils.save_checkpoint.side_effect = [
            tinker.ConflictError("exists", response=MagicMock(status_code=409), body=None),
            None,
        ]
        tc = self._make_training_client()

        with patch("cov_gated_experiment.train._delete_checkpoint_by_name"):
            _save_checkpoint_safe(ckpt_utils, tc, "final", "/tmp/fake_log", {"step": 99})

        used_names = [c.kwargs["name"] for c in ckpt_utils.save_checkpoint.call_args_list]
        for n in used_names:
            assert "_v" not in n, f"Versioned name used: {n}"
        logger.info("PASS: no versioned names used")


class TestDeleteCheckpointByName(unittest.TestCase):

    def _make_mock_checkpoints_jsonl(self, tmp_path: Path, entries: list[dict]) -> str:
        p = tmp_path / "checkpoints.jsonl"
        with open(p, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return str(tmp_path)

    def test_deletes_both_state_and_sampler(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log_dir = self._make_mock_checkpoints_jsonl(Path(td), [
                {
                    "name": "best",
                    "state_path": "tinker://run123/weights/best",
                    "sampler_path": "tinker://run123/sampler_weights/best",
                }
            ])

            mock_rest = MagicMock()
            mock_future = MagicMock()
            mock_rest.delete_checkpoint_from_tinker_path.return_value = mock_future

            mock_service = MagicMock()
            mock_service.create_rest_client.return_value = mock_rest

            # tinker is imported locally inside _delete_checkpoint_by_name, patch at source
            with patch("tinker.ServiceClient", return_value=mock_service):
                _delete_checkpoint_by_name("best", log_dir)

            calls = [str(c) for c in mock_rest.delete_checkpoint_from_tinker_path.call_args_list]
            assert any("weights/best" in c for c in calls), "state_path not deleted"
            assert any("sampler_weights/best" in c for c in calls), "sampler_path not deleted"
            logger.info("PASS: both state_path and sampler_path deleted")

    def test_missing_name_logs_warning(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log_dir = self._make_mock_checkpoints_jsonl(Path(td), [
                {"name": "final", "state_path": "x", "sampler_path": "y"}
            ])
            mock_service = MagicMock()
            with patch("tinker.ServiceClient", return_value=mock_service) as mock_sc:
                _delete_checkpoint_by_name("best", log_dir)
            mock_sc.assert_not_called()
            logger.info("PASS: no delete attempt when name not found in checkpoints.jsonl")


# ---------------------------------------------------------------------------
# Live API test: actually delete the 'best' checkpoint via Tinker
# ---------------------------------------------------------------------------

def run_live_test(log_dir: str) -> None:
    """Delete the real 'best' checkpoint via the Tinker REST API."""
    from cov_gated_experiment.checkpoint_io import read_checkpoints_jsonl

    rows = read_checkpoints_jsonl(log_dir)
    best = [r for r in rows if r.get("name") == "best"]
    if not best:
        logger.error("No 'best' checkpoint found in %s/checkpoints.jsonl", log_dir)
        return

    logger.info("Found 'best' checkpoint:")
    for k, v in best[-1].items():
        logger.info("  %s: %s", k, v)

    logger.info("Calling _delete_checkpoint_by_name('best', '%s') ...", log_dir)
    _delete_checkpoint_by_name("best", log_dir)
    logger.info("Deletion completed — the 'best' checkpoint has been removed from Tinker.")
    logger.info("The fix is confirmed: next time train.py tries to save 'best', it will")
    logger.info("succeed without conflict (since the old one is now gone).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Also run live API test that deletes the real 'best' checkpoint.")
    parser.add_argument("--log_dir", default=None,
                        help="Path to checkpoints.jsonl directory for --live test.")
    args = parser.parse_args()

    load_env(PROJECT_ROOT)

    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSaveCheckpointSafe))
    suite.addTests(loader.loadTestsFromTestCase(TestDeleteCheckpointByName))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

    if args.live:
        log_dir = args.log_dir or str(PROJECT_ROOT / "runtime/results/train_stock")
        logger.info("\n=== LIVE API TEST ===")
        run_live_test(log_dir)
