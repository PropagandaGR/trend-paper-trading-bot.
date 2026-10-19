import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import research_engine


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


class OOSLedgerGitPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bare = self.root / "origin.git"
        self.seed = self.root / "seed"
        self.first = self.root / "first"
        self.second = self.root / "second"
        git(self.root, "init", "--bare", str(self.bare))
        git(self.root, "init", str(self.seed))
        git(self.seed, "config", "user.name", "OOS test")
        git(self.seed, "config", "user.email", "oos-test@example.invalid")
        ledger_path = self.seed / "state" / "oos_evaluation_ledger.json"
        ledger_path.parent.mkdir(parents=True)
        ledger_path.write_text(json.dumps({
            "policy_version": "chronological-oos-v2", "windows": [],
            "last_test_start": None, "last_test_end": None, "last_result": None,
        }))
        git(self.seed, "add", "state/oos_evaluation_ledger.json")
        git(self.seed, "commit", "-m", "Seed empty OOS ledger")
        git(self.seed, "branch", "-M", "main")
        git(self.seed, "remote", "add", "origin", str(self.bare))
        git(self.seed, "push", "-u", "origin", "main")
        git(self.bare, "symbolic-ref", "HEAD", "refs/heads/main")
        git(self.root, "clone", str(self.bare), str(self.first))
        git(self.root, "clone", str(self.bare), str(self.second))

    def tearDown(self):
        self.temporary.cleanup()

    def test_remote_branch_reservation_is_durable_and_second_clone_cannot_reuse_it(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
        frame = pd.DataFrame({"close": range(12)}, index=index)
        env = patch.dict(os.environ, {"OOS_LEDGER_BRANCH": "oos-ledger"})
        with (
            env,
            patch.object(research_engine, "ROOT", self.first),
            patch.object(research_engine, "STATE", self.first / "state"),
        ):
            ok, reason, window = research_engine.reserve_oos_test_window(frame)
            self.assertTrue(ok, reason)

        persisted = json.loads(git(
            self.root, "--git-dir", str(self.bare), "show",
            "refs/heads/oos-ledger:state/oos_evaluation_ledger.json"
        ))
        self.assertEqual(persisted["windows"][0]["id"], window["id"])
        self.assertEqual(persisted["windows"][0]["status"], "reserved")

        # This checkout predates the reservation. The updater must fetch the
        # authoritative branch and reject the overlapping period before evaluation.
        with (
            patch.dict(os.environ, {"OOS_LEDGER_BRANCH": "oos-ledger"}),
            patch.object(research_engine, "ROOT", self.second),
            patch.object(research_engine, "STATE", self.second / "state"),
        ):
            ok, reason, _ = research_engine.reserve_oos_test_window(frame)
            self.assertFalse(ok)
            self.assertIn("overlaps or reuses", reason)

        with (
            patch.dict(os.environ, {"OOS_LEDGER_BRANCH": "oos-ledger"}),
            patch.object(research_engine, "ROOT", self.first),
            patch.object(research_engine, "STATE", self.first / "state"),
        ):
            research_engine.record_oos_test_result(False, window_id=window["id"])

        persisted = json.loads(git(
            self.root, "--git-dir", str(self.bare), "show",
            "refs/heads/oos-ledger:state/oos_evaluation_ledger.json"
        ))
        self.assertEqual(len(persisted["windows"]), 1)
        self.assertEqual(persisted["windows"][0]["status"], "failed")
        self.assertEqual(persisted["last_result"], "failed")


if __name__ == "__main__":
    unittest.main()
