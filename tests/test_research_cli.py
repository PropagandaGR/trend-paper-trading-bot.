import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import research_engine


class ResearchCliTests(unittest.TestCase):
    def _write_report(self, reports: Path, *, status="no_candidate", evaluations=12):
        reports.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        champion = None
        champion_version = None
        validated_candidates = 0
        if status == "validated":
            champion_version = "champion-test123456"
            validated_candidates = 1
            champion = {
                "symbol": "BTCUSDT", "interval": "5m", "strategy": "trend",
                "params": {"fast": 8, "slow": 30, "momentum": 6, "breakout": 12},
                "champion_version": champion_version,
                "validation_metadata": {
                    "created_at": now, "status": "validated",
                    "test_used_for_selection": False,
                    "validation_metrics": {"total_return_pct": 2.0},
                    "validation_stress_metrics": {"total_return_pct": 1.0},
                    "test_metrics": {"total_return_pct": -1.0},
                    "test_stress_metrics": {"total_return_pct": -2.0},
                    "cost_convention": {"fee_basis": "executed notional"},
                },
            }
        report = {
            "started_at": now,
            "completed_at": now,
            "run_status": "completed",
            "validation_status": status,
            "validation_result": {
                "status": status,
                "evaluations": evaluations,
                "validated_candidates": validated_candidates,
                "champion_version": champion_version,
                "reason": "No candidate passed validation gates" if status == "no_candidate" else None,
            },
            "evaluations": evaluations,
            "validated_count": validated_candidates,
            "champion": champion,
        }
        (reports / "research_latest.json").write_text(json.dumps(report))
        if champion is not None:
            state_dir = reports.parent / "state"
            snapshot_dir = state_dir / "champions"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "champion.json").write_text(json.dumps(champion))
            (snapshot_dir / f"{champion_version}.json").write_text(json.dumps(champion))
        return report

    def test_real_cli_entry_point_supports_help(self):
        result = subprocess.run(
            [sys.executable, str(Path(research_engine.__file__)), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--days", result.stdout)
        self.assertIn("--require-fresh-report", result.stdout)

    def test_cli_fails_when_no_candidate_passes_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(research_engine, "REPORTS", reports):
                with patch.object(
                    research_engine, "run_research",
                    side_effect=lambda days: self._write_report(reports, status="no_candidate"),
                ) as run:
                    self.assertEqual(research_engine.main(["--days", "30", "--require-fresh-report"]), 1)
                    run.assert_called_once_with(days=30)

    def test_cli_accepts_fresh_report_with_valid_champion_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            state_dir = Path(tmp) / "state"
            with patch.object(research_engine, "REPORTS", reports):
                with patch.object(research_engine, "STATE", state_dir):
                    with patch.object(
                        research_engine, "run_research",
                        side_effect=lambda days: self._write_report(reports, status="validated"),
                    ) as run:
                        self.assertEqual(research_engine.main(["--days", "30", "--require-fresh-report"]), 0)
                        run.assert_called_once_with(days=30)

    def test_cli_fails_if_research_does_not_write_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(research_engine, "REPORTS", reports):
                with patch.object(research_engine, "run_research", return_value={}):
                    self.assertEqual(research_engine.main(["--days", "30", "--require-fresh-report"]), 1)

    def test_cli_fails_if_report_has_no_candidate_evaluations(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            with patch.object(research_engine, "REPORTS", reports):
                with patch.object(
                    research_engine, "run_research",
                    side_effect=lambda days: self._write_report(reports, status="insufficient_data", evaluations=0),
                ):
                    self.assertEqual(research_engine.main(["--days", "30", "--require-fresh-report"]), 1)

    def test_artifact_gate_rejects_stale_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "research_latest.json"
            now = datetime.now(timezone.utc).isoformat()
            report_path.write_text(json.dumps({
                "started_at": now, "completed_at": now, "run_status": "completed",
                "validation_status": "no_candidate",
                "validation_result": {"status": "no_candidate", "evaluations": 1},
                "evaluations": 1, "champion": None,
            }))
            invocation_started = datetime.now(timezone.utc) + __import__("datetime").timedelta(seconds=5)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                research_engine.validate_research_artifacts(invocation_started, report_path)


    def test_artifact_gate_rejects_missing_active_champion_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            state_dir = Path(tmp) / "state"
            self._write_report(reports, status="validated")
            (state_dir / "champion.json").unlink()
            with patch.object(research_engine, "STATE", state_dir):
                with self.assertRaisesRegex(RuntimeError, "Active champion artifact is missing"):
                    research_engine.validate_research_artifacts(
                        datetime.now(timezone.utc), reports / "research_latest.json"
                    )

    def test_artifact_gate_rejects_mismatched_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            state_dir = Path(tmp) / "state"
            report = self._write_report(reports, status="validated")
            snapshot_path = state_dir / "champions" / f'{report["champion"]["champion_version"]}.json'
            snapshot_path.write_text(json.dumps({"champion_version": "champion-wrongversion"}))
            with patch.object(research_engine, "STATE", state_dir):
                with self.assertRaisesRegex(RuntimeError, "snapshot does not match"):
                    research_engine.validate_research_artifacts(
                        datetime.now(timezone.utc), reports / "research_latest.json"
                    )

    def test_validated_report_requires_versioned_champion_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            report = self._write_report(reports, status="validated")
            report["champion"] = {"symbol": "ETHUSDT"}
            report["validation_result"]["champion_version"] = "champion-bad"
            (reports / "research_latest.json").write_text(json.dumps(report))
            invocation_started = datetime.now(timezone.utc)
            with self.assertRaisesRegex(RuntimeError, "version is missing or invalid"):
                research_engine.validate_research_artifacts(invocation_started, reports / "research_latest.json")


if __name__ == "__main__":
    unittest.main()
