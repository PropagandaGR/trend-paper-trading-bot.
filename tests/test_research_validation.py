import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import research_engine


class ChronologicalSplitTests(unittest.TestCase):
    def test_splits_are_ordered_disjoint_and_60_20_20(self):
        index = pd.date_range("2026-01-01", periods=1000, freq="5min", tz="UTC")
        frame = pd.DataFrame({"close": range(1000)}, index=index)
        train, validation, test = research_engine.chronological_split(frame)
        self.assertEqual((len(train), len(validation), len(test)), (600, 200, 200))
        self.assertLess(train.index[-1], validation.index[0])
        self.assertLess(validation.index[-1], test.index[0])
        self.assertEqual(train.index[0], frame.index[0])
        self.assertEqual(test.index[-1], frame.index[-1])

    def test_unsorted_or_duplicate_timestamps_are_rejected(self):
        index = pd.to_datetime(["2026-01-01T00:05Z", "2026-01-01T00:00Z", "2026-01-01T00:05Z"])
        frame = pd.DataFrame({"close": [1, 2, 3]}, index=index)
        with self.assertRaises(ValueError):
            research_engine.chronological_split(frame)


class TestSetIsolationTests(unittest.TestCase):
    def test_negative_final_test_does_not_change_validation_selected_champion(self):
        index = pd.date_range("2026-01-01", periods=1200, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "open": [100.0] * len(index), "high": [101.0] * len(index),
            "low": [99.0] * len(index), "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
            "close_time": index + pd.Timedelta(minutes=5),
        }, index=index)
        params = {"fast": 2, "slow": 10, "momentum": 2, "breakout": 5}

        def fake_backtest(part, strategy, candidate, fee_bps=10, slippage_bps=5, interval="5m", starting_cash=15):
            if part.index[0] >= index[960]:
                return {"total_return_pct": -50.0, "max_drawdown_pct": 60.0, "sharpe": -2.0, "trades": 4}
            if part.index[0] >= index[720]:
                if fee_bps >= 20:
                    return {"total_return_pct": 2.0, "max_drawdown_pct": 5.0, "sharpe": 1.0, "trades": 4}
                return {"total_return_pct": 10.0, "max_drawdown_pct": 5.0, "sharpe": 2.0, "trades": 4}
            return {"total_return_pct": 3.0, "max_drawdown_pct": 3.0, "sharpe": 0.5, "trades": 3}

        with tempfile.TemporaryDirectory() as tmp:
            reports, state = Path(tmp) / "reports", Path(tmp) / "state"
            with (
                patch.object(research_engine, "REPORTS", reports),
                patch.object(research_engine, "STATE", state),
                patch.object(research_engine, "fetch_klines", return_value=frame),
                patch.object(research_engine, "resample_ohlcv", side_effect=lambda raw, rule: raw),
                patch.object(research_engine, "candidate_params", return_value=[params]),
                patch.object(research_engine, "backtest", side_effect=fake_backtest) as backtest,
            ):
                summary = research_engine.run_research(days=30)

            self.assertIsNotNone(summary["champion"])
            self.assertEqual(summary["champion"]["validation_metadata"]["test_metrics"]["total_return_pct"], -50.0)
            self.assertFalse(summary["champion"]["validation_metadata"]["test_used_for_selection"])
            self.assertEqual(summary["champion"]["validation_metadata"]["selection_metric"], "validation_score")
            self.assertTrue(summary["champion"]["champion_version"].startswith("champion-"))
            self.assertEqual(summary["champion"]["validation_metadata"]["periods"]["train"]["bars"], 720)
            self.assertEqual(summary["champion"]["validation_metadata"]["periods"]["validation"]["bars"], 240)
            self.assertEqual(summary["champion"]["validation_metadata"]["periods"]["test"]["bars"], 240)
            # All candidate evaluations use train/validation only. The test split
            # is passed to backtest only twice for the selected champion (base + stress).
            test_calls = [call for call in backtest.call_args_list if call.args[0].index[0] >= index[960]]
            self.assertEqual(len(test_calls), 2)
            self.assertTrue((state / "champion.json").exists())
            history_file = state / "champions" / (summary["champion"]["champion_version"] + ".json")
            self.assertTrue(history_file.exists())
            self.assertEqual(
                json.loads(history_file.read_text())["champion_version"],
                summary["champion"]["champion_version"],
            )


    def test_no_validated_candidate_clears_stale_active_champion(self):
        index = pd.date_range("2026-01-01", periods=1200, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "open": [100.0] * len(index), "high": [101.0] * len(index),
            "low": [99.0] * len(index), "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
            "close_time": index + pd.Timedelta(minutes=5),
        }, index=index)
        params = {"fast": 2, "slow": 10, "momentum": 2, "breakout": 5}

        with tempfile.TemporaryDirectory() as tmp:
            reports, state = Path(tmp) / "reports", Path(tmp) / "state"
            state.mkdir()
            (state / "champion.json").write_text(json.dumps({"symbol": "BTCUSDT", "stale": True}))

            def failed_candidate(*args, **kwargs):
                return {"total_return_pct": -5.0, "max_drawdown_pct": 30.0,
                        "sharpe": -1.0, "trades": 4}

            with (
                patch.object(research_engine, "REPORTS", reports),
                patch.object(research_engine, "STATE", state),
                patch.object(research_engine, "fetch_klines", return_value=frame),
                patch.object(research_engine, "resample_ohlcv", side_effect=lambda raw, rule: raw),
                patch.object(research_engine, "candidate_params", return_value=[params]),
                patch.object(research_engine, "backtest", side_effect=failed_candidate),
            ):
                summary = research_engine.run_research(days=30)

            self.assertIsNone(summary["champion"])
            self.assertIsNone(json.loads((state / "champion.json").read_text()))



class ResearchCliFreshnessTests(unittest.TestCase):
    def _valid_result(self, now):
        stamp = now.isoformat()
        champion = {
            "champion_version": "champion-test123456",
            "validation_metadata": {
                "status": "validated",
                "test_used_for_selection": False,
                "validation_metrics": {"total_return_pct": 2.0},
                "test_metrics": {"total_return_pct": -1.0},
            },
        }
        summary = {
            "started_at": stamp, "completed_at": stamp,
            "evaluations": 12, "validated_count": 1,
            "results": [{"passed_validation": True}], "champion": champion,
        }
        return summary

    def test_fresh_report_with_validation_and_isolated_test_is_accepted(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        summary = self._valid_result(now)
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "research_latest.json"
            report.write_text(json.dumps(summary))
            research_engine.validate_research_output(summary, report, now=now)

    def test_missing_report_fails(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "was not produced"):
                research_engine.validate_research_output(
                    self._valid_result(now), Path(tmp) / "missing.json", now=now
                )

    def test_stale_report_fails(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        summary = self._valid_result(now - timedelta(minutes=30))
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "research_latest.json"
            report.write_text(json.dumps(summary))
            with self.assertRaisesRegex(RuntimeError, "stale"):
                research_engine.validate_research_output(summary, report, now=now)

    def test_no_validated_candidate_fails(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        summary = self._valid_result(now)
        summary["validated_count"] = 0
        summary["champion"] = None
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "research_latest.json"
            report.write_text(json.dumps(summary))
            with self.assertRaisesRegex(RuntimeError, "no candidate that passed validation"):
                research_engine.validate_research_output(summary, report, now=now)


if __name__ == "__main__":
    unittest.main()
