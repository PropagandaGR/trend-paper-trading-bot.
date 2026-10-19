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
    def test_oos_window_is_consumed_once_and_overlap_is_rejected(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
        first = pd.DataFrame({"close": range(12)}, index=index)
        overlap = pd.DataFrame({"close": range(12)}, index=index + pd.Timedelta(minutes=5))
        later = pd.DataFrame({"close": range(12)}, index=index + pd.Timedelta(hours=1))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(research_engine, "STATE", Path(tmp) / "state"):
                ok, reason, window = research_engine.reserve_oos_test_window(first)
                self.assertTrue(ok, reason)
                self.assertEqual(window["start"], index[0].isoformat())
                research_engine.record_oos_test_result(False)
                ok, reason, _ = research_engine.reserve_oos_test_window(first)
                self.assertFalse(ok)
                self.assertIn("reuses", reason)
                ok, reason, _ = research_engine.reserve_oos_test_window(overlap)
                self.assertFalse(ok)
                self.assertIn("overlaps", reason)
                ok, reason, _ = research_engine.reserve_oos_test_window(later)
                self.assertTrue(ok, reason)

    def test_failed_quality_gates_never_reach_backtesting(self):
        index = pd.date_range("2026-01-01", periods=1200, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "open": [100.0] * len(index), "high": [101.0] * len(index),
            "low": [99.0] * len(index), "close": [100.0] * len(index),
            "volume": [1.0] * len(index), "close_time": index + pd.Timedelta(minutes=5),
        }, index=index)
        params = {"fast": 2, "slow": 10, "momentum": 2, "breakout": 5}
        failed_reports = [
            {"quality_gate_passed": False, "invalid_ohlc_rows": 1, "source_coverage_pct": 100.0,
             "converted_coverage_pct": 100.0, "fx_missing_pct_of_source": 0.0, "conflicting_duplicate_groups": 0},
            {"quality_gate_passed": False, "invalid_ohlc_rows": 0, "source_coverage_pct": 100.0,
             "converted_coverage_pct": 98.0, "fx_missing_pct_of_source": 2.0, "conflicting_duplicate_groups": 0},
            {"quality_gate_passed": False, "invalid_ohlc_rows": 0, "source_coverage_pct": 100.0,
             "converted_coverage_pct": 100.0, "fx_missing_pct_of_source": 0.0, "conflicting_duplicate_groups": 1},
            {"quality_gate_passed": False, "invalid_ohlc_rows": 0, "source_coverage_pct": 90.0,
             "converted_coverage_pct": 90.0, "fx_missing_pct_of_source": 0.0, "conflicting_duplicate_groups": 0},
        ]
        for quality in failed_reports:
            with self.subTest(quality=quality), tempfile.TemporaryDirectory() as tmp:
                bad_frame = frame.copy()
                bad_frame.attrs["data_quality"] = quality
                with (
                    patch.object(research_engine, "REPORTS", Path(tmp) / "reports"),
                    patch.object(research_engine, "STATE", Path(tmp) / "state"),
                    patch.object(research_engine, "get_research_batch", return_value=(["BTCUSDT"], {"status": "test"})),
                    patch.object(research_engine, "fetch_klines", return_value=bad_frame),
                    patch.object(research_engine, "candidate_params", return_value=[params]),
                    patch.object(research_engine, "backtest") as backtest,
                ):
                    summary = research_engine.run_research(days=30)
                    backtest.assert_not_called()
                    self.assertEqual(summary["evaluations"], 0)

    def test_negative_final_test_does_not_change_validation_selected_champion(self):
        index = pd.date_range("2026-01-01", periods=1200, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "open": [100.0] * len(index), "high": [101.0] * len(index),
            "low": [99.0] * len(index), "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
            "close_time": index + pd.Timedelta(minutes=5),
        }, index=index)
        frame.attrs["data_quality"] = {
            "quality_gate_passed": True, "source_coverage_pct": 100.0,
            "converted_coverage_pct": 100.0, "invalid_ohlc_rows": 0,
            "duplicate_candles_dropped": 0, "source_was_sorted": True,
        }
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

            self.assertIsNone(summary["champion"])
            self.assertEqual(summary["selected_candidate_test_gate"]["test_metrics"]["total_return_pct"], -50.0)
            self.assertFalse(summary["selected_candidate_test_gate"]["passed"])
            self.assertTrue(summary["selected_candidate_test_gate"]["failures"])
            self.assertIsNone(json.loads((state / "champion.json").read_text()))
            self.assertEqual(summary["selected_candidate_test_gate"]["selection_note"].find("does not trigger selection") >= 0, True)
            self.assertEqual(summary["effective_candidate_counts"],
                             {"trend": 1, "breakout": 1, "mean_reversion": 1})
            self.assertEqual(summary["total_unique_strategy_configs"], 3)
            self.assertEqual(set(summary["strategy_benchmark_summary"]),
                             {"trend", "breakout", "mean_reversion"})
            self.assertEqual(summary["selected_candidate_test_gate"]["candidate"]["strategy"], "trend")
            self.assertEqual(summary["results"][0]["periods"]["train"]["bars"], 720)
            self.assertEqual(summary["results"][0]["periods"]["validation"]["bars"], 240)
            self.assertEqual(summary["results"][0]["periods"]["test"]["bars"], 240)
            # Every candidate is ranked on train/validation only. The held-out test
            # is evaluated only for the validation winner and can veto promotion.
            test_calls = [call for call in backtest.call_args_list if call.args[0].index[0] >= index[960]]
            self.assertEqual(len(test_calls), 2)
            self.assertTrue((state / "champion.json").exists())
            self.assertIsNone(json.loads((state / "champion.json").read_text()))
            self.assertFalse((state / "champions").exists())


    def test_no_validated_candidate_clears_stale_active_champion(self):
        index = pd.date_range("2026-01-01", periods=1200, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "open": [100.0] * len(index), "high": [101.0] * len(index),
            "low": [99.0] * len(index), "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
            "close_time": index + pd.Timedelta(minutes=5),
        }, index=index)
        frame.attrs["data_quality"] = {
            "quality_gate_passed": True, "source_coverage_pct": 100.0,
            "converted_coverage_pct": 100.0, "invalid_ohlc_rows": 0,
            "duplicate_candles_dropped": 0, "source_was_sorted": True,
        }
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

    def test_incomplete_candle_coverage_fails_closed_before_evaluation(self):
        index = pd.date_range("2026-01-01", periods=1200, freq="5min", tz="UTC")
        frame = pd.DataFrame({
            "open": [100.0] * len(index), "high": [101.0] * len(index),
            "low": [99.0] * len(index), "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
            "close_time": index + pd.Timedelta(minutes=5),
        }, index=index)
        frame.attrs["data_quality"] = {
            "quality_gate_passed": False, "source_coverage_pct": 84.5,
            "converted_coverage_pct": 84.0, "invalid_ohlc_rows": 0,
            "duplicate_candles_dropped": 0, "conflicting_duplicate_groups": 0,
            "source_was_sorted": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            reports, state = Path(tmp) / "reports", Path(tmp) / "state"
            with (
                patch.object(research_engine, "REPORTS", reports),
                patch.object(research_engine, "STATE", state),
                patch.object(research_engine, "get_research_batch",
                             return_value=(["BTCUSDT"], {"status": "test"})),
                patch.object(research_engine, "fetch_klines", return_value=frame),
                patch.object(research_engine, "backtest") as backtest,
            ):
                summary = research_engine.run_research(days=30)

            backtest.assert_not_called()
            self.assertEqual(summary["evaluations"], 0)
            self.assertIsNone(summary["champion"])
            self.assertFalse(summary["data_quality"]["BTCUSDT"]["quality_gate_passed"])
            self.assertTrue(any("quality gate failed" in item["error"] for item in summary["errors"]))


class DurableOOSPolicyTests(unittest.TestCase):
    @staticmethod
    def _frame(index):
        frame = pd.DataFrame({
            "open": [100.0] * len(index), "high": [101.0] * len(index),
            "low": [99.0] * len(index), "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
            "close_time": index + pd.Timedelta(minutes=5),
        }, index=index)
        frame.attrs["data_quality"] = {
            "quality_gate_passed": True, "source_coverage_pct": 100.0,
            "converted_coverage_pct": 100.0, "fx_missing_pct_of_source": 0.0,
            "invalid_ohlc_rows": 0, "converted_invalid_ohlc_rows": 0,
            "conflicting_duplicate_groups": 0, "unique_timestamps": True,
            "monotonic_timestamps": True, "timezone_aware_timestamps": True,
            "unaligned_timestamps": 0, "converted_unaligned_timestamps": 0,
            "irregular_timestamp_gaps": 0, "max_consecutive_missing_slots": 0,
            "converted_max_consecutive_missing_slots": 0,
            "converted_unique_timestamps": True, "converted_monotonic_timestamps": True,
        }
        return frame

    @staticmethod
    def _ledger(start, end, status="failed"):
        return {
            "policy_version": "chronological-oos-v2",
            "windows": [{
                "id": "previous-consumed-window", "start": start.isoformat(),
                "end": end.isoformat(), "bars": int((end - start) / pd.Timedelta(minutes=5)) + 1,
                "status": status, "reserved_at": "test-fixture",
            }],
            "last_test_start": start.isoformat(), "last_test_end": end.isoformat(),
            "last_result": status,
        }

    def test_reservation_persistence_failure_fails_closed(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(research_engine, "STATE", Path(tmp) / "state"),
                patch.object(research_engine, "_update_oos_ledger",
                             side_effect=RuntimeError("ledger push rejected")),
            ):
                ok, reason, _ = research_engine.reserve_oos_test_window(
                    pd.DataFrame({"close": range(12)}, index=index)
                )
                self.assertFalse(ok)
                self.assertIn("failed closed", reason)
                self.assertIn("push rejected", reason)
                self.assertFalse((Path(tmp) / "state" / "oos_evaluation_ledger.json").exists())

    def test_concurrent_non_fast_forward_conflict_blocks_reservation(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(research_engine, "STATE", Path(tmp) / "state"),
                patch.object(research_engine, "_update_oos_ledger",
                             side_effect=RuntimeError("non-fast-forward push rejected after concurrent update")),
            ):
                ok, reason, _ = research_engine.reserve_oos_test_window(
                    pd.DataFrame({"close": range(12)}, index=index)
                )
                self.assertFalse(ok)
                self.assertIn("non-fast-forward", reason)

    def test_reservation_failure_prevents_held_out_backtest_and_promotion(self):
        index = pd.date_range("2026-01-01", periods=1200, freq="5min", tz="UTC")
        frame = self._frame(index)
        params = {"fast": 2, "slow": 10, "momentum": 2, "breakout": 5}

        def fake_backtest(part, strategy, candidate, fee_bps=60, slippage_bps=10,
                          interval="5m", starting_cash=15):
            return {
                "total_return_pct": 10.0 if fee_bps == 60 else 2.0,
                "max_drawdown_pct": 5.0, "sharpe": 1.0, "trades": 4,
            }

        with tempfile.TemporaryDirectory() as tmp:
            state, reports = Path(tmp) / "state", Path(tmp) / "reports"
            with (
                patch.object(research_engine, "STATE", state),
                patch.object(research_engine, "REPORTS", reports),
                patch.object(research_engine, "INTERVALS", {"5m": "5min"}),
                patch.object(research_engine, "get_research_batch", return_value=(["BTCUSDT"], {"status": "test"})),
                patch.object(research_engine, "fetch_klines", return_value=frame),
                patch.object(research_engine, "resample_ohlcv", side_effect=lambda raw, rule: raw),
                patch.object(research_engine, "candidate_params", return_value=[params]),
                patch.object(research_engine, "backtest", side_effect=fake_backtest) as backtest,
                patch.object(research_engine, "_update_oos_ledger",
                             side_effect=RuntimeError("durable ledger unavailable")),
            ):
                summary = research_engine.run_research(days=30)
            self.assertIsNone(summary["champion"])
            self.assertFalse(summary["selected_candidate_test_gate"]["evaluated"])
            self.assertIn("durable ledger unavailable", summary["selected_candidate_test_gate"]["failures"][0])
            self.assertTrue(all(len(call.args[0]) < 1200 for call in backtest.call_args_list))
            self.assertFalse((state / "champions").exists())

    def test_terminal_ledger_persistence_failure_prevents_champion_promotion(self):
        index = pd.date_range("2026-01-01", periods=1200, freq="5min", tz="UTC")
        frame = self._frame(index)
        params = {"fast": 2, "slow": 10, "momentum": 2, "breakout": 5}
        real_update = research_engine._update_oos_ledger
        calls = {"count": 0}

        def fail_terminal_write(mutator):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("terminal status push conflict")
            return real_update(mutator)

        def fake_backtest(part, strategy, candidate, fee_bps=60, slippage_bps=10,
                          interval="5m", starting_cash=15):
            return {
                "total_return_pct": 10.0 if fee_bps == 60 else 2.0,
                "max_drawdown_pct": 5.0, "sharpe": 1.0, "trades": 4,
            }

        with tempfile.TemporaryDirectory() as tmp:
            state, reports = Path(tmp) / "state", Path(tmp) / "reports"
            with (
                patch.object(research_engine, "STATE", state),
                patch.object(research_engine, "REPORTS", reports),
                patch.object(research_engine, "INTERVALS", {"5m": "5min"}),
                patch.object(research_engine, "get_research_batch", return_value=(["BTCUSDT"], {"status": "test"})),
                patch.object(research_engine, "fetch_klines", return_value=frame),
                patch.object(research_engine, "resample_ohlcv", side_effect=lambda raw, rule: raw),
                patch.object(research_engine, "candidate_params", return_value=[params]),
                patch.object(research_engine, "backtest", side_effect=fake_backtest),
                patch.object(research_engine, "_update_oos_ledger", side_effect=fail_terminal_write),
            ):
                with self.assertRaisesRegex(RuntimeError, "refusing promotion"):
                    research_engine.run_research(days=30)
            ledger = json.loads((state / "oos_evaluation_ledger.json").read_text())
            self.assertEqual(ledger["windows"][-1]["status"], "reserved")
            self.assertFalse((state / "champion.json").exists())
            self.assertFalse((state / "champions").exists())

    def test_prior_test_history_is_excluded_from_later_train_validation_and_test(self):
        index = pd.date_range("2026-01-01", periods=2400, freq="5min", tz="UTC")
        frame = self._frame(index)
        params = {"fast": 2, "slow": 10, "momentum": 2, "breakout": 5}
        old_start, old_end = index[1000], index[1199]
        observed_periods = []

        def fake_backtest(part, strategy, candidate, fee_bps=60, slippage_bps=10,
                          interval="5m", starting_cash=15):
            observed_periods.append((part.index[0], part.index[-1], fee_bps, slippage_bps))
            return {
                "total_return_pct": 10.0 if fee_bps == 60 else 2.0,
                "max_drawdown_pct": 5.0, "sharpe": 1.0, "trades": 4,
            }

        with tempfile.TemporaryDirectory() as tmp:
            state, reports = Path(tmp) / "state", Path(tmp) / "reports"
            state.mkdir()
            (state / "oos_evaluation_ledger.json").write_text(json.dumps(self._ledger(old_start, old_end)))
            with (
                patch.object(research_engine, "STATE", state),
                patch.object(research_engine, "REPORTS", reports),
                patch.object(research_engine, "INTERVALS", {"5m": "5min"}),
                patch.object(research_engine, "get_research_batch", return_value=(["BTCUSDT"], {"status": "test"})),
                patch.object(research_engine, "fetch_klines", return_value=frame),
                patch.object(research_engine, "resample_ohlcv", side_effect=lambda raw, rule: raw),
                patch.object(research_engine, "candidate_params", return_value=[params]),
                patch.object(research_engine, "backtest", side_effect=fake_backtest),
            ):
                summary = research_engine.run_research(days=30)
            self.assertTrue(observed_periods)
            self.assertTrue(all(start > old_end and end > old_end for start, end, _, _ in observed_periods))
            self.assertEqual(
                summary["out_of_sample_policy"]["previous_test_end_excluded_from_optimization"],
                old_end.isoformat(),
            )
            self.assertTrue(summary["out_of_sample_policy"]["test_history_excluded_from_training_validation"])
            ledger = json.loads((state / "oos_evaluation_ledger.json").read_text())
            self.assertEqual(len(ledger["windows"]), 2)
            self.assertGreater(pd.Timestamp(ledger["windows"][1]["start"]), old_end)
            self.assertEqual(ledger["windows"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
