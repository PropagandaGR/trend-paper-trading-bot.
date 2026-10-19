import unittest
import numpy as np
import pandas as pd

from research_engine import (backtest, candidate_passes_gates, desired_signal, performance, resample_ohlcv, deployment_gate_failures, candle_quality_report)


def sample_frame(n=500):
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = np.linspace(100, 150, n) + np.sin(np.arange(n) / 9)
    return pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998,
        "close": close, "volume": np.ones(n),
        "close_time": idx + pd.Timedelta(minutes=5)
    }, index=idx)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.params = {"fast": 8, "slow": 30, "momentum": 6, "breakout": 12,
                       "entry_momentum": 0.0, "exit_momentum": -0.005,
                       "rsi_entry": 25, "rsi_exit": 50, "stop_pct": 0.06}

    def test_resample_uses_only_complete_closed_candles(self):
        frame = sample_frame(12)  # 00:00 through 00:55, all source bars closed
        # Remove the final 5m bar: the 00:30-01:00 bucket must be excluded.
        incomplete = frame.iloc[:-1]
        bars = resample_ohlcv(incomplete, "30min")
        self.assertEqual(list(bars.index), [
            pd.Timestamp("2025-01-01 00:30:00", tz="UTC")
        ])
        self.assertEqual(bars.iloc[0]["close_time"],
                         pd.Timestamp("2025-01-01 00:30:00", tz="UTC"))
        # With all six source bars present, the next boundary is valid.
        complete = resample_ohlcv(frame, "30min")
        self.assertEqual(list(complete.index), [
            pd.Timestamp("2025-01-01 00:30:00", tz="UTC"),
            pd.Timestamp("2025-01-01 01:00:00", tz="UTC")
        ])

    def test_signal_has_no_lookahead_shift(self):
        frame = sample_frame()
        signals = desired_signal(frame, "trend", self.params)
        self.assertEqual(float(signals.iloc[0]), 0.0)
        self.assertEqual(len(signals), len(frame))

    def test_all_strategies_return_bounded_binary_positions(self):
        frame = sample_frame()
        for strategy in ("trend", "breakout", "mean_reversion"):
            positions = desired_signal(frame, strategy, self.params)
            self.assertTrue(set(positions.unique()).issubset({0.0, 1.0}))

    def test_cost_stress_cannot_improve_same_backtest(self):
        frame = sample_frame()
        base = backtest(frame, "trend", self.params, 10, 5)
        stressed = backtest(frame, "trend", self.params, 20, 10)
        self.assertLessEqual(stressed["total_return_pct"], base["total_return_pct"] + 1e-8)

    def test_candidate_gate_rejects_validation_spike_after_training_collapse(self):
        train = {"total_return_pct": -69.7706, "max_drawdown_pct": 70.0337, "sharpe": -27.88, "trades": 649}
        validation = {"total_return_pct": 164.6108, "max_drawdown_pct": 18.8077, "sharpe": 11.7, "trades": 212}
        stress = {"total_return_pct": 92.5315, "max_drawdown_pct": 23.0624, "sharpe": 8.214, "trades": 212}
        self.assertFalse(candidate_passes_gates(train, validation, stress))

    def test_candidate_gate_rejects_training_drawdown_at_or_above_limit(self):
        train = {"total_return_pct": 5.0, "max_drawdown_pct": 35.0, "sharpe": 0.5, "trades": 12}
        validation = {"total_return_pct": 10.0, "max_drawdown_pct": 10.0, "sharpe": 1.0, "trades": 4}
        stress = {"total_return_pct": 2.0, "max_drawdown_pct": 12.0, "sharpe": 0.5, "trades": 4}
        self.assertFalse(candidate_passes_gates(train, validation, stress))

    def test_candidate_gate_accepts_positive_train_and_validation_within_limits(self):
        train = {"total_return_pct": 5.0, "max_drawdown_pct": 10.0, "sharpe": 0.5, "trades": 12}
        validation = {"total_return_pct": 10.0, "max_drawdown_pct": 10.0, "sharpe": 1.0, "trades": 4}
        stress = {"total_return_pct": 2.0, "max_drawdown_pct": 12.0, "sharpe": 0.5, "trades": 4}
        self.assertTrue(candidate_passes_gates(train, validation, stress))

    def test_candidate_gate_requires_validation_outperformance_vs_benchmarks(self):
        train = {"total_return_pct": 5.0, "max_drawdown_pct": 10.0, "sharpe": 0.5, "trades": 12}
        validation = {"total_return_pct": 10.0, "max_drawdown_pct": 10.0, "sharpe": 1.0, "trades": 4}
        stress = {"total_return_pct": 2.0, "max_drawdown_pct": 12.0, "sharpe": 0.5, "trades": 4}
        cash = {"cash": {"total_return_pct": 0.0}, "buy_and_hold": {"total_return_pct": 12.0}}
        stress_cash = {"cash": {"total_return_pct": 0.0}, "buy_and_hold": {"total_return_pct": 1.0}}
        self.assertFalse(candidate_passes_gates(train, validation, stress, cash, stress_cash))
        cash["buy_and_hold"]["total_return_pct"] = 1.0
        self.assertTrue(candidate_passes_gates(train, validation, stress, cash, stress_cash))

    def test_held_out_gate_rejects_validation_spike_with_negative_test(self):
        test = {"total_return_pct": -10.0, "max_drawdown_pct": 30.0, "sharpe": -1.0, "trades": 4}
        stress = {"total_return_pct": -20.0, "max_drawdown_pct": 35.0, "sharpe": -2.0, "trades": 4}
        cash = {"cash": {"total_return_pct": 0.0}, "buy_and_hold": {"total_return_pct": -5.0}}
        stress_cash = {"cash": {"total_return_pct": 0.0}, "buy_and_hold": {"total_return_pct": -8.0}}
        failures = deployment_gate_failures(test, stress, cash, stress_cash)
        self.assertTrue(any("cash and buy-and-hold" in item for item in failures))
        self.assertTrue(any("drawdown" in item for item in failures))

    def test_candle_quality_gate_reports_long_missing_data_gaps(self):
        index = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC").delete([3, 4, 5, 6])
        frame = pd.DataFrame({
            "open": [100.0] * len(index), "high": [101.0] * len(index),
            "low": [99.0] * len(index), "close": [100.0] * len(index),
            "volume": [1.0] * len(index),
        }, index=index)
        quality = candle_quality_report(frame, "5m", days=1)
        self.assertEqual(quality["max_consecutive_missing_slots"], 4)
        self.assertFalse(quality["unique_timestamps"] is False)
        self.assertTrue(quality["monotonic_timestamps"])

    def test_performance_empty_is_safe(self):
        result = performance(pd.Series(dtype=float))
        self.assertEqual(result["total_return_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
