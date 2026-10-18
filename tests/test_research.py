import unittest
import numpy as np
import pandas as pd

from research_engine import backtest, desired_signal, performance


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

    def test_performance_empty_is_safe(self):
        result = performance(pd.Series(dtype=float))
        self.assertEqual(result["total_return_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
