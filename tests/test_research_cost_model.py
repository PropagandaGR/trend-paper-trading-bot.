import unittest
from unittest.mock import patch

import pandas as pd

import research_engine


class ResearchCostModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "open": [100.0] * 100, "high": [100.0] * 100,
            "low": [100.0] * 100, "close": [100.0] * 100,
            "volume": [1.0] * 100,
        }, index=pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC"))
        self.params = {"fast": 2, "slow": 10, "momentum": 2, "breakout": 5}

    def test_baseline_cost_is_charged_on_entry_and_exit(self):
        positions = pd.Series([0.0] * 20 + [1.0] * 30 + [0.0] * 50, index=self.frame.index)
        with patch.object(research_engine, "desired_signal", return_value=positions):
            metrics = research_engine.backtest(self.frame, "trend", self.params, 10, 5)
        self.assertEqual(metrics["trades"], 2)
        self.assertAlmostEqual(metrics["total_return_pct"], -0.2998, delta=0.001)

    def test_stress_cost_is_charged_on_entry_and_exit(self):
        positions = pd.Series([0.0] * 20 + [1.0] * 30 + [0.0] * 50, index=self.frame.index)
        with patch.object(research_engine, "desired_signal", return_value=positions):
            metrics = research_engine.backtest(self.frame, "trend", self.params, 20, 10)
        self.assertEqual(metrics["trades"], 2)
        self.assertAlmostEqual(metrics["total_return_pct"], -0.5991, delta=0.001)


if __name__ == "__main__":
    unittest.main()
