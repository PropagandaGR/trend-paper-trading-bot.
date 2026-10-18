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

    def test_baseline_cost_is_charged_on_both_sides_using_paper_fill_math(self):
        positions = pd.Series([0.0] * 20 + [1.0] * 30 + [0.0] * 50, index=self.frame.index)
        with patch.object(research_engine, "desired_signal", return_value=positions):
            metrics = research_engine.backtest(self.frame, "trend", self.params, 10, 5, "5m")
        self.assertEqual(metrics["trades"], 2)
        self.assertAlmostEqual(metrics["total_return_pct"], -0.2996, delta=0.01)

    def test_stress_cost_is_charged_on_both_sides_using_paper_fill_math(self):
        positions = pd.Series([0.0] * 20 + [1.0] * 30 + [0.0] * 50, index=self.frame.index)
        with patch.object(research_engine, "desired_signal", return_value=positions):
            metrics = research_engine.backtest(self.frame, "trend", self.params, 20, 10, "5m")
        self.assertEqual(metrics["trades"], 2)
        self.assertAlmostEqual(metrics["total_return_pct"], -0.598, delta=0.02)

    def test_sharpe_annualization_matches_each_timeframe(self):
        self.assertEqual(research_engine.bars_per_year("5m"), 105120)
        self.assertEqual(research_engine.bars_per_year("10m"), 52560)
        self.assertEqual(research_engine.bars_per_year("15m"), 35040)
        self.assertEqual(research_engine.bars_per_year("30m"), 17520)
        returns = pd.Series([0.01, -0.004, 0.006, -0.002] * 20)
        five_minute = research_engine.performance(returns, "5m")
        thirty_minute = research_engine.performance(returns, "30m")
        self.assertGreater(five_minute["sharpe"], thirty_minute["sharpe"])
        self.assertAlmostEqual(
            five_minute["sharpe"] / thirty_minute["sharpe"],
            (research_engine.bars_per_year("5m") /
             research_engine.bars_per_year("30m")) ** 0.5,
            delta=0.01,
        )


if __name__ == "__main__":
    unittest.main()
