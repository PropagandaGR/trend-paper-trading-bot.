import os
import unittest
from unittest.mock import patch

import pandas as pd

import research_engine
from portfolio_accounting import _cost_setting


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
            metrics = research_engine.backtest(self.frame, "trend", self.params, 60, 10, "5m")
        self.assertEqual(metrics["trades"], 2)
        self.assertLess(metrics["total_return_pct"], -1.0)
        with self.assertRaisesRegex(ValueError, "fee_bps must be"):
            research_engine.backtest(self.frame, "trend", self.params, 10, 5, "5m")

    def test_stress_cost_is_charged_on_both_sides_using_paper_fill_math(self):
        positions = pd.Series([0.0] * 20 + [1.0] * 30 + [0.0] * 50, index=self.frame.index)
        with patch.object(research_engine, "desired_signal", return_value=positions):
            metrics = research_engine.backtest(self.frame, "trend", self.params, 100, 20, "5m")
        self.assertEqual(metrics["trades"], 2)
        self.assertLess(metrics["total_return_pct"], -1.0)

    def test_default_costs_are_conservative_and_stress_costs_are_higher(self):
        self.assertEqual(research_engine.DEFAULT_FEE_BPS, 60.0)
        self.assertEqual(research_engine.DEFAULT_SLIPPAGE_BPS, 10.0)
        self.assertGreater(research_engine.STRESS_FEE_BPS, research_engine.DEFAULT_FEE_BPS)
        self.assertGreater(research_engine.STRESS_SLIPPAGE_BPS, research_engine.DEFAULT_SLIPPAGE_BPS)

    def test_environment_overrides_cannot_lower_any_approved_cost_floor(self):
        for name, floor in (("PAPER_FEE_BPS", 60.0), ("PAPER_SLIPPAGE_BPS", 10.0),
                            ("PAPER_STRESS_FEE_BPS", 100.0), ("PAPER_STRESS_SLIPPAGE_BPS", 20.0)):
            with patch.dict(os.environ, {name: str(floor - 1)}):
                with self.assertRaisesRegex(ValueError, "at least the conservative floor"):
                    _cost_setting(name, floor)

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

    def test_backtest_latches_daily_loss_halt_and_does_not_reenter(self):
        frame = self.frame.copy()
        frame.loc[frame.index[40]:, "close"] = 80.0
        positions = pd.Series([0.0] * 20 + [1.0] * 80, index=frame.index)
        with patch.object(research_engine, "desired_signal", return_value=positions):
            metrics = research_engine.backtest(frame, "trend", self.params, 10, 5, "5m")

        # One buy and one risk liquidation; the continuing long signal cannot re-enter.
        self.assertEqual(metrics["trades"], 2)
        self.assertLess(metrics["total_return_pct"], -5.0)

    def test_strategy_grid_deduplicates_irrelevant_parameters(self):
        first = {
            "fast": 8, "slow": 30, "momentum": 6, "breakout": 12,
            "entry_momentum": 0.0, "exit_momentum": -0.005,
            "rsi_entry": 20, "rsi_exit": 45, "stop_pct": 0.04,
        }
        second = {
            **first, "breakout": 24, "rsi_entry": 25,
            "rsi_exit": 50, "stop_pct": 0.06,
        }
        self.assertEqual(len(research_engine.unique_candidates_for_strategy("trend", [first, second])), 1)
        self.assertEqual(len(research_engine.unique_candidates_for_strategy("breakout", [first, second])), 2)
        self.assertEqual(len(research_engine.unique_candidates_for_strategy("mean_reversion", [first, second])), 2)

    def test_cash_and_buy_and_hold_benchmarks_report_window_costs_exposure_and_missing_policy(self):
        cash = research_engine.cash_benchmark(self.frame, "5m", starting_cash=15.0)
        buy_hold = research_engine.buy_and_hold_benchmark(self.frame, "5m", 60, 10, 15.0)
        self.assertEqual(cash["total_return_pct"], 0.0)
        self.assertEqual(cash["trades"], 0)
        self.assertEqual(cash["period_start"], buy_hold["period_start"])
        self.assertEqual(cash["period_end"], buy_hold["period_end"])
        self.assertEqual(cash["bars"], buy_hold["bars"])
        self.assertEqual(cash["exposure_fraction"], 0.0)
        self.assertEqual(buy_hold["exposure_fraction"], 0.50)
        self.assertEqual(buy_hold["fee_bps_per_side"], 60.0)
        self.assertEqual(buy_hold["slippage_bps_per_side"], 10.0)
        self.assertEqual(buy_hold["trades"], 1)
        self.assertEqual(buy_hold["missing_data_policy"],
                         "No interpolation; use exact supplied timestamps and report missing slots.")

    def test_benchmarks_fail_closed_on_missing_close_data(self):
        broken = self.frame.copy()
        broken.iloc[10, broken.columns.get_loc("close")] = float("nan")
        with self.assertRaisesRegex(ValueError, "missing data is not imputed"):
            research_engine.cash_benchmark(broken, "5m")
        with self.assertRaisesRegex(ValueError, "missing data is not imputed"):
            research_engine.buy_and_hold_benchmark(broken, "5m")


if __name__ == "__main__":
    unittest.main()
