import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from paper_trader import (MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT, MAX_POSITION_FRACTION,
                          champion_validation_error, fresh_price_map, update_risk_metrics)


def valid_champion(created_at=None):
    return {
        "symbol": "ETHUSDT",
        "interval": "30m",
        "strategy": "mean_reversion",
        "params": {"fast": 8, "slow": 48, "momentum": 12, "breakout": 12,
                   "rsi_entry": 20, "rsi_exit": 50, "stop_pct": 0.06},
        "champion_version": "champion-test1234567890",
        "validation_metadata": {
            "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
            "status": "validated",
            "test_used_for_selection": False,
            "validation_score": 1.25,
            "train_metrics": {"total_return_pct": 5.0, "max_drawdown_pct": 10.0, "sharpe": 0.5, "trades": 12},
            "periods": {
                "train": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-02T00:00:00+00:00"},
                "validation": {"start": "2026-01-03T00:00:00+00:00", "end": "2026-01-04T00:00:00+00:00"},
                "test": {"start": "2026-01-05T00:00:00+00:00", "end": "2026-01-06T00:00:00+00:00"},
            },
            "validation_metrics": {"total_return_pct": 1.0, "max_drawdown_pct": 2.0, "sharpe": 0.5, "trades": 3},
            "validation_stress_metrics": {"total_return_pct": 0.5, "max_drawdown_pct": 3.0, "sharpe": 0.2, "trades": 3},
            "test_metrics": {"total_return_pct": 0.2, "max_drawdown_pct": 4.0, "sharpe": 0.1, "trades": 2},
            "test_stress_metrics": {"total_return_pct": 0.1, "max_drawdown_pct": 5.0, "sharpe": 0.05, "trades": 2},
            "cost_convention": {
                "fee_basis": "executed notional",
                "fill_timing": "closed candle close after one-candle signal shift",
                "fee_bps_per_side": 10,
                "slippage_bps_per_side": 5,
                "stress_fee_bps_per_side": 20,
                "stress_slippage_bps_per_side": 10,
            },
        },
    }


class PaperTraderChampionGateTests(unittest.TestCase):
    def test_accepts_fresh_versioned_validated_champion(self):
        self.assertIsNone(champion_validation_error(valid_champion()))

    def test_rejects_legacy_champion_without_metadata(self):
        self.assertIsNotNone(champion_validation_error({
            "symbol": "BTCUSDT", "interval": "5m", "strategy": "trend",
            "params": {"fast": 8, "slow": 30, "momentum": 6, "breakout": 12},
        }))

    def test_rejects_stale_champion(self):
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        self.assertIn("stale", champion_validation_error(valid_champion(old)))

    def test_rejects_champion_if_final_test_was_used_for_selection(self):
        champion = valid_champion()
        champion["validation_metadata"]["test_used_for_selection"] = True
        self.assertIn("final-test isolation", champion_validation_error(champion))

    def test_rejects_champion_with_negative_training_return(self):
        champion = valid_champion()
        champion["validation_metadata"]["train_metrics"]["total_return_pct"] = -69.7706
        self.assertIn("training return must be positive", champion_validation_error(champion))

    def test_rejects_champion_with_excessive_training_drawdown(self):
        champion = valid_champion()
        champion["validation_metadata"]["train_metrics"]["max_drawdown_pct"] = 70.0337
        self.assertIn("training drawdown must be below 35%", champion_validation_error(champion))

    def test_final_test_loss_is_report_only_not_an_acceptance_gate(self):
        champion = valid_champion()
        champion["validation_metadata"]["test_metrics"]["total_return_pct"] = -31.84
        champion["validation_metadata"]["test_stress_metrics"]["total_return_pct"] = -48.82
        self.assertIsNone(champion_validation_error(champion))

    def test_rejects_incomplete_validation_metrics(self):
        champion = valid_champion()
        del champion["validation_metadata"]["validation_stress_metrics"]
        self.assertIn("validation_stress_metrics", champion_validation_error(champion))


class PaperTraderAdversarialSafetyTests(unittest.TestCase):
    def test_rejects_negative_or_out_of_range_strategy_parameters(self):
        champion = valid_champion()
        champion["params"]["fast"] = -5
        self.assertIn("fast", champion_validation_error(champion))
        champion = valid_champion()
        champion["params"]["fast"] = 90
        champion["params"]["slow"] = 30
        self.assertIn("fast < slow", champion_validation_error(champion))

    def test_rejects_non_finite_validation_metrics(self):
        champion = valid_champion()
        champion["validation_metadata"]["test_metrics"]["sharpe"] = float("nan")
        self.assertIn("finite number", champion_validation_error(champion))

    def test_rejects_overlapping_final_test_period(self):
        champion = valid_champion()
        champion["validation_metadata"]["periods"]["test"]["start"] = "2026-01-03T12:00:00+00:00"
        self.assertIn("overlap", champion_validation_error(champion))

    def test_rejects_invalid_mean_reversion_threshold_order(self):
        champion = valid_champion()
        champion["params"]["rsi_entry"] = 60
        champion["params"]["rsi_exit"] = 50
        self.assertIn("entry must be below RSI exit", champion_validation_error(champion))

    def test_fresh_price_map_rejects_stale_future_and_invalid_marks(self):
        now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        frames = {
            "BTCUSDT": pd.DataFrame({"close": [100.0], "close_time": [pd.Timestamp("2026-01-01T11:50:00Z")]}),
            "ETHUSDT": pd.DataFrame({"close": [200.0], "close_time": [pd.Timestamp("2026-01-01T11:40:00Z")]}),
            "SOLUSDT": pd.DataFrame({"close": [300.0], "close_time": [pd.Timestamp("2026-01-01T12:05:00Z")]}),
            "DOGEUSDT": pd.DataFrame({"close": [float("nan")], "close_time": [pd.Timestamp("2026-01-01T11:59:00Z")]}),
        }
        prices, timestamps = fresh_price_map(frames, now)
        self.assertEqual(prices, {"BTCUSDT": 100.0})
        self.assertIn("BTCUSDT", timestamps)
        self.assertNotIn("ETHUSDT", prices)
        self.assertNotIn("SOLUSDT", prices)
        self.assertNotIn("DOGEUSDT", prices)

    def test_daily_loss_limit_latches_halt(self):
        now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        state = {"risk_day": "2026-01-01", "daily_start_equity": 15.0, "peak_equity": 15.0}
        risk = update_risk_metrics(state, 14.0, now)
        self.assertGreater(risk["daily_loss_pct"], MAX_DAILY_LOSS_PCT)
        self.assertTrue(risk["halted"])
        self.assertIn("daily_loss_limit", risk["halt_reason"])

    def test_drawdown_limit_latches_halt(self):
        now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        state = {"risk_day": "2026-01-01", "daily_start_equity": 20.0, "peak_equity": 20.0}
        risk = update_risk_metrics(state, 17.9, now)
        self.assertGreater(risk["drawdown_pct"], MAX_DRAWDOWN_PCT)
        self.assertTrue(risk["halted"])
        self.assertIn("drawdown_limit", risk["halt_reason"])

    def test_position_budget_is_configured_below_full_equity(self):
        self.assertLess(MAX_POSITION_FRACTION, 1.0)
        self.assertEqual(MAX_POSITION_FRACTION, 0.50)


if __name__ == "__main__":
    unittest.main()
