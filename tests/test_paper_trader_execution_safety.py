import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import paper_trader


def make_champion():
    return {
        "symbol": "BTCUSDT",
        "interval": "5m",
        "strategy": "trend",
        "params": {"fast": 8, "slow": 30, "momentum": 6, "breakout": 12,
                   "entry_momentum": 0.0, "exit_momentum": -0.005},
        "champion_version": "champion-a1b2c3d4e5f6",
        "validation_metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "validated",
            "selection_metric": "validation_score",
            "validation_score": 1.0,
            "test_used_for_selection": False,
            "periods": {
                "train": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-01-02T00:00:00+00:00"},
                "validation": {"start": "2026-01-03T00:00:00+00:00", "end": "2026-01-04T00:00:00+00:00"},
                "test": {"start": "2026-01-05T00:00:00+00:00", "end": "2026-01-06T00:00:00+00:00"},
            },
            "validation_metrics": {"total_return_pct": 1.0, "max_drawdown_pct": 2.0, "sharpe": 0.5, "trades": 3},
            "validation_stress_metrics": {"total_return_pct": 0.5, "max_drawdown_pct": 3.0, "sharpe": 0.2, "trades": 3},
            "test_metrics": {"total_return_pct": -1.0, "max_drawdown_pct": 4.0, "sharpe": -0.5, "trades": 2},
            "test_stress_metrics": {"total_return_pct": -2.0, "max_drawdown_pct": 5.0, "sharpe": -0.8, "trades": 2},
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


def make_frame(price, age_minutes=1):
    closed_at = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=age_minutes)
    index = pd.DatetimeIndex([closed_at - pd.Timedelta(minutes=10), closed_at - pd.Timedelta(minutes=5)])
    return pd.DataFrame({
        "open": [price, price], "high": [price * 1.01, price * 1.01],
        "low": [price * 0.99, price * 0.99], "close": [price, price],
        "volume": [10.0, 10.0],
        "close_time": [closed_at - pd.Timedelta(minutes=5), closed_at],
    }, index=index)


class PaperTraderExecutionSafetyTests(unittest.TestCase):
    def _run_in_temp_repo(self, state, champion, frame_map, signal_value=1.0):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        state_dir, reports_dir = root / "state", root / "reports"
        state_dir.mkdir()
        reports_dir.mkdir()
        state_file, champion_file, status_file = (
            state_dir / "paper_state.json", state_dir / "champion.json", reports_dir / "paper_status.json"
        )
        state_file.write_text(json.dumps(state))
        champion_file.write_text(json.dumps(champion))
        def fetch(symbol, interval, days):
            return frame_map[symbol]
        feature_frame = pd.DataFrame({
            "close": [100.0, 100.0], "rsi": [50.0, 50.0],
            "ema_fast": [100.0, 101.0], "ema_slow": [100.0, 100.0],
            "momentum": [0.0, 0.01],
        }, index=frame_map[champion["symbol"]].index)
        with patch.object(paper_trader, "STATE", state_dir), \
             patch.object(paper_trader, "REPORTS", reports_dir), \
             patch.object(paper_trader, "STATE_FILE", state_file), \
             patch.object(paper_trader, "CHAMPION_FILE", champion_file), \
             patch.object(paper_trader, "STATUS_FILE", status_file), \
             patch.object(paper_trader, "fetch_klines", side_effect=fetch), \
             patch.object(paper_trader, "desired_signal",
                          return_value=pd.Series([0.0, signal_value], index=frame_map[champion["symbol"]].index)), \
             patch.object(paper_trader, "add_features", return_value=feature_frame):
            status = paper_trader.run_once()
        final_state = json.loads(state_file.read_text())
        return status, final_state

    def test_stale_champion_price_cannot_trigger_buy(self):
        champion = make_champion()
        stale_frame = make_frame(100.0, age_minutes=45)
        status, state = self._run_in_temp_repo(
            {"cash": 15.0, "quantity": 0.0, "symbol": None, "entry_price": None,
             "last_processed_bars": {}, "starting_cash": 15.0},
            champion, {"BTCUSDT": stale_frame}, signal_value=1.0,
        )
        self.assertEqual(status["actions"], [])
        self.assertIn("Fresh champion price unavailable", status["message"])
        self.assertEqual(state["cash"], 15.0)
        self.assertEqual(state["quantity"], 0.0)

    def test_daily_loss_breach_sells_existing_position_and_latches_halt(self):
        champion = make_champion()
        now = datetime.now(timezone.utc)
        state = {
            "cash": 0.0, "quantity": 0.1, "symbol": "ETHUSDT", "entry_price": 150.0,
            "last_processed_bars": {}, "starting_cash": 15.0,
            "risk_day": now.date().isoformat(), "daily_start_equity": 15.0, "peak_equity": 15.0,
        }
        status, final_state = self._run_in_temp_repo(
            state, champion, {"BTCUSDT": make_frame(100.0), "ETHUSDT": make_frame(100.0)},
            signal_value=1.0,
        )
        self.assertTrue(status["risk"]["halted"])
        self.assertIn("daily_loss_limit", status["risk"]["halt_reason"])
        self.assertEqual([x["action"] for x in status["actions"]], ["PAPER_SELL"])
        self.assertEqual(final_state["quantity"], 0.0)
        self.assertIsNone(final_state["symbol"])

    def test_new_position_uses_at_most_half_of_equity(self):
        champion = make_champion()
        status, state = self._run_in_temp_repo(
            {"cash": 15.0, "quantity": 0.0, "symbol": None, "entry_price": None,
             "last_processed_bars": {}, "starting_cash": 15.0},
            champion, {"BTCUSDT": make_frame(100.0)}, signal_value=1.0,
        )
        self.assertEqual([x["action"] for x in status["actions"]], ["PAPER_BUY"])
        invested = state["quantity"] * 100.0
        self.assertLessEqual(invested, 7.5)
        self.assertGreaterEqual(state["cash"], 7.49)


if __name__ == "__main__":
    unittest.main()
