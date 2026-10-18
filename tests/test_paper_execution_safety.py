import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import paper_trader
from paper_margin import open_position


def candles(price: float, *, stale_minutes: int = 0) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=stale_minutes)
    end_open = now - pd.Timedelta(minutes=5)
    index = pd.date_range(end=end_open, periods=240, freq="5min")
    values = np.full(len(index), price, dtype=float)
    return pd.DataFrame({
        "open": values, "high": values * 1.001, "low": values * 0.999,
        "close": values, "volume": np.full(len(index), 100.0),
        "close_time": index + pd.Timedelta(minutes=5),
    }, index=index)


class PaperExecutionSafetyTests(unittest.TestCase):
    def test_champion_requires_pass_and_fresh_timestamp(self):
        good = {"passed_validation": True, "validated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "symbol": "BTCUSDT", "interval": "5m", "strategy": "trend", "params": {"fast": 8, "slow": 30, "momentum": 6, "breakout": 12, "entry_momentum": 0.0, "exit_momentum": -0.005, "rsi_entry": 20, "rsi_exit": 50, "stop_pct": 0.06}, "test": {"total_return_pct": 1.0, "max_drawdown_pct": 5.0, "trades": 4}, "stress": {"total_return_pct": 0.5}}
        self.assertTrue(paper_trader._valid_champion(good)[0])
        self.assertFalse(paper_trader._valid_champion({**good, "passed_validation": False})[0])
        stale = {**good, "validated_at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).isoformat()}
        self.assertFalse(paper_trader._valid_champion(stale)[0])
        self.assertFalse(paper_trader._valid_champion({k: v for k, v in good.items() if k != "validated_at"})[0])

    def test_stale_candles_are_rejected(self):
        fresh, _ = paper_trader._prepare_bars(candles(100), "5m")
        stale, reason = paper_trader._prepare_bars(candles(100, stale_minutes=30), "5m")
        self.assertIsNotNone(fresh)
        self.assertIsNone(stale)
        self.assertIn("stale", reason.lower())

    def test_coin_switch_closes_using_held_asset_price(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path, champion_path = root / "paper_state.json", root / "champion.json"
            reports = root / "reports"
            reports.mkdir()
            state = {"cash": 15.0, "quantity": 0.0, "symbol": None,
                     "entry_price": None, "last_processed_bars": {},
                     "starting_cash": 15.0, "completed_trades": 0, "realized_pnl": 0.0,
                     "side": 0, "margin": 0.0, "leverage": 1.0, "entry_fee": 0.0}
            open_position(state, "BTCUSDT", 1, 100.0, leverage=1.0)
            state_path.write_text(json.dumps(state))
            champion = {"passed_validation": True, "validated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "symbol": "ETHUSDT", "interval": "5m", "strategy": "trend",
                        "allow_short": False,
                        "test": {"total_return_pct": 1.0, "max_drawdown_pct": 5.0, "trades": 4},
                        "stress": {"total_return_pct": 0.5},
                        "params": {"fast": 8, "slow": 30, "momentum": 6, "breakout": 12,
                                   "entry_momentum": 0.0, "exit_momentum": -0.005,
                                   "rsi_entry": 20, "rsi_exit": 50, "stop_pct": 0.06}}
            champion_path.write_text(json.dumps(champion))
            data = {"BTCUSDT": candles(100), "ETHUSDT": candles(1000), "SOLUSDT": candles(50)}
            with patch.multiple(
                paper_trader,
                STATE=root, REPORTS=reports, STATE_FILE=state_path,
                CHAMPION_FILE=champion_path, STATUS_FILE=reports / "paper_status.json",
            ), patch.object(paper_trader, "fetch_klines", side_effect=lambda symbol, *_: data[symbol]), \
                 patch.object(paper_trader, "desired_signal", side_effect=lambda bars, *_args, **_kwargs: pd.Series(0.0, index=bars.index)):
                status = paper_trader.run_once()
            close_actions = [a for a in status["actions"] if a["action"] == "PAPER_SELL"]
            self.assertEqual(len(close_actions), 1)
            self.assertAlmostEqual(close_actions[0]["price"], 100.0 * (1 - paper_trader.SLIPPAGE_BPS / 10000), places=5)
            self.assertNotAlmostEqual(close_actions[0]["price"], 1000.0, places=1)
            self.assertIsNone(status["open_position"])

    def test_unvalidated_champion_cannot_trade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            state_path, champion_path = root / "paper_state.json", root / "champion.json"
            state_path.write_text(json.dumps({"cash": 15.0, "quantity": 0.0, "symbol": None,
                                              "last_processed_bars": {}, "starting_cash": 15.0,
                                              "completed_trades": 0, "realized_pnl": 0.0}))
            champion_path.write_text(json.dumps({"passed_validation": False,
                "validated_at": pd.Timestamp.now(tz="UTC").isoformat(), "symbol": "BTCUSDT",
                "interval": "5m", "strategy": "trend", "params": {}}))
            data = {symbol: candles(100) for symbol in paper_trader.SYMBOLS}
            with patch.multiple(paper_trader, STATE=root, REPORTS=reports, STATE_FILE=state_path,
                                CHAMPION_FILE=champion_path, STATUS_FILE=reports / "paper_status.json"), \
                 patch.object(paper_trader, "fetch_klines", side_effect=lambda symbol, *_: data[symbol]):
                status = paper_trader.run_once()
            self.assertEqual(status["actions"], [])
            self.assertFalse(status["champion_valid"])
            self.assertEqual(status["cash_usdt"], 15.0)

    def test_daily_loss_limit_blocks_new_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            state_path, champion_path = root / "paper_state.json", root / "champion.json"
            today = pd.Timestamp.now(tz="UTC").date().isoformat()
            state_path.write_text(json.dumps({
                "cash": 14.0, "quantity": 0.0, "symbol": None, "last_processed_bars": {},
                "starting_cash": 15.0, "completed_trades": 0, "realized_pnl": 0.0,
                "high_water_equity": 15.0, "day_start_equity": 15.0, "risk_day_utc": today,
            }))
            champion = {"passed_validation": True, "validated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        "symbol": "BTCUSDT", "interval": "5m", "strategy": "trend",
                        "allow_short": False,
                        "test": {"total_return_pct": 1.0, "max_drawdown_pct": 5.0, "trades": 4},
                        "stress": {"total_return_pct": 0.5},
                        "params": {"fast": 8, "slow": 30, "momentum": 6, "breakout": 12,
                                   "entry_momentum": 0.0, "exit_momentum": -0.005,
                                   "rsi_entry": 20, "rsi_exit": 50, "stop_pct": 0.06}}
            champion_path.write_text(json.dumps(champion))
            data = {symbol: candles(100) for symbol in paper_trader.SYMBOLS}
            with patch.multiple(paper_trader, STATE=root, REPORTS=reports, STATE_FILE=state_path,
                                CHAMPION_FILE=champion_path, STATUS_FILE=reports / "paper_status.json"), \
                 patch.object(paper_trader, "fetch_klines", side_effect=lambda symbol, *_: data[symbol]), \
                 patch.object(paper_trader, "desired_signal", side_effect=lambda bars, *_args, **_kwargs: pd.Series(1.0, index=bars.index)):
                status = paper_trader.run_once()
            self.assertEqual(status["actions"], [])
            self.assertTrue(status["risk_limits"]["risk_halt"])
            self.assertEqual(status["cash_usdt"], 14.0)


if __name__ == "__main__":
    unittest.main()
