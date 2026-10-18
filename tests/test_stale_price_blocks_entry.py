import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import paper_trader


def frame(price: float, stale: bool = False) -> pd.DataFrame:
    close_time = pd.Timestamp.now(tz="UTC").floor("5min")
    if stale:
        close_time -= pd.Timedelta(minutes=30)
    index = pd.date_range(end=close_time - pd.Timedelta(minutes=5), periods=240, freq="5min")
    values = np.full(len(index), price, dtype=float)
    return pd.DataFrame({
        "open": values, "high": values * 1.001, "low": values * 0.999,
        "close": values, "volume": values * 0 + 1,
        "close_time": index + pd.Timedelta(minutes=5),
    }, index=index)


class StalePriceBlocksEntryTests(unittest.TestCase):
    def test_stale_target_price_never_opens_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            state_path, champion_path = root / "paper_state.json", root / "champion.json"
            state_path.write_text(json.dumps({
                "cash": 15.0, "quantity": 0.0, "symbol": None,
                "last_processed_bars": {}, "starting_cash": 15.0,
                "completed_trades": 0, "realized_pnl": 0.0,
            }))
            champion_path.write_text(json.dumps({
                "passed_validation": True,
                "validated_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "symbol": "BTCUSDT", "interval": "5m", "strategy": "trend",
                "allow_short": False,
                "test": {"total_return_pct": 1.0, "max_drawdown_pct": 5.0, "trades": 4},
                "stress": {"total_return_pct": 0.5},
                "params": {"fast": 8, "slow": 30, "momentum": 6, "breakout": 12,
                           "entry_momentum": 0.0, "exit_momentum": -0.005,
                           "rsi_entry": 20, "rsi_exit": 50, "stop_pct": 0.06},
            }))
            data = {"BTCUSDT": frame(100, stale=True),
                    "ETHUSDT": frame(1000), "SOLUSDT": frame(50)}
            with patch.multiple(
                paper_trader, STATE=root, REPORTS=reports, STATE_FILE=state_path,
                CHAMPION_FILE=champion_path, STATUS_FILE=reports / "paper_status.json",
            ), patch.object(paper_trader, "fetch_klines", side_effect=lambda symbol, *_: data[symbol]), \
                 patch.object(paper_trader, "desired_signal", side_effect=lambda bars, *_args, **_kwargs: pd.Series(1.0, index=bars.index)):
                status = paper_trader.run_once()
            self.assertEqual(status["actions"], [])
            self.assertEqual(status["cash_usdt"], 15.0)
            self.assertIsNone(status["open_position"])
            self.assertFalse(status["price_checks"]["BTCUSDT"]["usable"])


if __name__ == "__main__":
    unittest.main()
