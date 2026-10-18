import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import paper_trader


class PaperTraderWrongSymbolIntegrationTests(unittest.TestCase):
    def test_symbol_change_sells_using_held_asset_mark_not_champion_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir, reports_dir = root / "state", root / "reports"
            state_dir.mkdir()
            reports_dir.mkdir()
            state_file = state_dir / "paper_state.json"
            champion_file = state_dir / "champion.json"
            status_file = reports_dir / "paper_status.json"
            state_file.write_text(json.dumps({
                "cash": 0.0, "quantity": 0.5, "symbol": "ETHUSDT",
                "entry_price": 2_000.0, "last_processed_bars": {},
                "starting_cash": 15.0, "completed_trades": 0, "realized_pnl": 0.0
            }))
            champion_file.write_text(json.dumps({
                "symbol": "BTCUSDT", "interval": "5m", "strategy": "trend",
                "params": {"fast": 2, "slow": 3, "momentum": 1, "breakout": 2},
                "champion_version": "champion-test1234567890",
                "validation_metadata": {
                    "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "status": "validated", "test_used_for_selection": False,
                    "validation_metrics": {"total_return_pct": 1.0},
                    "validation_stress_metrics": {"total_return_pct": 0.5},
                    "test_metrics": {"total_return_pct": 0.2},
                    "test_stress_metrics": {"total_return_pct": 0.1},
                    "cost_convention": {"fee_basis": "executed notional"}
                }
            }))
            index = pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC")
            eth = pd.DataFrame({"close": [1_990.0, 2_000.0]}, index=index)
            btc = pd.DataFrame({"close": [59_000.0, 60_000.0]}, index=index)
            features = pd.DataFrame({
                "rsi": [50.0, 50.0], "ema_fast": [1.0, 1.0],
                "ema_slow": [1.0, 1.0], "momentum": [0.0, 0.0]
            }, index=index)
            signals = pd.Series([0.0, 0.0], index=index)

            with (
                patch.object(paper_trader, "STATE", state_dir),
                patch.object(paper_trader, "REPORTS", reports_dir),
                patch.object(paper_trader, "STATE_FILE", state_file),
                patch.object(paper_trader, "CHAMPION_FILE", champion_file),
                patch.object(paper_trader, "STATUS_FILE", status_file),
                patch.object(paper_trader, "fetch_klines", side_effect=[btc, eth]),
                patch.object(paper_trader, "desired_signal", return_value=signals),
                patch.object(paper_trader, "add_features", return_value=features),
            ):
                status = paper_trader.run_once()

            sells = [a for a in status["actions"] if a["action"] == "PAPER_SELL"]
            self.assertEqual(len(sells), 1)
            self.assertEqual(sells[0]["symbol"], "ETHUSDT")
            self.assertAlmostEqual(sells[0]["price"], 1_999.0, places=7)
            self.assertNotAlmostEqual(sells[0]["price"], 59_970.0, places=3)
            self.assertIsNone(status["open_position"])
            self.assertAlmostEqual(status["cash_usdt"], sells[0]["net_proceeds_usdt"], places=7)


if __name__ == "__main__":
    unittest.main()
