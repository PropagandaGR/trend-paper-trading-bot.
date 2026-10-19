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
                    "oos_evaluation_policy_version": "chronological-oos-v1",
                    "validation_score": 1.0,
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
                    "validation_benchmarks": {"cash": {"total_return_pct": 0.0}, "buy_and_hold": {"total_return_pct": -0.5}},
                    "validation_stress_benchmarks": {"cash": {"total_return_pct": 0.0}, "buy_and_hold": {"total_return_pct": -1.0}},
                    "test_benchmarks": {"cash": {"total_return_pct": 0.0}, "buy_and_hold": {"total_return_pct": -0.5}},
                    "test_stress_benchmarks": {"cash": {"total_return_pct": 0.0}, "buy_and_hold": {"total_return_pct": -1.0}},
                    "test_deployment_gate": {"passed": True, "failures": []},
                    "cost_convention": {
                        "fee_basis": "executed notional",
                        "fill_timing": "closed candle close after one-candle signal shift",
                        "fee_bps_per_side": 60.0, "slippage_bps_per_side": 10.0,
                        "stress_fee_bps_per_side": 100.0, "stress_slippage_bps_per_side": 20.0
                    }
                }
            }))
            index = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5),
                                  periods=2, freq="5min")
            closes = index + pd.Timedelta(minutes=5)
            eth = pd.DataFrame({"close": [1_990.0, 2_000.0], "close_time": closes}, index=index)
            btc = pd.DataFrame({"close": [59_000.0, 60_000.0], "close_time": closes}, index=index)
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
            self.assertAlmostEqual(sells[0]["price"], 1_998.0, places=7)
            self.assertNotAlmostEqual(sells[0]["price"], 59_970.0, places=3)
            self.assertIsNone(status["open_position"])
            self.assertAlmostEqual(status["cash_usdt"], sells[0]["net_proceeds_usdt"], places=7)


if __name__ == "__main__":
    unittest.main()
