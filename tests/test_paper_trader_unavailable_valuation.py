import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import paper_trader


class UnavailablePositionValuationTests(unittest.TestCase):
    def _run_with_state(self, state, champion, fetch_side_effect):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        state_dir = root / "state"
        reports_dir = root / "reports"
        state_dir.mkdir()
        reports_dir.mkdir()
        state_file = state_dir / "paper_state.json"
        champion_file = state_dir / "champion.json"
        status_file = reports_dir / "paper_status.json"
        state_file.write_text(json.dumps(state))
        champion_file.write_text(json.dumps(champion))
        with (
            patch.object(paper_trader, "STATE", state_dir),
            patch.object(paper_trader, "REPORTS", reports_dir),
            patch.object(paper_trader, "STATE_FILE", state_file),
            patch.object(paper_trader, "CHAMPION_FILE", champion_file),
            patch.object(paper_trader, "STATUS_FILE", status_file),
            patch.object(paper_trader, "fetch_klines", side_effect=fetch_side_effect),
        ):
            status = paper_trader.run_once()
        persisted_state = json.loads(state_file.read_text())
        return status, persisted_state, json.loads(status_file.read_text())

    def _held_position(self):
        return {
            "cash": 0.0,
            "quantity": 0.25,
            "symbol": "BTCUSDT",
            "entry_price": 60000.0,
            "last_processed_bars": {},
            "starting_cash": 15.0,
            "completed_trades": 0,
            "realized_pnl": 0.0,
        }

    def test_no_market_data_does_not_report_cash_as_total_equity(self):
        def unavailable(*args, **kwargs):
            raise RuntimeError("public market data unavailable")

        status, persisted, written = self._run_with_state(
            self._held_position(), None, unavailable
        )

        for report in (status, written):
            self.assertIsNone(report["equity_usdt"])
            self.assertIsNone(report["total_pnl_usdt"])
            self.assertEqual(report["valuation_status"], "unavailable")
            self.assertEqual(report["open_position"]["symbol"], "BTCUSDT")
            self.assertEqual(report["open_position"]["quantity"], 0.25)
            self.assertIsNone(report["open_position"]["mark_price"])
            self.assertIn("cannot be valued", report["message"])
        self.assertEqual(persisted["quantity"], 0.25)
        self.assertEqual(persisted["symbol"], "BTCUSDT")
        self.assertEqual(status["actions"], [])

    def test_missing_held_symbol_mark_returns_unknown_valuation_without_crashing(self):
        champion = {"symbol": "ETHUSDT"}

        def one_symbol_only(symbol, *args, **kwargs):
            if symbol == "ETHUSDT":
                return pd.DataFrame()
            raise RuntimeError("BTC mark unavailable")

        status, persisted, _ = self._run_with_state(
            self._held_position(), champion, one_symbol_only
        )

        self.assertIsNone(status["equity_usdt"])
        self.assertEqual(status["valuation_status"], "unavailable")
        self.assertEqual(status["open_position"]["symbol"], "BTCUSDT")
        self.assertIsNone(status["open_position"]["mark_price"])
        self.assertEqual(persisted["quantity"], 0.25)
        self.assertEqual(status["actions"], [])


if __name__ == "__main__":
    unittest.main()
