import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paper_trader


def valid_state(**overrides):
    state = {
        "cash": 15.0,
        "quantity": 0.0,
        "symbol": None,
        "entry_price": None,
        "last_processed_bars": {},
        "starting_cash": 15.0,
        "completed_trades": 0,
        "realized_pnl": 0.0,
    }
    state.update(overrides)
    return state


class PaperStateValidationTests(unittest.TestCase):
    def test_missing_state_file_initializes_the_15_usdt_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper_state.json"
            with patch.object(paper_trader, "STATE_FILE", path):
                state = paper_trader.load_paper_state()
        self.assertEqual(state["cash"], 15.0)
        self.assertEqual(state["starting_cash"], 15.0)
        self.assertEqual(state["quantity"], 0.0)

    def test_corrupt_existing_state_fails_closed_without_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper_state.json"
            path.write_text("{not valid json")
            with patch.object(paper_trader, "STATE_FILE", path):
                with self.assertRaisesRegex(RuntimeError, "unreadable"):
                    paper_trader.load_paper_state()
            self.assertEqual(path.read_text(), "{not valid json")

    def test_negative_cash_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper_state.json"
            path.write_text(json.dumps(valid_state(cash=-1.0)))
            with patch.object(paper_trader, "STATE_FILE", path):
                with self.assertRaisesRegex(RuntimeError, "cash/quantity"):
                    paper_trader.load_paper_state()

    def test_flat_state_cannot_retain_a_position_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper_state.json"
            path.write_text(json.dumps(valid_state(symbol="BTCUSDT")))
            with patch.object(paper_trader, "STATE_FILE", path):
                with self.assertRaisesRegex(RuntimeError, "Flat paper state"):
                    paper_trader.load_paper_state()

    def test_existing_risk_halt_is_preserved(self):
        state_data = valid_state(risk_halt=True, risk_halt_reason="daily_loss_limit")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper_state.json"
            path.write_text(json.dumps(state_data))
            with patch.object(paper_trader, "STATE_FILE", path):
                state = paper_trader.load_paper_state()
        self.assertTrue(state["risk_halt"])
        self.assertEqual(state["risk_halt_reason"], "daily_loss_limit")


if __name__ == "__main__":
    unittest.main()
