import unittest

from paper_margin import close_position, mark_position, open_position


class MarginLedgerReconciliationTests(unittest.TestCase):
    def test_cash_plus_margin_and_pnl_reconcile_to_equity(self):
        state = {"cash": 15.0, "quantity": 0.0, "symbol": None,
                 "completed_trades": 0, "realized_pnl": 0.0}
        open_position(state, "BTCUSDT", 1, 100.0, leverage=2.0)
        marked = mark_position(state, 105.0)
        self.assertAlmostEqual(marked["equity"], state["cash"] + state["margin"] +
                               state["side"] * state["quantity"] * (105.0 - state["entry_price"]))
        before_close_cash = state["cash"]
        closed = close_position(state, 105.0)
        self.assertIsNotNone(closed)
        self.assertAlmostEqual(state["cash"], 15.0 + state["realized_pnl"], places=7)
        self.assertEqual(state["quantity"], 0.0)
        self.assertNotEqual(state["cash"], before_close_cash)

    def test_nonfinite_prices_are_rejected(self):
        state = {"cash": 15.0, "quantity": 0.0}
        with self.assertRaises(ValueError):
            open_position(state, "BTCUSDT", 1, float("nan"))
        open_position(state, "BTCUSDT", 1, 100.0)
        with self.assertRaises(ValueError):
            mark_position(state, float("inf"))


if __name__ == "__main__":
    unittest.main()
