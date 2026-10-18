import unittest

from paper_margin import close_position, mark_position, open_position


class PaperMarginTests(unittest.TestCase):
    def make_state(self, cash=15.0):
        return {"cash": cash, "quantity": 0.0, "symbol": None,
                "completed_trades": 0, "realized_pnl": 0.0}

    def test_long_profit_after_costs(self):
        state = self.make_state()
        opened = open_position(state, "BTCUSDT", 1, 100.0, leverage=1.0)
        self.assertEqual(opened["action"], "PAPER_BUY")
        closed = close_position(state, 110.0)
        self.assertEqual(closed["action"], "PAPER_SELL")
        self.assertGreater(state["cash"], 15.0)
        self.assertEqual(state["quantity"], 0.0)

    def test_short_profit_when_price_falls(self):
        state = self.make_state()
        opened = open_position(state, "BTCUSDT", -1, 100.0, leverage=2.0)
        self.assertEqual(opened["action"], "PAPER_SHORT")
        closed = close_position(state, 90.0)
        self.assertEqual(closed["action"], "PAPER_COVER")
        self.assertGreater(state["cash"], 15.0)

    def test_leverage_cap_and_invalid_side(self):
        state = self.make_state()
        with self.assertRaises(ValueError):
            open_position(state, "BTCUSDT", 1, 100.0, leverage=3.0)
        with self.assertRaises(ValueError):
            open_position(state, "BTCUSDT", 0, 100.0)

    def test_no_overlapping_positions(self):
        state = self.make_state()
        open_position(state, "BTCUSDT", 1, 100.0)
        with self.assertRaises(ValueError):
            open_position(state, "ETHUSDT", -1, 100.0)

    def test_equity_and_liquidation_flag(self):
        state = self.make_state()
        open_position(state, "BTCUSDT", 1, 100.0, leverage=2.0)
        normal = mark_position(state, 99.0)
        adverse = mark_position(state, 40.0)
        self.assertGreater(normal["equity"], 0)
        self.assertTrue(adverse["liquidation"])

    def test_round_trip_costs_reduce_profit(self):
        state = self.make_state()
        open_position(state, "BTCUSDT", 1, 100.0, fee_bps=0, slippage_bps=0)
        zero_cost = close_position(state, 101.0, fee_bps=0, slippage_bps=0)["net_pnl"]
        state2 = self.make_state()
        open_position(state2, "BTCUSDT", 1, 100.0, fee_bps=10, slippage_bps=5)
        with_costs = close_position(state2, 101.0, fee_bps=10, slippage_bps=5)["net_pnl"]
        self.assertGreater(zero_cost, with_costs)


if __name__ == "__main__":
    unittest.main()
