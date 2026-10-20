import unittest

from portfolio_accounting import (MAX_POSITION_FRACTION, buy_with_costs, entry_budget,
                                  mark_to_market, sell_with_costs)


class SharedEntryBudgetTests(unittest.TestCase):
    def test_entry_budget_caps_new_position_at_half_equity_with_buffer(self):
        self.assertAlmostEqual(entry_budget(15.0, 15.0), 15.0 * MAX_POSITION_FRACTION * 0.995)
        self.assertAlmostEqual(entry_budget(4.0, 15.0), 4.0)
        self.assertEqual(entry_budget(0.0, 15.0), 0.0)

    def test_entry_budget_rejects_invalid_inputs(self):
        for args in ((-1.0, 15.0), (15.0, -1.0), (15.0, 15.0, 1.5, 0.9)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    entry_budget(*args)


class WrongSymbolValuationRegressionTests(unittest.TestCase):
    def test_held_symbol_uses_its_own_mark_not_champion_mark(self):
        equity, mark = mark_to_market(
            cash=2.0,
            quantity=0.5,
            position_symbol="BTCUSDT",
            mark_prices={"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0},
        )
        self.assertEqual(mark, 60_000.0)
        self.assertEqual(equity, 30_002.0)

    def test_missing_held_symbol_mark_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "BTCUSDT"):
            mark_to_market(2.0, 0.5, "BTCUSDT", {"ETHUSDT": 3_000.0})

    def test_flat_portfolio_does_not_require_a_mark(self):
        self.assertEqual(mark_to_market(15.0, 0.0, None, {}), (15.0, 0.0))


class LedgerAccountingRegressionTests(unittest.TestCase):
    def test_buy_spends_cash_once_and_entry_basis_includes_fee_and_slippage(self):
        result = buy_with_costs(15.0, 100.0, fee_bps=60, slippage_bps=10)
        self.assertGreater(result.quantity, 0)
        self.assertAlmostEqual(result.cash_remaining, 0.0, places=9)
        self.assertAlmostEqual(
            result.quantity * result.entry_cost_per_unit, 15.0, places=9
        )
        self.assertAlmostEqual(result.execution_price, 100.1, places=8)

    def test_sell_realized_pnl_equals_net_proceeds_minus_entry_basis(self):
        buy = buy_with_costs(15.0, 100.0, fee_bps=60, slippage_bps=10)
        sell = sell_with_costs(
            buy.quantity, 110.0, buy.entry_cost_per_unit,
            fee_bps=60, slippage_bps=10,
        )
        self.assertAlmostEqual(
            sell.realized_pnl,
            sell.net_proceeds - buy.quantity * buy.entry_cost_per_unit,
            places=10,
        )
        self.assertAlmostEqual(sell.execution_price, 109.89, places=8)

    def test_round_trip_costs_match_per_side_bps_convention(self):
        buy = buy_with_costs(15.0, 100.0, fee_bps=60, slippage_bps=10)
        sell = sell_with_costs(
            buy.quantity, 100.0, buy.entry_cost_per_unit,
            fee_bps=60, slippage_bps=10,
        )
        self.assertAlmostEqual(
            100 * (1 - sell.net_proceeds / 15.0), 1.3903, delta=0.02
        )

    def test_invalid_negative_or_non_finite_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            buy_with_costs(15.0, 100.0, fee_bps=-1, slippage_bps=5)
        with self.assertRaises(ValueError):
            sell_with_costs(1.0, 100.0, 100.0, fee_bps=60, slippage_bps=-1)
        with self.assertRaises(ValueError):
            buy_with_costs(15.0, 100.0, fee_bps=float("nan"), slippage_bps=5)
        with self.assertRaises(ValueError):
            mark_to_market(0.0, 1.0, "BTCUSDT", {"BTCUSDT": float("nan")})


if __name__ == "__main__":
    unittest.main()
