"""Pure, deterministic accounting primitives for the paper-only portfolio."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuyResult:
    quantity: float
    entry_cost_per_unit: float
    cash_remaining: float
    execution_price: float
    fee: float
    slippage: float


@dataclass(frozen=True)
class SellResult:
    net_proceeds: float
    realized_pnl: float
    execution_price: float
    fee: float
    slippage: float


def _validate_costs(fee_bps: float, slippage_bps: float) -> tuple[float, float]:
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("fee_bps and slippage_bps must be non-negative")
    return fee_bps / 10_000.0, slippage_bps / 10_000.0


def buy_with_costs(cash: float, market_price: float, fee_bps: float = 10.0,
                   slippage_bps: float = 5.0) -> BuyResult:
    """Buy with all available cash; entry basis includes execution slippage and fee."""
    fee_rate, slippage_rate = _validate_costs(fee_bps, slippage_bps)
    if cash < 0 or market_price <= 0:
        raise ValueError("cash must be non-negative and market_price must be positive")
    execution_price = market_price * (1.0 + slippage_rate)
    # Fee is charged on executed notional, in addition to the execution price.
    quantity = cash / (execution_price * (1.0 + fee_rate)) if cash else 0.0
    gross = quantity * execution_price
    fee = gross * fee_rate
    slippage = quantity * (execution_price - market_price)
    remaining = cash - gross - fee
    if abs(remaining) < 1e-10:
        remaining = 0.0
    return BuyResult(quantity, (gross + fee) / quantity if quantity else 0.0,
                     remaining, execution_price, fee, slippage)


def sell_with_costs(quantity: float, market_price: float, entry_cost_per_unit: float,
                    fee_bps: float = 10.0, slippage_bps: float = 5.0) -> SellResult:
    """Sell a long position; realized P/L uses net proceeds minus all-in entry basis."""
    fee_rate, slippage_rate = _validate_costs(fee_bps, slippage_bps)
    if quantity < 0 or market_price <= 0 or entry_cost_per_unit < 0:
        raise ValueError("quantity must be non-negative and prices/cost basis valid")
    execution_price = market_price * (1.0 - slippage_rate)
    gross = quantity * execution_price
    fee = gross * fee_rate
    net = gross - fee
    slippage = quantity * (market_price - execution_price)
    return SellResult(net, net - quantity * entry_cost_per_unit,
                      execution_price, fee, slippage)


def mark_to_market(cash: float, quantity: float, position_symbol: str | None,
                   mark_prices: dict[str, float]) -> tuple[float, float]:
    """Return (equity, position_mark); never value a holding with another symbol's price."""
    if cash < 0 or quantity < 0:
        raise ValueError("cash and quantity must be non-negative")
    if quantity == 0 or position_symbol is None:
        return cash, 0.0
    price = mark_prices.get(position_symbol)
    if price is None or price <= 0:
        raise ValueError(f"Missing valid mark price for held symbol {position_symbol}")
    return cash + quantity * price, price
