"""Pure, deterministic accounting primitives for the paper-only portfolio."""

from __future__ import annotations

from dataclasses import dataclass
import math


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


MAX_POSITION_FRACTION = 0.50
ENTRY_BUDGET_BUFFER = 0.995


def entry_budget(cash: float, equity: float,
                 max_fraction: float = MAX_POSITION_FRACTION,
                 buffer: float = ENTRY_BUDGET_BUFFER) -> float:
    """Shared paper/backtest entry budget, leaving room for execution costs."""
    if not all(math.isfinite(value) for value in (cash, equity, max_fraction, buffer)):
        raise ValueError("cash, equity, max_fraction, and buffer must be finite")
    if cash < 0 or equity < 0 or not 0 <= max_fraction <= 1 or not 0 <= buffer <= 1:
        raise ValueError("cash/equity must be non-negative and fractions must be in [0, 1]")
    return min(cash, equity * max_fraction * buffer)


def _validate_costs(fee_bps: float, slippage_bps: float) -> tuple[float, float]:
    if (not math.isfinite(fee_bps) or not math.isfinite(slippage_bps)
            or fee_bps < 0 or slippage_bps < 0):
        raise ValueError("fee_bps and slippage_bps must be finite and non-negative")
    return fee_bps / 10_000.0, slippage_bps / 10_000.0


def buy_with_costs(cash: float, market_price: float, fee_bps: float = 10.0,
                   slippage_bps: float = 5.0) -> BuyResult:
    """Buy with all available cash; entry basis includes execution slippage and fee."""
    fee_rate, slippage_rate = _validate_costs(fee_bps, slippage_bps)
    if not math.isfinite(cash) or not math.isfinite(market_price) or cash < 0 or market_price <= 0:
        raise ValueError("cash must be finite and non-negative; market_price must be finite and positive")
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
    if (not all(math.isfinite(value) for value in (quantity, market_price, entry_cost_per_unit))
            or quantity < 0 or market_price <= 0 or entry_cost_per_unit < 0):
        raise ValueError("quantity and cost basis must be finite/non-negative; market price finite/positive")
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
    if not math.isfinite(cash) or not math.isfinite(quantity) or cash < 0 or quantity < 0:
        raise ValueError("cash and quantity must be finite and non-negative")
    if quantity == 0 or position_symbol is None:
        return cash, 0.0
    price = mark_prices.get(position_symbol)
    if price is None or not math.isfinite(price) or price <= 0:
        raise ValueError(f"Missing valid mark price for held symbol {position_symbol}")
    return cash + quantity * price, price
