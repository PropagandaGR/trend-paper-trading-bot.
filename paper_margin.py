"""Paper-only isolated-margin ledger and risk helpers.

This module has no exchange/order API. Liquidation is approximate and is
intended for conservative simulation safeguards, not exchange replication.
"""
from __future__ import annotations

import math
from typing import Any

FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
MAX_LEVERAGE = 2.0
MAINTENANCE_MARGIN_RATE = 0.05


def _finite_positive(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def open_position(
    state: dict[str, Any], symbol: str, side: int, mark_price: float,
    leverage: float = 1.0, fee_bps: float = FEE_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Open a long (side=1) or short (side=-1) isolated-margin paper position."""
    mark_price = _finite_positive(mark_price, "mark_price")
    leverage = float(leverage)
    fee_bps, slippage_bps = float(fee_bps), float(slippage_bps)
    if side not in (-1, 1):
        raise ValueError("side must be 1 (long) or -1 (short)")
    if not math.isfinite(leverage) or not 1.0 <= leverage <= MAX_LEVERAGE:
        raise ValueError(f"leverage must be between 1 and {MAX_LEVERAGE}x")
    if not symbol:
        raise ValueError("symbol is required")
    if not math.isfinite(fee_bps) or fee_bps < 0 or not math.isfinite(slippage_bps) or slippage_bps < 0:
        raise ValueError("fees and slippage must be finite and non-negative")
    if float(state.get("quantity", 0.0)) != 0.0:
        raise ValueError("close the current position before opening another")
    cash = float(state.get("cash", 0.0))
    if not math.isfinite(cash) or cash <= 0:
        raise ValueError("insufficient or invalid free cash")
    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0
    fill = mark_price * (1 + side * slip_rate)
    # Keep enough free cash to absorb the entry fee and adverse entry slippage,
    # so marked gross exposure does not immediately exceed the equity cap.
    slippage_cost_rate = slip_rate / (1.0 + slip_rate) if side == 1 else slip_rate / (1.0 - slip_rate)
    margin = cash / (1.0 + leverage * (fee_rate + slippage_cost_rate))
    notional = margin * leverage
    quantity = notional / fill
    entry_fee = notional * fee_rate
    state["cash"] = cash - margin - entry_fee
    state.update({
        "symbol": symbol, "side": side, "quantity": quantity,
        "entry_price": fill, "margin": margin, "leverage": leverage,
        "entry_fee": entry_fee,
    })
    return {"action": "PAPER_BUY" if side == 1 else "PAPER_SHORT",
            "symbol": symbol, "side": side, "price": fill,
            "quantity": quantity, "notional": notional, "leverage": leverage,
            "margin_reserved": margin, "fee": entry_fee}


def mark_position(state: dict[str, Any], mark_price: float) -> dict[str, float | bool]:
    """Mark isolated margin to market and flag approximate liquidation."""
    mark_price = _finite_positive(mark_price, "mark_price")
    qty = float(state.get("quantity", 0.0))
    cash = float(state.get("cash", 0.0))
    if not math.isfinite(qty) or qty < 0 or not math.isfinite(cash):
        raise ValueError("invalid position quantity or cash ledger")
    if qty == 0:
        return {"unrealized_pnl": 0.0, "equity": cash, "liquidation": False,
                "gross_exposure": 0.0, "margin_ratio": 0.0}
    side = int(state.get("side", 0))
    if side not in (-1, 1):
        raise ValueError("open position has invalid side")
    entry = _finite_positive(state["entry_price"], "entry_price")
    margin = float(state["margin"])
    leverage = float(state.get("leverage", 1.0))
    if not math.isfinite(margin) or margin <= 0 or not 1.0 <= leverage <= MAX_LEVERAGE:
        raise ValueError("invalid margin or leverage in position ledger")
    pnl = side * qty * (mark_price - entry)
    notional = qty * mark_price
    margin_equity = margin + pnl
    liquidation = margin_equity <= notional * MAINTENANCE_MARGIN_RATE
    equity = cash + margin_equity
    return {"unrealized_pnl": pnl, "equity": equity, "liquidation": liquidation,
            "gross_exposure": notional,
            "margin_ratio": margin_equity / notional if notional else 0.0}


def close_position(
    state: dict[str, Any], mark_price: float, reason: str = "signal",
    fee_bps: float = FEE_BPS, slippage_bps: float = SLIPPAGE_BPS,
) -> dict[str, Any] | None:
    """Close a position and reconcile free cash, isolated margin and realized P/L."""
    mark_price = _finite_positive(mark_price, "mark_price")
    qty = float(state.get("quantity", 0.0))
    if qty == 0:
        return None
    if not math.isfinite(qty) or qty < 0:
        raise ValueError("invalid position quantity")
    side = int(state.get("side", 0))
    if side not in (-1, 1):
        raise ValueError("open position has invalid side")
    fee_bps, slippage_bps = float(fee_bps), float(slippage_bps)
    if not math.isfinite(fee_bps) or fee_bps < 0 or not math.isfinite(slippage_bps) or slippage_bps < 0:
        raise ValueError("fees and slippage must be finite and non-negative")
    slip_rate = slippage_bps / 10000.0
    fee_rate = fee_bps / 10000.0
    fill = mark_price * (1 - side * slip_rate)
    entry = _finite_positive(state["entry_price"], "entry_price")
    margin = float(state["margin"])
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("invalid reserved margin")
    gross_pnl = side * qty * (fill - entry)
    exit_fee = qty * fill * fee_rate
    entry_fee = float(state.get("entry_fee", 0.0))
    # Isolated margin cannot pay out less than zero. Cap losses to reserved
    # margin, and make the ledger P/L agree with the actual cash movement.
    payout = max(0.0, margin + gross_pnl - exit_fee)
    net_pnl = payout - margin - entry_fee
    state["cash"] = float(state.get("cash", 0.0)) + payout
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + net_pnl
    state["completed_trades"] = int(state.get("completed_trades", 0)) + 1
    result = {"action": "PAPER_SELL" if side == 1 else "PAPER_COVER",
              "symbol": state.get("symbol"), "side": side, "price": fill,
              "quantity": qty, "gross_pnl": gross_pnl, "net_pnl": net_pnl,
              "entry_fee": entry_fee, "exit_fee": exit_fee, "payout": payout,
              "reason": reason}
    for key, value in (("quantity", 0.0), ("side", 0), ("entry_price", None),
                       ("margin", 0.0), ("leverage", 1.0), ("entry_fee", 0.0),
                       ("symbol", None)):
        state[key] = value
    return result
