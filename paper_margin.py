"""Paper-only isolated-margin execution helpers.

No exchange APIs or live order code are used. Position sizes are capped by
available equity and MAX_LEVERAGE. Liquidation is an intentionally conservative
bar-close approximation, not an exchange-accurate liquidation engine.
"""
from __future__ import annotations

from typing import Any

FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
MAX_LEVERAGE = 2.0
MAINTENANCE_MARGIN_RATE = 0.05


def open_position(
    state: dict[str, Any], symbol: str, side: int, mark_price: float,
    leverage: float = 1.0, fee_bps: float = FEE_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
) -> dict[str, Any]:
    """Open a long (side=1) or short (side=-1) isolated-margin paper position."""
    if side not in (-1, 1):
        raise ValueError("side must be 1 (long) or -1 (short)")
    if mark_price <= 0:
        raise ValueError("mark_price must be positive")
    if not 1.0 <= leverage <= MAX_LEVERAGE:
        raise ValueError(f"leverage must be between 1 and {MAX_LEVERAGE}x")
    if float(state.get("quantity", 0.0)) != 0.0:
        raise ValueError("close the current position before opening another")
    cash = float(state.get("cash", 0.0))
    if cash <= 0:
        raise ValueError("insufficient free cash")
    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0
    fill = mark_price * (1 + side * slip_rate)
    # Reserve nearly all free cash as isolated margin while leaving enough to pay
    # the entry fee; notional is capped at leverage * margin.
    margin = cash / (1.0 + leverage * fee_rate)
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
            "quantity": quantity, "leverage": leverage,
            "margin_reserved": margin, "fee": entry_fee}


def mark_position(state: dict[str, Any], mark_price: float) -> dict[str, float | bool]:
    """Mark isolated margin to market and flag approximate liquidation."""
    qty = float(state.get("quantity", 0.0))
    if qty == 0:
        return {"unrealized_pnl": 0.0, "equity": float(state.get("cash", 0.0)),
                "liquidation": False}
    side = int(state.get("side", 1))
    entry = float(state["entry_price"])
    margin = float(state["margin"])
    leverage = float(state.get("leverage", 1.0))
    pnl = side * qty * (mark_price - entry)
    notional = qty * mark_price
    liquidation = margin + pnl <= notional * MAINTENANCE_MARGIN_RATE
    equity = float(state.get("cash", 0.0)) + margin + pnl
    return {"unrealized_pnl": pnl, "equity": equity, "liquidation": liquidation}


def close_position(
    state: dict[str, Any], mark_price: float, reason: str = "signal",
    fee_bps: float = FEE_BPS, slippage_bps: float = SLIPPAGE_BPS,
) -> dict[str, Any] | None:
    """Close a long or short, returning isolated margin plus net P/L to free cash."""
    qty = float(state.get("quantity", 0.0))
    if qty == 0:
        return None
    side = int(state.get("side", 1))
    slip_rate = slippage_bps / 10000.0
    fee_rate = fee_bps / 10000.0
    fill = mark_price * (1 - side * slip_rate)
    entry = float(state["entry_price"])
    margin = float(state["margin"])
    gross_pnl = side * qty * (fill - entry)
    exit_fee = qty * fill * fee_rate
    entry_fee = float(state.get("entry_fee", 0.0))
    net_pnl = gross_pnl - entry_fee - exit_fee
    state["cash"] = float(state.get("cash", 0.0)) + max(0.0, margin + gross_pnl - exit_fee)
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + net_pnl
    state["completed_trades"] = int(state.get("completed_trades", 0)) + 1
    result = {"action": "PAPER_SELL" if side == 1 else "PAPER_COVER",
              "symbol": state.get("symbol"), "side": side, "price": fill,
              "quantity": qty, "gross_pnl": gross_pnl, "net_pnl": net_pnl,
              "fee": exit_fee, "reason": reason}
    for key, value in (("quantity", 0.0), ("side", 0), ("entry_price", None),
                       ("margin", 0.0), ("leverage", 1.0), ("entry_fee", 0.0),
                       ("symbol", None)):
        state[key] = value
    return result
