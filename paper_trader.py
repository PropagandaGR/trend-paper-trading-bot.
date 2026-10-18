from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research_engine import (
    INTERVALS, REPORTS, STATE, SYMBOLS, fetch_klines, desired_signal, utc_now
)

STARTING_CASH = 15.0
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
STATE_FILE = STATE / "paper_state.json"
CHAMPION_FILE = STATE / "champion.json"
STATUS_FILE = REPORTS / "paper_status.json"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def run_once() -> dict[str, Any]:
    STATE.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    state = read_json(STATE_FILE, {
        "cash": STARTING_CASH, "quantity": 0.0, "symbol": None,
        "entry_price": None, "last_processed_bars": {}, "starting_cash": STARTING_CASH,
        "completed_trades": 0, "realized_pnl": 0.0
    })
    champion = read_json(CHAMPION_FILE, None)
    status = {
        "updated_at": utc_now(), "mode": "PAPER_ONLY",
        "starting_cash_usdt": STARTING_CASH, "cash_usdt": round(float(state["cash"]), 8),
        "equity_usdt": round(float(state["cash"]), 8), "unrealized_pnl_usdt": 0.0,
        "realized_pnl_usdt": round(float(state.get("realized_pnl", 0.0)), 8),
        "completed_trades": int(state.get("completed_trades", 0)),
        "open_position": None, "champion_available": bool(champion),
        "actions": [], "data_errors": [],
        "safety": {"real_orders": False, "exchange_credentials_used": False,
                   "leverage": False, "short_selling": False}
    }
    frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        try:
            frames[symbol] = fetch_klines(symbol, "5m", 5)
        except Exception as exc:
            status["data_errors"].append({"symbol": symbol, "error": str(exc)})
    if not frames:
        status["updated_at"] = utc_now()
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status

    # No strategy has passed validation yet? Stay in cash; research cannot alter the
    # portfolio until it produces a validated champion.
    if not champion:
        state["last_checked_at"] = utc_now()
        STATE_FILE.write_text(json.dumps(state, indent=2))
        status["equity_usdt"] = round(float(state["cash"]), 8)
        status["message"] = "No validated champion yet; capital remains in simulated cash."
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status

    symbol = champion["symbol"]
    if symbol not in frames:
        status["message"] = "Champion symbol data unavailable; no simulated order made."
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status
    interval = champion.get("interval", "5m")
    bars = frames[symbol]
    if interval != "5m":
        bars = bars.resample(INTERVALS[interval], label="right", closed="right").agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum", "close_time": "last"
        }).dropna(subset=["open", "high", "low", "close"])
    params = champion["params"]
    signals = desired_signal(bars, champion["strategy"], params)
    if len(bars) < 2:
        status["message"] = "Waiting for sufficient closed candles."
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status
    latest_time = bars.index[-1].isoformat()
    cursor = state.setdefault("last_processed_bars", {}).get(symbol)
    latest_price = float(bars.close.iloc[-1])
    target_long = bool(signals.iloc[-1] > 0)
    position_symbol = state.get("symbol")
    qty = float(state.get("quantity", 0.0))
    cash = float(state.get("cash", STARTING_CASH))
    cost_rate = (FEE_BPS + SLIPPAGE_BPS) / 10000.0

    if cursor != latest_time:
        # One-position, long-only portfolio. All fills are simulated at the latest
        # closed candle price with a conservative cost haircut.
        if qty > 0 and (position_symbol != symbol or not target_long):
            fill = latest_price * (1 - cost_rate)
            proceeds = qty * fill
            entry = float(state.get("entry_price") or fill)
            pnl = proceeds - qty * entry
            cash += proceeds
            state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + pnl
            state["completed_trades"] = int(state.get("completed_trades", 0)) + 1
            status["actions"].append({"action": "PAPER_SELL", "symbol": position_symbol,
                                      "price": round(fill, 8), "quantity": qty,
                                      "reason": "validated champion exit or symbol change"})
            qty, position_symbol = 0.0, None
            state["entry_price"] = None
        if qty <= 0 and target_long:
            # Invest available cash, including the 15 USDT starting portfolio.
            fill = latest_price * (1 + cost_rate)
            spend = cash
            qty = max(0.0, spend / fill)
            cash = 0.0
            position_symbol = symbol
            state["entry_price"] = fill
            status["actions"].append({"action": "PAPER_BUY", "symbol": symbol,
                                      "price": round(fill, 8), "quantity": qty,
                                      "reason": "validated champion entry"})
        state.setdefault("last_processed_bars", {})[symbol] = latest_time
    state.update({"cash": cash, "quantity": qty, "symbol": position_symbol,
                  "last_checked_at": utc_now()})
    state["last_price"] = latest_price
    equity = cash + (qty * latest_price if position_symbol else 0.0)
    entry_price = float(state.get("entry_price") or latest_price)
    unrealized = (latest_price - entry_price) * qty if position_symbol else 0.0
    status.update({
        "updated_at": utc_now(), "cash_usdt": round(cash, 8),
        "equity_usdt": round(equity, 8),
        "total_pnl_usdt": round(equity - STARTING_CASH, 8),
        "unrealized_pnl_usdt": round(unrealized, 8),
        "open_position": ({"symbol": position_symbol, "quantity": round(qty, 10),
                           "mark_price": round(latest_price, 8),
                           "entry_price": round(entry_price, 8)} if position_symbol else None),
        "champion": {"symbol": symbol, "interval": interval,
                     "strategy": champion["strategy"], "params": params},
        "last_processed_bar": latest_time,
        "message": "Simulated portfolio only; fills are estimates, not exchange orders."
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    print(json.dumps(run_once(), indent=2))
