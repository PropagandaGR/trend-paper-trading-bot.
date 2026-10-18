from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_accounting import buy_with_costs, mark_to_market, sell_with_costs

from research_engine import (
    INTERVALS, REPORTS, STATE, SYMBOLS, fetch_klines, desired_signal, add_features, resample_ohlcv, utc_now
)

STARTING_CASH = 15.0
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
STATE_FILE = STATE / "paper_state.json"
CHAMPION_FILE = STATE / "champion.json"
STATUS_FILE = REPORTS / "paper_status.json"


def champion_validation_error(champion: Any, now: datetime | None = None) -> str | None:
    """Return a fail-closed rejection reason for any unvalidated/stale champion."""
    if not isinstance(champion, dict):
        return "No champion is available."
    if champion.get("symbol") not in SYMBOLS:
        return "Champion symbol is unsupported."
    if champion.get("interval") not in INTERVALS:
        return "Champion interval is unsupported."
    if champion.get("strategy") not in {"trend", "breakout", "mean_reversion"}:
        return "Champion strategy is unsupported."
    params = champion.get("params")
    if not isinstance(params, dict) or not {"fast", "slow", "momentum", "breakout"}.issubset(params):
        return "Champion parameters are incomplete."
    if not isinstance(champion.get("champion_version"), str) or not champion["champion_version"].startswith("champion-"):
        return "Champion version is missing."
    metadata = champion.get("validation_metadata")
    if not isinstance(metadata, dict) or metadata.get("status") != "validated":
        return "Champion has no explicit validation metadata."
    if metadata.get("test_used_for_selection") is not False:
        return "Champion metadata does not confirm final-test isolation."
    for field in ("validation_metrics", "validation_stress_metrics", "test_metrics", "test_stress_metrics"):
        if not isinstance(metadata.get(field), dict):
            return f"Champion validation metadata is missing {field}."
    cost = metadata.get("cost_convention")
    if not isinstance(cost, dict) or cost.get("fee_basis") != "executed notional":
        return "Champion cost convention is missing or unsupported."
    created_at = metadata.get("created_at")
    if not isinstance(created_at, str):
        return "Champion validation timestamp is missing."
    try:
        validated_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return "Champion validation timestamp is invalid."
    if validated_at.tzinfo is None:
        return "Champion validation timestamp must include a timezone."
    now = now or datetime.now(timezone.utc)
    validated_at = validated_at.astimezone(timezone.utc)
    age = now.astimezone(timezone.utc) - validated_at
    if age < -timedelta(minutes=5):
        return "Champion validation timestamp is in the future."
    if age > timedelta(hours=24):
        return "Champion validation is stale (older than 24 hours)."
    return None


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
    champion_error = champion_validation_error(champion)
    champion_accepted = champion_error is None
    status = {
        "updated_at": utc_now(), "mode": "PAPER_ONLY",
        "starting_cash_usdt": STARTING_CASH, "cash_usdt": round(float(state["cash"]), 8),
        "equity_usdt": round(float(state["cash"]), 8), "unrealized_pnl_usdt": 0.0,
        "realized_pnl_usdt": round(float(state.get("realized_pnl", 0.0)), 8),
        "completed_trades": int(state.get("completed_trades", 0)),
        "open_position": None, "champion_available": champion_accepted, "champion_rejection_reason": champion_error,
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
    if not champion_accepted:
        state["last_checked_at"] = utc_now()
        STATE_FILE.write_text(json.dumps(state, indent=2))
        status["equity_usdt"] = round(float(state["cash"]), 8)
        status["message"] = champion_error or "No validated champion; capital remains in simulated cash."
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
        bars = resample_ohlcv(bars, INTERVALS[interval])
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
    features = add_features(
        bars, int(params["fast"]), int(params["slow"]),
        int(params["momentum"]), int(params["breakout"])
    )
    latest_features = features.iloc[-1]
    signal_diagnostics = {
        "symbol": symbol,
        "strategy": champion["strategy"],
        "interval": interval,
        "latest_bar": latest_time,
        "last_processed_bar_before_run": cursor,
        "new_bar_to_process": cursor != latest_time,
        "close_usdt": round(latest_price, 8),
        "signal_long": target_long,
        "current_signal": float(signals.iloc[-1]),
        "previous_signal": float(signals.iloc[-2]),
        "rsi": round(float(latest_features["rsi"]), 4),
        "ema_fast": round(float(latest_features["ema_fast"]), 8),
        "ema_slow": round(float(latest_features["ema_slow"]), 8),
        "momentum": round(float(latest_features["momentum"]), 8)
            if pd.notna(latest_features["momentum"]) else None,
    }
    if champion["strategy"] == "mean_reversion":
        rsi_entry = float(params["rsi_entry"])
        below_ema = bool(latest_features["close"] < latest_features["ema_slow"])
        rsi_low = bool(latest_features["rsi"] < rsi_entry)
        signal_diagnostics.update({
            "rsi_entry_threshold": rsi_entry,
            "rsi_below_entry_threshold": rsi_low,
            "close_below_ema_slow": below_ema,
            "raw_entry_conditions_met": rsi_low and below_ema,
        })
    if cursor == latest_time:
        signal_diagnostics["explanation"] = "This candle was already processed; no duplicate simulated order is placed."
    elif target_long:
        signal_diagnostics["explanation"] = "The strategy signal is long; the paper trader should enter or remain invested on this new candle."
    elif champion["strategy"] == "mean_reversion":
        failed = []
        if not signal_diagnostics["rsi_below_entry_threshold"]:
            failed.append(f"RSI is not below {signal_diagnostics['rsi_entry_threshold']}")
        if not signal_diagnostics["close_below_ema_slow"]:
            failed.append("close is not below the slow EMA")
        signal_diagnostics["explanation"] = (
            "No entry: " + " and ".join(failed) + "."
            if failed else
            "The current candle meets raw entry conditions, but the strategy's one-candle signal shift means the active signal may not have switched long yet."
        )
    else:
        signal_diagnostics["explanation"] = "The latest strategy signal is flat, so the bot stays in cash rather than forcing a trade."
    position_symbol = state.get("symbol")
    qty = float(state.get("quantity", 0.0))
    cash = float(state.get("cash", STARTING_CASH))
    # Keep marks symbol-scoped. Never reuse the champion's price to value a
    # different asset held from an earlier champion.
    current_prices = {
        name: float(frame.close.iloc[-1])
        for name, frame in frames.items()
        if not frame.empty and float(frame.close.iloc[-1]) > 0
    }
    cached_prices = state.get("last_prices", {})
    mark_prices = {**cached_prices, **current_prices}
    state["last_prices"] = mark_prices
    if qty > 0 and position_symbol not in mark_prices:
        status["data_errors"].append({
            "symbol": position_symbol,
            "error": "Missing valid mark for held position; refusing cross-symbol valuation or trade."
        })
        status["message"] = "Held position cannot be safely valued; no simulated order made."
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status

    if cursor != latest_time:
        # One-position, long-only portfolio. All fills are simulated at the latest
        # closed candle price with a conservative cost haircut.
        if qty > 0 and (position_symbol != symbol or not target_long):
            held_mark = float(mark_prices[position_symbol])
            entry = float(state.get("entry_price") or held_mark)
            sale = sell_with_costs(qty, held_mark, entry, FEE_BPS, SLIPPAGE_BPS)
            cash += sale.net_proceeds
            state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + sale.realized_pnl
            state["completed_trades"] = int(state.get("completed_trades", 0)) + 1
            status["actions"].append({
                "action": "PAPER_SELL", "symbol": position_symbol,
                "price": round(sale.execution_price, 8), "quantity": qty,
                "fee_usdt": round(sale.fee, 8), "slippage_usdt": round(sale.slippage, 8),
                "net_proceeds_usdt": round(sale.net_proceeds, 8),
                "realized_pnl_usdt": round(sale.realized_pnl, 8),
                "reason": "validated champion exit or symbol change"
            })
            qty, position_symbol = 0.0, None
            state["entry_price"] = None
        if qty <= 0 and target_long:
            # Buy quantity is calculated after modeled execution slippage and fee.
            buy = buy_with_costs(cash, latest_price, FEE_BPS, SLIPPAGE_BPS)
            qty = max(0.0, buy.quantity)
            cash = buy.cash_remaining
            position_symbol = symbol
            state["entry_price"] = buy.entry_cost_per_unit
            status["actions"].append({
                "action": "PAPER_BUY", "symbol": symbol,
                "price": round(buy.execution_price, 8), "quantity": qty,
                "fee_usdt": round(buy.fee, 8), "slippage_usdt": round(buy.slippage, 8),
                "reason": "validated champion entry"
            })
        state.setdefault("last_processed_bars", {})[symbol] = latest_time
    state.update({"cash": cash, "quantity": qty, "symbol": position_symbol,
                  "last_checked_at": utc_now()})
    equity, position_mark = mark_to_market(cash, qty, position_symbol, mark_prices)
    entry_price = float(state.get("entry_price") or position_mark or latest_price)
    unrealized = (position_mark - entry_price) * qty if position_symbol and qty > 0 else 0.0
    status.update({
        "updated_at": utc_now(), "cash_usdt": round(cash, 8),
        "equity_usdt": round(equity, 8),
        "total_pnl_usdt": round(equity - STARTING_CASH, 8),
        "unrealized_pnl_usdt": round(unrealized, 8),
        "open_position": ({"symbol": position_symbol, "quantity": round(qty, 10),
                           "mark_price": round(position_mark, 8),
                           "entry_price": round(entry_price, 8)} if position_symbol and qty > 0 else None),
        "champion": {"symbol": symbol, "interval": interval,
                     "strategy": champion["strategy"], "params": params},
        "last_processed_bar": latest_time,
        "signal_diagnostics": signal_diagnostics,
        "message": "Simulated portfolio only; fills are estimates, not exchange orders."
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    print(json.dumps(run_once(), indent=2))
