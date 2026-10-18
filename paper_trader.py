from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from paper_margin import close_position, mark_position, open_position, MAX_LEVERAGE

from research_engine import (
    INTERVALS, REPORTS, STATE, SYMBOLS, fetch_klines, desired_signal, add_features, resample_ohlcv, utc_now
)

STARTING_CASH = 15.0
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
PAPER_ALLOW_SHORTS = os.getenv("PAPER_ALLOW_SHORTS", "true").lower() in ("1", "true", "yes")
PAPER_LEVERAGE = float(os.getenv("PAPER_LEVERAGE", "1.0"))
if not 1.0 <= PAPER_LEVERAGE <= MAX_LEVERAGE:
    raise ValueError(f"PAPER_LEVERAGE must be between 1.0 and {MAX_LEVERAGE}x")
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
                   "paper_leverage_enabled": PAPER_LEVERAGE > 1.0, "paper_shorts_enabled": PAPER_ALLOW_SHORTS, "max_paper_leverage": MAX_LEVERAGE}
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
    target_side = int(signals.iloc[-1])
    champion_allows_shorts = bool(champion.get("allow_short", False))
    if not PAPER_ALLOW_SHORTS or not champion_allows_shorts:
        target_side = max(0, target_side)
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
        "signal_long": target_side > 0,\n        "signal_short": target_side < 0,\n        "target_side": target_side,
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
    state.setdefault("side", 1 if qty > 0 else 0)
    state.setdefault("margin", 0.0)
    state.setdefault("leverage", 1.0)
    state.setdefault("entry_fee", 0.0)
    if qty > 0 and not state.get("margin") and state.get("entry_price"):
        # Migrate a legacy spot position into 1x isolated-margin paper accounting.
        state["margin"] = qty * float(state["entry_price"])
        state["side"] = 1
        state["leverage"] = 1.0
    if cursor != latest_time:
        liquidated = False
        if qty > 0:
            mark = mark_position(state, latest_price)
            if mark["liquidation"]:
                closed = close_position(state, latest_price, reason="simulated liquidation",
                                        fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
                if closed:
                    status["actions"].append(closed)
                qty, position_symbol = 0.0, None
                liquidated = True
            elif position_symbol != symbol or target_side != int(state.get("side", 0)):
                closed = close_position(state, latest_price, reason="validated champion exit or side change",
                                        fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
                if closed:
                    status["actions"].append(closed)
                qty, position_symbol = 0.0, None
        if qty <= 0 and target_side != 0 and not liquidated:
            opened = open_position(state, symbol, target_side, latest_price,
                                   leverage=PAPER_LEVERAGE, fee_bps=FEE_BPS,
                                   slippage_bps=SLIPPAGE_BPS)
            status["actions"].append(opened)
            qty = float(state["quantity"])
            position_symbol = symbol
        state.setdefault("last_processed_bars", {})[symbol] = latest_time
    state["last_checked_at"] = utc_now()
    state["last_price"] = latest_price
    qty = float(state.get("quantity", 0.0))
    mark = mark_position(state, latest_price)
    equity = float(mark["equity"])
    unrealized = float(mark["unrealized_pnl"])
    entry_price = float(state.get("entry_price") or latest_price)
    position_symbol = state.get("symbol") if qty > 0 else None
    state["cash"] = float(state.get("cash", 0.0))
    status.update({
        "updated_at": utc_now(), "cash_usdt": round(state["cash"], 8),
        "equity_usdt": round(equity, 8),
        "total_pnl_usdt": round(equity - STARTING_CASH, 8),
        "realized_pnl_usdt": round(float(state.get("realized_pnl", 0.0)), 8),
        "unrealized_pnl_usdt": round(unrealized, 8),
        "completed_trades": int(state.get("completed_trades", 0)),
        "open_position": ({
            "symbol": position_symbol, "side": "long" if int(state.get("side", 0)) == 1 else "short",
            "quantity": round(qty, 10), "mark_price": round(latest_price, 8),
            "entry_price": round(entry_price, 8),
            "leverage": float(state.get("leverage", 1.0)),
            "margin_reserved_usdt": round(float(state.get("margin", 0.0)), 8)
        } if position_symbol else None),
        "paper_risk_config": {"shorts_enabled": PAPER_ALLOW_SHORTS,
                              "leverage": PAPER_LEVERAGE, "max_leverage": MAX_LEVERAGE},
        "champion": {"symbol": symbol, "interval": interval,
                     "strategy": champion["strategy"], "allow_short": champion_allows_shorts,
                     "params": params},
        "last_processed_bar": latest_time,
        "signal_diagnostics": signal_diagnostics,
        "message": "Paper-only isolated-margin simulation; liquidation and fills are estimates, not exchange execution."
    })
        STATE_FILE.write_text(json.dumps(state, indent=2))
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    print(json.dumps(run_once(), indent=2))
