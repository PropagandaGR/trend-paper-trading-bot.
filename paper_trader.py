from __future__ import annotations

import json
import math
import os
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paper_margin import close_position, mark_position, open_position, MAX_LEVERAGE
from research_engine import (
    INTERVALS, REPORTS, STATE, SYMBOLS, add_features, desired_signal,
    fetch_klines, resample_ohlcv, utc_now,
)

STARTING_CASH = 15.0
FEE_BPS = 10.0
SLIPPAGE_BPS = 5.0
PAPER_ALLOW_SHORTS = os.getenv("PAPER_ALLOW_SHORTS", "true").lower() in ("1", "true", "yes")
PAPER_LEVERAGE = float(os.getenv("PAPER_LEVERAGE", "1.0"))
MAX_CHAMPION_AGE_HOURS = 24.0
MAX_DAILY_LOSS_PCT = 5.0
MAX_TOTAL_DRAWDOWN_PCT = 15.0
MAX_GROSS_EXPOSURE_MULTIPLE = 2.0
STATE_FILE = STATE / "paper_state.json"
CHAMPION_FILE = STATE / "champion.json"
STATUS_FILE = REPORTS / "paper_status.json"

if not math.isfinite(PAPER_LEVERAGE) or not 1.0 <= PAPER_LEVERAGE <= MAX_LEVERAGE:
    raise ValueError(f"PAPER_LEVERAGE must be between 1.0 and {MAX_LEVERAGE}x")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _valid_champion(champion: Any) -> tuple[bool, str]:
    if not isinstance(champion, dict):
        return False, "No champion record is available."
    if champion.get("passed_validation") is not True:
        return False, "Champion is missing an explicit passed_validation=true flag."
    stamp = champion.get("validated_at")
    if not stamp:
        return False, "Champion has no validation timestamp; legacy champion is blocked."
    try:
        validated_at = pd.Timestamp(stamp)
        if validated_at.tzinfo is None:
            validated_at = validated_at.tz_localize("UTC")
        validated_at = validated_at.tz_convert("UTC")
    except (TypeError, ValueError):
        return False, "Champion validation timestamp is invalid."
    age_hours = (_now() - validated_at).total_seconds() / 3600
    if age_hours < -0.05:
        return False, "Champion validation timestamp is in the future."
    if age_hours > MAX_CHAMPION_AGE_HOURS:
        return False, f"Champion validation is stale ({age_hours:.1f} hours old)."
    if champion.get("symbol") not in SYMBOLS:
        return False, "Champion symbol is not in the allowed universe."
    if champion.get("interval", "5m") not in INTERVALS:
        return False, "Champion interval is unsupported."
    if champion.get("strategy") not in ("trend", "breakout", "mean_reversion"):
        return False, "Champion strategy is unsupported."
    if not isinstance(champion.get("params"), dict):
        return False, "Champion parameters are missing."
    return True, "Champion validation is current."


def _prepare_bars(frame: pd.DataFrame, interval: str) -> tuple[pd.DataFrame | None, str]:
    if frame is None or frame.empty:
        return None, "No candles returned."
    bars = frame if interval == "5m" else resample_ohlcv(frame, INTERVALS[interval])
    if bars.empty or "close_time" not in bars.columns:
        return None, "No complete closed candle is available."
    close_time = pd.to_datetime(bars["close_time"].iloc[-1], utc=True, errors="coerce")
    if pd.isna(close_time):
        return None, "Latest candle close timestamp is invalid."
    now = _now()
    if close_time > now:
        return None, "Latest candle is not closed yet."
    interval_delta = pd.Timedelta(INTERVALS[interval])
    max_age = interval_delta * 2 + pd.Timedelta(minutes=1)
    age = now - close_time
    if age > max_age:
        return None, f"Latest candle is stale ({age.total_seconds() / 60:.1f} minutes old)."
    try:
        price = float(bars["close"].iloc[-1])
    except (TypeError, ValueError):
        return None, "Latest close price is not numeric."
    if not math.isfinite(price) or price <= 0:
        return None, "Latest close price is not finite and positive."
    for col in ("open", "high", "low", "close"):
        values = pd.to_numeric(bars[col].tail(3), errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all() or (values <= 0).any():
            return None, f"Recent OHLC data contains invalid {col} values."
    return bars, "Price and candle freshness checks passed."


def _snapshot(state: dict[str, Any], price: float | None = None) -> dict[str, Any]:
    qty = float(state.get("quantity", 0.0))
    if qty <= 0:
        equity = float(state.get("cash", STARTING_CASH))
        return {"equity": equity, "unrealized_pnl": 0.0, "gross_exposure": 0.0,
                "liquidation": False}
    if price is None:
        return {"equity": float(state.get("last_equity", state.get("cash", 0.0))),
                "unrealized_pnl": float(state.get("last_unrealized_pnl", 0.0)),
                "gross_exposure": float(state.get("last_gross_exposure", 0.0)),
                "liquidation": False}
    return mark_position(state, price)


def _save(state: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    STATE_FILE.write_text(json.dumps(state, indent=2))
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    return status


def run_once() -> dict[str, Any]:
    STATE.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    state = read_json(STATE_FILE, {
        "cash": STARTING_CASH, "quantity": 0.0, "symbol": None,
        "entry_price": None, "last_processed_bars": {}, "starting_cash": STARTING_CASH,
        "completed_trades": 0, "realized_pnl": 0.0, "side": 0,
        "margin": 0.0, "leverage": 1.0, "entry_fee": 0.0,
    })
    state.setdefault("last_processed_bars", {})
    state.setdefault("completed_trades", 0)
    state.setdefault("realized_pnl", 0.0)
    state.setdefault("starting_cash", STARTING_CASH)
    state.setdefault("high_water_equity", float(state.get("cash", STARTING_CASH)))
    state.setdefault("risk_day_utc", _now().date().isoformat())
    state.setdefault("day_start_equity", float(state.get("cash", STARTING_CASH)))
    champion = read_json(CHAMPION_FILE, None)
    champion_ok, champion_reason = _valid_champion(champion)
    status: dict[str, Any] = {
        "updated_at": utc_now(), "mode": "PAPER_ONLY",
        "starting_cash_usdt": STARTING_CASH, "cash_usdt": round(float(state.get("cash", 0.0)), 8),
        "equity_usdt": round(float(state.get("last_equity", state.get("cash", 0.0))), 8),
        "realized_pnl_usdt": round(float(state.get("realized_pnl", 0.0)), 8),
        "unrealized_pnl_usdt": round(float(state.get("last_unrealized_pnl", 0.0)), 8),
        "completed_trades": int(state.get("completed_trades", 0)),
        "open_position": None, "champion_available": bool(champion),
        "champion_valid": champion_ok, "champion_validation": champion_reason,
        "actions": [], "data_errors": [], "price_checks": {},
        "safety": {"real_orders": False, "exchange_credentials_used": False,
                   "paper_leverage_enabled": PAPER_LEVERAGE > 1.0,
                   "paper_shorts_enabled": PAPER_ALLOW_SHORTS,
                   "max_paper_leverage": MAX_LEVERAGE},
    }

    # Fetch each allowed asset once. Never substitute one symbol's price for another.
    raw_frames: dict[str, pd.DataFrame] = {}
    for asset in SYMBOLS:
        try:
            raw_frames[asset] = fetch_klines(asset, "5m", 5)
        except Exception as exc:
            status["data_errors"].append({"symbol": asset, "error": str(exc)})

    current_symbol = state.get("symbol") if float(state.get("quantity", 0.0)) > 0 else None
    current_interval = champion.get("interval", "5m") if champion_ok else "5m"
    prepared: dict[str, pd.DataFrame] = {}
    prepared_prices: dict[str, float] = {}
    for asset, raw in raw_frames.items():
        # Prepare each held/target symbol using its own timeframe and price.
        interval = current_interval if champion_ok and asset == champion["symbol"] else "5m"
        try:
            bars, reason = _prepare_bars(raw, interval)
        except Exception as exc:
            bars, reason = None, f"Candle preparation failed: {exc}"
        status["price_checks"][asset] = {"usable": bars is not None, "reason": reason,
                                          "interval": interval}
        if bars is not None:
            prepared[asset] = bars
            prepared_prices[asset] = float(bars["close"].iloc[-1])

    # Legacy open spot positions can be migrated safely to 1x accounting only.
    qty = float(state.get("quantity", 0.0))
    if qty > 0 and not state.get("margin") and state.get("entry_price"):
        state["margin"] = qty * float(state["entry_price"])
        state["side"] = 1
        state["leverage"] = 1.0
        state["entry_fee"] = 0.0

    # Mark the existing position ONLY with its own symbol's fresh mark.
    held_price = prepared_prices.get(current_symbol) if current_symbol else None
    if qty > 0 and held_price is None:
        state["last_checked_at"] = utc_now()
        status.update({
            "cash_usdt": round(float(state.get("cash", 0.0)), 8),
            "equity_usdt": round(float(state.get("last_equity", state.get("cash", 0.0))), 8),
            "open_position": {"symbol": current_symbol, "side": "long" if int(state.get("side", 1)) == 1 else "short",
                              "quantity": qty, "mark_price": None, "price_status": "stale_or_unavailable"},
            "message": "Held-asset price is stale or unavailable. Position is frozen; no mark, exit, or new trade is fabricated.",
            "risk_limits": {"daily_loss_pct": MAX_DAILY_LOSS_PCT,
                            "max_drawdown_pct": MAX_TOTAL_DRAWDOWN_PCT,
                            "max_gross_exposure_multiple": MAX_GROSS_EXPOSURE_MULTIPLE},
        })
        return _save(state, status)

    snap = _snapshot(state, held_price)
    equity = float(snap["equity"])
    state["last_equity"] = equity
    state["last_unrealized_pnl"] = float(snap["unrealized_pnl"])
    state["last_gross_exposure"] = float(snap.get("gross_exposure", 0.0))
    today = _now().date().isoformat()
    if state.get("risk_day_utc") != today:
        state["risk_day_utc"] = today
        state["day_start_equity"] = equity
        state["risk_halt_day_utc"] = None
    state["high_water_equity"] = max(float(state.get("high_water_equity", equity)), equity)
    day_start = max(float(state.get("day_start_equity", equity)), 1e-12)
    high_water = max(float(state.get("high_water_equity", equity)), 1e-12)
    daily_loss_pct = max(0.0, (day_start - equity) / day_start * 100)
    drawdown_pct = max(0.0, (high_water - equity) / high_water * 100)
    exposure_limit_breached = (
        qty > 0 and float(snap.get("gross_exposure", 0.0)) >
        max(equity, 0.0) * MAX_GROSS_EXPOSURE_MULTIPLE
    )
    risk_halt = (
        daily_loss_pct >= MAX_DAILY_LOSS_PCT or
        drawdown_pct >= MAX_TOTAL_DRAWDOWN_PCT or
        exposure_limit_breached
    )
    if risk_halt:
        state["risk_halt_day_utc"] = today

    # If the risk limit is breached, flatten only when the held asset has a fresh mark.
    if qty > 0 and risk_halt:
        closed = close_position(state, held_price, reason="portfolio loss safeguard",
                                fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
        if closed:
            status["actions"].append(closed)
        qty = 0.0
        current_symbol = None
        snap = _snapshot(state)
        equity = float(snap["equity"])
        state["last_equity"] = equity
        state["last_unrealized_pnl"] = 0.0
        state["last_gross_exposure"] = 0.0

    # A stale/invalid champion, stale target price, or loss halt blocks entries/exits.
    target_symbol = champion.get("symbol") if champion_ok else None
    target_bars = prepared.get(target_symbol) if target_symbol else None
    target_price = prepared_prices.get(target_symbol) if target_symbol else None
    can_consider_trade = bool(champion_ok and target_bars is not None and target_price is not None and not risk_halt)
    if champion_ok and target_bars is None:
        status["message"] = "Champion market data is stale or unavailable; no simulated trade is allowed."
    elif not champion_ok:
        status["message"] = f"Trading blocked: {champion_reason}"
    elif risk_halt:
        status["message"] = "Portfolio loss safeguard is active; new entries are blocked."

    if can_consider_trade:
        params = champion["params"]
        signals = desired_signal(target_bars, champion["strategy"], params,
                                 allow_short=bool(champion.get("allow_short", False)) and PAPER_ALLOW_SHORTS)
        latest_time = pd.Timestamp(target_bars.index[-1]).isoformat()
        target_side = int(signals.iloc[-1])
        if not PAPER_ALLOW_SHORTS or not bool(champion.get("allow_short", False)):
            target_side = max(0, target_side)
        cursor = state["last_processed_bars"].get(target_symbol)
        new_bar = cursor != latest_time
        status["signal_diagnostics"] = {
            "symbol": target_symbol, "interval": champion.get("interval", "5m"),
            "latest_bar": latest_time, "last_processed_bar_before_run": cursor,
            "new_bar_to_process": new_bar, "target_side": target_side,
            "price_usdt": target_price, "strategy": champion["strategy"],
        }
        if new_bar:
            # Exit at the held asset's own fresh price before considering a switch.
            if qty > 0 and (current_symbol != target_symbol or target_side != int(state.get("side", 0)) or target_side == 0):
                closed = close_position(state, held_price, reason="validated signal exit or symbol switch",
                                        fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
                if closed:
                    status["actions"].append(closed)
                qty = 0.0
                current_symbol = None
            # A daily halt persists through the UTC day, even if equity later recovers.
            if state.get("risk_halt_day_utc") == today:
                risk_halt = True
            if qty <= 0 and target_side != 0 and not risk_halt:
                free_cash = float(state.get("cash", 0.0))
                pre_open_equity = free_cash
                requested_notional = free_cash * PAPER_LEVERAGE
                if pre_open_equity <= 0 or requested_notional > pre_open_equity * MAX_GROSS_EXPOSURE_MULTIPLE:
                    status["message"] = "Exposure safeguard blocked the new position."
                else:
                    opened = open_position(state, target_symbol, target_side, target_price,
                                           leverage=PAPER_LEVERAGE, fee_bps=FEE_BPS,
                                           slippage_bps=SLIPPAGE_BPS)
                    status["actions"].append(opened)
                    qty = float(state["quantity"])
                    current_symbol = target_symbol
                    snap = mark_position(state, target_price)
                    equity = float(snap["equity"])
                    if float(snap["gross_exposure"]) > equity * MAX_GROSS_EXPOSURE_MULTIPLE:
                        # Defensive rollback: never keep a position beyond portfolio exposure cap.
                        close_position(state, target_price, reason="exposure cap rollback",
                                       fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS)
                        qty, current_symbol = 0.0, None
                        status["actions"].append({"action": "PAPER_ENTRY_BLOCKED", "reason": "gross exposure cap"})
                    state["last_equity"] = float(_snapshot(state, target_price if qty > 0 else None)["equity"])
            state["last_processed_bars"][target_symbol] = latest_time

    # Refresh the held position's mark using only its own symbol's fresh price.
    qty = float(state.get("quantity", 0.0))
    current_symbol = state.get("symbol") if qty > 0 else None
    final_price = prepared_prices.get(current_symbol) if current_symbol else None
    snap = _snapshot(state, final_price)
    equity = float(snap["equity"])
    state["last_equity"] = equity
    state["last_unrealized_pnl"] = float(snap["unrealized_pnl"])
    state["last_gross_exposure"] = float(snap.get("gross_exposure", 0.0))
    state["last_checked_at"] = utc_now()
    state["high_water_equity"] = max(float(state.get("high_water_equity", equity)), equity)
    day_start = max(float(state.get("day_start_equity", equity)), 1e-12)
    high_water = max(float(state.get("high_water_equity", equity)), 1e-12)
    daily_loss_pct = max(0.0, (day_start - equity) / day_start * 100)
    drawdown_pct = max(0.0, (high_water - equity) / high_water * 100)
    status.update({
        "updated_at": utc_now(), "cash_usdt": round(float(state.get("cash", 0.0)), 8),
        "equity_usdt": round(equity, 8),
        "total_pnl_usdt": round(equity - STARTING_CASH, 8),
        "realized_pnl_usdt": round(float(state.get("realized_pnl", 0.0)), 8),
        "unrealized_pnl_usdt": round(float(snap["unrealized_pnl"]), 8),
        "completed_trades": int(state.get("completed_trades", 0)),
        "gross_exposure_usdt": round(float(snap.get("gross_exposure", 0.0)), 8),
        "risk_limits": {
            "daily_loss_pct": MAX_DAILY_LOSS_PCT, "daily_loss_now_pct": round(daily_loss_pct, 4),
            "max_drawdown_pct": MAX_TOTAL_DRAWDOWN_PCT, "drawdown_now_pct": round(drawdown_pct, 4),
            "max_gross_exposure_multiple": MAX_GROSS_EXPOSURE_MULTIPLE,
            "risk_halt": bool(risk_halt or state.get("risk_halt_day_utc") == today),
        },
        "open_position": ({
            "symbol": current_symbol, "side": "long" if int(state.get("side", 0)) == 1 else "short",
            "quantity": round(qty, 10), "mark_price": round(float(final_price), 8) if final_price is not None else None,
            "entry_price": round(float(state.get("entry_price") or 0.0), 8),
            "leverage": float(state.get("leverage", 1.0)),
            "margin_reserved_usdt": round(float(state.get("margin", 0.0)), 8),
        } if current_symbol else None),
        "paper_risk_config": {"shorts_enabled": PAPER_ALLOW_SHORTS,
                              "leverage": PAPER_LEVERAGE, "max_leverage": MAX_LEVERAGE},
        "champion": ({"symbol": champion.get("symbol"), "interval": champion.get("interval", "5m"),
                      "strategy": champion.get("strategy"), "allow_short": bool(champion.get("allow_short", False))}
                     if champion_ok else None),
        "message": status.get("message", "Paper-only simulation. No exchange orders are possible."),
    })
    return _save(state, status)


if __name__ == "__main__":
    print(json.dumps(run_once(), indent=2))
