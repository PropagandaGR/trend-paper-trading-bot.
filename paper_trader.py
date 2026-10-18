from __future__ import annotations

import json
import math
import re
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
MAX_POSITION_FRACTION = 0.50
MAX_DAILY_LOSS_PCT = 5.0
MAX_DRAWDOWN_PCT = 10.0
MAX_PRICE_AGE_SECONDS = 15 * 60


def champion_validation_error(champion: Any, now: datetime | None = None) -> str | None:
    """Fail closed on unsupported, malformed, stale, or incompletely validated champions."""
    if not isinstance(champion, dict):
        return "No champion is available."
    symbol = champion.get("symbol")
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]{2,15}USDT", symbol):
        return "Champion symbol is unsupported."
    if champion.get("interval") not in INTERVALS:
        return "Champion interval is unsupported."
    strategy = champion.get("strategy")
    if strategy not in {"trend", "breakout", "mean_reversion"}:
        return "Champion strategy is unsupported."
    params = champion.get("params")
    if not isinstance(params, dict):
        return "Champion parameters are incomplete."
    required = {"fast", "slow", "momentum", "breakout"}
    if not required.issubset(params):
        return "Champion parameters are incomplete."

    def finite_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    for name, low, high in (("fast", 2, 100), ("slow", 3, 300),
                            ("momentum", 1, 100), ("breakout", 2, 300)):
        value = params.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            return f"Champion parameter {name} is outside its allowed integer range."
    if params["fast"] >= params["slow"]:
        return "Champion parameters require fast < slow."
    for name in ("entry_momentum", "exit_momentum"):
        if name in params and (not finite_number(params[name]) or abs(float(params[name])) > 1):
            return f"Champion parameter {name} must be finite and within [-1, 1]."
    if strategy == "mean_reversion":
        if not finite_number(params.get("rsi_entry")) or not 0 <= float(params["rsi_entry"]) <= 100:
            return "Champion rsi_entry must be finite and within [0, 100]."
        if not finite_number(params.get("rsi_exit")) or not 0 <= float(params["rsi_exit"]) <= 100:
            return "Champion rsi_exit must be finite and within [0, 100]."
        if float(params["rsi_entry"]) >= float(params["rsi_exit"]):
            return "Champion mean-reversion RSI entry must be below RSI exit."
    if "stop_pct" in params and (
        not finite_number(params["stop_pct"]) or not 0.001 <= float(params["stop_pct"]) <= 0.25
    ):
        return "Champion stop_pct must be finite and within [0.001, 0.25]."

    version = champion.get("champion_version")
    if not isinstance(version, str) or not re.fullmatch(r"champion-[a-f0-9]{8,64}|champion-[A-Za-z0-9_-]{6,64}", version):
        return "Champion version is missing or invalid."
    metadata = champion.get("validation_metadata")
    if not isinstance(metadata, dict) or metadata.get("status") != "validated":
        return "Champion has no explicit validation metadata."
    if metadata.get("test_used_for_selection") is not False:
        return "Champion metadata does not confirm final-test isolation."
    for field in ("validation_metrics", "validation_stress_metrics", "test_metrics", "test_stress_metrics"):
        metrics = metadata.get(field)
        if not isinstance(metrics, dict):
            return f"Champion validation metadata is missing {field}."
        for metric in ("total_return_pct", "max_drawdown_pct", "sharpe", "trades"):
            if not finite_number(metrics.get(metric)):
                return f"Champion {field}.{metric} must be a finite number."
        if float(metrics["max_drawdown_pct"]) < 0 or float(metrics["trades"]) < 0:
            return f"Champion {field} contains invalid drawdown/trade metrics."
    validation_score = metadata.get("validation_score")
    if not finite_number(validation_score):
        return "Champion validation_score must be finite."
    cost = metadata.get("cost_convention")
    if (not isinstance(cost, dict) or cost.get("fee_basis") != "executed notional"
            or cost.get("fill_timing") != "closed candle close after one-candle signal shift"):
        return "Champion cost convention is missing or unsupported."
    for field in ("fee_bps_per_side", "slippage_bps_per_side",
                  "stress_fee_bps_per_side", "stress_slippage_bps_per_side"):
        if not finite_number(cost.get(field)) or float(cost[field]) < 0:
            return f"Champion cost convention field {field} is invalid."
    periods = metadata.get("periods")
    if not isinstance(periods, dict) or not all(isinstance(periods.get(k), dict) for k in ("train", "validation", "test")):
        return "Champion chronological period metadata is incomplete."
    try:
        train_end = datetime.fromisoformat(str(periods["train"]["end"]).replace("Z", "+00:00"))
        validation_start = datetime.fromisoformat(str(periods["validation"]["start"]).replace("Z", "+00:00"))
        validation_end = datetime.fromisoformat(str(periods["validation"]["end"]).replace("Z", "+00:00"))
        test_start = datetime.fromisoformat(str(periods["test"]["start"]).replace("Z", "+00:00"))
        if train_end.tzinfo is None or validation_start.tzinfo is None or validation_end.tzinfo is None or test_start.tzinfo is None:
            raise ValueError("naive period timestamp")
        if not train_end < validation_start <= validation_end < test_start:
            return "Champion train/validation/test periods overlap or are out of order."
    except (ValueError, TypeError, KeyError):
        return "Champion chronological period timestamps are invalid."

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
    age = now.astimezone(timezone.utc) - validated_at.astimezone(timezone.utc)
    if age < -timedelta(minutes=5):
        return "Champion validation timestamp is in the future."
    if age > timedelta(hours=24):
        return "Champion validation is stale (older than 24 hours)."
    return None


def fresh_price_map(frames: dict[str, pd.DataFrame], now: datetime | None = None) -> tuple[dict[str, float], dict[str, str]]:
    """Return only finite positive marks whose candle close is no more than 15 minutes old."""
    current = pd.Timestamp(now or datetime.now(timezone.utc))
    if current.tzinfo is None:
        raise ValueError("fresh-price check requires a timezone-aware clock")
    current = current.tz_convert("UTC") if current.tzinfo else current.tz_localize("UTC")
    prices: dict[str, float] = {}
    timestamps: dict[str, str] = {}
    for symbol, frame in frames.items():
        if frame.empty or "close_time" not in frame.columns or "close" not in frame.columns:
            continue
        try:
            price = float(frame["close"].iloc[-1])
            closed_at = pd.to_datetime(frame["close_time"].iloc[-1], utc=True)
            age_seconds = (current - closed_at).total_seconds()
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(price) and price > 0 and -60 <= age_seconds <= MAX_PRICE_AGE_SECONDS:
            prices[symbol] = price
            timestamps[symbol] = closed_at.isoformat()
    return prices, timestamps


def update_risk_metrics(state: dict[str, Any], equity: float,
                        now: datetime | None = None) -> dict[str, Any]:
    """Persist UTC daily-start and peak-equity metrics; latch any loss/drawdown breach."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("risk clock must be timezone-aware")
    day = current.astimezone(timezone.utc).date().isoformat()
    if not math.isfinite(equity) or equity < 0:
        raise ValueError("equity must be finite and non-negative")
    if state.get("risk_day") != day or not finite_positive(state.get("daily_start_equity")):
        state["risk_day"] = day
        state["daily_start_equity"] = max(equity, 1e-12)
    if not finite_positive(state.get("peak_equity")):
        state["peak_equity"] = max(equity, 1e-12)
    state["peak_equity"] = max(float(state["peak_equity"]), equity)
    daily_start = float(state["daily_start_equity"])
    peak = float(state["peak_equity"])
    daily_loss_pct = max(0.0, (daily_start - equity) / daily_start * 100.0)
    drawdown_pct = max(0.0, (peak - equity) / peak * 100.0)
    reasons = []
    if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
        reasons.append("daily_loss_limit")
    if drawdown_pct >= MAX_DRAWDOWN_PCT:
        reasons.append("drawdown_limit")
    if reasons:
        state["risk_halt"] = True
        state["risk_halt_reason"] = ",".join(reasons)
    return {
        "daily_loss_pct": round(daily_loss_pct, 6),
        "drawdown_pct": round(drawdown_pct, 6),
        "daily_loss_limit_pct": MAX_DAILY_LOSS_PCT,
        "drawdown_limit_pct": MAX_DRAWDOWN_PCT,
        "max_position_fraction": MAX_POSITION_FRACTION,
        "halted": bool(state.get("risk_halt", False)),
        "halt_reason": state.get("risk_halt_reason"),
    }


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0



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
                   "leverage": False, "short_selling": False,
                   "max_position_fraction": MAX_POSITION_FRACTION,
                   "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
                   "max_drawdown_pct": MAX_DRAWDOWN_PCT,
                   "max_price_age_seconds": MAX_PRICE_AGE_SECONDS}
    }
    frames: dict[str, pd.DataFrame] = {}
    symbols_to_fetch = []
    for candidate in (champion.get("symbol") if isinstance(champion, dict) else None, state.get("symbol")):
        if isinstance(candidate, str) and re.fullmatch(r"[A-Z0-9]{2,15}USDT", candidate) and candidate not in symbols_to_fetch:
            symbols_to_fetch.append(candidate)
    if not symbols_to_fetch:
        symbols_to_fetch = list(SYMBOLS[:1])
    for symbol in symbols_to_fetch:
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
        # Invalid champions must not trade, but an existing paper position still
        # needs an honest symbol-specific mark in the status report.
        current_prices, price_timestamps = fresh_price_map(frames)
        mark_prices = current_prices
        state["last_prices"] = current_prices
        state["last_price_timestamps"] = price_timestamps
        qty = float(state.get("quantity", 0.0))
        held_symbol = state.get("symbol")
        cash = float(state.get("cash", STARTING_CASH))
        if qty > 0 and held_symbol not in mark_prices:
            status["data_errors"].append({
                "symbol": held_symbol,
                "error": "Missing valid mark for held position; refusing cross-symbol valuation or trade."
            })
            status["message"] = "Champion rejected and held position cannot be safely valued; no simulated order made."
        else:
            equity, mark = mark_to_market(cash, qty, held_symbol, mark_prices)
            risk = update_risk_metrics(state, equity)
            exposure_fraction = (qty * mark / equity) if qty > 0 and equity > 0 else 0.0
            if exposure_fraction > MAX_POSITION_FRACTION:
                state["risk_halt"] = True
                state["risk_halt_reason"] = "position_exposure_limit"
                risk["halted"] = True
                risk["halt_reason"] = state["risk_halt_reason"]
            if risk["halted"] and qty > 0:
                entry = float(state.get("entry_price") or mark)
                sale = sell_with_costs(qty, mark, entry, FEE_BPS, SLIPPAGE_BPS)
                cash += sale.net_proceeds
                state["realized_pnl"] = float(state.get("realized_pnl", 0.0)) + sale.realized_pnl
                state["completed_trades"] = int(state.get("completed_trades", 0)) + 1
                status["actions"].append({
                    "action": "PAPER_SELL", "symbol": held_symbol,
                    "price": round(sale.execution_price, 8), "quantity": qty,
                    "fee_usdt": round(sale.fee, 8), "slippage_usdt": round(sale.slippage, 8),
                    "net_proceeds_usdt": round(sale.net_proceeds, 8),
                    "realized_pnl_usdt": round(sale.realized_pnl, 8),
                    "reason": "risk circuit breaker: " + str(risk["halt_reason"]),
                })
                qty, held_symbol, mark = 0.0, None, 0.0
                state["quantity"], state["symbol"], state["entry_price"] = 0.0, None, None
                equity = cash
                risk = update_risk_metrics(state, equity)
                risk["halted"] = True
                risk["halt_reason"] = state.get("risk_halt_reason")
            entry = float(state.get("entry_price") or mark or 0.0)
            status.update({
                "cash_usdt": round(cash, 8),
                "equity_usdt": round(equity, 8),
                "total_pnl_usdt": round(equity - STARTING_CASH, 8),
                "unrealized_pnl_usdt": round((mark - entry) * qty if qty > 0 else 0.0, 8),
                "open_position": ({"symbol": held_symbol, "quantity": round(qty, 10),
                                   "mark_price": round(mark, 8), "entry_price": round(entry, 8)}
                                  if qty > 0 else None),
                "risk": risk,
                "message": ("Risk circuit breaker is latched; no new simulated buys are allowed."
                            if risk["halted"] else champion_error or "No validated champion; no simulated order made."),
            })
        state["last_checked_at"] = utc_now()
        STATE_FILE.write_text(json.dumps(state, indent=2))
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
    current_prices, price_timestamps = fresh_price_map(frames)
    mark_prices = current_prices
    state["last_prices"] = current_prices
    state["last_price_timestamps"] = price_timestamps
    if symbol not in current_prices:
        status["data_errors"].append({
            "symbol": symbol,
            "error": "Champion price is stale, missing, or invalid; no simulated order made."
        })
        status["message"] = "Fresh champion price unavailable; no simulated order made."
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status
    if qty > 0 and position_symbol not in mark_prices:
        status["data_errors"].append({
            "symbol": position_symbol,
            "error": "Missing valid mark for held position; refusing cross-symbol valuation or trade."
        })
        status["message"] = "Held position cannot be safely valued; no simulated order made."
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status

    equity_before, held_mark = mark_to_market(cash, qty, position_symbol, mark_prices)
    risk = update_risk_metrics(state, equity_before)
    exposure_fraction = (qty * held_mark / equity_before) if qty > 0 and equity_before > 0 else 0.0
    if exposure_fraction > MAX_POSITION_FRACTION:
        state["risk_halt"] = True
        state["risk_halt_reason"] = "position_exposure_limit"
        risk = update_risk_metrics(state, equity_before)
        risk["halted"] = True
        risk["halt_reason"] = state["risk_halt_reason"]
    if risk["halted"]:
        if qty > 0 and position_symbol in mark_prices:
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
                "reason": "risk circuit breaker: " + str(risk["halt_reason"]),
            })
            qty, position_symbol = 0.0, None
            state["entry_price"] = None
        equity_after = cash
        risk = update_risk_metrics(state, equity_after)
        risk["halted"] = True
        risk["halt_reason"] = state.get("risk_halt_reason")
        state.update({"cash": cash, "quantity": qty, "symbol": position_symbol,
                      "last_checked_at": utc_now()})
        status.update({
            "cash_usdt": round(cash, 8), "equity_usdt": round(equity_after, 8),
            "total_pnl_usdt": round(equity_after - STARTING_CASH, 8),
            "risk": risk,
            "message": "Risk circuit breaker is latched; no new simulated buys are allowed. Manual review/reset required.",
        })
        STATE_FILE.write_text(json.dumps(state, indent=2))
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
            # Cap each new long to 50% of current equity; leave the rest in simulated cash.
            spend_cash = min(cash, max(0.0, equity_before * MAX_POSITION_FRACTION))
            buy = buy_with_costs(spend_cash, latest_price, FEE_BPS, SLIPPAGE_BPS)
            qty = max(0.0, buy.quantity)
            cash = max(0.0, cash - spend_cash + buy.cash_remaining)
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
        "risk": update_risk_metrics(state, equity),
        "message": "Simulated portfolio only; fills are estimates, not exchange orders."
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    return status


if __name__ == "__main__":
    print(json.dumps(run_once(), indent=2))
