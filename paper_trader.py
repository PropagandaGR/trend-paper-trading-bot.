from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_accounting import (DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
                                  MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT,
                                  MAX_POSITION_FRACTION, STRESS_FEE_BPS,
                                  STRESS_SLIPPAGE_BPS, buy_with_costs, entry_budget,
                                  mark_to_market, sell_with_costs)

from research_engine import (
    INTERVALS, REPORTS, STATE, SYMBOLS, fetch_klines, desired_signal, add_features, resample_ohlcv, utc_now
)

STARTING_CASH = 15.0
FEE_BPS = DEFAULT_FEE_BPS
SLIPPAGE_BPS = DEFAULT_SLIPPAGE_BPS
STATE_FILE = STATE / "paper_state.json"
CHAMPION_FILE = STATE / "champion.json"
STATUS_FILE = REPORTS / "paper_status.json"
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
    for field in ("train_metrics", "validation_metrics", "validation_stress_metrics", "test_metrics", "test_stress_metrics"):
        metrics = metadata.get(field)
        if not isinstance(metrics, dict):
            return f"Champion validation metadata is missing {field}."
        for metric in ("total_return_pct", "max_drawdown_pct", "sharpe", "trades"):
            if not finite_number(metrics.get(metric)):
                return f"Champion {field}.{metric} must be a finite number."
        if float(metrics["max_drawdown_pct"]) < 0 or float(metrics["trades"]) < 0:
            return f"Champion {field} contains invalid drawdown/trade metrics."
    # Reject strategies that only look good in validation after collapsing in training.
    # The final test is deliberately not an acceptance criterion to preserve its isolation.
    if float(metadata["train_metrics"]["total_return_pct"]) <= 0:
        return "Champion failed training robustness gate: training return must be positive."
    if float(metadata["train_metrics"]["max_drawdown_pct"]) >= 35:
        return "Champion failed training robustness gate: training drawdown must be below 35%."
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
    expected_costs = {
        "fee_bps_per_side": FEE_BPS,
        "slippage_bps_per_side": SLIPPAGE_BPS,
        "stress_fee_bps_per_side": STRESS_FEE_BPS,
        "stress_slippage_bps_per_side": STRESS_SLIPPAGE_BPS,
    }
    for field, expected in expected_costs.items():
        if not math.isclose(float(cost[field]), expected, rel_tol=0.0, abs_tol=1e-9):
            return f"Champion cost convention {field} does not match current paper execution settings."
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


def missed_candle_count(bars: pd.DataFrame, cursor: str | None) -> int | None:
    """Count intermediate closed candles since the last processed candle.

    None means the cursor is malformed, in the future, or falls between known
    candle timestamps, so the caller must fail closed rather than guess.
    """
    if cursor is None:
        return 0
    cursor_time = pd.to_datetime(cursor, utc=True, errors="coerce")
    times = pd.to_datetime(bars.index, utc=True, errors="coerce")
    if pd.isna(cursor_time) or len(times) == 0 or times.isna().any():
        return None
    if cursor_time > times[-1]:
        return None
    if cursor_time == times[-1]:
        return 0
    later_count = int((times > cursor_time).sum())
    if later_count <= 0:
        return 0
    if cursor_time < times[0]:
        # The cursor fell outside the fetched window. The exact gap is unknown,
        # but there is at least one missed decision candle; do not backfill trades.
        return max(1, later_count - 1)
    if not bool((times == cursor_time).any()):
        return None
    return max(0, later_count - 1)



def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def default_paper_state() -> dict[str, Any]:
    return {
        "cash": STARTING_CASH, "quantity": 0.0, "symbol": None,
        "entry_price": None, "last_processed_bars": {},
        "starting_cash": STARTING_CASH, "completed_trades": 0, "realized_pnl": 0.0,
    }


def load_paper_state() -> dict[str, Any]:
    """Load portfolio state without silently resetting a corrupt existing ledger."""
    if not STATE_FILE.exists():
        return default_paper_state()
    try:
        state = json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Paper state is unreadable; refusing to trade: {STATE_FILE}") from exc
    if not isinstance(state, dict):
        raise RuntimeError("Paper state must be a JSON object; refusing to trade.")
    required = {"cash", "quantity", "symbol", "entry_price", "starting_cash"}
    if not required.issubset(state):
        missing = sorted(required - set(state))
        raise RuntimeError(f"Paper state is missing required fields {missing}; refusing to trade.")
    for field in ("cash", "quantity", "starting_cash", "realized_pnl"):
        if field == "realized_pnl" and field not in state:
            state[field] = 0.0
        value = state.get(field)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(float(value))):
            raise RuntimeError(f"Paper state field {field} must be finite; refusing to trade.")
    if float(state["cash"]) < 0 or float(state["quantity"]) < 0 or float(state["starting_cash"]) <= 0:
        raise RuntimeError("Paper state cash/quantity/starting equity is invalid; refusing to trade.")
    if not math.isclose(float(state["starting_cash"]), STARTING_CASH, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("Paper state starting equity does not match the 15 USDT project invariant.")
    quantity = float(state["quantity"])
    symbol = state.get("symbol")
    entry = state.get("entry_price")
    if quantity > 0:
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9]{2,15}USDT", symbol):
            raise RuntimeError("Open paper position has an invalid symbol; refusing to trade.")
        if not finite_positive(entry):
            raise RuntimeError("Open paper position has an invalid entry price; refusing to trade.")
    elif symbol is not None or entry is not None:
        raise RuntimeError("Flat paper state must not retain a position symbol or entry price.")
    cursors = state.setdefault("last_processed_bars", {})
    if not isinstance(cursors, dict):
        raise RuntimeError("Paper state candle cursors must be an object; refusing to trade.")
    for cursor_symbol, cursor in cursors.items():
        if (not isinstance(cursor_symbol, str)
                or not re.fullmatch(r"[A-Z0-9]{2,15}USDT", cursor_symbol)
                or not isinstance(cursor, str)):
            raise RuntimeError("Paper state contains an invalid candle cursor; refusing to trade.")
        timestamp = pd.to_datetime(cursor, utc=True, errors="coerce")
        if pd.isna(timestamp):
            raise RuntimeError("Paper state contains a malformed candle timestamp; refusing to trade.")
    completed = state.setdefault("completed_trades", 0)
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
        raise RuntimeError("Paper state completed_trades must be a non-negative integer.")
    if "risk_halt" in state and not isinstance(state["risk_halt"], bool):
        raise RuntimeError("Paper state risk_halt must be boolean; refusing to trade.")
    if state.get("risk_halt") and not isinstance(state.get("risk_halt_reason"), str):
        raise RuntimeError("Latched paper risk halt is missing its reason; refusing to trade.")
    for field in ("daily_start_equity", "peak_equity"):
        if field in state and not finite_positive(state[field]):
            raise RuntimeError(f"Paper state field {field} must be positive; refusing to trade.")
    return state


def _sync_ledger_status(status: dict[str, Any], state: dict[str, Any]) -> None:
    """Keep the published status ledger fields aligned with persisted state."""
    status["realized_pnl_usdt"] = round(float(state.get("realized_pnl", 0.0)), 8)
    status["completed_trades"] = int(state.get("completed_trades", 0))


def _set_unvalued_position_status(status: dict[str, Any], state: dict[str, Any],
                                  message: str) -> None:
    """Publish an explicit unknown valuation instead of presenting cash as total equity."""
    cash = float(state.get("cash", STARTING_CASH))
    quantity = float(state.get("quantity", 0.0))
    symbol = state.get("symbol")
    has_position = quantity > 0
    entry = state.get("entry_price")
    status.update({
        "cash_usdt": round(cash, 8),
        "equity_usdt": None if has_position else round(cash, 8),
        "total_pnl_usdt": None if has_position else round(cash - STARTING_CASH, 8),
        "unrealized_pnl_usdt": None if has_position else 0.0,
        "open_position": ({
            "symbol": symbol,
            "quantity": round(quantity, 10),
            "mark_price": None,
            "entry_price": float(entry) if finite_positive(entry) else None,
            "valuation_status": "unavailable",
        } if has_position else None),
        "valuation_status": "unavailable" if has_position else "cash_only",
        "message": message,
    })
    if has_position:
        status["risk"] = {
            "halted": bool(state.get("risk_halt", False)),
            "halt_reason": state.get("risk_halt_reason"),
        }


def _run_once_impl() -> dict[str, Any]:
    STATE.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    state = load_paper_state()
    champion = read_json(CHAMPION_FILE, None)
    champion_error = champion_validation_error(champion)
    champion_accepted = champion_error is None
    status = {
        "updated_at": utc_now(), "mode": "PAPER_ONLY",
        "starting_cash_usdt": STARTING_CASH, "cash_usdt": round(float(state["cash"]), 8),
        "equity_usdt": round(float(state["cash"]), 8), "unrealized_pnl_usdt": 0.0,
        "realized_pnl_usdt": round(float(state.get("realized_pnl", 0.0)), 8),
        "completed_trades": int(state.get("completed_trades", 0)),
        "open_position": None, "champion_available": champion_accepted,
        "champion_version": champion.get("champion_version") if isinstance(champion, dict) else None,
        "champion_rejection_reason": champion_error,
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
        _set_unvalued_position_status(
            status, state,
            "Market data unavailable; held position cannot be valued; no simulated order made."
            if float(state.get("quantity", 0.0)) > 0
            else "Market data unavailable; no simulated order made.",
        )
        _sync_ledger_status(status, state)
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
            _set_unvalued_position_status(
                status, state,
                "Champion rejected and held position cannot be safely valued; no simulated order made.",
            )
            state.update({"cash": cash, "quantity": qty, "symbol": held_symbol,
                          "last_checked_at": utc_now()})
            STATE_FILE.write_text(json.dumps(state, indent=2))
            _sync_ledger_status(status, state)
            STATUS_FILE.write_text(json.dumps(status, indent=2))
            return status
        else:
            equity, mark = mark_to_market(cash, qty, held_symbol, mark_prices)
            risk = update_risk_metrics(state, equity)
            # The 50% limit constrains new entries only; appreciation is not a loss-risk halt.
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
        state.update({"cash": cash, "quantity": qty, "symbol": held_symbol,
                      "entry_price": state.get("entry_price") if qty > 0 else None,
                      "last_checked_at": utc_now()})
        STATE_FILE.write_text(json.dumps(state, indent=2))
        _sync_ledger_status(status, state)
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status

    symbol = champion["symbol"]
    if symbol not in frames:
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
                "error": "Held-asset mark is stale or unavailable; no trade or unsafe valuation allowed."
            })
            _set_unvalued_position_status(
                status, state,
                "Champion data unavailable and held position cannot be safely marked.",
            )
            state.update({"cash": cash, "quantity": qty, "symbol": held_symbol,
                          "last_checked_at": utc_now()})
            STATE_FILE.write_text(json.dumps(state, indent=2))
            _sync_ledger_status(status, state)
            STATUS_FILE.write_text(json.dumps(status, indent=2))
            return status
        else:
            equity, mark = mark_to_market(cash, qty, held_symbol, mark_prices)
            risk = update_risk_metrics(state, equity)
            # The 50% limit constrains new entries only; appreciation is not a loss-risk halt.
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
                qty, held_symbol = 0.0, None
                state["quantity"], state["symbol"], state["entry_price"] = 0.0, None, None
                equity = cash
                risk = update_risk_metrics(state, equity)
                risk["halted"] = True
                risk["halt_reason"] = state.get("risk_halt_reason")
            status.update({
                "cash_usdt": round(cash, 8), "equity_usdt": round(equity, 8),
                "total_pnl_usdt": round(equity - STARTING_CASH, 8), "risk": risk,
            })
            status["message"] = ("Risk circuit breaker is latched; no new simulated buys are allowed."
                                 if risk["halted"] else
                                 "Champion data unavailable; no simulated order made.")
        state.update({"cash": cash, "quantity": qty, "symbol": held_symbol,
                      "last_checked_at": utc_now()})
        STATE_FILE.write_text(json.dumps(state, indent=2))
        _sync_ledger_status(status, state)
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
        _sync_ledger_status(status, state)
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
    champion_price_fresh = symbol in current_prices
    if qty > 0 and position_symbol not in mark_prices:
        status["data_errors"].append({
            "symbol": position_symbol,
            "error": "Missing valid mark for held position; refusing cross-symbol valuation or trade."
        })
        _set_unvalued_position_status(
            status, state,
            "Held position cannot be safely valued; no simulated order made.",
        )
        _sync_ledger_status(status, state)
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status

    equity_before, held_mark = mark_to_market(cash, qty, position_symbol, mark_prices)
    risk = update_risk_metrics(state, equity_before)
    # MAX_POSITION_FRACTION is an entry cap, not a continuously marked circuit breaker.
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
        _sync_ledger_status(status, state)
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status

    try:
        strategy_close_time = pd.to_datetime(bars["close_time"].iloc[-1], utc=True)
        strategy_age_seconds = (pd.Timestamp.now(tz="UTC") - strategy_close_time).total_seconds()
        strategy_bar_fresh = -60 <= strategy_age_seconds <= MAX_PRICE_AGE_SECONDS
    except (KeyError, IndexError, TypeError, ValueError):
        strategy_bar_fresh = False
    if not champion_price_fresh or not strategy_bar_fresh:
        status["data_errors"].append({
            "symbol": symbol,
            "error": "Champion price or signal candle is stale, missing, or invalid; no simulated order made."
        })
        status["message"] = "Fresh champion price or signal candle unavailable; no simulated order made."
        state.update({"cash": cash, "quantity": qty, "symbol": position_symbol,
                      "last_checked_at": utc_now()})
        STATE_FILE.write_text(json.dumps(state, indent=2))
        _sync_ledger_status(status, state)
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        return status

    missed = missed_candle_count(bars, cursor)
    signal_diagnostics["missed_candles_since_last_run"] = missed
    execution_message = "Simulated portfolio only; fills are estimates, not exchange orders."
    if cursor is not None and (missed is None or missed > 0):
        # Never invent retroactive fills after the scheduler/data feed skipped
        # candles. Close an existing position at the current fresh held-asset
        # mark, advance the cursor, and wait for the next live decision candle.
        if qty > 0 and position_symbol in mark_prices:
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
                "reason": "safety exit after missed strategy candles",
            })
            qty, position_symbol = 0.0, None
            state["entry_price"] = None
        state.setdefault("last_processed_bars", {})[symbol] = latest_time
        status["missed_candles"] = {
            "symbol": symbol, "count": missed, "cursor_was_valid": missed is not None,
            "last_processed_bar": cursor, "resumed_at_bar": latest_time,
        }
        execution_message = (
            "Missed strategy candles detected; skipped entries and closed any open paper position for safety. "
            "No retroactive simulated orders were created."
        )
    else:
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
                # Leave a small buffer so fees/slippage do not push post-fill marked
                # exposure above the circuit-breaker threshold on the next run.
                spend_cash = entry_budget(cash, equity_before)
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
                     "strategy": champion["strategy"], "params": params,
                     "champion_version": champion.get("champion_version")},
        "last_processed_bar": latest_time,
        "signal_diagnostics": signal_diagnostics,
        "risk": update_risk_metrics(state, equity),
        "message": execution_message
    })
    STATE_FILE.write_text(json.dumps(state, indent=2))
    _sync_ledger_status(status, state)
    STATUS_FILE.write_text(json.dumps(status, indent=2))
    return status


def _recent_event_ids(path: Path, max_bytes: int = 1_000_000) -> set[str]:
    """Read a bounded tail of the append-only event file for crash-safe deduplication."""
    if not path.exists():
        return set()
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        chunk = handle.read().decode("utf-8", errors="replace")
    lines = chunk.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line may be a truncated JSON record
    ids: set[str] = set()
    for line in lines[-5000:]:
        try:
            event = json.loads(line)
            event_id = event.get("event_id")
            if isinstance(event_id, str):
                ids.add(event_id)
        except (json.JSONDecodeError, AttributeError):
            continue
    return ids


def _append_paper_events(status: dict[str, Any], error: str | None = None) -> None:
    """Append idempotent run/action records so state changes can be independently audited."""
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "paper_events.jsonl"
    state = read_json(STATE_FILE, {})
    if not isinstance(state, dict):
        state = {}
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    event_time = str(status.get("updated_at") or utc_now())
    champion = status.get("champion") if isinstance(status.get("champion"), dict) else {}
    champion_version = status.get("champion_version") or champion.get("champion_version")
    bar_time = status.get("last_processed_bar")
    events: list[dict[str, Any]] = []

    actions = status.get("actions") if isinstance(status.get("actions"), list) else []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        action_payload = {
            "event_type": "PAPER_ACTION",
            "timestamp": event_time,
            "champion_version": champion_version,
            "bar_timestamp": bar_time or state.get("last_processed_bars", {}).get(action.get("symbol")),
            "action": action,
        }
        canonical = json.dumps(action_payload, sort_keys=True, separators=(",", ":"), default=str)
        action_payload["event_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        action_payload.update({
            "workflow_run_id": run_id, "workflow_run_attempt": attempt,
            "portfolio": {
                "cash_usdt": status.get("cash_usdt"),
                "equity_usdt": status.get("equity_usdt"),
                "total_pnl_usdt": status.get("total_pnl_usdt"),
                "unrealized_pnl_usdt": status.get("unrealized_pnl_usdt"),
            },
            "risk": status.get("risk"),
            "state_snapshot": {
                key: state.get(key) for key in (
                    "cash", "quantity", "symbol", "entry_price", "realized_pnl",
                    "completed_trades", "risk_halt", "risk_halt_reason",
                    "daily_start_equity", "peak_equity",
                )
            },
        })
        events.append(action_payload)

    run_payload = {
        "event_type": "PAPER_ERROR" if error else "PAPER_RUN",
        "timestamp": event_time,
        "workflow_run_id": run_id,
        "workflow_run_attempt": attempt,
        "champion_version": champion_version,
        "bar_timestamp": bar_time,
        "message": error or status.get("message"),
        "data_errors": status.get("data_errors", []),
        "portfolio": {
            "cash_usdt": status.get("cash_usdt"),
            "equity_usdt": status.get("equity_usdt"),
            "total_pnl_usdt": status.get("total_pnl_usdt"),
            "realized_pnl_usdt": status.get("realized_pnl_usdt"),
            "unrealized_pnl_usdt": status.get("unrealized_pnl_usdt"),
            "completed_trades": status.get("completed_trades"),
            "open_position": status.get("open_position"),
        },
        "risk": status.get("risk"),
        "state_snapshot": {
            key: state.get(key) for key in (
                "cash", "quantity", "symbol", "entry_price", "realized_pnl",
                "completed_trades", "risk_halt", "risk_halt_reason",
                "daily_start_equity", "peak_equity",
            )
        },
    }
    run_canonical = json.dumps(run_payload, sort_keys=True, separators=(",", ":"), default=str)
    run_payload["event_id"] = hashlib.sha256(
        f"{run_id}|{attempt}|{run_canonical}".encode("utf-8")
    ).hexdigest()
    events.append(run_payload)

    existing_ids = _recent_event_ids(path)
    new_events = [event for event in events if event["event_id"] not in existing_ids]
    if not new_events:
        return
    with path.open("a", encoding="utf-8") as handle:
        for event in new_events:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_once() -> dict[str, Any]:
    """Run the paper trader and persist an audit record for every outcome."""
    try:
        status = _run_once_impl()
    except Exception as exc:
        _append_paper_events({}, error=f"{type(exc).__name__}: {exc}")
        raise
    _append_paper_events(status)
    return status


if __name__ == "__main__":
    print(json.dumps(run_once(), indent=2))
