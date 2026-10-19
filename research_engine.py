from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from portfolio_accounting import buy_with_costs, sell_with_costs
from universe import get_coinbase_product_id, get_research_batch

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
STATE = ROOT / "state"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
INTERVALS = {"5m": "5min", "10m": "10min", "15m": "15min", "30m": "30min"}
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Autonomous-Paper-Trading-Lab/1.0"})
_USDT_USD_CANDLE_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_coinbase(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch Coinbase Exchange public USD candles without account credentials."""
    product = get_coinbase_product_id(symbol)
    if not product:
        raise ValueError(f"No Coinbase fallback mapping for {symbol}")
    granularity = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}.get(interval)
    if granularity is None:
        raise ValueError(f"Unsupported Coinbase candle interval: {interval}")
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=days)
    # Coinbase caps each response at 300 candles; page through the period.
    rows: list[list[Any]] = []
    chunk = pd.Timedelta(seconds=granularity * 280)
    cursor = start
    while cursor < end:
        stop = min(cursor + chunk, end)
        response = SESSION.get(
            f"https://api.exchange.coinbase.com/products/{product}/candles",
            params={"granularity": granularity, "start": cursor.isoformat(),
                    "end": stop.isoformat()},
            timeout=20,
        )
        response.raise_for_status()
        batch = response.json()
        rows.extend(batch)
        cursor = stop
        time.sleep(0.12)
    if not rows:
        raise RuntimeError(f"No Coinbase candles returned for {product} {interval}")
    frame = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    frame = frame.drop_duplicates("time").sort_values("time")
    frame["open_time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame["close_time"] = frame["open_time"] + pd.to_timedelta(granularity, unit="s")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.set_index("open_time")
    return frame.loc[frame["close_time"] < pd.Timestamp.now(tz="UTC"),
                     ["open", "high", "low", "close", "volume", "close_time"]]


def _convert_usd_candles_to_usdt(frame: pd.DataFrame, fx_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert USD-quoted OHLC prices to approximate USDT prices using USDT-USD closes.

    Coinbase's USDT-USD close is USD per USDT. Dividing each asset's USD OHLC
    by that same-candle FX close gives a consistent USDT-denominated price
    series. This is an approximation, not tick-level FX conversion.
    """
    if frame.empty or fx_frame.empty:
        raise RuntimeError("Cannot convert candles without asset and USDT-USD data")
    # Do not forward-fill USDT-USD across missing candles: stale FX can silently
    # distort the entire asset price series. Only exact same-candle timestamps qualify.
    fx = pd.to_numeric(fx_frame["close"], errors="coerce").reindex(frame.index)
    valid = fx.notna() & np.isfinite(fx) & (fx > 0)
    missing_fx_candles = int((~valid).sum())
    converted = frame.loc[valid].copy()
    fx = fx.loc[valid]
    for column in ("open", "high", "low", "close"):
        converted[column] = converted[column] / fx
    converted.attrs["source"] = "coinbase"
    converted.attrs["quote_currency"] = "USDT"
    converted.attrs["fx_conversion"] = "same-candle USDT-USD close only; approximate"
    converted.attrs["fx_missing_candles"] = missing_fx_candles
    return converted


def fetch_klines(symbol: str, interval: str = "5m", days: int = 90) -> pd.DataFrame:
    """Fetch closed Coinbase candles and express asset prices in USDT.

    Binance is deliberately not used as a fallback. Candle data is public and
    does not require API credentials. USD candles are converted using Coinbase
    USDT-USD candle closes so the simulated USDT ledger uses matching units.
    """
    frame = _fetch_coinbase(symbol, interval, days)
    if frame.empty:
        raise RuntimeError(f"Coinbase returned no closed candles for {symbol} {interval}")
    if symbol != "USDTUSD":
        cache_key = (interval, days)
        if cache_key not in _USDT_USD_CANDLE_CACHE:
            _USDT_USD_CANDLE_CACHE[cache_key] = _fetch_coinbase("USDTUSD", interval, days)
        fx_frame = _USDT_USD_CANDLE_CACHE[cache_key]
        frame = _convert_usd_candles_to_usdt(frame, fx_frame)
        if frame.empty:
            raise RuntimeError(
                f"No same-candle USDT-USD FX closes available to convert {symbol} {interval}"
            )
    frame.attrs["source"] = "coinbase"
    frame.attrs["quote_currency"] = "USDT"
    return frame.sort_index()

def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Build only complete higher-timeframe candles from closed 5m source bars.

    Source indices are candle open times. Left-closed/right-labeled buckets
    group [18:00, 18:30) into the candle labeled 18:30, without pulling the
    18:30 candle into the candle that just ended. Require every expected 5m
    source candle and a final close timestamp at the bucket boundary (allowing
    Coinbase candle close timestamps are derived from their open times.
    """
    rule_delta = pd.Timedelta(rule)
    source_delta = pd.Timedelta(minutes=5)
    if rule_delta < source_delta or rule_delta % source_delta != pd.Timedelta(0):
        raise ValueError(f"Resample rule must be a multiple of 5 minutes: {rule}")
    expected_rows = int(rule_delta / source_delta)
    grouped = frame.resample(rule, label="right", closed="left")
    result = grouped.agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "close_time": "last"
    }).dropna(subset=["open", "high", "low", "close"])
    counts = grouped["close"].count()
    complete = counts.reindex(result.index).eq(expected_rows)
    boundary_closed = result["close_time"] >= (result.index - pd.Timedelta(seconds=1))
    return result.loc[complete & boundary_closed]


def add_features(frame: pd.DataFrame, fast: int, slow: int, momentum: int,
                 breakout: int, rsi_period: int = 14) -> pd.DataFrame:
    x = frame.copy()
    x["ema_fast"] = x.close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    x["ema_slow"] = x.close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    x["momentum"] = x.close.pct_change(momentum)
    x["prior_high"] = x.high.shift(1).rolling(breakout, min_periods=breakout).max()
    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    x["rsi"] = x["rsi"].fillna(50)
    return x


def desired_signal(x: pd.DataFrame, strategy: str, params: dict[str, Any]) -> pd.Series:
    fast, slow = int(params["fast"]), int(params["slow"])
    mom = int(params["momentum"])
    breakout = int(params["breakout"])
    f = add_features(x, fast, slow, mom, breakout)
    if strategy == "trend":
        enter = (f.ema_fast > f.ema_slow) & (f.momentum > float(params["entry_momentum"]))
        exit_ = (f.ema_fast < f.ema_slow) | (f.momentum < float(params["exit_momentum"]))
    elif strategy == "breakout":
        enter = (f.close > f.prior_high) & (f.momentum > 0)
        exit_ = (f.close < f.ema_fast) | (f.momentum < float(params["exit_momentum"]))
    elif strategy == "mean_reversion":
        enter = (f.rsi < float(params["rsi_entry"])) & (f.close < f.ema_slow)
        exit_ = (f.rsi >= float(params["rsi_exit"])) | (f.close < f.ema_slow * (1 - float(params["stop_pct"])))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    # Keep positions open until a strategy-specific exit, instead of requiring
    # the entry condition to remain true. Signals are shifted to avoid look-ahead.
    position = np.zeros(len(f), dtype=float)
    active = False
    for i in range(len(f)):
        if active and bool(exit_.iloc[i]):
            active = False
        elif not active and bool(enter.iloc[i]):
            active = True
        position[i] = 1.0 if active else 0.0
    return pd.Series(position, index=f.index).shift(1).fillna(0.0)


def bars_per_year(interval: str) -> int:
    """Annualization factor for crypto bars, using 365 calendar days per year."""
    if interval not in INTERVALS:
        raise ValueError(f"Unsupported research interval: {interval}")
    minutes = int(pd.Timedelta(INTERVALS[interval]).total_seconds() // 60)
    return int(365 * 24 * 60 // minutes)


def performance(returns: pd.Series, interval: str = "5m",
                trades: int = 0) -> dict[str, float]:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "sharpe": 0.0, "trades": int(trades)}
    equity_values = np.concatenate(([1.0], (1 + returns).cumprod().to_numpy()))
    running_peak = np.maximum.accumulate(equity_values)
    drawdown = equity_values / running_peak - 1
    std = float(returns.std(ddof=0))
    sharpe = (float(returns.mean() / std * math.sqrt(bars_per_year(interval)))
              if std > 0 else 0.0)
    return {"total_return_pct": round((float(equity_values[-1]) - 1) * 100, 4),
            "max_drawdown_pct": round(abs(float(drawdown.min())) * 100, 4),
            "sharpe": round(sharpe, 3), "trades": int(trades)}


def backtest(frame: pd.DataFrame, strategy: str, params: dict[str, Any],
             fee_bps: float = DEFAULT_FEE_BPS,
             slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
             interval: str = "5m", starting_cash: float = 15.0) -> dict[str, float]:
    """Simulate closed-candle fills using paper-trader per-side costs."""
    minimum = max(int(params["slow"]) * 3, int(params["breakout"]) + 50)
    if len(frame) < minimum:
        return performance(pd.Series(dtype=float), interval, trades=0)
    if not math.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("starting_cash must be finite and positive")
    if (not math.isfinite(fee_bps) or not math.isfinite(slippage_bps)
            or fee_bps < 0 or slippage_bps < 0):
        raise ValueError("fee_bps and slippage_bps must be finite and non-negative")
    prices = pd.to_numeric(frame["close"], errors="coerce")
    if prices.isna().any() or (~np.isfinite(prices)).any() or (prices <= 0).any():
        raise ValueError("backtest requires finite, positive close prices")

    target = desired_signal(frame, strategy, params).reindex(frame.index).fillna(0.0)
    cash, quantity, entry_basis, changes = float(starting_cash), 0.0, 0.0, 0
    equity_values: list[float] = []
    for timestamp, market_price in prices.items():
        price = float(market_price)
        wants_long = bool(float(target.loc[timestamp]) > 0)
        if wants_long and quantity <= 0:
            buy = buy_with_costs(cash, price, fee_bps, slippage_bps)
            quantity, cash, entry_basis = buy.quantity, buy.cash_remaining, buy.entry_cost_per_unit
            changes += 1
        elif not wants_long and quantity > 0:
            sale = sell_with_costs(quantity, price, entry_basis, fee_bps, slippage_bps)
            cash += sale.net_proceeds
            quantity, entry_basis = 0.0, 0.0
            changes += 1
        equity_values.append(cash + quantity * price)

    equity = pd.Series(equity_values, index=frame.index, dtype=float)
    period_returns = equity.pct_change()
    if not period_returns.empty:
        period_returns.iloc[0] = equity.iloc[0] / starting_cash - 1.0
    return performance(period_returns, interval, trades=changes)

def candidate_params() -> list[dict[str, Any]]:
    candidates = []
    for fast, slow, momentum, breakout, entry, exit_mom in itertools.product(
        (8, 12, 20), (30, 48, 72), (6, 12, 24), (12, 24, 48),
        (-0.002, 0.0, 0.002), (-0.01, -0.005, 0.0)
    ):
        if fast >= slow:
            continue
        candidates.append({"fast": fast, "slow": slow, "momentum": momentum,
                           "breakout": breakout, "entry_momentum": entry,
                           "exit_momentum": exit_mom, "rsi_entry": 30,
                           "rsi_exit": 50, "stop_pct": 0.06})
    # Include a small, deterministic mean-reversion parameter grid.
    for fast, slow, momentum, breakout, rsi_entry, rsi_exit, stop_pct in itertools.product(
        (8, 12), (30, 48), (6, 12), (12, 24), (20, 25, 30), (45, 50, 55), (0.04, 0.06)
    ):
        if fast < slow and rsi_entry < rsi_exit:
            candidates.append({"fast": fast, "slow": slow, "momentum": momentum,
                               "breakout": breakout, "entry_momentum": 0.0,
                               "exit_momentum": -0.005, "rsi_entry": rsi_entry,
                               "rsi_exit": rsi_exit, "stop_pct": stop_pct})
    # Limit run time while retaining deterministic coverage.
    return candidates[::max(1, len(candidates) // 48)][:48]


TRAIN_MIN_RETURN_PCT = 0.0
TRAIN_MAX_DRAWDOWN_PCT = 35.0


def score(metrics: dict[str, float]) -> float:
    # Prefer return while explicitly penalizing drawdown and excessive churn.
    return metrics["total_return_pct"] - 0.75 * metrics["max_drawdown_pct"] - 0.01 * metrics["trades"]


def candidate_passes_gates(train: dict[str, float], validation: dict[str, float],
                           validation_stress: dict[str, float]) -> bool:
    """Require baseline training robustness as well as positive validation performance.

    The final test is deliberately excluded: it remains report-only and is never
    used to rank or accept candidates. These gates are necessary, not proof of
    future profitability or a substitute for forward paper trading.
    """
    return (
        train["total_return_pct"] > TRAIN_MIN_RETURN_PCT
        and train["max_drawdown_pct"] < TRAIN_MAX_DRAWDOWN_PCT
        and validation["total_return_pct"] > 0
        and validation_stress["total_return_pct"] > 0
        and validation["max_drawdown_pct"] < 25
        and validation["trades"] >= 2
    )


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.60,
                       validation_fraction: float = 0.20
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ordered, non-overlapping train/validation/final-test periods."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("training and validation fractions must leave a test period")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("split input must be chronologically sorted with unique timestamps")
    n = len(frame)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    train, validation, test = (frame.iloc[:train_end].copy(),
                               frame.iloc[train_end:validation_end].copy(),
                               frame.iloc[validation_end:].copy())
    if train.empty or validation.empty or test.empty:
        raise ValueError("training, validation, and test periods must all be non-empty")
    return train, validation, test


def _period_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    return {"start": frame.index[0].isoformat() if len(frame) else None,
            "end": frame.index[-1].isoformat() if len(frame) else None,
            "bars": int(len(frame))}


def run_research(days: int = 90) -> dict[str, Any]:
    REPORTS.mkdir(exist_ok=True)
    STATE.mkdir(exist_ok=True)
    started = utc_now()
    results: list[dict[str, Any]] = []
    candidates = candidate_params()
    errors: list[dict[str, Any]] = []
    evaluation_frames: dict[tuple[str, str], pd.DataFrame] = {}
    try:
        research_symbols, universe_metadata = get_research_batch()
    except Exception as exc:
        research_symbols = []
        universe_metadata = {"status": "unavailable", "source": "CoinGecko top-100 intersected with Coinbase USD spot products"}
        errors.append({"universe": str(exc)})
    for symbol in research_symbols:
        try:
            raw = fetch_klines(symbol, "5m", days).sort_index()
            for interval, rule in INTERVALS.items():
                bars = raw if interval == "5m" else resample_ohlcv(raw, rule)
                if len(bars) < 1100:
                    errors.append({"symbol": symbol, "interval": interval,
                                   "error": "Insufficient bars for independent train/validation/test evaluation"})
                    continue
                evaluation_frames[(symbol, interval)] = bars.copy()
                train, validation, test = chronological_split(bars)
                for strategy in ("trend", "breakout", "mean_reversion"):
                    for params in candidates:
                        train_m = backtest(train, strategy, params, 10, 5, interval)
                        validation_m = backtest(validation, strategy, params, 10, 5, interval)
                        validation_stress = backtest(validation, strategy, params, 20, 10, interval)
                        passed = candidate_passes_gates(train_m, validation_m, validation_stress)
                        results.append({
                            "symbol": symbol, "interval": interval, "strategy": strategy,
                            "params": params, "train": train_m, "validation": validation_m,
                            "validation_stress": validation_stress,
                            "validation_score": score(validation_m),
                            "passed_validation": bool(passed),
                            "periods": {"train": _period_metadata(train),
                                        "validation": _period_metadata(validation),
                                        "test": _period_metadata(test)},
                        })
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    eligible = [r for r in results if r.get("passed_validation")]
    eligible.sort(key=lambda r: r["validation_score"], reverse=True)
    selected = eligible[0] if eligible else None
    champion = None
    if selected is not None:
        # The final test is only evaluated after the validation-only selection is fixed.
        symbol, interval = selected["symbol"], selected["interval"]
        bars = evaluation_frames[(symbol, interval)]
        train, validation, test = chronological_split(bars)
        test_metrics = backtest(test, selected["strategy"], selected["params"],
                                DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS, interval)
        test_stress = backtest(test, selected["strategy"], selected["params"], 20, 10, interval)
        version_material = json.dumps({
            "symbol": symbol, "interval": interval, "strategy": selected["strategy"],
            "params": selected["params"], "train": _period_metadata(train),
            "validation": _period_metadata(validation), "test": _period_metadata(test),
            "created_at": started,
        }, sort_keys=True)
        import hashlib
        version_hash = hashlib.sha256(version_material.encode()).hexdigest()[:12]
        champion = {
            "symbol": symbol, "interval": interval, "strategy": selected["strategy"],
            "params": selected["params"], "champion_version": f"champion-{version_hash}",
            "validation_metadata": {
                "created_at": started, "status": "validated",
                "selection_metric": "validation_score",
                "validation_score": selected["validation_score"],
                "validation_criteria": {
                    "training_return_positive": True,
                    "training_max_drawdown_below_pct": TRAIN_MAX_DRAWDOWN_PCT,
                    "validation_return_positive": True,
                    "validation_stress_return_positive": True,
                    "validation_max_drawdown_below_pct": 25,
                    "minimum_validation_position_changes": 2,
                    "final_test_used_for_selection": False,
                },
                "periods": {"train": _period_metadata(train),
                            "validation": _period_metadata(validation),
                            "test": _period_metadata(test)},
                "train_metrics": selected["train"],
                "validation_metrics": selected["validation"],
                "validation_stress_metrics": selected["validation_stress"],
                "test_metrics": test_metrics, "test_stress_metrics": test_stress,
                "test_used_for_selection": False,
                "cost_convention": {
                    "fee_bps_per_side": DEFAULT_FEE_BPS,
                    "slippage_bps_per_side": DEFAULT_SLIPPAGE_BPS,
                    "stress_fee_bps_per_side": 20, "stress_slippage_bps_per_side": 10,
                    "fill_timing": "closed candle close after one-candle signal shift",
                    "fee_basis": "executed notional",
                    "slippage": "execution price adjusted per side",
                },
                "sharpe_annualization": {"bars_per_year": bars_per_year(interval),
                                         "interval": interval, "calendar_days_per_year": 365},
                "warning": "Test metrics are report-only and were not used to rank, select, or validate this champion.",
            },
        }
    validation_status = "validated" if champion else ("no_candidate" if results else "insufficient_data")
    summary = {
        "started_at": started, "completed_at": utc_now(), "days": days,
        "run_status": "completed" if results else "failed",
        "validation_status": validation_status,
        "validation_result": {
            "status": validation_status,
            "evaluations": len(results),
            "validated_candidates": len(eligible),
            "champion_version": champion["champion_version"] if champion else None,
            "reason": None if champion else (
                "No candidate passed validation gates" if results
                else "No candidate evaluations completed; check market data and history coverage"
            ),
        },
        "symbols": list(research_symbols), "universe": universe_metadata, "intervals": list(INTERVALS),
        "candidate_count": len(candidates), "evaluations": len(results),
        "validated_count": len(eligible), "champion": champion,
        "selection_policy": "Require positive training return, training max drawdown under 35%, positive validation and validation-stress returns, validation drawdown under 25%, and at least two validation position changes; rank eligible candidates by validation score only. Final test metrics are report-only and never used for selection.",
        "results": sorted(results, key=lambda r: r["validation_score"], reverse=True)[:30],
        "errors": errors,
        "warning": "Test metrics are not used for selection. Repeated runs over the same dates can still contaminate broader research decisions.",
    }
    (REPORTS / "research_latest.json").write_text(json.dumps(summary, indent=2))
    with (REPORTS / "experiments.jsonl").open("a") as handle:
        handle.write(json.dumps({"at": utc_now(), "days": days, "evaluations": len(results),
                                 "validated": len(eligible),
                                 "champion_version": champion["champion_version"] if champion else None},
                                default=str) + "\n")
    if champion:
        # Keep immutable, version-addressable snapshots as well as the current champion.
        version_dir = STATE / "champions"
        version_dir.mkdir(parents=True, exist_ok=True)
        version_file = version_dir / f"{champion['champion_version']}.json"
        if not version_file.exists():
            version_file.write_text(json.dumps(champion, indent=2))
        (STATE / "champion.json").write_text(json.dumps(champion, indent=2))
    else:
        # Clear the active pointer rather than leaving a stale champion tradable.
        (STATE / "champion.json").write_text("null")
    return summary

def validate_research_artifacts(invocation_started: datetime, report_path: Path | None = None) -> dict[str, Any]:
    """Fail closed unless this invocation produced a fresh, substantive report."""
    report_path = report_path or (REPORTS / "research_latest.json")
    if not report_path.is_file():
        raise RuntimeError(f"Research report was not produced: {report_path}")
    if report_path.stat().st_mtime < invocation_started.timestamp() - 2:
        raise RuntimeError("Research report is stale; this invocation did not refresh it")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Research report is unreadable: {exc}") from exc

    def parse_timestamp(value: Any, field: str) -> datetime:
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"Research report is missing {field}")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError(f"Research report has invalid {field}") from exc
        if parsed.tzinfo is None:
            raise RuntimeError(f"Research report {field} must include a timezone")
        return parsed.astimezone(timezone.utc)

    report_started = parse_timestamp(report.get("started_at"), "started_at")
    report_completed = parse_timestamp(report.get("completed_at"), "completed_at")
    if report_started < invocation_started.replace(microsecond=0) - pd.Timedelta(seconds=2).to_pytimedelta():
        raise RuntimeError("Research report started before this invocation")
    if report_completed < report_started:
        raise RuntimeError("Research report completed_at precedes started_at")
    if report.get("run_status") != "completed":
        raise RuntimeError(f"Research run did not complete successfully: {report.get('run_status')}")
    if not isinstance(report.get("evaluations"), int) or report["evaluations"] <= 0:
        raise RuntimeError("Research produced no candidate evaluations")
    validation = report.get("validation_result")
    if not isinstance(validation, dict) or validation.get("status") != "validated":
        raise RuntimeError("Research did not produce a validated champion")
    if validation.get("status") != report.get("validation_status"):
        raise RuntimeError("Research validation status is inconsistent")
    if validation.get("evaluations") != report.get("evaluations"):
        raise RuntimeError("Research validation result does not match evaluation count")
    if validation.get("validated_candidates", 0) < 1 or report.get("validated_count", 0) < 1:
        raise RuntimeError("Research produced no candidate that passed validation")
    champion = report.get("champion")
    if not isinstance(champion, dict):
        raise RuntimeError("Validation says a champion passed, but no champion was reported")
    champion_version = champion.get("champion_version")
    if not isinstance(champion_version, str) or not re.fullmatch(r"champion-[A-Za-z0-9_-]{6,64}", champion_version):
        raise RuntimeError("Validated champion version is missing or invalid")
    metadata = champion.get("validation_metadata")
    if not isinstance(metadata, dict) or metadata.get("status") != "validated":
        raise RuntimeError("Champion is missing explicit validated metadata")
    if metadata.get("test_used_for_selection") is not False:
        raise RuntimeError("Champion metadata does not prove final-test isolation")
    for field in ("train_metrics", "validation_metrics", "validation_stress_metrics", "test_metrics", "test_stress_metrics"):
        if not isinstance(metadata.get(field), dict):
            raise RuntimeError(f"Champion validation metadata is missing {field}")
    train_metrics = metadata["train_metrics"]
    if train_metrics.get("total_return_pct", float("-inf")) <= TRAIN_MIN_RETURN_PCT:
        raise RuntimeError("Champion failed training robustness gate: training return must be positive")
    if train_metrics.get("max_drawdown_pct", float("inf")) >= TRAIN_MAX_DRAWDOWN_PCT:
        raise RuntimeError("Champion failed training robustness gate: training drawdown must be below 35%")
    cost = metadata.get("cost_convention")
    if not isinstance(cost, dict) or cost.get("fee_basis") != "executed notional":
        raise RuntimeError("Champion cost convention is missing or invalid")
    created_at = parse_timestamp(metadata.get("created_at"), "champion validation created_at")
    if created_at < report_started or created_at > report_completed:
        raise RuntimeError("Champion validation timestamp is outside the research run")
    if validation.get("champion_version") != champion_version:
        raise RuntimeError("Validation result champion version does not match the champion")

    # A report alone is not sufficient evidence that the strategy is deployable
    # to the paper trader. Verify the active pointer and immutable version snapshot
    # both match this exact invocation's champion.
    active_path = STATE / "champion.json"
    if not active_path.is_file():
        raise RuntimeError(f"Active champion artifact is missing: {active_path}")
    try:
        active_champion = json.loads(active_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Active champion artifact is unreadable: {exc}") from exc
    if active_champion != champion:
        raise RuntimeError("Active champion artifact does not match the fresh research report")

    snapshot_path = STATE / "champions" / f"{champion_version}.json"
    if not snapshot_path.is_file():
        raise RuntimeError(f"Immutable champion snapshot is missing: {snapshot_path}")
    try:
        snapshot = json.loads(snapshot_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Immutable champion snapshot is unreadable: {exc}") from exc
    if snapshot != champion:
        raise RuntimeError("Immutable champion snapshot does not match the fresh research report")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Coinbase-only strategy research and validate the resulting artifacts.")
    parser.add_argument("--days", type=int, default=90, help="Historical window in days (30–365; default: 90)")
    parser.add_argument(
        "--require-fresh-report", action="store_true",
        help="Explicitly require a fresh report and complete validation result (always enforced in CI).",
    )
    args = parser.parse_args(argv)
    if not 30 <= args.days <= 365:
        parser.error("--days must be between 30 and 365")

    invocation_started = datetime.now(timezone.utc)
    try:
        run_research(days=args.days)
        report = validate_research_artifacts(invocation_started)
    except Exception as exc:
        import sys
        print(f"Research failed artifact validation: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "run_status": report["run_status"],
        "validation_status": report["validation_status"],
        "evaluations": report["evaluations"],
        "validated_count": report["validated_count"],
        "champion_version": report["validation_result"].get("champion_version"),
        "report": str(REPORTS / "research_latest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
