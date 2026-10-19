from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from portfolio_accounting import (DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS,
                                  MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT,
                                  MAX_POSITION_FRACTION, STRESS_FEE_BPS,
                                  STRESS_SLIPPAGE_BPS, buy_with_costs, entry_budget,
                                  sell_with_costs, _validate_costs)
from universe import get_coinbase_product_id, get_research_batch

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
STATE = ROOT / "state"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
INTERVALS = {"5m": "5min", "10m": "10min", "15m": "15min", "30m": "30min"}
MIN_CANDLE_COVERAGE_PCT = 99.0
MAX_CONSECUTIVE_MISSING_SLOTS = 3
MAX_FX_MISSING_PCT = 1.0
OOS_POLICY_VERSION = "chronological-oos-v2"
OOS_LEDGER_FILE = STATE / "oos_evaluation_ledger.json"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Autonomous-Paper-Trading-Lab/1.0"})
_USDT_USD_CANDLE_CACHE: dict[tuple[str, int], pd.DataFrame] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_coinbase(symbol: str, interval: str, days: int, *,
                    window_start: pd.Timestamp | None = None,
                    window_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Fetch Coinbase Exchange public USD candles over aligned UTC boundaries."""
    product = get_coinbase_product_id(symbol)
    if not product:
        raise ValueError(f"No Coinbase fallback mapping for {symbol}")
    granularity = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}.get(interval)
    if granularity is None:
        raise ValueError(f"Unsupported Coinbase candle interval: {interval}")
    if window_start is None or window_end is None:
        window_start, window_end = _aligned_candle_window(interval, days)
    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Coinbase window boundaries must be ordered timezone-aware timestamps")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.floor(f"{granularity}s") or end != end.floor(f"{granularity}s"):
        raise ValueError("Coinbase window boundaries must align to candle intervals")
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
    duplicate_candles_dropped = int(frame["time"].duplicated().sum())
    response_was_sorted = bool(frame["time"].is_monotonic_increasing)
    conflicting_duplicate_groups = 0
    duplicated = frame.loc[frame["time"].duplicated(keep=False)]
    if not duplicated.empty:
        for _, group in duplicated.groupby("time", sort=False):
            if group[["low", "high", "open", "close", "volume"]].nunique(dropna=False).gt(1).any():
                conflicting_duplicate_groups += 1
    frame = frame.drop_duplicates("time").sort_values("time")
    frame["open_time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame["close_time"] = frame["open_time"] + pd.to_timedelta(granularity, unit="s")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.set_index("open_time")
    closed = frame.loc[
        (frame.index >= start) & (frame.index < end) & (frame["close_time"] <= end),
        ["open", "high", "low", "close", "volume", "close_time"]
    ]
    closed.attrs["duplicate_candles_dropped"] = duplicate_candles_dropped
    closed.attrs["conflicting_duplicate_groups"] = conflicting_duplicate_groups
    closed.attrs["response_was_sorted"] = response_was_sorted
    closed.attrs["source_was_sorted"] = bool(closed.index.is_monotonic_increasing)
    return closed


def _max_consecutive_missing_slots(index: pd.Index, granularity: int) -> int:
    """Return the longest run of missing interval slots between observed candles."""
    if len(index) < 2:
        return 0
    stamps = pd.DatetimeIndex(index)
    differences = stamps.to_series().diff().dropna().dt.total_seconds().to_numpy()
    if not len(differences):
        return 0
    slot_counts = np.rint(differences / granularity).astype(int)
    return int(max(0, int(slot_counts.max()) - 1))


def _aligned_candle_window(interval: str, days: int,
                           now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return aligned UTC candle-open boundaries for the requested [start, end) window."""
    seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}.get(interval)
    if seconds is None or days <= 0:
        raise ValueError("Unsupported interval or non-positive requested days")
    current = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    if current.tzinfo is None:
        raise ValueError("Candle window clock must be timezone-aware")
    end = current.tz_convert("UTC").floor(f"{seconds}s")
    return end - pd.Timedelta(days=days), end


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for missing in values:
        current = current + 1 if bool(missing) else 0
        longest = max(longest, current)
    return longest


def candle_quality_report(frame: pd.DataFrame, interval: str = "5m", days: int = 90,
                          now: pd.Timestamp | None = None,
                          window_start: pd.Timestamp | None = None,
                          window_end: pd.Timestamp | None = None) -> dict[str, Any]:
    """Measure coverage against every expected candle in an aligned UTC [start, end) window."""
    granularity = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}.get(interval)
    if granularity is None:
        raise ValueError(f"Unsupported candle-quality interval: {interval}")
    if window_start is None or window_end is None:
        window_start, window_end = _aligned_candle_window(interval, days, now)
    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Candle quality boundaries must be ordered timezone-aware timestamps")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.floor(f"{granularity}s") or end != end.floor(f"{granularity}s"):
        raise ValueError("Candle quality boundaries must align to candle intervals")
    expected = pd.date_range(start=start, end=end, freq=f"{granularity}s", inclusive="left")
    raw_index = pd.DatetimeIndex(frame.index)
    timezone_aware = raw_index.tz is not None
    normalized = raw_index.tz_localize("UTC") if raw_index.tz is None else raw_index.tz_convert("UTC")
    expected_mask = expected.isin(normalized)
    missing_mask = ~expected_mask
    expected_slots = len(expected)
    observed_expected = int(expected_mask.sum())
    coverage = observed_expected / expected_slots * 100 if expected_slots else 0.0
    invalid_ohlc_rows = 0
    if not frame.empty:
        required = ["open", "high", "low", "close", "volume"]
        if not set(required).issubset(frame.columns):
            invalid_ohlc_rows = len(frame)
        else:
            values = frame[required].apply(pd.to_numeric, errors="coerce")
            valid = np.isfinite(values.to_numpy()).all(axis=1)
            valid &= (values[["open", "high", "low", "close"]] > 0).all(axis=1).to_numpy()
            valid &= (values["high"].to_numpy() >= values[["open", "close"]].max(axis=1).to_numpy())
            valid &= (values["low"].to_numpy() <= values[["open", "close"]].min(axis=1).to_numpy())
            valid &= (values["high"].to_numpy() >= values["low"].to_numpy())
            valid &= (values["volume"].to_numpy() >= 0)
            invalid_ohlc_rows = int((~valid).sum())
    diffs = normalized.to_series().diff().dropna().dt.total_seconds().to_numpy() if len(normalized) > 1 else np.array([])
    irregular = int(np.count_nonzero(np.abs(diffs / granularity - np.rint(diffs / granularity)) > 1e-6))
    missing_at_start = 0
    while missing_at_start < len(missing_mask) and missing_mask[missing_at_start]:
        missing_at_start += 1
    missing_at_end = 0
    while missing_at_end < len(missing_mask) and missing_mask[len(missing_mask) - 1 - missing_at_end]:
        missing_at_end += 1
    unaligned = sum(1 for stamp in normalized if stamp < start or stamp >= end or stamp not in expected)
    return {
        "interval": interval, "requested_days": int(days),
        "window_start_utc": start.isoformat(), "window_end_utc_exclusive": end.isoformat(),
        "expected_window_slots": int(expected_slots), "observed_source_candles": int(len(frame)),
        "observed_expected_slots": observed_expected, "missing_expected_slots": int(missing_mask.sum()),
        "missing_at_start_slots": int(missing_at_start), "missing_at_end_slots": int(missing_at_end),
        "source_coverage_pct": round(coverage, 4),
        "duplicate_candles_dropped": int(frame.attrs.get("duplicate_candles_dropped", 0)),
        "conflicting_duplicate_groups": int(frame.attrs.get("conflicting_duplicate_groups", 0)),
        "response_was_sorted": bool(frame.attrs.get("response_was_sorted", frame.index.is_monotonic_increasing)),
        "source_was_sorted": bool(frame.attrs.get("source_was_sorted", frame.index.is_monotonic_increasing)),
        "unique_timestamps": not raw_index.has_duplicates, "monotonic_timestamps": bool(raw_index.is_monotonic_increasing),
        "timezone_aware_timestamps": timezone_aware, "unaligned_timestamps": int(unaligned),
        "irregular_timestamp_gaps": irregular, "max_consecutive_missing_slots": _longest_true_run(missing_mask),
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "coverage_note": "Coverage is measured against all expected aligned UTC candles; missing candles are never interpolated.",
    }


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
    window_start, window_end = _aligned_candle_window(interval, days)
    frame = _fetch_coinbase(symbol, interval, days, window_start=window_start, window_end=window_end)
    if frame.empty:
        raise RuntimeError(f"Coinbase returned no closed candles for {symbol} {interval}")
    source_quality = candle_quality_report(frame, interval, days, window_start=window_start, window_end=window_end)
    source_candle_count = len(frame)
    fx_missing_candles = 0
    if symbol != "USDTUSD":
        cache_key = (interval, days, window_start.isoformat(), window_end.isoformat())
        if cache_key not in _USDT_USD_CANDLE_CACHE:
            _USDT_USD_CANDLE_CACHE[cache_key] = _fetch_coinbase(
                "USDTUSD", interval, days, window_start=window_start, window_end=window_end
            )
        fx_frame = _USDT_USD_CANDLE_CACHE[cache_key]
        frame = _convert_usd_candles_to_usdt(frame, fx_frame)
        fx_missing_candles = int(frame.attrs.get("fx_missing_candles", 0))
        if frame.empty:
            raise RuntimeError(
                f"No same-candle USDT-USD FX closes available to convert {symbol} {interval}"
            )
    quality = dict(source_quality)
    converted_quality = candle_quality_report(
        frame, interval, days, window_start=window_start, window_end=window_end
    )
    quality.update({
        "source_candle_count": source_candle_count,
        "converted_candle_count": int(len(frame)),
        "converted_coverage_pct": converted_quality["source_coverage_pct"],
        "converted_missing_expected_slots": converted_quality["missing_expected_slots"],
        "converted_invalid_ohlc_rows": converted_quality["invalid_ohlc_rows"],
        "fx_missing_candles": fx_missing_candles,
        "fx_missing_pct_of_source": round(fx_missing_candles / max(source_candle_count, 1) * 100.0, 4),
        "converted_max_consecutive_missing_slots": converted_quality["max_consecutive_missing_slots"],
        "converted_unique_timestamps": converted_quality["unique_timestamps"],
        "converted_monotonic_timestamps": converted_quality["monotonic_timestamps"],
        "converted_unaligned_timestamps": converted_quality["unaligned_timestamps"],
    })
    quality["quality_gate_passed"] = bool(
        quality["source_coverage_pct"] >= MIN_CANDLE_COVERAGE_PCT
        and quality["converted_coverage_pct"] >= MIN_CANDLE_COVERAGE_PCT
        and quality["fx_missing_pct_of_source"] <= MAX_FX_MISSING_PCT
        and quality["invalid_ohlc_rows"] == 0
        and quality["converted_invalid_ohlc_rows"] == 0
        and quality["conflicting_duplicate_groups"] == 0
        and quality["unique_timestamps"]
        and quality["monotonic_timestamps"]
        and quality["timezone_aware_timestamps"]
        and quality["unaligned_timestamps"] == 0
        and quality["irregular_timestamp_gaps"] == 0
        and quality["max_consecutive_missing_slots"] <= MAX_CONSECUTIVE_MISSING_SLOTS
        and quality["converted_max_consecutive_missing_slots"] <= MAX_CONSECUTIVE_MISSING_SLOTS
        and quality["converted_unique_timestamps"]
        and quality["converted_monotonic_timestamps"]
        and quality["converted_unaligned_timestamps"] == 0
    )
    frame.attrs["data_quality"] = quality
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


def _benchmark_window_metadata(frame: pd.DataFrame, exposure_fraction: float,
                               starting_cash: float, fee_bps: float,
                               slippage_bps: float, interval: str) -> dict[str, Any]:
    index = pd.DatetimeIndex(frame.index)
    if not frame.empty and (index.tz is None or not index.is_monotonic_increasing or index.has_duplicates):
        raise ValueError("benchmarks require unique, monotonic, timezone-aware timestamps")
    missing_slots = 0
    start = end = None
    if not frame.empty:
        seconds = int(pd.Timedelta(INTERVALS[interval]).total_seconds())
        expected = pd.date_range(index[0], index[-1], freq=f"{seconds}s")
        missing_slots = int(len(expected.difference(index)))
        start, end = index[0].isoformat(), index[-1].isoformat()
    return {
        "period_start": start, "period_end": end, "bars": int(len(frame)),
        "missing_candle_slots": missing_slots,
        "missing_data_policy": "No interpolation; use exact supplied timestamps and report missing slots.",
        "exposure_fraction": float(exposure_fraction), "starting_cash_usdt": float(starting_cash),
        "fee_bps_per_side": float(fee_bps), "slippage_bps_per_side": float(slippage_bps),
    }


def _validate_benchmark_prices(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    if "close" not in frame:
        raise ValueError("benchmark requires close prices")
    prices = pd.to_numeric(frame["close"], errors="coerce")
    if prices.isna().any() or (~np.isfinite(prices)).any() or (prices <= 0).any():
        raise ValueError("benchmark requires finite, positive close prices; missing data is not imputed")
    return prices


def cash_benchmark(frame: pd.DataFrame, interval: str = "5m",
                   starting_cash: float = 15.0) -> dict[str, Any]:
    """Report a zero-exposure cash benchmark over the exact supplied timestamps."""
    _validate_benchmark_prices(frame)
    result: dict[str, Any] = performance(pd.Series(np.zeros(len(frame), dtype=float), index=frame.index), interval, trades=0)
    result.update(_benchmark_window_metadata(frame, 0.0, starting_cash, 0.0, 0.0, interval))
    return result


def buy_and_hold_benchmark(frame: pd.DataFrame, interval: str = "5m",
                           fee_bps: float = DEFAULT_FEE_BPS,
                           slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
                           starting_cash: float = 15.0) -> dict[str, Any]:
    """Buy once at the shared 50%-equity budget; do not interpolate or liquidate at end."""
    _validate_costs(fee_bps, slippage_bps)
    prices = _validate_benchmark_prices(frame)
    if not math.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("starting_cash must be finite and positive")
    if frame.empty:
        result: dict[str, Any] = performance(pd.Series(dtype=float), interval, trades=0)
        result.update(_benchmark_window_metadata(frame, 0.0, starting_cash, fee_bps, slippage_bps, interval))
        return result
    spend_cash = entry_budget(starting_cash, starting_cash)
    buy = buy_with_costs(spend_cash, float(prices.iloc[0]), fee_bps, slippage_bps)
    cash = max(0.0, starting_cash - spend_cash + buy.cash_remaining)
    equity = pd.Series(cash + buy.quantity * prices, index=frame.index, dtype=float)
    returns = equity.pct_change()
    if not returns.empty:
        returns.iloc[0] = equity.iloc[0] / starting_cash - 1.0
    result: dict[str, Any] = performance(returns, interval, trades=1 if buy.quantity > 0 else 0)
    result.update(_benchmark_window_metadata(frame, MAX_POSITION_FRACTION, starting_cash, fee_bps, slippage_bps, interval))
    return result


def backtest(frame: pd.DataFrame, strategy: str, params: dict[str, Any],
             fee_bps: float = DEFAULT_FEE_BPS,
             slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
             interval: str = "5m", starting_cash: float = 15.0) -> dict[str, float]:
    """Simulate closed-candle fills using paper-trader per-side costs."""
    _validate_costs(fee_bps, slippage_bps)
    quality = frame.attrs.get("data_quality")
    if isinstance(quality, dict) and quality.get("quality_gate_passed") is not True:
        raise ValueError("backtest refused: input failed the candle/FX data-quality gate")
    minimum = max(int(params["slow"]) * 3, int(params["breakout"]) + 50)
    if len(frame) < minimum:
        return performance(pd.Series(dtype=float), interval, trades=0)
    if not math.isfinite(starting_cash) or starting_cash <= 0:
        raise ValueError("starting_cash must be finite and positive")
    required = ["open", "high", "low", "close", "volume"]
    if not set(required).issubset(frame.columns):
        raise ValueError("backtest requires complete OHLCV columns")
    values = frame[required].apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(values.to_numpy()).all()
            or (values[["open", "high", "low", "close"]] <= 0).any().any()
            or (values["high"] < values[["open", "close"]].max(axis=1)).any()
            or (values["low"] > values[["open", "close"]].min(axis=1)).any()
            or (values["high"] < values["low"]).any()
            or (values["volume"] < 0).any()):
        raise ValueError("backtest refused: invalid OHLCV rows")
    prices = pd.to_numeric(frame["close"], errors="coerce")
    if prices.isna().any() or (~np.isfinite(prices)).any() or (prices <= 0).any():
        raise ValueError("backtest requires finite, positive close prices")

    target = desired_signal(frame, strategy, params).reindex(frame.index).fillna(0.0)
    cash, quantity, entry_basis, changes = float(starting_cash), 0.0, 0.0, 0
    equity_values: list[float] = []
    risk_day: str | None = None
    daily_start_equity = float(starting_cash)
    peak_equity = float(starting_cash)
    halted = False
    for timestamp, market_price in prices.items():
        price = float(market_price)
        stamp = pd.Timestamp(timestamp)
        if stamp.tzinfo is None:
            raise ValueError("backtest requires timezone-aware candle timestamps")
        day = stamp.tz_convert("UTC").date().isoformat()
        equity_before = cash + quantity * price
        if day != risk_day:
            risk_day = day
            daily_start_equity = max(equity_before, 1e-12)
        peak_equity = max(peak_equity, equity_before)
        daily_loss_pct = max(0.0, (daily_start_equity - equity_before) / daily_start_equity * 100.0)
        drawdown_pct = max(0.0, (peak_equity - equity_before) / peak_equity * 100.0)
        if daily_loss_pct >= MAX_DAILY_LOSS_PCT or drawdown_pct >= MAX_DRAWDOWN_PCT:
            halted = True

        # Match the paper trader's latched circuit breakers: liquidate at a
        # current mark once triggered and do not re-enter during this period.
        if halted:
            if quantity > 0:
                sale = sell_with_costs(quantity, price, entry_basis, fee_bps, slippage_bps)
                cash += sale.net_proceeds
                quantity, entry_basis = 0.0, 0.0
                changes += 1
            equity_values.append(cash)
            continue

        wants_long = bool(float(target.loc[timestamp]) > 0)
        if wants_long and quantity <= 0:
            # Match the paper trader's shared entry budget rather than investing
            # the entire simulated portfolio in every position.
            spend_cash = entry_budget(cash, equity_before)
            buy = buy_with_costs(spend_cash, price, fee_bps, slippage_bps)
            quantity = buy.quantity
            cash = max(0.0, cash - spend_cash + buy.cash_remaining)
            entry_basis = buy.entry_cost_per_unit
            changes += 1
        elif not wants_long and quantity > 0:
            sale = sell_with_costs(quantity, price, entry_basis, fee_bps, slippage_bps)
            cash += sale.net_proceeds
            quantity, entry_basis = 0.0, 0.0
            changes += 1

        equity_after = cash + quantity * price
        peak_equity = max(peak_equity, equity_after)
        after_daily_loss = max(0.0, (daily_start_equity - equity_after) / daily_start_equity * 100.0)
        after_drawdown = max(0.0, (peak_equity - equity_after) / peak_equity * 100.0)
        if after_daily_loss >= MAX_DAILY_LOSS_PCT or after_drawdown >= MAX_DRAWDOWN_PCT:
            # The paper trader records the latch after the decision and liquidates
            # on the next decision cycle; mirror that sequencing here.
            halted = True
        equity_values.append(equity_after)

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


def effective_strategy_params(strategy: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return only parameters that can change a strategy's rules."""
    keys = {
        "trend": ("fast", "slow", "momentum", "entry_momentum", "exit_momentum"),
        "breakout": ("fast", "momentum", "breakout", "exit_momentum"),
        "mean_reversion": ("slow", "rsi_entry", "rsi_exit", "stop_pct"),
    }
    if strategy not in keys:
        raise ValueError(f"Unknown strategy: {strategy}")
    defaults = {"entry_momentum": 0.0, "exit_momentum": -0.005,
                "rsi_entry": 30, "rsi_exit": 50, "stop_pct": 0.06}
    return {key: params.get(key, defaults.get(key)) for key in keys[strategy]}


def unique_candidates_for_strategy(strategy: str,
                                   candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate grid entries whose strategy-effective parameter sets are identical."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        effective = effective_strategy_params(strategy, candidate)
        key = json.dumps(effective, sort_keys=True, separators=(",", ":"), default=str)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


TRAIN_MIN_RETURN_PCT = 0.0
TRAIN_MAX_DRAWDOWN_PCT = 35.0


def score(metrics: dict[str, float], benchmarks: dict[str, dict[str, float]] | None = None) -> float:
    """Rank using validation-only excess return, drawdown, and turnover."""
    benchmark_return = max(
        (item.get("total_return_pct", 0.0) for item in (benchmarks or {}).values()),
        default=0.0,
    )
    excess_return = metrics["total_return_pct"] - benchmark_return
    return excess_return - 0.75 * metrics["max_drawdown_pct"] - 0.01 * metrics["trades"]


def _beats_benchmarks(metrics: dict[str, float],
                      benchmarks: dict[str, dict[str, float]] | None) -> bool:
    if not isinstance(benchmarks, dict) or not {"cash", "buy_and_hold"}.issubset(benchmarks):
        return False
    try:
        candidate_return = float(metrics["total_return_pct"])
        benchmark_returns = [float(benchmarks[name]["total_return_pct"])
                             for name in ("cash", "buy_and_hold")]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in [candidate_return, *benchmark_returns]) and all(
        candidate_return > benchmark_return for benchmark_return in benchmark_returns
    )


def candidate_passes_gates(train: dict[str, float], validation: dict[str, float],
                           validation_stress: dict[str, float],
                           validation_benchmarks: dict[str, dict[str, float]] | None = None,
                           validation_stress_benchmarks: dict[str, dict[str, float]] | None = None) -> bool:
    """Require training robustness and after-cost validation advantage over cash and buy-and-hold.

    The final test is excluded from selection. After validation-only ranking fixes
    one candidate, a separate held-out deployment veto decides whether it can be promoted.
    """
    core_gates = (
        train["total_return_pct"] > TRAIN_MIN_RETURN_PCT
        and train["max_drawdown_pct"] < TRAIN_MAX_DRAWDOWN_PCT
        and validation["total_return_pct"] > 0
        and validation_stress["total_return_pct"] > 0
        and validation["max_drawdown_pct"] < 25
        and validation["trades"] >= 2
    )
    if not core_gates:
        return False
    if validation_benchmarks is None and validation_stress_benchmarks is None:
        return True
    return (
        _beats_benchmarks(validation, validation_benchmarks)
        and _beats_benchmarks(validation_stress, validation_stress_benchmarks)
    )


def deployment_gate_failures(test_metrics: dict[str, float],
                             test_stress_metrics: dict[str, float],
                             test_benchmarks: dict[str, dict[str, float]],
                             test_stress_benchmarks: dict[str, dict[str, float]]) -> list[str]:
    """Out-of-sample veto; never used to rank candidates or fall back to another one."""
    failures: list[str] = []
    for label, metrics, benchmarks in (
        ("test", test_metrics, test_benchmarks),
        ("stress_test", test_stress_metrics, test_stress_benchmarks),
    ):
        if not _beats_benchmarks(metrics, benchmarks):
            failures.append(f"{label} return must beat both cash and buy-and-hold after costs")
        try:
            drawdown = float(metrics.get("max_drawdown_pct", float("nan")))
            trades = float(metrics.get("trades", float("nan")))
        except (AttributeError, TypeError, ValueError):
            failures.append(f"{label} metrics are missing valid drawdown/trade values")
            continue
        if not math.isfinite(drawdown) or drawdown >= 25:
            failures.append(f"{label} drawdown must be below 25%")
        if not math.isfinite(trades) or trades < 2:
            failures.append(f"{label} must contain at least two position changes")
    return failures


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


def _empty_oos_ledger() -> dict[str, Any]:
    return {
        "policy_version": OOS_POLICY_VERSION,
        "windows": [],
        "last_test_start": None,
        "last_test_end": None,
        "last_result": None,
    }


def _validate_oos_ledger(ledger: Any) -> dict[str, Any]:
    if not isinstance(ledger, dict) or ledger.get("policy_version") != OOS_POLICY_VERSION:
        raise RuntimeError("Out-of-sample ledger missing or incompatible with chronological-oos-v2; refusing research.")
    windows = ledger.get("windows")
    if not isinstance(windows, list):
        raise RuntimeError("Out-of-sample ledger has no valid append-only windows list; refusing research.")
    previous_end: pd.Timestamp | None = None
    seen_ids: set[str] = set()
    for item in windows:
        if not isinstance(item, dict):
            raise RuntimeError("Out-of-sample ledger contains a malformed window.")
        try:
            start = pd.Timestamp(item["start"])
            end = pd.Timestamp(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Out-of-sample ledger contains invalid window boundaries.") from exc
        if start.tzinfo is None or end.tzinfo is None or start > end:
            raise RuntimeError("Out-of-sample ledger boundaries must be ordered and timezone-aware.")
        if previous_end is not None and start <= previous_end:
            raise RuntimeError("Out-of-sample ledger windows overlap or are not strictly chronological.")
        window_id = item.get("id")
        if not isinstance(window_id, str) or not window_id or window_id in seen_ids:
            raise RuntimeError("Out-of-sample ledger contains a missing or duplicate window ID.")
        if item.get("status") not in {"reserved", "passed", "failed"}:
            raise RuntimeError("Out-of-sample ledger contains an unknown window status.")
        seen_ids.add(window_id)
        previous_end = end.tz_convert("UTC")
    if windows:
        last = windows[-1]
        if ledger.get("last_test_end") != last.get("end") or ledger.get("last_test_start") != last.get("start"):
            raise RuntimeError("Out-of-sample ledger summary disagrees with its append-only window history.")
    return ledger


def _require_nonempty_oos_history(ledger: dict[str, Any]) -> dict[str, Any]:
    """Reject an empty ledger unless a separately reviewed recovery policy exists."""
    if not ledger.get("windows") or not ledger.get("last_test_end"):
        raise RuntimeError(
            "OOS history is empty or uncertain; no reviewed recovery/exclusion policy "
            "is recorded; refusing research."
        )
    return ledger


def _load_local_oos_ledger() -> dict[str, Any]:
    path = STATE / "oos_evaluation_ledger.json"
    if not path.is_file():
        raise RuntimeError(
            "Authoritative OOS ledger is missing; consumed test history is unknown; "
            "refusing research."
        )
    try:
        ledger = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Out-of-sample ledger is unreadable; refusing research.") from exc
    return _require_nonempty_oos_history(_validate_oos_ledger(ledger))


def _atomic_write_local_ledger(ledger: dict[str, Any]) -> None:
    path = STATE / "oos_evaluation_ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    temporary.replace(path)


def _git_checked(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"Safe OOS ledger persistence failed ({' '.join(args)}): {result.stderr.strip()}")
    return result.stdout.strip()


def _update_oos_ledger(mutator) -> dict[str, Any]:
    """Atomically update local state or persist by non-force push to a dedicated ledger branch.

    A normal fast-forward push is the concurrency compare-and-swap: if another run
    advances the branch after our fetch, the push fails and the caller must fail closed.
    """
    branch = os.environ.get("OOS_LEDGER_BRANCH", "").strip()
    if not branch:
        ledger = _load_local_oos_ledger()
        updated = mutator(ledger)
        _validate_oos_ledger(updated)
        _atomic_write_local_ledger(updated)
        return updated
    if branch == "main" or branch.startswith("main/") or not re.fullmatch(r"[A-Za-z0-9._/-]+", branch):
        raise RuntimeError("OOS_LEDGER_BRANCH must name a dedicated non-main branch.")

    with tempfile.TemporaryDirectory(prefix="oos-ledger-") as temporary_dir:
        worktree = Path(temporary_dir) / "worktree"
        worktree_added = False
        try:
            remote_ref = f"refs/remotes/origin/{branch}"
            remote_exists = bool(_git_checked(
                ["ls-remote", "--heads", "origin", f"refs/heads/{branch}"], ROOT
            ))
            if remote_exists:
                _git_checked(["fetch", "origin", f"refs/heads/{branch}:{remote_ref}"], ROOT)
                _git_checked(["worktree", "add", "--detach", str(worktree), remote_ref], ROOT)
                worktree_added = True
                ledger_path = worktree / "state" / "oos_evaluation_ledger.json"
                if not ledger_path.is_file():
                    raise RuntimeError("Dedicated OOS ledger branch exists but its ledger file is missing.")
                try:
                    current = _validate_oos_ledger(json.loads(ledger_path.read_text()))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError("Dedicated OOS ledger branch is unreadable; refusing update.") from exc
            else:
                _git_checked(["worktree", "add", "--detach", str(worktree), "HEAD"], ROOT)
                worktree_added = True
                ledger_path = worktree / "state" / "oos_evaluation_ledger.json"
                local_path = STATE / "oos_evaluation_ledger.json"
                if not local_path.is_file():
                    raise RuntimeError(
                        "Dedicated OOS ledger branch is missing and no authoritative local ledger "
                        "exists; refusing to initialize empty history."
                    )
                try:
                    current = _validate_oos_ledger(json.loads(local_path.read_text()))
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError("Cannot bootstrap dedicated branch from unreadable local ledger.") from exc
            current = _require_nonempty_oos_history(current)
            updated = mutator(current)
            _validate_oos_ledger(updated)
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_path.write_text(json.dumps(updated, indent=2, sort_keys=True))
            _git_checked(["config", "user.name", "github-actions[bot]"], worktree)
            _git_checked(["config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], worktree)
            _git_checked(["add", "--", "state/oos_evaluation_ledger.json"], worktree)
            _git_checked(["commit", "-m", "Reserve/update one-way out-of-sample ledger"], worktree)
            # Never force-push and never target main. A race or non-fast-forward
            # rejection leaves the previous reservation intact and prevents evaluation.
            _git_checked(["push", "origin", f"HEAD:refs/heads/{branch}"], worktree)
            _atomic_write_local_ledger(updated)
            return updated
        finally:
            if worktree_added:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
                )


def _ledger_last_test_end(ledger: dict[str, Any]) -> pd.Timestamp | None:
    raw = ledger.get("last_test_end")
    if not raw:
        return None
    stamp = pd.Timestamp(raw)
    if stamp.tzinfo is None:
        raise RuntimeError("OOS ledger last_test_end is timezone-naive; refusing optimization.")
    return stamp.tz_convert("UTC")


def reserve_oos_test_window(test: pd.DataFrame) -> tuple[bool, str | None, dict[str, str]]:
    """Persist a unique, chronological reservation before evaluating the held-out test."""
    if test.empty or not isinstance(test.index, pd.DatetimeIndex):
        return False, "Held-out test window is empty or has invalid timestamps.", {}
    index = test.index
    if index.tz is None or not index.is_monotonic_increasing or index.has_duplicates:
        return False, "Held-out test timestamps must be unique, monotonic, and timezone-aware.", {}
    start, end = index[0].tz_convert("UTC"), index[-1].tz_convert("UTC")
    identity = hashlib.sha256(
        f"{OOS_POLICY_VERSION}|{start.isoformat()}|{end.isoformat()}|{len(test)}".encode()
    ).hexdigest()[:20]
    window = {"id": identity, "start": start.isoformat(), "end": end.isoformat(), "bars": str(len(test))}

    def reserve(ledger: dict[str, Any]) -> dict[str, Any]:
        _validate_oos_ledger(ledger)
        last_end = _ledger_last_test_end(ledger)
        if last_end is not None and start <= last_end:
            raise ValueError(
                f"Test window overlaps or reuses historical OOS data (policy {OOS_POLICY_VERSION}); "
                "all bars through the previous test end are excluded from later optimization."
            )
        for previous in ledger["windows"]:
            prev_start, prev_end = pd.Timestamp(previous["start"]), pd.Timestamp(previous["end"])
            if start <= prev_end and end >= prev_start:
                raise ValueError("Concurrent or overlapping OOS evaluation rejected.")
        updated = dict(ledger)
        windows = list(ledger["windows"])
        windows.append({
            **window, "bars": int(len(test)), "status": "reserved", "reserved_at": utc_now()
        })
        updated.update({
            "policy_version": OOS_POLICY_VERSION, "windows": windows,
            "last_test_start": start.isoformat(), "last_test_end": end.isoformat(),
            "last_result": "reserved",
        })
        return updated

    try:
        _update_oos_ledger(reserve)
    except (ValueError, RuntimeError, OSError) as exc:
        return False, f"Out-of-sample reservation failed closed: {exc}", window
    return True, None, window


def record_oos_test_result(passed: bool, window_id: str | None = None) -> None:
    """Persist only terminal status. A failed status write leaves a consumed reservation."""
    def record(ledger: dict[str, Any]) -> dict[str, Any]:
        _validate_oos_ledger(ledger)
        windows = [dict(item) for item in ledger["windows"]]
        candidates = [item for item in windows if item.get("status") == "reserved"
                      and (window_id is None or item.get("id") == window_id)]
        if not candidates:
            raise RuntimeError("No matching persisted OOS reservation is available to finalize.")
        target = candidates[-1]
        target["status"] = "passed" if passed else "failed"
        target["evaluated_at"] = utc_now()
        updated = dict(ledger)
        updated["windows"] = windows
        if target.get("id") == windows[-1].get("id"):
            updated["last_result"] = target["status"]
        return updated

    try:
        _update_oos_ledger(record)
    except (ValueError, RuntimeError, OSError) as exc:
        raise RuntimeError(f"Could not safely persist OOS evaluation status; refusing promotion: {exc}") from exc


def run_research(days: int = 90) -> dict[str, Any]:
    REPORTS.mkdir(exist_ok=True)
    STATE.mkdir(exist_ok=True)
    # The most recent held-out period is permanently excluded from all later
    # training/validation. Only newly arrived candles after this boundary may
    # enter the next optimization cycle.
    initial_oos_ledger = _load_local_oos_ledger()
    previous_test_end = _ledger_last_test_end(initial_oos_ledger)
    started = utc_now()
    results: list[dict[str, Any]] = []
    candidates = candidate_params()
    strategies = ("trend", "breakout", "mean_reversion")
    strategy_candidates = {
        strategy: unique_candidates_for_strategy(strategy, candidates)
        for strategy in strategies
    }
    effective_candidate_counts = {strategy: len(items) for strategy, items in strategy_candidates.items()}
    errors: list[dict[str, Any]] = []
    data_quality: dict[str, Any] = {}
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
            if previous_test_end is not None:
                raw = raw.loc[raw.index > previous_test_end].copy()
                if raw.empty:
                    errors.append({
                        "symbol": symbol,
                        "error": "No new candles exist after the previously consumed OOS boundary; historical test data is excluded.",
                    })
                    continue
                # Recompute quality on the post-test slice, including missing
                # candles immediately after the consumed boundary.
                post_start = previous_test_end + pd.Timedelta(minutes=5)
                post_end = raw.index[-1] + pd.Timedelta(minutes=5)
                post_quality = candle_quality_report(
                    raw, "5m", days=1, window_start=post_start, window_end=post_end
                )
                original_quality = raw.attrs.get("data_quality")
                if isinstance(original_quality, dict):
                    quality = dict(original_quality)
                    post_coverage = post_quality["source_coverage_pct"]
                    quality.update({
                        "source_coverage_pct": min(float(quality.get("source_coverage_pct", 0)), post_coverage),
                        "converted_coverage_pct": post_coverage,
                        "missing_expected_slots": post_quality["missing_expected_slots"],
                        "converted_missing_expected_slots": post_quality["missing_expected_slots"],
                        "missing_at_start_slots": post_quality["missing_at_start_slots"],
                        "missing_at_end_slots": post_quality["missing_at_end_slots"],
                        "invalid_ohlc_rows": post_quality["invalid_ohlc_rows"],
                        "converted_invalid_ohlc_rows": post_quality["invalid_ohlc_rows"],
                        "unique_timestamps": post_quality["unique_timestamps"],
                        "monotonic_timestamps": post_quality["monotonic_timestamps"],
                        "timezone_aware_timestamps": post_quality["timezone_aware_timestamps"],
                        "unaligned_timestamps": post_quality["unaligned_timestamps"],
                        "irregular_timestamp_gaps": post_quality["irregular_timestamp_gaps"],
                        "max_consecutive_missing_slots": post_quality["max_consecutive_missing_slots"],
                        "converted_max_consecutive_missing_slots": post_quality["max_consecutive_missing_slots"],
                        "converted_unique_timestamps": post_quality["unique_timestamps"],
                        "converted_monotonic_timestamps": post_quality["monotonic_timestamps"],
                        "converted_unaligned_timestamps": post_quality["unaligned_timestamps"],
                        "optimization_history_start": post_start.isoformat(),
                        "previous_oos_test_end_exclusive_boundary": previous_test_end.isoformat(),
                    })
                    quality["quality_gate_passed"] = bool(
                        quality["source_coverage_pct"] >= MIN_CANDLE_COVERAGE_PCT
                        and quality["converted_coverage_pct"] >= MIN_CANDLE_COVERAGE_PCT
                        and quality.get("fx_missing_pct_of_source", 100) <= MAX_FX_MISSING_PCT
                        and quality["invalid_ohlc_rows"] == 0
                        and quality["converted_invalid_ohlc_rows"] == 0
                        and quality.get("conflicting_duplicate_groups", 1) == 0
                        and quality["unique_timestamps"] and quality["monotonic_timestamps"]
                        and quality["timezone_aware_timestamps"] and quality["unaligned_timestamps"] == 0
                        and quality["irregular_timestamp_gaps"] == 0
                        and quality["max_consecutive_missing_slots"] <= MAX_CONSECUTIVE_MISSING_SLOTS
                    )
                    raw.attrs["data_quality"] = quality
            quality = raw.attrs.get("data_quality")
            if not isinstance(quality, dict):
                data_quality[symbol] = {"status": "missing_quality_report"}
                errors.append({
                    "symbol": symbol,
                    "error": "Candle source did not provide a data-quality report; refusing research on unverified coverage.",
                })
                continue
            data_quality[symbol] = quality
            if not quality.get("quality_gate_passed", False):
                errors.append({
                    "symbol": symbol,
                    "error": "Candle quality gate failed: require at least 99% source/converted coverage, <=1% missing FX, no invalid OHLCV or conflicting duplicates, unique monotonic timestamps, and no gap longer than three bars.",
                    "data_quality": quality,
                })
                continue
            for interval, rule in INTERVALS.items():
                bars = raw if interval == "5m" else resample_ohlcv(raw, rule)
                if len(bars) < 1100:
                    errors.append({"symbol": symbol, "interval": interval,
                                   "error": "Insufficient bars for independent train/validation/test evaluation"})
                    continue
                closes = pd.to_numeric(bars["close"], errors="coerce")
                if (not bars.index.is_monotonic_increasing or bars.index.has_duplicates
                        or not np.isfinite(closes).all() or (closes <= 0).any()):
                    errors.append({"symbol": symbol, "interval": interval,
                                   "error": "Resampled data failed unique, monotonic, finite-positive close validation"})
                    continue
                evaluation_frames[(symbol, interval)] = bars.copy()
                train, validation, test = chronological_split(bars)
                validation_benchmarks = {
                    "cash": cash_benchmark(validation, interval),
                    "buy_and_hold": buy_and_hold_benchmark(
                        validation, interval, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS
                    ),
                }
                validation_stress_benchmarks = {
                    "cash": cash_benchmark(validation, interval),
                    "buy_and_hold": buy_and_hold_benchmark(
                        validation, interval, STRESS_FEE_BPS, STRESS_SLIPPAGE_BPS
                    ),
                }
                for strategy in strategies:
                    for params in strategy_candidates[strategy]:
                        train_m = backtest(train, strategy, params, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS, interval)
                        validation_m = backtest(validation, strategy, params, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS, interval)
                        validation_stress = backtest(validation, strategy, params, STRESS_FEE_BPS, STRESS_SLIPPAGE_BPS, interval)
                        passed = candidate_passes_gates(
                            train_m, validation_m, validation_stress,
                            validation_benchmarks, validation_stress_benchmarks,
                        )
                        results.append({
                            "symbol": symbol, "interval": interval, "strategy": strategy,
                            "params": params,
                            "effective_params": effective_strategy_params(strategy, params),
                            "train": train_m, "validation": validation_m,
                            "validation_stress": validation_stress,
                            "validation_benchmarks": validation_benchmarks,
                            "validation_stress_benchmarks": validation_stress_benchmarks,
                            "validation_score": score(validation_m, validation_benchmarks),
                            "passed_validation": bool(passed),
                            "validation_rejection_reasons": [
                                reason for condition, reason in (
                                    (train_m["total_return_pct"] <= TRAIN_MIN_RETURN_PCT, "training return is not positive"),
                                    (train_m["max_drawdown_pct"] >= TRAIN_MAX_DRAWDOWN_PCT, "training drawdown is >= 35%"),
                                    (validation_m["total_return_pct"] <= 0, "validation return is not positive after costs"),
                                    (validation_stress["total_return_pct"] <= 0, "stressed validation return is not positive"),
                                    (validation_m["max_drawdown_pct"] >= 25, "validation drawdown is >= 25%"),
                                    (validation_m["trades"] < 2, "validation has fewer than two position changes"),
                                    (not _beats_benchmarks(validation_m, validation_benchmarks), "validation does not beat cash and buy-and-hold"),
                                    (not _beats_benchmarks(validation_stress, validation_stress_benchmarks), "stressed validation does not beat cash and stressed buy-and-hold"),
                                ) if condition
                            ],
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
    selected_test_gate: dict[str, Any] | None = None
    if selected is not None:
        # Ranking is frozen on training/validation. A test window is reserved
        # before evaluation and cannot be reused after either pass or failure.
        symbol, interval = selected["symbol"], selected["interval"]
        bars = evaluation_frames[(symbol, interval)]
        train, validation, test = chronological_split(bars)
        can_evaluate, reservation_error, test_window = reserve_oos_test_window(test)
        test_gate_failures: list[str] = []
        if not can_evaluate:
            test_gate_failures = [reservation_error or "Out-of-sample test window unavailable."]
            selected_test_gate = {
                "candidate": {"symbol": symbol, "interval": interval,
                              "strategy": selected["strategy"], "params": selected["params"]},
                "policy_version": OOS_POLICY_VERSION, "test_window": test_window,
                "evaluated": False, "passed": False, "failures": test_gate_failures,
                "selection_note": "Ranking used training/validation only; the versioned OOS policy forbids reusing this test window. No fallback was selected.",
            }
        else:
            test_metrics = backtest(test, selected["strategy"], selected["params"],
                                    DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS, interval)
            test_stress = backtest(test, selected["strategy"], selected["params"],
                                   STRESS_FEE_BPS, STRESS_SLIPPAGE_BPS, interval)
            test_benchmarks = {
                "cash": cash_benchmark(test, interval),
                "buy_and_hold": buy_and_hold_benchmark(
                    test, interval, DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS
                ),
            }
            test_stress_benchmarks = {
                "cash": cash_benchmark(test, interval),
                "buy_and_hold": buy_and_hold_benchmark(
                    test, interval, STRESS_FEE_BPS, STRESS_SLIPPAGE_BPS
                ),
            }
            test_gate_failures = deployment_gate_failures(
                test_metrics, test_stress, test_benchmarks, test_stress_benchmarks
            )
            selected_test_gate = {
                "candidate": {"symbol": symbol, "interval": interval,
                              "strategy": selected["strategy"], "params": selected["params"]},
                "policy_version": OOS_POLICY_VERSION, "test_window": test_window,
                "evaluated": True, "passed": not test_gate_failures, "failures": test_gate_failures,
                "test_metrics": test_metrics, "test_stress_metrics": test_stress,
                "test_benchmarks": test_benchmarks, "test_stress_benchmarks": test_stress_benchmarks,
                "selection_note": "Candidate was fixed by validation score before test evaluation; test failure does not trigger selection of another candidate.",
            }
            record_oos_test_result(not test_gate_failures, window_id=test_window["id"])
        selected["test_deployment_gate"] = {
            "passed": not test_gate_failures, "failures": test_gate_failures
        }
        if not test_gate_failures and can_evaluate:
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
                "params": selected["params"],
                "effective_params": effective_strategy_params(selected["strategy"], selected["params"]),
                "champion_version": f"champion-{version_hash}",
                "validation_metadata": {
                    "created_at": started, "status": "validated",
                    "selection_metric": "validation_excess_return_score",
                    "validation_score": selected["validation_score"],
                    "validation_criteria": {
                        "training_return_positive": True,
                        "training_max_drawdown_below_pct": TRAIN_MAX_DRAWDOWN_PCT,
                        "validation_return_positive": True,
                        "validation_stress_return_positive": True,
                        "validation_max_drawdown_below_pct": 25,
                        "minimum_validation_position_changes": 2,
                        "beats_cash_and_buy_and_hold": True,
                        "stressed_validation_beats_cash_and_buy_and_hold": True,
                        "final_test_used_for_selection": False,
                    },
                    "periods": {"train": _period_metadata(train),
                                "validation": _period_metadata(validation),
                                "test": _period_metadata(test)},
                    "train_metrics": selected["train"],
                    "validation_metrics": selected["validation"],
                    "validation_stress_metrics": selected["validation_stress"],
                    "validation_benchmarks": selected["validation_benchmarks"],
                    "validation_stress_benchmarks": selected["validation_stress_benchmarks"],
                    "test_metrics": test_metrics, "test_stress_metrics": test_stress,
                    "test_benchmarks": test_benchmarks,
                    "test_stress_benchmarks": test_stress_benchmarks,
                    "test_deployment_gate": {"passed": True, "failures": []},
                    "test_used_for_selection": False,
                    "oos_evaluation_policy_version": OOS_POLICY_VERSION,
                    "oos_test_window": selected_test_gate.get("test_window") if selected_test_gate else None,
                    "previous_test_end_excluded_from_optimization": (
                        previous_test_end.isoformat() if previous_test_end is not None else None
                    ),
                    "cost_convention": {
                        "fee_bps_per_side": DEFAULT_FEE_BPS,
                        "slippage_bps_per_side": DEFAULT_SLIPPAGE_BPS,
                        "stress_fee_bps_per_side": STRESS_FEE_BPS,
                        "stress_slippage_bps_per_side": STRESS_SLIPPAGE_BPS,
                        "fill_timing": "closed candle close after one-candle signal shift",
                        "fee_basis": "executed notional",
                        "slippage": "execution price adjusted per side",
                    },
                    "sharpe_annualization": {"bars_per_year": bars_per_year(interval),
                                             "interval": interval, "calendar_days_per_year": 365},
                    "warning": "The untouched test was used only as a one-way deployment veto after validation selection was frozen; it was never used to rank candidates or choose a fallback.",
                },
            }
    strategy_benchmark_summary: dict[str, Any] = {}
    for strategy in strategies:
        strategy_rows = [row for row in results if row["strategy"] == strategy]
        ranked_rows = sorted(strategy_rows, key=lambda row: row["validation_score"], reverse=True)
        best = ranked_rows[0] if ranked_rows else None
        strategy_benchmark_summary[strategy] = {
            "evaluated_candidates": len(strategy_rows),
            "validation_eligible_candidates": sum(bool(row["passed_validation"]) for row in strategy_rows),
            "best_validation_candidate": ({
                "symbol": best["symbol"], "interval": best["interval"],
                "params": best["params"], "validation": best["validation"],
                "validation_stress": best["validation_stress"],
                "cash_benchmark": best.get("validation_benchmarks", {}).get("cash"),
                "buy_and_hold_benchmark": best.get("validation_benchmarks", {}).get("buy_and_hold"),
                "stress_cash_benchmark": best.get("validation_stress_benchmarks", {}).get("cash"),
                "stress_buy_and_hold_benchmark": best.get("validation_stress_benchmarks", {}).get("buy_and_hold"),
                "passed_validation": best["passed_validation"],
                "rejection_reasons": best.get("validation_rejection_reasons", []),
            } if best else None),
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
                ("Top validation-selected candidate failed the untouched-test deployment gate: "
                 + "; ".join((selected_test_gate or {}).get("failures", []))
                 + ". No fallback was selected to preserve test isolation.")
                if selected_test_gate is not None else (
                    "No candidate passed training/validation benchmark gates" if results
                    else "No candidate evaluations completed; check market data and history coverage"
                )
            ),
        },
        "symbols": list(research_symbols), "universe": universe_metadata, "intervals": list(INTERVALS),
        "minimum_candle_coverage_pct": MIN_CANDLE_COVERAGE_PCT,
        "maximum_consecutive_missing_slots": MAX_CONSECUTIVE_MISSING_SLOTS,
        "maximum_fx_missing_pct": MAX_FX_MISSING_PCT,
        "cost_assumptions": {
            "fee_bps_per_side": DEFAULT_FEE_BPS,
            "slippage_bps_per_side": DEFAULT_SLIPPAGE_BPS,
            "stress_fee_bps_per_side": STRESS_FEE_BPS,
            "stress_slippage_bps_per_side": STRESS_SLIPPAGE_BPS,
        },
        "data_quality": data_quality,
        "candidate_count": len(candidates),
        "effective_candidate_counts": effective_candidate_counts,
        "total_unique_strategy_configs": sum(effective_candidate_counts.values()),
        "evaluations": len(results),
        "validated_count": len(eligible), "champion": champion,
        "strategy_benchmark_summary": strategy_benchmark_summary,
        "selected_candidate_test_gate": selected_test_gate,
        "selection_policy": "Require positive training return, training drawdown below 35%, positive baseline/stress validation returns, validation drawdown below 25%, at least two position changes, and validation outperformance versus cash and buy-and-hold under baseline and stress costs. Rank using validation-only excess-return score. Freeze the top candidate before evaluating the untouched test; the test can veto promotion but never rank candidates or trigger fallback.",
        "out_of_sample_policy": {
            "version": OOS_POLICY_VERSION, "chronological": True,
            "one_evaluation_per_non_overlapping_window": True,
            "rejected_windows_are_consumed": True,
            "candidate_ranking_uses": ["train", "validation"],
            "test_failure_feedback_to_optimizer": False,
            "ledger": f"refs/heads/{os.environ.get('OOS_LEDGER_BRANCH', 'local-state')}",
            "previous_test_end_excluded_from_optimization": (
                previous_test_end.isoformat() if previous_test_end is not None else None
            ),
            "test_history_excluded_from_training_validation": True,
            "new_windows_require_strictly_later_data": True,
            "reservation_persisted_before_test_evaluation": True,
        },
        "results": sorted(results, key=lambda r: r["validation_score"], reverse=True)[:30],
        "errors": errors,
        "warning": "The untouched test is used only as a one-way deployment veto after validation selection is frozen. Repeatedly inspecting the same test period and then changing rules can contaminate it; a new later holdout is required for further tuning.",
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
    policy = report.get("out_of_sample_policy")
    if (not isinstance(policy, dict) or policy.get("version") != OOS_POLICY_VERSION
            or policy.get("one_evaluation_per_non_overlapping_window") is not True
            or policy.get("rejected_windows_are_consumed") is not True
            or policy.get("test_failure_feedback_to_optimizer") is not False
            or policy.get("test_history_excluded_from_training_validation") is not True
            or policy.get("new_windows_require_strictly_later_data") is not True
            or policy.get("reservation_persisted_before_test_evaluation") is not True):
        raise RuntimeError("Research report does not declare the approved versioned OOS isolation policy")
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
    if metadata.get("oos_evaluation_policy_version") != OOS_POLICY_VERSION:
        raise RuntimeError("Champion is missing the current versioned out-of-sample policy")
    for field in ("train_metrics", "validation_metrics", "validation_stress_metrics", "test_metrics", "test_stress_metrics"):
        if not isinstance(metadata.get(field), dict):
            raise RuntimeError(f"Champion validation metadata is missing {field}")
    train_metrics = metadata["train_metrics"]
    if not isinstance(metadata.get("test_benchmarks"), dict) or not isinstance(metadata.get("test_stress_benchmarks"), dict):
        raise RuntimeError("Champion is missing held-out cash/buy-and-hold benchmark comparisons")
    test_gate = deployment_gate_failures(
        metadata["test_metrics"], metadata["test_stress_metrics"],
        metadata["test_benchmarks"], metadata["test_stress_benchmarks"],
    )
    if test_gate or not isinstance(metadata.get("test_deployment_gate"), dict) or metadata["test_deployment_gate"].get("passed") is not True:
        raise RuntimeError("Champion failed held-out deployment gate: " + "; ".join(test_gate or ["gate metadata missing"]))
    if not isinstance(metadata.get("validation_benchmarks"), dict) or not isinstance(metadata.get("validation_stress_benchmarks"), dict):
        raise RuntimeError("Champion is missing validation cash/buy-and-hold benchmark comparisons")
    if not candidate_passes_gates(
        metadata["train_metrics"], metadata["validation_metrics"],
        metadata["validation_stress_metrics"], metadata["validation_benchmarks"],
        metadata["validation_stress_benchmarks"],
    ):
        raise RuntimeError("Champion failed training robustness gate or training/validation benchmark gates")
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
