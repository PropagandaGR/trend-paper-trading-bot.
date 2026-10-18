from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
STATE = ROOT / "state"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
INTERVALS = {"5m": "5min", "10m": "10min", "15m": "15min", "30m": "30min"}
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Autonomous-Paper-Trading-Lab/1.0"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frame_from_binance_rows(rows: list[list[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
    ])
    frame = frame.drop_duplicates("open_time").sort_values("open_time")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.set_index("open_time")
    return frame.loc[frame["close_time"] < pd.Timestamp.now(tz="UTC")]


def _fetch_binance(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    cursor = end_ms - int(days * 86400 * 1000)
    rows: list[list[Any]] = []
    # Try the global and US public endpoints. Some hosted runners are region-blocked.
    last_error: Exception | None = None
    for base in ("https://api.binance.com", "https://api.binance.us"):
        try:
            rows = []
            page_cursor = cursor
            while page_cursor < end_ms:
                response = SESSION.get(
                    f"{base}/api/v3/klines",
                    params={"symbol": symbol, "interval": interval, "startTime": page_cursor,
                            "endTime": end_ms, "limit": 1000},
                    timeout=20,
                )
                response.raise_for_status()
                batch = response.json()
                if not batch:
                    break
                rows.extend(batch)
                next_cursor = int(batch[-1][0]) + 1
                if next_cursor <= page_cursor:
                    break
                page_cursor = next_cursor
                if len(batch) < 1000:
                    break
                time.sleep(0.08)
            if rows:
                return _frame_from_binance_rows(rows)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Binance public endpoints unavailable for {symbol} {interval}: {last_error}")


def _fetch_coinbase(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fallback to Coinbase Exchange public USD candles when Binance is region-blocked."""
    product = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD"}.get(symbol)
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
        time.sleep(0.06)
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


def fetch_klines(symbol: str, interval: str = "5m", days: int = 90) -> pd.DataFrame:
    """Fetch public closed candles; fallback when Binance is geo-blocked on CI."""
    errors = []
    for provider in (_fetch_binance, _fetch_coinbase):
        try:
            frame = provider(symbol, interval, days)
            if not frame.empty:
                frame.attrs["source"] = "coinbase" if provider is _fetch_coinbase else "binance"
                return frame.sort_index()
            errors.append(f"{provider.__name__}: empty data")
        except Exception as exc:
            errors.append(f"{provider.__name__}: {exc}")
    raise RuntimeError(f"All public data providers failed for {symbol} {interval}: " + " | ".join(errors))


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    return frame.resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "close_time": "last"
    }).dropna(subset=["open", "high", "low", "close"])


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


def performance(returns: pd.Series) -> dict[str, float]:
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "sharpe": 0.0, "trades": 0}
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    std = float(returns.std(ddof=0))
    # Five-minute bars annualized; Sharpe is only a rough diagnostic.
    sharpe = float(returns.mean() / std * math.sqrt(365 * 24 * 12)) if std > 0 else 0.0
    changes = returns.attrs.get("trades", 0)
    return {"total_return_pct": round((float(equity.iloc[-1]) - 1) * 100, 4),
            "max_drawdown_pct": round(abs(float(drawdown.min())) * 100, 4),
            "sharpe": round(sharpe, 3), "trades": int(changes)}


def backtest(frame: pd.DataFrame, strategy: str, params: dict[str, Any],
             fee_bps: float = DEFAULT_FEE_BPS,
             slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> dict[str, float]:
    if len(frame) < max(int(params["slow"]) * 3, int(params["breakout"]) + 50):
        return performance(pd.Series(dtype=float))
    pos = desired_signal(frame, strategy, params)
    # Signal uses prior closed candle; execute at next candle open, then mark
    # to the next close. Apply cost to each entry/exit change.
    market_return = frame.close.pct_change().fillna(0.0)
    changes = pos.diff().abs().fillna(pos.abs())
    cost = (fee_bps + slippage_bps) / 10000.0
    net = pos * market_return - changes * cost
    net.attrs["trades"] = int((changes > 0).sum())
    return performance(net)


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


def score(metrics: dict[str, float]) -> float:
    # Prefer return while explicitly penalizing drawdown and excessive churn.
    return metrics["total_return_pct"] - 0.75 * metrics["max_drawdown_pct"] - 0.01 * metrics["trades"]


def run_research(days: int = 90) -> dict[str, Any]:
    REPORTS.mkdir(exist_ok=True)
    STATE.mkdir(exist_ok=True)
    started = utc_now()
    results: list[dict[str, Any]] = []
    candidates = candidate_params()
    for symbol in SYMBOLS:
        try:
            raw = fetch_klines(symbol, "5m", days)
            for interval, rule in INTERVALS.items():
                bars = raw if interval == "5m" else resample_ohlcv(raw, rule)
                if len(bars) < 500:
                    continue
                split = int(len(bars) * 0.70)
                train, test = bars.iloc[:split], bars.iloc[split:]
                for strategy in ("trend", "breakout", "mean_reversion"):
                    for params in candidates:
                        train_m = backtest(train, strategy, params, 10, 5)
                        test_m = backtest(test, strategy, params, 10, 5)
                        stress_m = backtest(test, strategy, params, 20, 10)
                        # Out-of-sample candidate must survive costs, not just fit train.
                        passed = (test_m["total_return_pct"] > 0 and
                                  stress_m["total_return_pct"] > 0 and
                                  test_m["max_drawdown_pct"] < 25 and
                                  test_m["trades"] >= 2)
                        results.append({
                            "symbol": symbol, "interval": interval, "strategy": strategy,
                            "params": params, "train": train_m, "test": test_m,
                            "stress": stress_m, "score": score(test_m),
                            "passed_validation": bool(passed)
                        })
        except Exception as exc:
            results.append({"symbol": symbol, "error": str(exc)})
    valid = [r for r in results if r.get("passed_validation")]
    valid.sort(key=lambda r: r["score"], reverse=True)
    champion = valid[0] if valid else None
    summary = {
        "started_at": started, "completed_at": utc_now(), "days": days,
        "symbols": list(SYMBOLS), "intervals": list(INTERVALS),
        "candidate_count": len(candidates), "evaluations": len(results),
        "validated_count": len(valid), "champion": champion,
        "selection_policy": "positive unseen-period return at 15bps round-trip baseline and 30bps stress; max drawdown under 25%; minimum two position changes",
        "results": sorted([r for r in results if "score" in r],
                          key=lambda r: r["score"], reverse=True)[:30],
        "errors": [r for r in results if "error" in r],
        "warning": "Backtest results are simulations and may overfit. No candidate is promoted unless it passes held-out and cost-stress checks."
    }
    (REPORTS / "research_latest.json").write_text(json.dumps(summary, indent=2))
    with (REPORTS / "experiments.jsonl").open("a") as handle:
        handle.write(json.dumps({"at": utc_now(), "days": days, "evaluations": len(results),
                                 "validated": len(valid), "champion": champion}, default=str) + "\n")
    if champion:
        (STATE / "champion.json").write_text(json.dumps(champion, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    result = run_research(max(30, min(args.days, 365)))
    champ = result["champion"]
    print(json.dumps({"completed_at": result["completed_at"],
                      "evaluations": result["evaluations"],
                      "validated_count": result["validated_count"],
                      "champion": champ}, indent=2))


if __name__ == "__main__":
    main()
