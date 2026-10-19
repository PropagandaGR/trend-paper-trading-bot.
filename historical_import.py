#!/usr/bin/env python3
"""Resumable, month-partitioned Coinbase public-candle importer for multi-year research.

Examples:
  python historical_import.py --symbol BTCUSDT --interval 1h --start 2023-01-01 --end 2026-01-01
  python historical_import.py --symbol ETHUSDT --interval 5m --start 2025-01-01 --end 2026-01-01

The end timestamp is exclusive. Output stays local under data/historical/ and is git-ignored.
No exchange credentials or order permissions are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from research_engine import (
    MIN_CANDLE_COVERAGE_PCT,
    _convert_usd_candles_to_usdt,
    _fetch_coinbase_range,
    candle_quality_report,
    get_coinbase_product_id,
)

GRANULARITY_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600,
}


def parse_utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def month_windows(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield contiguous half-open windows, split at UTC month boundaries."""
    cursor = start
    while cursor < end:
        if cursor.month == 12:
            next_month = pd.Timestamp(year=cursor.year + 1, month=1, day=1, tz="UTC")
        else:
            next_month = pd.Timestamp(year=cursor.year, month=cursor.month + 1, day=1, tz="UTC")
        stop = min(end, next_month)
        yield cursor, stop
        cursor = stop


def range_quality(asset: pd.DataFrame, converted: pd.DataFrame, start: pd.Timestamp,
                  end: pd.Timestamp, interval: str, min_coverage_pct: float) -> dict[str, Any]:
    seconds = GRANULARITY_SECONDS[interval]
    expected_slots = int((end - start).total_seconds() // seconds)
    if expected_slots <= 0:
        raise ValueError("Requested range is shorter than one candle interval")
    source_report = candle_quality_report(asset, interval, days=1)
    converted_report = candle_quality_report(converted, interval, days=1)
    source_coverage = min(100.0, len(asset) / expected_slots * 100.0)
    converted_coverage = min(100.0, len(converted) / expected_slots * 100.0)
    fx_missing = max(0, len(asset) - len(converted))
    return {
        "expected_slots": expected_slots,
        "source_candles": int(len(asset)),
        "converted_candles": int(len(converted)),
        "source_coverage_pct": round(source_coverage, 4),
        "converted_coverage_pct": round(converted_coverage, 4),
        "fx_missing_candles": fx_missing,
        "invalid_source_ohlc_rows": int(source_report["invalid_ohlc_rows"]),
        "invalid_converted_ohlc_rows": int(converted_report["invalid_ohlc_rows"]),
        "duplicate_candles_dropped": int(asset.attrs.get("duplicate_candles_dropped", 0)),
        "conflicting_duplicate_groups": int(asset.attrs.get("conflicting_duplicate_groups", 0)),
        "source_was_sorted": bool(asset.index.is_monotonic_increasing),
        "converted_was_sorted": bool(converted.index.is_monotonic_increasing),
        "coverage_note": "Missing slots may be no-trade intervals or unavailable candles; they are not interpolated.",
        "quality_gate_passed": bool(
            source_coverage >= min_coverage_pct
            and converted_coverage >= min_coverage_pct
            and source_report["invalid_ohlc_rows"] == 0
            and converted_report["invalid_ohlc_rows"] == 0
            and int(asset.attrs.get("conflicting_duplicate_groups", 0)) == 0
            and asset.index.is_monotonic_increasing
            and converted.index.is_monotonic_increasing
        ),
    }


def _window_paths(root: Path, symbol: str, interval: str,
                  start: pd.Timestamp) -> tuple[Path, Path]:
    folder = root / symbol / interval
    key = start.strftime("%Y-%m")
    return folder / f"{key}.csv", folder / f"{key}.manifest.json"


def import_range(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp,
                 output_dir: Path, min_coverage_pct: float = MIN_CANDLE_COVERAGE_PCT,
                 force: bool = False) -> dict[str, Any]:
    symbol = symbol.upper()
    if interval not in GRANULARITY_SECONDS:
        raise ValueError(f"Unsupported interval: {interval}")
    product = get_coinbase_product_id(symbol)
    if not product:
        raise ValueError(f"No Coinbase USD spot product mapping for {symbol}")
    step = pd.Timedelta(seconds=GRANULARITY_SECONDS[interval])
    start = start.floor(step)
    end = end.floor(step)
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("start/end must define a non-empty timezone-aware UTC range")
    if end > pd.Timestamp.now(tz="UTC").floor(step):
        end = pd.Timestamp.now(tz="UTC").floor(step)
    if end <= start:
        raise ValueError("No closed candles exist in the requested range")

    root = output_dir
    month_results: list[dict[str, Any]] = []
    for window_start, window_end in month_windows(start, end):
        csv_path, manifest_path = _window_paths(root, symbol, interval, window_start)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        expected_manifest = {
            "symbol": symbol, "product_id": product, "interval": interval,
            "requested_start": window_start.isoformat(), "requested_end_exclusive": window_end.isoformat(),
        }
        if not force and csv_path.exists() and manifest_path.exists():
            try:
                old = json.loads(manifest_path.read_text())
                existing_checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
                if (
                    all(old.get(key) == value for key, value in expected_manifest.items())
                    and old.get("status") == "complete"
                    and old.get("checksum_sha256") == existing_checksum
                ):
                    month_results.append({"month": window_start.strftime("%Y-%m"), "status": "skipped_verified",
                                          "checksum_sha256": existing_checksum})
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            asset = _fetch_coinbase_range(symbol, interval, window_start, window_end)
            if symbol == "USDTUSD":
                converted = asset.copy()
                converted.attrs.update(asset.attrs)
                fx_product = None
            else:
                fx = _fetch_coinbase_range("USDTUSD", interval, window_start, window_end)
                converted = _convert_usd_candles_to_usdt(asset, fx)
                fx_product = get_coinbase_product_id("USDTUSD")
            quality = range_quality(asset, converted, window_start, window_end, interval, min_coverage_pct)
            converted.to_csv(csv_path, index_label="open_time")
            checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            manifest = {
                **expected_manifest,
                "fx_product_id": fx_product,
                "fetched_at": fetched_at,
                "status": "complete" if quality["quality_gate_passed"] else "rejected_quality",
                "quality": quality,
                "checksum_sha256": checksum,
                "storage_format": "CSV",
                "credentials_used": False,
                "orders_placed": False,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            month_results.append({"month": window_start.strftime("%Y-%m"), "status": manifest["status"],
                                  "source_candles": len(asset), "converted_candles": len(converted),
                                  "source_coverage_pct": quality["source_coverage_pct"],
                                  "converted_coverage_pct": quality["converted_coverage_pct"],
                                  "checksum_sha256": checksum})
        except Exception as exc:
            manifest = {
                **expected_manifest, "fetched_at": fetched_at, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "credentials_used": False, "orders_placed": False,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            month_results.append({"month": window_start.strftime("%Y-%m"), "status": "failed",
                                  "error": manifest["error"]})

    failed = [item for item in month_results if item["status"] in {"failed", "rejected_quality"}]
    return {
        "symbol": symbol, "product_id": product, "interval": interval,
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "output_dir": str(root / symbol / interval),
        "month_count": len(month_results), "failed_month_count": len(failed),
        "status": "complete" if not failed else "incomplete",
        "months": month_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="USDT-quoted symbol, e.g. BTCUSDT")
    parser.add_argument("--interval", choices=sorted(GRANULARITY_SECONDS), default="1h")
    parser.add_argument("--start", required=True, help="UTC ISO timestamp/date, inclusive")
    parser.add_argument("--end", required=True, help="UTC ISO timestamp/date, exclusive")
    parser.add_argument("--output-dir", type=Path, default=Path("data/historical"))
    parser.add_argument("--min-coverage", type=float, default=MIN_CANDLE_COVERAGE_PCT)
    parser.add_argument("--force", action="store_true", help="Re-fetch even verified monthly partitions")
    args = parser.parse_args()
    start, end = parse_utc(args.start), parse_utc(args.end)
    if not 0 < args.min_coverage <= 100:
        parser.error("--min-coverage must be in (0, 100]")
    summary = import_range(args.symbol, args.interval, start, end, args.output_dir,
                           args.min_coverage, args.force)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
