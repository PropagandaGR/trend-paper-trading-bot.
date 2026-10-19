import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import historical_import


def candles(index):
    return pd.DataFrame({
        "open": [100.0] * len(index),
        "high": [101.0] * len(index),
        "low": [99.0] * len(index),
        "close": [100.0] * len(index),
        "volume": [1.0] * len(index),
        "close_time": index + pd.Timedelta(hours=1),
    }, index=index)


class HistoricalImportTests(unittest.TestCase):
    def test_month_windows_are_contiguous_and_end_exclusive(self):
        start = pd.Timestamp("2025-12-15T00:00:00Z")
        end = pd.Timestamp("2026-02-01T00:00:00Z")
        windows = list(historical_import.month_windows(start, end))
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0][0], start)
        self.assertEqual(windows[-1][1], end)
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(previous[1], current[0])
        self.assertTrue(all(left < right for left, right in windows))

    def test_range_quality_rejects_incomplete_history(self):
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        end = pd.Timestamp("2025-01-02T00:00:00Z")
        frame = candles(pd.DatetimeIndex([start]))
        quality = historical_import.range_quality(frame, frame.copy(), start, end, "1h", 95.0)
        self.assertEqual(quality["expected_slots"], 24)
        self.assertLess(quality["source_coverage_pct"], 95.0)
        self.assertFalse(quality["quality_gate_passed"])

    def test_verified_monthly_partition_is_skipped_without_network(self):
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        end = pd.Timestamp("2025-02-01T00:00:00Z")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "BTCUSDT" / "1h" / "2025-01.csv"
            manifest_path = root / "BTCUSDT" / "1h" / "2025-01.manifest.json"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("open_time,open,high,low,close,volume,close_time\n")
            manifest_path.write_text(json.dumps({
                "symbol": "BTCUSDT", "product_id": "BTC-USD", "interval": "1h",
                "requested_start": start.isoformat(), "requested_end_exclusive": end.isoformat(),
                "status": "complete", "checksum_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            }))
            with (
                patch.object(historical_import, "get_coinbase_product_id", return_value="BTC-USD"),
                patch.object(historical_import, "_fetch_coinbase_range") as fetch,
            ):
                result = historical_import.import_range("BTCUSDT", "1h", start, end, root)
            fetch.assert_not_called()
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["months"][0]["status"], "skipped_verified")

    def test_parse_utc_assumes_utc_for_naive_date(self):
        self.assertEqual(historical_import.parse_utc("2024-01-01").isoformat(),
                         "2024-01-01T00:00:00+00:00")

    def test_checksum_mismatch_forces_partition_refetch(self):
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        end = pd.Timestamp("2025-02-01T00:00:00Z")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "BTCUSDT" / "1h" / "2025-01.csv"
            manifest_path = root / "BTCUSDT" / "1h" / "2025-01.manifest.json"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("stale partition")
            manifest_path.write_text(json.dumps({
                "symbol": "BTCUSDT", "product_id": "BTC-USD", "interval": "1h",
                "requested_start": start.isoformat(), "requested_end_exclusive": end.isoformat(),
                "status": "complete", "checksum_sha256": "not-the-real-checksum",
            }))
            with (
                patch.object(historical_import, "get_coinbase_product_id", return_value="BTC-USD"),
                patch.object(historical_import, "_fetch_coinbase_range",
                             side_effect=RuntimeError("refetch requested")) as fetch,
            ):
                result = historical_import.import_range("BTCUSDT", "1h", start, end, root)
            fetch.assert_called_once()
            self.assertEqual(result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
