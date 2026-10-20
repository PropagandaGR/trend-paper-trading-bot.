import unittest
from unittest.mock import patch

import pandas as pd

from research_engine import (_aligned_candle_window, _convert_usd_candles_to_usdt,
                             candle_quality_report, fetch_klines)


def candle_frame(index, close):
    return pd.DataFrame(
        {
            "open": [close - 1.0],
            "high": [close + 2.0],
            "low": [close - 2.0],
            "close": [close],
            "volume": [3.0],
            "close_time": [index[0] + pd.Timedelta(minutes=5)],
        },
        index=index,
    )


class CoinbaseDataSourceTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.DatetimeIndex(
            [pd.Timestamp("2026-01-01T00:00:00Z")], name="open_time"
        )

    def test_usd_prices_are_converted_to_usdt(self):
        asset = candle_frame(self.index, 100.0)
        fx = candle_frame(self.index, 0.99)  # USD per USDT
        converted = _convert_usd_candles_to_usdt(asset, fx)

        self.assertAlmostEqual(converted.iloc[0]["close"], 100.0 / 0.99)
        self.assertAlmostEqual(converted.iloc[0]["open"], 99.0 / 0.99)
        self.assertEqual(converted.iloc[0]["volume"], 3.0)
        self.assertEqual(converted.attrs["quote_currency"], "USDT")

    def test_fetch_klines_uses_coinbase_asset_and_fx_candles(self):
        asset = candle_frame(self.index, 100.0)
        fx = candle_frame(self.index, 1.0)
        with patch("research_engine._fetch_coinbase", side_effect=[asset, fx]) as fetch:
            result = fetch_klines("BTCUSDT", "5m", 1)

        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(fetch.call_args_list[0].args[0], "BTCUSDT")
        self.assertEqual(fetch.call_args_list[1].args[0], "USDTUSD")
        self.assertEqual(result.attrs["source"], "coinbase")
        self.assertEqual(result.attrs["quote_currency"], "USDT")
        self.assertAlmostEqual(result.iloc[0]["close"], 100.0)
        self.assertFalse(result.attrs["data_quality"]["quality_gate_passed"])
        self.assertEqual(result.attrs["data_quality"]["source_candle_count"], 1)

    def test_quality_report_flags_invalid_ohlc(self):
        asset = candle_frame(self.index, 100.0)
        asset.loc[self.index[0], "high"] = 98.0
        start = pd.Timestamp("2026-01-01T00:00:00Z")
        end = start + pd.Timedelta(minutes=15)
        report = candle_quality_report(asset, "5m", days=1, window_start=start, window_end=end)
        self.assertEqual(report["invalid_ohlc_rows"], 1)
        self.assertAlmostEqual(report["source_coverage_pct"], 100.0 / 3.0, places=3)
        self.assertEqual(report["missing_at_start_slots"], 0)
        self.assertEqual(report["missing_at_end_slots"], 2)

    def test_aligned_utc_window_and_edge_gap_coverage_are_exact(self):
        now = pd.Timestamp("2026-01-01T00:17:00Z")
        start, end = _aligned_candle_window("5m", 1, now)
        self.assertEqual(end, pd.Timestamp("2026-01-01T00:15:00Z"))
        self.assertEqual(start, end - pd.Timedelta(days=1))
        window_start = end - pd.Timedelta(minutes=30)
        index = pd.date_range(window_start + pd.Timedelta(minutes=10), periods=4, freq="5min")
        frame = pd.DataFrame({
            "open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4,
            "close": [100.0] * 4, "volume": [1.0] * 4,
        }, index=index)
        report = candle_quality_report(frame, "5m", days=1, window_start=window_start, window_end=end)
        self.assertEqual(report["expected_window_slots"], 6)
        self.assertEqual(report["observed_expected_slots"], 4)
        self.assertEqual(report["missing_expected_slots"], 2)
        self.assertEqual(report["missing_at_start_slots"], 2)
        self.assertEqual(report["missing_at_end_slots"], 0)
        self.assertAlmostEqual(report["source_coverage_pct"], 400.0 / 6.0, places=3)

    def test_conflicting_duplicate_groups_are_reported(self):
        stamp = pd.Timestamp("2026-01-01T00:00:00Z")
        frame = pd.DataFrame({
            "open": [99.0, 100.0], "high": [102.0, 103.0], "low": [98.0, 99.0],
            "close": [100.0, 101.0], "volume": [3.0, 4.0],
        }, index=pd.DatetimeIndex([stamp, stamp]))
        frame.attrs["conflicting_duplicate_groups"] = 1
        report = candle_quality_report(
            frame, "5m", days=1, window_start=stamp, window_end=stamp + pd.Timedelta(minutes=10)
        )
        self.assertEqual(report["conflicting_duplicate_groups"], 1)
        self.assertFalse(report["unique_timestamps"])

    def test_usdtusd_symbol_does_not_recursively_convert(self):
        fx = candle_frame(self.index, 0.995)
        with patch("research_engine._fetch_coinbase", return_value=fx) as fetch:
            result = fetch_klines("USDTUSD", "5m", 1)

        self.assertEqual(fetch.call_args.args[:3], ("USDTUSD", "5m", 1))
        self.assertAlmostEqual(result.iloc[0]["close"], 0.995)

    def test_missing_same_candle_fx_is_dropped_instead_of_forward_filled(self):
        index = pd.DatetimeIndex([
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-01T00:05:00Z"),
        ], name="open_time")
        asset = pd.DataFrame({
            "open": [99.0, 100.0], "high": [102.0, 103.0],
            "low": [98.0, 99.0], "close": [100.0, 101.0],
            "volume": [3.0, 4.0],
            "close_time": index + pd.Timedelta(minutes=5),
        }, index=index)
        fx = candle_frame(index[:1], 1.0)

        converted = _convert_usd_candles_to_usdt(asset, fx)

        self.assertEqual(list(converted.index), [index[0]])
        self.assertEqual(converted.attrs["fx_missing_candles"], 1)
        self.assertIn("same-candle USDT-USD close only", converted.attrs["fx_conversion"])


if __name__ == "__main__":
    unittest.main()
