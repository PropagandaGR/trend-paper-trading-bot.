import unittest
from unittest.mock import patch

import pandas as pd

from research_engine import _convert_usd_candles_to_usdt, candle_quality_report, fetch_klines


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
        report = candle_quality_report(asset, "5m", days=1)
        self.assertEqual(report["invalid_ohlc_rows"], 1)
        self.assertLess(report["source_coverage_pct"], 1.0)

    def test_usdtusd_symbol_does_not_recursively_convert(self):
        fx = candle_frame(self.index, 0.995)
        with patch("research_engine._fetch_coinbase", return_value=fx) as fetch:
            result = fetch_klines("USDTUSD", "5m", 1)

        fetch.assert_called_once_with("USDTUSD", "5m", 1)
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
