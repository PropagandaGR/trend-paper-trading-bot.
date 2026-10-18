import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import universe


class Top100UniverseTests(unittest.TestCase):
    def setUp(self):
        universe._coinbase_usd_products.cache_clear()
        universe.get_top100_coinbase_universe.cache_clear()

    def tearDown(self):
        universe._coinbase_usd_products.cache_clear()
        universe.get_top100_coinbase_universe.cache_clear()

    def _responses(self):
        products = [
            {"id": "BTC-USD", "base_currency": "BTC", "quote_currency": "USD", "status": "online"},
            {"id": "ETH-USD", "base_currency": "ETH", "quote_currency": "USD", "status": "online"},
            {"id": "SOL-USD", "base_currency": "SOL", "quote_currency": "USD", "status": "online"},
            {"id": "DOGE-USD", "base_currency": "DOGE", "quote_currency": "USD", "status": "online"},
            {"id": "USDT-USD", "base_currency": "USDT", "quote_currency": "USD", "status": "online"},
            {"id": "FAKE-EUR", "base_currency": "FAKE", "quote_currency": "EUR", "status": "online"},
            {"id": "OFF-USD", "base_currency": "OFF", "quote_currency": "USD", "status": "offline"},
            {"id": "LOW-USD", "base_currency": "LOW", "quote_currency": "USD", "status": "online"},
        ]
        markets = [
            {"id": "bitcoin", "symbol": "btc", "market_cap_rank": 1, "market_cap": 1000, "total_volume": 100_000_000},
            {"id": "ethereum", "symbol": "eth", "market_cap_rank": 2, "market_cap": 900, "total_volume": 50_000_000},
            {"id": "tether", "symbol": "usdt", "market_cap_rank": 3, "market_cap": 800, "total_volume": 200_000_000},
            {"id": "solana", "symbol": "sol", "market_cap_rank": 4, "market_cap": 700, "total_volume": 20_000_000},
            {"id": "dogecoin", "symbol": "doge", "market_cap_rank": 5, "market_cap": 600, "total_volume": 10_000_000},
            {"id": "fake", "symbol": "fake", "market_cap_rank": 6, "market_cap": 500, "total_volume": 20_000_000},
            {"id": "offline", "symbol": "off", "market_cap_rank": 7, "market_cap": 400, "total_volume": 20_000_000},
            {"id": "low-volume", "symbol": "low", "market_cap_rank": 8, "market_cap": 300, "total_volume": 1_000_000},
        ]
        first, second = Mock(), Mock()
        first.json.return_value = products
        first.raise_for_status.return_value = None
        second.json.return_value = markets
        second.raise_for_status.return_value = None
        return first, second

    def test_universe_intersects_top100_with_online_coinbase_usd_products(self):
        product_response, market_response = self._responses()
        with patch.object(universe.SESSION, "get", side_effect=[product_response, market_response]):
            result = universe.get_top100_coinbase_universe()
        self.assertEqual([x["symbol"] for x in result], ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"])
        self.assertEqual([x["market_cap_rank"] for x in result], [1, 2, 4, 5])
        self.assertNotIn("USDTUSDT", [x["symbol"] for x in result])
        self.assertNotIn("FAKEUSDT", [x["symbol"] for x in result])
        self.assertNotIn("OFFUSDT", [x["symbol"] for x in result])
        self.assertNotIn("LOWUSDT", [x["symbol"] for x in result])
        self.assertTrue(all(x["global_24h_volume_usd"] >= universe.MIN_GLOBAL_24H_VOLUME_USD for x in result))

    def test_research_batch_rotates_deterministically(self):
        product_response, market_response = self._responses()
        with patch.object(universe.SESSION, "get", side_effect=[product_response, market_response]):
            first, meta = universe.get_research_batch(
                batch_size=2, at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
            )
        self.assertEqual(first, ["BTCUSDT", "ETHUSDT"])
        self.assertEqual(meta["eligible_coinbase_markets"], 4)
        self.assertEqual(meta["market_cap_rank_limit"], 100)
        self.assertEqual(meta["batch_index"], 0)
        second, second_meta = universe.get_research_batch(
            batch_size=2, at=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        )
        self.assertEqual(second, ["SOLUSDT", "DOGEUSDT"])
        self.assertEqual(second_meta["batch_index"], 1)

    def test_invalid_batch_size_is_rejected(self):
        with self.assertRaises(ValueError):
            universe.get_research_batch(batch_size=0)


if __name__ == "__main__":
    unittest.main()
