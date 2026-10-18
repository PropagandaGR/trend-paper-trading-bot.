"""Public, market-cap-ranked crypto universe intersected with Coinbase USD spot markets.

The universe describes eligible markets, not buy recommendations. Research evaluates a
small rotating batch each run so 100-coin coverage does not turn one job into thousands
of serial candle requests. Paper trading remains one-position, long-only, and simulated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import re
from typing import Any

import requests

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
COINBASE_PRODUCTS_URL = "https://api.exchange.coinbase.com/products"
TOP_MARKET_CAP_LIMIT = 100
RESEARCH_BATCH_SIZE = 5
STABLECOIN_SYMBOLS = {
    "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDS", "USDD",
    "PYUSD", "USDP", "EURC", "EURT", "UST", "USTC", "FRAX",
}
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Autonomous-Paper-Trading-Lab/1.0"})


@lru_cache(maxsize=1)
def _coinbase_usd_products() -> dict[str, str]:
    response = SESSION.get(COINBASE_PRODUCTS_URL, timeout=20)
    response.raise_for_status()
    products = response.json()
    result: dict[str, str] = {}
    for product in products:
        if not isinstance(product, dict):
            continue
        base = str(product.get("base_currency", "")).upper()
        product_id = str(product.get("id", ""))
        if (product.get("quote_currency") == "USD"
                and product.get("status") == "online"
                and not product.get("trading_disabled", False)
                and re.fullmatch(r"[A-Z0-9]{2,15}", base)
                and product_id == f"{base}-USD"):
            result[base] = product_id
    return result


def get_coinbase_product_id(symbol: str) -> str | None:
    """Resolve a canonical BASEUSDT symbol to a currently listed Coinbase BASE-USD product."""
    if symbol == "USDTUSD":
        return "USDT-USD"
    if not isinstance(symbol, str) or not symbol.endswith("USDT"):
        return None
    base = symbol[:-4]
    if not re.fullmatch(r"[A-Z0-9]{2,15}", base):
        return None
    return _coinbase_usd_products().get(base)


@lru_cache(maxsize=1)
def get_top100_coinbase_universe() -> tuple[dict[str, Any], ...]:
    """Return CoinGecko's top 100 by market cap intersected with Coinbase USD spot markets."""
    products = _coinbase_usd_products()
    response = SESSION.get(
        COINGECKO_MARKETS_URL,
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": TOP_MARKET_CAP_LIMIT,
            "page": 1,
            "sparkline": "false",
        },
        timeout=25,
    )
    response.raise_for_status()
    markets = response.json()
    eligible: list[dict[str, Any]] = []
    seen_bases: set[str] = set()
    for item in markets:
        if not isinstance(item, dict):
            continue
        base = str(item.get("symbol", "")).upper()
        rank = item.get("market_cap_rank")
        if (not isinstance(rank, int) or rank < 1 or base in STABLECOIN_SYMBOLS
                or base not in products or base in seen_bases
                or not re.fullmatch(r"[A-Z0-9]{2,15}", base)):
            continue
        seen_bases.add(base)
        eligible.append({
            "symbol": f"{base}USDT",
            "base": base,
            "product_id": products[base],
            "market_cap_rank": rank,
            "market_cap_usd": item.get("market_cap"),
            "source": "CoinGecko top-100 market-cap ranking intersected with Coinbase USD spot markets",
        })
    return tuple(sorted(eligible, key=lambda item: item["market_cap_rank"]))


def get_research_batch(batch_size: int = RESEARCH_BATCH_SIZE,
                       at: datetime | None = None) -> tuple[list[str], dict[str, Any]]:
    """Return one deterministic rotating batch from the eligible top-100 universe."""
    if not 1 <= batch_size <= TOP_MARKET_CAP_LIMIT:
        raise ValueError("batch_size must be between 1 and 100")
    universe = list(get_top100_coinbase_universe())
    if not universe:
        raise RuntimeError("No top-100 cryptocurrencies have an eligible Coinbase USD spot market")
    now = at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("batch rotation time must be timezone-aware")
    batches = max(1, (len(universe) + batch_size - 1) // batch_size)
    batch_index = now.astimezone(timezone.utc).hour % batches
    start = batch_index * batch_size
    selected = universe[start:start + batch_size]
    return [item["symbol"] for item in selected], {
        "source": "CoinGecko market-cap top 100 intersected with online Coinbase USD spot products",
        "market_cap_rank_limit": TOP_MARKET_CAP_LIMIT,
        "eligible_coinbase_markets": len(universe),
        "batch_size": batch_size,
        "batch_index": batch_index,
        "rotation_hour_utc": now.astimezone(timezone.utc).hour,
        "symbols": selected,
    }
