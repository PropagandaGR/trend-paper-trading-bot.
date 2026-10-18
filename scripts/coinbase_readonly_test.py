#!/usr/bin/env python3
"""Read-only Coinbase Advanced Trade connectivity check. Never places orders."""

import os
import sys

from coinbase.rest import RESTClient


def main() -> int:
    api_key = os.environ.get("COINBASE_API_KEY", "").strip()
    api_secret = os.environ.get("COINBASE_API_SECRET", "").strip()

    if not api_key or not api_secret:
        print("FAIL: COINBASE_API_KEY or COINBASE_API_SECRET GitHub secret is missing.")
        return 2

    try:
        client = RESTClient(api_key=api_key, api_secret=api_secret, timeout=15)
        response = client.get_accounts()
        accounts = getattr(response, "accounts", None)
        if accounts is None and isinstance(response, dict):
            accounts = response.get("accounts", [])

        if accounts is None:
            print("FAIL: Coinbase responded, but the SDK response format was unexpected.")
            return 1

        print("SUCCESS: Coinbase Advanced Trade authenticated read-only request completed.")
        print(f"Accounts visible to this API key: {len(accounts)}")
        print("No orders, transfers, or other account changes were requested.")
        return 0
    except Exception as exc:
        message = str(exc)
        for secret in (api_key, api_secret):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        print(f"FAIL: Coinbase read-only connection test failed ({type(exc).__name__}).")
        print(message[:500])
        return 1


if __name__ == "__main__":
    sys.exit(main())
