# Autonomous Crypto Paper-Trading Lab

A **new, isolated research project** for continuous live-market paper trading and autonomous strategy discovery. It is not connected to a broker and cannot place real orders.

## Safety and scope
- Simulated starting equity: **15 USDT**.
- Spot-only, long/flat: no leverage, no shorting.
- Public Coinbase Exchange market data only; no API keys or secrets are needed for candle data.
- Research explores parameterized trend, breakout, and mean-reversion strategies for BTC, ETH, and SOL using Coinbase USD-quoted candles. The simulated ledger remains denominated in USDT, so USD/USDT quote differences must be considered when interpreting results.
- Candidate strategies must pass chronological train/validation/test splits, realistic fees/slippage stress, and benchmark comparisons before being selected for the paper portfolio.
- The live paper trader uses the currently validated champion only. Research experiments never directly modify portfolio cash or positions.
- Results are simulated estimates, not proof of future profitability.

## What runs
- **Paper trader**: every 5 minutes, fetches closed candles, updates a persistent $15 simulated portfolio and writes one status report per run.
- **Research engine**: hourly, downloads recent history, tests candidate configurations across 5/10/15/30-minute bars, evaluates costs and out-of-sample performance, and records experiments.
- **Weekly research**: broadens the history window for a more robust evaluation.

GitHub Actions schedules are best-effort, not a guaranteed always-on server. Scheduled workflows can be delayed or disabled by GitHub; the repository's Actions settings and usage limits apply.

## Run locally
Python 3.11+ recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python research_engine.py --days 90
python paper_trader.py --once
```

## Reports and state
- `reports/paper_status.json`: latest paper portfolio status
- `reports/research_latest.json`: latest search and validation summary
- `reports/experiments.jsonl`: append-only research experiment log
- `state/paper_state.json`: simulated portfolio and candle cursors
- `state/champion.json`: latest candidate that passed validation

All generated state is simulated. Do not add exchange credentials. The system deliberately has no real-order code.
