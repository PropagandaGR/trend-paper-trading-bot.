# Autonomous Crypto Paper-Trading Lab

A **new, isolated research project** for continuous live-market paper trading and autonomous strategy discovery. It is not connected to a broker and cannot place real orders.

## Safety and scope
- Simulated starting equity: **15 USDT**.
- Paper margin engine defaults to 1x; leverage above 1x is simulation-only and capped at 2x. Shorting is paper-only. Set repository Actions variables `PAPER_LEVERAGE` to `2.0` to test 2x, and `PAPER_ALLOW_SHORTS` to `false` to disable shorts.
- Liquidation is approximated at completed-bar marks and is not an exchange liquidation model.
- Paper execution blocks stale/unvalidated champions and stale/non-positive price marks; champion validation must be newer than 24 hours and include positive held-out and fee-stress evidence.
- Portfolio safeguards: 5% UTC-day loss limit, 15% peak-to-equity drawdown limit, and gross exposure capped at 2x marked equity. A loss/exposure/liquidation halt blocks new entries for the UTC day.
- On coin switches, an open position is closed using a fresh mark for its own symbol; unavailable held-asset prices freeze execution rather than substituting another asset's price.
- Paper simulation supports long and short positions with configurable isolated leverage (1x–2x maximum). No live derivatives or real orders.
- Public Binance market data only; no API keys or secrets.
- Research explores parameterized trend, breakout, and mean-reversion strategies on BTCUSDT, ETHUSDT, and SOLUSDT.
- Candidate strategies must pass chronological train/validation/test splits, realistic fees/slippage stress, and benchmark comparisons before being selected for the paper portfolio.
- The paper trader uses the currently validated champion only. Research experiments never directly modify portfolio cash or positions.
- Margin accounting helpers and unit tests live in `paper_margin.py` and `tests/test_paper_margin.py`; these tests do not by themselves prove the scheduled workflow or full strategy integration.
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
python paper_trader.py
```

## Reports and state
- `reports/paper_status.json`: latest paper portfolio status
- `reports/research_latest.json`: latest search and validation summary
- `reports/experiments.jsonl`: append-only research experiment log
- `state/paper_state.json`: simulated portfolio and candle cursors
- `state/champion.json`: latest candidate that passed validation

All generated state is simulated. Do not add exchange credentials. The system deliberately has no real-order code.
