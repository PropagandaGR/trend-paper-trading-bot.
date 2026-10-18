# Autonomous Crypto Paper-Trading Lab

A **new, isolated research project** for continuous live-market paper trading and autonomous strategy discovery. It is not connected to a broker and cannot place real orders.

## Safety and scope
- Simulated starting equity: **15 USDT**.
- Spot-only, long/flat: no leverage, no shorting.
- Public Coinbase Exchange market data only; no API keys or secrets are needed for candle data.
- Research explores parameterized trend, breakout, and mean-reversion strategies for BTC, ETH, and SOL using Coinbase USD-quoted candles converted to approximate USDT prices with same-candle Coinbase USDT-USD closes. This conversion is approximate, not tick-level FX accounting.
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
python research_engine.py --days 90 --require-fresh-report
python paper_trader.py --once
```

## Reports and state
- `reports/paper_status.json`: latest paper portfolio status
- `reports/research_latest.json`: latest search and validation summary
- `reports/experiments.jsonl`: append-only research experiment log
- `state/paper_state.json`: simulated portfolio and candle cursors
- `state/champion.json`: latest fresh candidate that passed validation, including `champion_version` and `validation_metadata`

All generated state is simulated. Do not add exchange credentials. The system deliberately has no real-order code. Coinbase public candles are the only market-data source; if Coinbase data or the USDT-USD conversion candles are unavailable, the run should fail rather than silently switch providers.

## Validation

Pull requests run `python -m unittest discover -s tests -v` through the GitHub Actions Tests workflow. The research CLI and autonomous research workflow fail unless the current invocation writes a fresh report, performs substantive evaluations, and produces a versioned champion with complete validation metadata. A no-candidate run is recorded in the report but fails the workflow gate. The workflow independently verifies report freshness, champion metadata, final-test isolation, and the matching immutable snapshot before committing artifacts. Paper trading rejects missing, stale (over 24 hours), or incompletely validated champions. The Coinbase Public Data Smoke Test also checks recent public candles and USDT price conversion; it requires public network access and no exchange credentials.
## Research validation and backtest contract

- Research uses chronological **60% training / 20% validation / 20% final test** periods per symbol and timeframe.
- Candidate eligibility and ranking use validation metrics only, including a validation-only stressed-cost run. The final test is evaluated only after a champion is selected and is report-only; it is never used to rank candidates or decide validation eligibility.
- Backtests execute at the closed candle close after the strategy's one-candle signal shift, matching the paper trader's timing. Both use per-side slippage-adjusted execution prices and fees on executed notional.
- Sharpe annualization uses the timeframe's bars per calendar year (365 days), recorded with the champion metadata.
- `state/champion.json` includes a unique `champion_version` and `validation_metadata`: data-period boundaries, train/validation/test metrics, validation criteria, cost convention, test-isolation flag and Sharpe annualization details.
- If no candidate meets validation gates, the active champion pointer is cleared rather than retaining a stale strategy. Immutable champion snapshots remain in version history.\n- A final-test result is not a guarantee of future performance. Repeatedly inspecting the same test period and then changing strategy rules can contaminate it; use a new, later holdout for any subsequent research cycle.
- Regression tests cover split ordering, final-test isolation, cost convention and timeframe-specific Sharpe. A test is considered passed only when a recorded CI run or reproducible local output confirms it.
