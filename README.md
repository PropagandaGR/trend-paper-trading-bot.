# Autonomous Crypto Paper-Trading Lab

A **new, isolated research project** for continuous live-market paper trading and autonomous strategy discovery. It is not connected to a broker and cannot place real orders.

## Safety and scope
- Simulated starting equity: **15 USDT**.
- Spot-only, long/flat: no leverage, no shorting.
- Public Coinbase Exchange market data only; no API keys or secrets are needed for candle data.
- Research explores parameterized trend, breakout, and mean-reversion strategies across a rotating batch from the CoinGecko market-cap top 100, intersected with currently online Coinbase USD spot markets. Stablecoins, duplicate/unsupported symbols, and coins below US$5 million global 24-hour volume are excluded. The volume threshold is a broad liquidity screen, not Coinbase-specific depth; the universe is not a recommendation to buy any asset.
- Coinbase USD-quoted candles are converted to approximate USDT prices only when an exact same-candle Coinbase USDT-USD close exists; missing FX candles are dropped rather than forward-filled from stale rates. This conversion remains approximate, not tick-level FX accounting, and missing-candle coverage must be audited.
- Execution-cost defaults are deliberately conservative for an unknown Coinbase spot taker tier: 60 bps fee + 10 bps slippage per side; stress is 100 bps fee + 20 bps slippage per side. Research and paper execution share these settings. Only override them via `PAPER_FEE_BPS`, `PAPER_SLIPPAGE_BPS`, `PAPER_STRESS_FEE_BPS`, and `PAPER_STRESS_SLIPPAGE_BPS` after verifying the account tier and intended order type; a champion evaluated under different costs is rejected.
- To keep research runs bounded, each run evaluates up to five eligible coins and rotates through the ranked universe by UTC hour. Coverage of all Coinbase-supported top-100 assets therefore takes multiple runs; the active champion is selected only from the current batch.
- Candidate strategies must show positive training return with training max drawdown below 35%, then pass positive validation and stressed-validation returns, validation drawdown below 25%, and minimum-trade gates before being selected. Final-test metrics remain report-only and are never used to rank or accept candidates. These gates reduce obvious regime overfit but do not prove profitability.
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

Pull requests run `python -m unittest discover -s tests -v` through the GitHub Actions Tests workflow. The research CLI and autonomous research workflow fail unless the current invocation writes a fresh report, performs substantive evaluations, and produces a versioned champion with complete validation metadata and positive training return / training drawdown below 35%. The paper trader independently rejects champions that fail those training gates. Final-test results are not acceptance gates to preserve final-test isolation. A no-candidate run is recorded in the report but fails the workflow gate. The workflow independently verifies report freshness, champion metadata, final-test isolation, and the matching immutable snapshot before committing artifacts. Paper trading rejects missing, stale (over 24 hours), malformed, unsupported, or incompletely validated champions; when a prior candle cursor shows missed or ambiguous decision candles, it does not backfill imaginary fills, closes any open paper position at a fresh held-asset mark, advances the cursor, and waits for the next live decision candle; checks finite metrics, parameter ranges, chronological non-overlapping periods and cost metadata; rejects stale/missing price marks rather than reusing cached prices; caps new simulated entries at 50% of equity (position appreciation above that entry fraction is reported risk exposure, not by itself a circuit-breaker event); and latches a circuit breaker at a 5% UTC-day loss or 10% peak-to-trough drawdown. A risk halt requires manual review/reset in `state/paper_state.json`. Price marks must be no more than 15 minutes old. These controls are simulation-only and do not establish live-trading readiness. The Coinbase Public Data Smoke Test also checks recent public candles and USDT price conversion; it requires public network access and no exchange credentials.

## Research validation and backtest contract

- Research uses chronological **60% training / 20% validation / 20% final test** periods per symbol and timeframe.
- Candidate eligibility and ranking use validation metrics only, including a validation-only stressed-cost run. The final test is evaluated only after a champion is selected and is report-only; it is never used to rank candidates or decide validation eligibility.
- Backtests execute at the closed candle close after the strategy's one-candle signal shift. Research and paper execution share the same entry-budget function, per-side fee/slippage accounting primitives, and latched UTC-day loss / peak-drawdown halt semantics. The simulator approximates the workflow's polling cadence as one decision per candle; verify parity through regression tests before treating results as production evidence.
- Sharpe annualization uses the timeframe's bars per calendar year (365 days), recorded with the champion metadata.
- `state/champion.json` includes a unique `champion_version` and `validation_metadata`: data-period boundaries, train/validation/test metrics, validation criteria, cost convention, test-isolation flag and Sharpe annualization details.
- If no candidate meets validation gates, the active champion pointer is cleared rather than retaining a stale strategy. Immutable champion snapshots remain in version history.
- A final-test result is not a guarantee of future performance. Repeatedly inspecting the same test period and then changing strategy rules can contaminate it; use a new, later holdout for any subsequent research cycle.
- Regression tests cover split ordering, final-test isolation, cost convention and timeframe-specific Sharpe. A test is considered passed only when a recorded CI run or reproducible local output confirms it.
