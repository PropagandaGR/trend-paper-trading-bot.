# Crypto Trading AI — Core Master Specification

**Repository:** `PropagandaGR/trend-paper-trading-bot.`  
**Canonical branch:** `main`  
**Current integration PR:** [#16 — Consolidate paper-trading safety and market-data fixes](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/16)  
**Last audited:** 2026-10-19 UTC

## Mission and non-negotiable constraints

Build an auditable autonomous crypto research and paper-trading lab. The objective is to establish trustworthy data, accounting, evaluation, risk controls, and operational reliability before considering any real execution.

- Starting simulated cash/equity: **15 USDT**.
- Coinbase public market data; candle retrieval must not require API credentials.
- Spot-only, long/flat. No leverage, no short positions.
- No real orders or live execution until security controls and risk controls have been independently tested and explicitly approved.
- Never place secrets in source, artifacts, commits, logs, or paper-trading state.
- Never claim a test passed without the exact CI run or reproducible test evidence.
- Never reset a risk halt simply to make trading resume. Investigate the trigger and approve a documented reset only after the underlying issue is fixed.
- Final test periods must remain untouched by candidate selection. A final-test result is not a profitability guarantee.

## Architecture and workstreams

### Core / integration
Owns the specification, architecture, task priority, commit/PR references, verified evidence, known bugs, risk register, rollback plan, and release decision. Only the canonical repository above is in scope; the older `trend-paper-trading-bot-_1` repository is historical and must not receive changes.

### Research AI
Current engine is **rule-based parameter search**, not a neural network or reinforcement-learning system. It evaluates trend, breakout, and mean-reversion configurations over Coinbase candles, with chronological train/validation/test periods and stressed-cost checks. It currently rotates a small batch of up to five symbols from the CoinGecko market-cap top 100 intersected with online Coinbase USD spot products. X/social sentiment, investor-account analysis, and learned ML models are not implemented; do not add them before data integrity and evaluation methodology are trustworthy.

### Paper Trading
One simulated long position at a time, persistent simulated cash/quantity/entry basis, Coinbase public candles converted to approximate USDT, symbol-specific marks, per-side fees/slippage, stale-price checks, and a latched 5% UTC-day loss / 10% peak-to-trough drawdown halt. A 50% equity fraction caps **new entries**; ordinary appreciation above that fraction is not itself a circuit-breaker event.

### Testing & Audit
Every safety claim needs a regression test and evidence from the current code head. The audit checklist at [docs/PRIORITIZED_AUDIT_CHECKLIST.md](PRIORITIZED_AUDIT_CHECKLIST.md) is an evidence log: unchecked means not passed. A successful unit-test run does not establish profitability or live readiness.

### Live Execution
**Disabled and out of scope for this release.** No real-order code is approved. Future work, if explicitly approved, must be isolated behind a separate adapter and independently test mocked/read-only behavior, order-state tracking, idempotency, balance/position reconciliation, permission boundaries, secret handling, and emergency stops before any live-order authorization.

## Current verified state

### Integration branch / PR #16
- Head commit: `5648d121bc1acb665acc0625ee9dd71c0186e99f`.
- [Tests run #39211762123](https://github.com/PropagandaGR/trend-paper-trading-bot./actions/runs/39211762123): **69 tests passed** (`Ran 69 tests`, `OK`).
- [Coinbase public-data smoke test #39211762206](https://github.com/PropagandaGR/trend-paper-trading-bot./actions/runs/39211762206): success.
- These results apply to that exact PR head. The PR is still a **draft**, unmerged, and requires review. No live readiness is implied.

### Latest saved research artifact observed during audit
- Research completed at `2026-10-19T06:31:16Z`, 90-day window, 2,880 evaluations, 256 candidates passed the current research gates.
- Champion: `SKYUSDT`, 15-minute trend, version `champion-25b9692ed031`.
- Training: +2.8423%, 18.8739% max drawdown.
- Validation: +25.965%, 7.371% max drawdown; stressed validation +22.0592%.
- Untouched test: **−20.663%**, max drawdown 27.2751%; stressed test **−23.9261%**.
- Test metrics are report-only and did not select the candidate. These returns are backtest report figures, not live results and not a candle-by-candle replay of the $15 paper ledger.

### Paper portfolio / risk halt
The last status artifact inspected showed approximately 15.0840 USDT equity, +0.0840 USDT realized P&L, one completed simulated trade, no open position, and a latched `position_exposure_limit` halt. PR #16 removes that trigger path from the code; it deliberately does **not** edit persisted portfolio state or clear the existing halt. Before resuming, review the saved state and its history, verify current mainline state, and perform a documented/manual reset only after approval.

## Recent implementation references

- [PR #12 — Block paper deployment when held-out tests fail](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/12): draft proposal; do not merge as-is because repeatedly using the same final test as a deployment selector risks holdout contamination.
- [PR #13 — Reject fragile strategy champions](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/13): training robustness gate and entry buffer.
- [PR #15 — Fail closed on missed decision candles](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/15): avoids retroactive fills after missed candles.
- [PR #16 — Consolidation branch](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/16): carries forward PR #13/#15 changes and fixes the exposure-halt semantics, strict same-candle FX conversion, and shared entry sizing. Keep draft until the remaining review gates are met.

## Known gaps and risk register

| Priority | Gap / risk | Required closure evidence |
|---|---|---|
| P0 | Existing halt state cannot be safely cleared from code change alone | Inspect current persisted state, explain historical trigger, verify corrected code on main, approve explicit reset procedure |
| P0 | Historical backtest still lacks paper trader's latched daily-loss and peak-drawdown circuit breakers | Shared deterministic simulator; regression tests covering UTC day reset, loss breach, drawdown breach, liquidation, and no trades after halt |
| P0 | Candle coverage and FX alignment are not yet reported as a complete integrity contract | Expected-vs-observed slots, missing/duplicate/out-of-order candles, OHLC sanity, explicit gaps policy, FX coverage and maximum-age checks |
| P0 | No append-only paper execution/risk event journal | Immutable per-run events with candle time, symbol, champion version, action, fill estimate, fee, slippage, quantity, cash/equity, P&L, risk state, workflow run ID |
| P1 | Cost assumptions are likely too optimistic for Coinbase market/taker execution | Configurable account-tier fee model, conservative default when tier unknown, report the exact assumptions, baseline and stress tests |
| P1 | Research uses generic parameter dictionaries; irrelevant parameters create duplicate effective candidates | Per-strategy grids, effective-signal/configuration deduplication, report unique configurations and true evaluation count |
| P1 | No same-asset buy-and-hold/cash benchmark in latest report despite README claims | Add same-period benchmarks and test their calculation; otherwise correct docs |
| P1 | One final 20% test split can become contaminated after repeated inspection | Preserve current historical report as evidence; add rolling walk-forward folds and a future forward-paper qualification period, without using the same holdout repeatedly for selection |
| P1 | Champion may change symbol/strategy on each rotating research batch | Compare against incumbent, require stable multi-fold qualification and hysteresis, and forward paper qualification before switching |
| P1 | Research data window is limited to 30–365 days; no durable multi-year store | Paginated historical importer, UTC manifests, retry/rate limits, coverage/gap checks, persistent CSV/Parquet storage outside Git blobs |
| P1 | GitHub scheduled workflows are best-effort and have shown observed gaps | Heartbeat/last-success artifact, stale-run alert, workflow-enabled checks, concurrency review, and a watchdog; never represent GitHub Actions as guaranteed always-on |
| P2 | Independent review and rollback evidence remain incomplete | Independent reviewer checks data, accounting, selection, and halt semantics; record commit, test evidence, risks, and rollback |
| P2 | No ML or X/social sentiment integration | Defer until data quality, execution parity, walk-forward evaluation, and audit trail are complete |

## Next approved work sequence

1. Complete PR #16 review and CI verification; do not merge without review.
2. Implement shared backtest risk semantics so backtest and paper trading use the same sizing, costs, timing, daily-loss and drawdown behavior.
3. Add data-quality coverage/gap and FX-freshness reports; reject materially incomplete data.
4. Add append-only execution/risk events and state schema/reconciliation checks.
5. Make fees/slippage configurable and conservative; use actual account/order assumptions only when confirmed.
6. Replace generic strategy grids with strategy-specific grids, deduplication, benchmarks, rolling walk-forward and incumbent stability.
7. Build multi-year historical import/storage and re-run research with reproducible manifests.
8. Add scheduled-run heartbeat and alerting; verify actual run history instead of trusting cron configuration.
9. Run independent Testing & Audit review and record evidence. Keep all execution paper-only until explicit separate approval.

## Release gate

Do **not** enable real orders or call the bot profitable unless the required safety tests, research validation, paper operation, security review, independent audit, rollback plan, and explicit approval are all recorded. A passing test suite only proves the tested assertions passed on the tested commit.
