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
- Last code/test-verified Core head before this documentation update: `4fcc4a374f50f92cd26bbec080a3cda0ea8d4859`.
- [Tests run #39214793383](https://github.com/PropagandaGR/trend-paper-trading-bot./actions/runs/39214793383): **87 tests passed**.
- [Coinbase public-data smoke test #39214793420](https://github.com/PropagandaGR/trend-paper-trading-bot./actions/runs/39214793420): success.
- These results apply to that exact Core head. This master-spec update will trigger a new CI run; PR #16 remains a **draft**, unmerged, and requires review. No live readiness is implied.

### Latest saved research artifact observed during audit
- Research completed at `2026-10-19T06:31:16Z`, 90-day window, 2,880 evaluations, 256 candidates passed the current research gates.
- Champion: `SKYUSDT`, 15-minute trend, version `champion-25b9692ed031`.
- Training: +2.8423%, 18.8739% max drawdown.
- Validation: +25.965%, 7.371% max drawdown; stressed validation +22.0592%.
- Untouched test: **−20.663%**, max drawdown 27.2751%; stressed test **−23.9261%**.
- Test metrics are report-only and did not select the candidate. This artifact used the previous 10 bps fee / 5 bps slippage assumptions and predates the new 60/10 bps default-cost policy, 95% data-coverage gate, strategy-effective deduplication, and benchmark reporting. Treat it as historical evidence only; the paper trader will reject its cost metadata under the new policy until a fresh research run produces a compatible champion. These are backtest report figures, not live results or a replay of the $15 ledger.

### Paper portfolio / risk halt
The last status artifact inspected showed approximately 15.0840 USDT equity, +0.0840 USDT realized P&L, one completed simulated trade, no open position, and a latched `position_exposure_limit` halt. PR #16 removes that trigger path from the code; it deliberately does **not** edit persisted portfolio state or clear the existing halt. Existing corrupt/inconsistent state now fails closed instead of silently resetting. An append-only, idempotent paper event journal records subsequent runs/actions/errors; it cannot reconstruct events from before its rollout. The cost model changed, so the currently saved champion is expected to fail cost-metadata validation. Do not resume until a fresh quality-gated research run produces a compatible champion, the saved halt has been reviewed, and a documented/manual reset is explicitly approved.

## Recent implementation references

- [PR #12 — Block paper deployment when held-out tests fail](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/12): draft proposal; do not merge as-is because repeatedly using the same final test as a deployment selector risks holdout contamination.
- [PR #13 — Reject fragile strategy champions](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/13): training robustness gate and entry buffer.
- [PR #15 — Fail closed on missed decision candles](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/15): avoids retroactive fills after missed candles.
- [PR #17](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/17) was closed as superseded. [PR #18 — Resumable multi-year Coinbase candle importer](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/18) is the current draft stacked on Core; **92 tests passed** on head `7c06132a9e050ef95f132efab8e31a2581dadaa6` ([run #39214658596](https://github.com/PropagandaGR/trend-paper-trading-bot./actions/runs/39214658596)) and Coinbase smoke passed ([run #39214658532](https://github.com/PropagandaGR/trend-paper-trading-bot./actions/runs/39214658532)). It includes a manual Actions workflow that uploads month CSVs/manifests as an artifact. The importer has not been run over real multi-year data.
- [PR #16 — Consolidation branch](https://github.com/PropagandaGR/trend-paper-trading-bot./pull/16): carries forward PR #13/#15 changes and fixes exposure-halt semantics, strict same-candle FX conversion, shared entry sizing, backtest risk halts, conservative shared cost assumptions, a 95% source/converted candle-coverage gate, fail-closed state validation, an append-only idempotent paper event journal, strategy-effective candidate deduplication, and report-only cash/buy-and-hold benchmarks. Latest verified code head passed 83 tests and the Coinbase smoke test. Keep draft until fresh data-quality research, remaining audit gates, and independent review are complete.

## Known gaps and risk register

| Priority | Gap / risk | Required closure evidence |
|---|---|---|
| P0 | Existing halt state cannot be safely cleared from code change alone | Inspect current persisted state, explain historical trigger, verify corrected code on main, approve explicit reset procedure |
| P0 | Backtest risk semantics are now implemented but still need broader parity review against paper polling/cursor behavior | Shared deterministic simulator; tests cover loss breach, liquidation, and no re-entry; still verify UTC day boundaries, gap handling, and parity under long/flat and symbol changes |
| P0 | Candle coverage report and 95% quality gate are implemented, but no fresh mainline research run has yet demonstrated the actual coverage outcome | Run research on the current code; inspect source/converted coverage and FX gaps; explain no-trade versus API gaps; retain a reproducible data manifest |
| P0 | Append-only paper event journal is implemented and unit-tested, but only future runs can be reconstructed; no historical events exist for the prior halt | Verify the journal is committed by the scheduled workflow; reconcile a sample of paper status/state to action events; retain manual evidence for the pre-journal halt |
| P1 | Conservative shared fee/slippage settings now default to 60/10 bps per side and stress 100/20, but the user's exact account/order-tier fee remains unconfirmed | Confirm intended Coinbase product/order type and account tier; only then override the shared environment settings; maintain matching champion metadata |
| P1 | Strategy-effective parameter deduplication and per-strategy counts are implemented, but the grid is still generated from a shared generic candidate list and does not deduplicate identical signals on actual data | Replace generic grid with explicit strategy-specific grids; add signal-hash deduplication and report both unique parameter sets and unique signal paths |
| P1 | Same-period cash and buy-and-hold benchmarks are now included in selected champion's final-test metadata and are report-only | Verify benchmark metadata in a fresh report; add benchmark comparison to rolling walk-forward reports without using the final test for selection |
| P1 | One final 20% test split can become contaminated after repeated inspection | Preserve current historical report as evidence; add rolling walk-forward folds and a future forward-paper qualification period, without using the same holdout repeatedly for selection |
| P1 | Champion may change symbol/strategy on each rotating research batch | Compare against incumbent, require stable multi-fold qualification and hysteresis, and forward paper qualification before switching |
| P1 | Research data window is limited to 30–365 days; no durable multi-year store | Paginated historical importer, UTC manifests, retry/rate limits, coverage/gap checks, persistent CSV/Parquet storage outside Git blobs |
| P1 | A 15-minute heartbeat watchdog now checks paper status/state timestamps and opens one issue if either is missing or older than 20 minutes | After merge, verify issue permissions and run a controlled stale-heartbeat test; GitHub schedules remain best-effort and are not guaranteed always-on |
| P2 | Independent review and rollback evidence remain incomplete | Independent reviewer checks data, accounting, selection, and halt semantics; record commit, test evidence, risks, and rollback |
| P2 | No ML or X/social sentiment integration | Defer until data quality, execution parity, walk-forward evaluation, and audit trail are complete |

## Next approved work sequence

1. Complete PR #16 review and CI verification; do not merge without review.
2. Review PR #18 on top of PR #16; do not merge it independently to `main` before its base is approved.
3. Run the multi-year importer for at least one symbol over more than one year, inspect monthly coverage manifests, and verify a resume/checksum path.
4. Run the research pipeline under the 60/10 bps defaults and 95% candle-coverage gate; inspect whether Coinbase data passes. Expect a fail-closed no-champion outcome if coverage is incomplete.
5. Complete parity review for backtest/paper timing, UTC-day boundaries, missed candles, and risk-halt state.
6. Verify append-only events are persisted by the scheduled workflow; independently reconcile state/status against a sample of events.
7. Replace the shared generic grid with explicit strategy-specific grids and signal-equivalence deduplication; add rolling walk-forward evaluation and incumbent/challenger stability.
8. Confirm the Coinbase product/order type and actual account fee tier, retaining conservative defaults until verified.
6. Replace generic strategy grids with strategy-specific grids, deduplication, benchmarks, rolling walk-forward and incumbent stability.
7. Build multi-year historical import/storage and re-run research with reproducible manifests.
8. Add scheduled-run heartbeat and alerting; verify actual run history instead of trusting cron configuration.
9. Run independent Testing & Audit review and record evidence. Keep all execution paper-only until explicit separate approval.

## Release gate

Do **not** enable real orders or call the bot profitable unless the required safety tests, research validation, paper operation, security review, independent audit, rollback plan, and explicit approval are all recorded. A passing test suite only proves the tested assertions passed on the tested commit.
