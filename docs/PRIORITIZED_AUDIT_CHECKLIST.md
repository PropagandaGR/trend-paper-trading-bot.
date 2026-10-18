# Prioritized Audit Checklist — Paper Portfolio Accounting

This checklist is an evidence log. An unchecked item is **not passed**. A checkbox may be checked only with a linked commit, exact test command, and recorded CI run or independently reproducible output.

## P0 — Portfolio valuation integrity
- [ ] **P0.1 Wrong-symbol mark:** equity and unrealized P/L use a price for the held position's exact symbol, never the current champion's symbol by accident.
  - Evidence required: regression test with two symbols at materially different prices; held symbol's valuation must match its own price.
- [ ] **P0.2 Missing mark protection:** if the held symbol has no current or explicitly symbol-tagged cached mark, emit a valuation error and do not substitute another asset's price.
  - Evidence required: test missing held-symbol price.
- [ ] **P0.3 Symbol-scoped cursors/prices:** processed candle cursors and cached marks are keyed by symbol; a champion symbol change cannot overwrite the held asset's mark.
  - Evidence required: state migration and symbol-switch tests.

## P0 — Ledger accounting invariants
- [ ] **P0.4 Buy cash conservation:** quantity is derived from available cash after modeled slippage and fee; cash + marked position value is reconciled.
- [ ] **P0.5 Sell proceeds and realized P/L:** realized P/L equals net sale proceeds after fee minus all-in entry cost basis.
- [ ] **P0.6 No negative cash or quantity:** reject invalid prices, negative quantities, and invalid fee/slippage rates.
- [ ] **P0.7 Trade idempotency:** reprocessing the same closed candle cannot place a duplicate simulated order or duplicate a ledger event.
- [ ] **P0.8 State reconciliation:** starting equity + realized P/L + unrealized P/L reconciles to equity within a documented rounding tolerance.

## P1 — Cost model verification
- [ ] **P1.1 Units:** basis points are converted using 1 bp = 1/10,000; fee and slippage are charged on each executed side.
- [ ] **P1.2 Round trip:** baseline 10 bp fee + 5 bp slippage per side is approximately 30 bp total for a buy and sell, before price movement and compounding.
- [ ] **P1.3 Stress case:** 20 bp fee + 10 bp slippage per side is approximately 60 bp per round trip.
- [ ] **P1.4 Research/paper parity:** backtest and paper accounting use documented, comparable per-side cost assumptions.
- [ ] **P1.5 Boundary cases:** zero costs, fractional quantities, tiny notional, and invalid negative rates are tested.

## P1 — Strategy research validity
- [ ] **P1.6 Chronological split:** training, parameter selection, and final out-of-sample periods are distinct and time ordered.
- [ ] **P1.7 Benchmark:** report buy-and-hold on the same held-out period and asset.
- [ ] **P1.8 Data integrity:** Coinbase candles are closed, unique, sorted, correctly quoted, and aligned with USDT-USD conversion data.
- [ ] **P1.9 Execution realism:** signal timestamp, order fill timestamp, fee/slippage model, and mark-to-market conventions are consistent.
- [ ] **P1.10 Robustness:** baseline and stressed cost results, drawdown, trade count, and failed candidates are retained.

## P2 — Operational safety and reproducibility
- [ ] **P2.1 Paper-only invariant:** no private exchange credentials or real-order endpoint is used.
- [ ] **P2.2 Scheduled-run evidence:** GitHub Actions run history confirms the workflow actually executes; workflow file existence alone is not proof.
- [ ] **P2.3 Clean-state replay:** tests run from a clean checkout without depending on generated state.
- [ ] **P2.4 Independent review:** a reviewer distinct from the implementation checks valuation, ledger math, and test evidence.
- [ ] **P2.5 Promotion gate:** no candidate is promoted and no live execution is enabled until all required tests and independent approvals are recorded.

## Evidence record
For each checked item, record: commit SHA, test file and test name, exact command, CI run URL and conclusion, and independent reviewer/date. Do not infer success from a PR being open, a workflow being configured, or a test merely existing.
