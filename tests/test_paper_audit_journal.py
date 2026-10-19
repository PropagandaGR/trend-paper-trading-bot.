import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paper_trader


class PaperAuditJournalTests(unittest.TestCase):
    def test_action_and_run_events_are_append_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            state_file = Path(tmp) / "state" / "paper_state.json"
            state_file.parent.mkdir()
            state_file.write_text(json.dumps({
                "cash": 7.5, "quantity": 0.074, "symbol": "BTCUSDT",
                "entry_price": 100.15, "realized_pnl": 0.0,
                "completed_trades": 0, "risk_halt": False,
                "risk_halt_reason": None, "last_processed_bars": {"BTCUSDT": "2026-10-19T06:00:00+00:00"},
            }))
            status = {
                "updated_at": "2026-10-19T06:00:01+00:00",
                "champion_version": "champion-abcdef123456",
                "last_processed_bar": "2026-10-19T06:00:00+00:00",
                "actions": [{
                    "action": "PAPER_BUY", "symbol": "BTCUSDT", "price": 100.15,
                    "quantity": 0.074, "fee_usdt": 0.045, "slippage_usdt": 0.037,
                    "reason": "validated champion entry",
                }],
                "cash_usdt": 7.5, "equity_usdt": 14.9, "total_pnl_usdt": -0.1,
                "realized_pnl_usdt": 0.0, "unrealized_pnl_usdt": -0.1,
                "completed_trades": 0, "open_position": {"symbol": "BTCUSDT"},
                "risk": {"halted": False, "halt_reason": None}, "data_errors": [],
                "message": "paper simulation",
            }
            with (
                patch.object(paper_trader, "REPORTS", reports),
                patch.object(paper_trader, "STATE_FILE", state_file),
                patch.dict("os.environ", {"GITHUB_RUN_ID": "test-run-1", "GITHUB_RUN_ATTEMPT": "1"}),
            ):
                paper_trader._append_paper_events(status)
                paper_trader._append_paper_events(status)

            events = [json.loads(line) for line in (reports / "paper_events.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 2)
            self.assertEqual({event["event_type"] for event in events}, {"PAPER_ACTION", "PAPER_RUN"})
            action = next(event for event in events if event["event_type"] == "PAPER_ACTION")
            self.assertEqual(action["champion_version"], "champion-abcdef123456")
            self.assertEqual(action["bar_timestamp"], "2026-10-19T06:00:00+00:00")
            self.assertEqual(action["workflow_run_id"], "test-run-1")
            self.assertEqual(len({event["event_id"] for event in events}), 2)

    def test_run_once_logs_failure_then_rethrows(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports = Path(tmp) / "reports"
            state_file = Path(tmp) / "paper_state.json"
            with (
                patch.object(paper_trader, "REPORTS", reports),
                patch.object(paper_trader, "STATE_FILE", state_file),
                patch.object(paper_trader, "_run_once_impl", side_effect=RuntimeError("bad state")),
            ):
                with self.assertRaisesRegex(RuntimeError, "bad state"):
                    paper_trader.run_once()

            events = [json.loads(line) for line in (reports / "paper_events.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "PAPER_ERROR")
            self.assertIn("bad state", events[0]["message"])


if __name__ == "__main__":
    unittest.main()
