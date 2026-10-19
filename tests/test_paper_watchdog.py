import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paper_watchdog import check_heartbeat


class PaperWatchdogTests(unittest.TestCase):
    def _write_pair(self, root, status_time, state_time):
        status = root / "paper_status.json"
        state = root / "paper_state.json"
        status.write_text(json.dumps({"updated_at": status_time}))
        state.write_text(json.dumps({"last_checked_at": state_time}))
        return status, state

    def test_fresh_status_and_state_pass(self):
        now = datetime(2026, 10, 19, 7, 0, tzinfo=timezone.utc)
        stamp = (now - timedelta(minutes=5)).isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            status, state = self._write_pair(Path(tmp), stamp, stamp)
            self.assertEqual(check_heartbeat(status, state, now), [])

    def test_stale_status_is_reported(self):
        now = datetime(2026, 10, 19, 7, 0, tzinfo=timezone.utc)
        stale = (now - timedelta(minutes=30)).isoformat()
        fresh = (now - timedelta(minutes=2)).isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            status, state = self._write_pair(Path(tmp), stale, fresh)
            issues = check_heartbeat(status, state, now)
        self.assertEqual(len(issues), 1)
        self.assertIn("paper_status.json.updated_at", issues[0])
        self.assertIn("1800s", issues[0])

    def test_missing_or_malformed_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status, state = root / "missing-status.json", root / "bad-state.json"
            state.write_text("{bad json")
            issues = check_heartbeat(status, state, datetime(2026, 10, 19, tzinfo=timezone.utc))
        self.assertEqual(len(issues), 2)
        self.assertIn("could not be validated", issues[0])
        self.assertIn("could not be validated", issues[1])

    def test_naive_timestamp_is_interpreted_as_utc(self):
        now = datetime(2026, 10, 19, 7, 0, tzinfo=timezone.utc)
        stamp = (now - timedelta(minutes=1)).replace(tzinfo=None).isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            status, state = self._write_pair(Path(tmp), stamp, stamp)
            self.assertEqual(check_heartbeat(status, state, now), [])


if __name__ == "__main__":
    unittest.main()
