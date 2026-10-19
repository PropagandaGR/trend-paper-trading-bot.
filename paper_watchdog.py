"""Check whether the paper-trading workflow has produced fresh state/status artifacts."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_HEARTBEAT_AGE_SECONDS = 20 * 60


def _parse_timestamp(value: Any) -> datetime:
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def check_heartbeat(status_path: Path, state_path: Path, now: datetime | None = None,
                    max_age_seconds: int = MAX_HEARTBEAT_AGE_SECONDS) -> list[str]:
    """Return issues when either paper status or state timestamp is stale/invalid."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("watchdog clock must be timezone-aware")
    current = current.astimezone(timezone.utc)
    issues: list[str] = []
    for path, field in ((status_path, "updated_at"), (state_path, "last_checked_at")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON root must be an object")
            raw = data.get(field)
            if raw is None:
                raise ValueError(f"missing {field}")
            stamp = _parse_timestamp(raw)
            age = (current - stamp).total_seconds()
            if age < -60 or age > max_age_seconds:
                issues.append(
                    f"{path}.{field} age is {age:.0f}s "
                    f"(limit {max_age_seconds}s; value={raw})"
                )
        except Exception as exc:
            issues.append(f"{path} could not be validated: {type(exc).__name__}: {exc}")
    return issues


def main() -> int:
    status_path = Path("reports/paper_status.json")
    state_path = Path("state/paper_state.json")
    issues = check_heartbeat(status_path, state_path)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
        handle.write(f"stale={'true' if issues else 'false'}\n")
    run_url = (
        f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', '')}"
        f"/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    )
    body = (
        "The paper-trading heartbeat is stale or invalid.\n\n"
        + "\n".join(f"- {item}" for item in issues)
        + f"\n\nWatchdog run: {run_url}\n"
        + "\nNo real orders are enabled. Inspect workflow history, Coinbase public data, "
        "and persisted paper state before changing or resetting anything.\n"
    )
    Path(os.environ["RUNNER_TEMP"], "paper-heartbeat-issue.txt").write_text(body, encoding="utf-8")
    if issues:
        print("Heartbeat is stale:\n" + "\n".join(issues))
    else:
        print("Paper status and state timestamps are fresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
