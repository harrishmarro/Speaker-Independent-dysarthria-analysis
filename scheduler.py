#!/usr/bin/env python3
"""Persistent eligibility gate for scheduled research."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATE = Path(__file__).resolve().parent / "experiments" / "schedule_state.json"


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("scheduler timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def is_due(last_completed: datetime | None, now: datetime, hours: int) -> bool:
    if hours < 1:
        raise ValueError("hours must be positive")
    return last_completed is None or now >= last_completed + timedelta(hours=hours)


def is_due_minutes(last_completed: datetime | None, now: datetime, minutes: int) -> bool:
    """Return whether the configured minute interval has elapsed."""
    if minutes < 1:
        raise ValueError("minutes must be positive")
    return last_completed is None or now >= last_completed + timedelta(minutes=minutes)


def read_last_completed() -> datetime | None:
    if not STATE.exists():
        return None
    value = json.loads(STATE.read_text(encoding="utf-8")).get("last_completed_utc")
    return parse_time(value) if value else None


def mark_completed(now: datetime) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    normalized = now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    STATE.write_text(json.dumps({"last_completed_utc": normalized}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("due", "mark"))
    parser.add_argument("--hours", type=int)
    parser.add_argument("--minutes", type=int)
    parser.add_argument("--now")
    arguments = parser.parse_args()
    now = parse_time(arguments.now) if arguments.now else datetime.now(timezone.utc)
    if arguments.command == "due":
        if arguments.hours is not None and arguments.minutes is not None:
            raise ValueError("choose either --hours or --minutes")
        if arguments.minutes is not None:
            due = is_due_minutes(read_last_completed(), now, arguments.minutes)
        else:
            due = is_due(read_last_completed(), now, arguments.hours or 5)
        print(f"due={'true' if due else 'false'}")
    else:
        mark_completed(now)
        print(f"marked={now.isoformat()}")


if __name__ == "__main__":
    main()
