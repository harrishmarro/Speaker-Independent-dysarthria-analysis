from datetime import datetime, timezone

from scheduler import is_due


def test_first_cycle_is_due():
    assert is_due(None, datetime(2026, 10, 26, tzinfo=timezone.utc), 5)


def test_cycle_waits_for_full_five_hours():
    previous = datetime(2026, 10, 26, 8, 30, tzinfo=timezone.utc)
    assert not is_due(previous, datetime(2026, 10, 26, 13, 29, tzinfo=timezone.utc), 5)
    assert is_due(previous, datetime(2026, 10, 26, 13, 30, tzinfo=timezone.utc), 5)
