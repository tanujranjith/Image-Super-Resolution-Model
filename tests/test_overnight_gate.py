"""Conditional approval is one-night-only and excludes incomplete/failed car runs."""

from datetime import datetime, timezone, timedelta

from scripts.overnight import eligibility


def test_five_am_gate_and_failure_states():
    deadline = datetime(2026, 10, 17, 9, tzinfo=timezone.utc)
    before = deadline - timedelta(minutes=1)
    complete = {"phase": "complete", "completed_utc": before.isoformat()}
    assert eligibility(complete, before, deadline) == "start"
    assert eligibility(complete, deadline, deadline) == "deadline_passed"
    assert eligibility({"phase": "training-1"}, before, deadline) == "wait"
    assert eligibility({"phase": "failed"}, before, deadline) == "car_failed"
    assert eligibility(complete, deadline + timedelta(days=1), deadline) == "deadline_passed"
