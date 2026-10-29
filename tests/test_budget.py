"""Budget checks are deterministic and never perform an optimizer step."""

import pytest
from superres import budget


def test_budget_requires_explicit_timezone():
    with pytest.raises(ValueError, match="offset"):
        budget.deadline_timestamp("2026-10-19T15:00:00")
    assert budget.deadline_timestamp("2026-10-19T15:00:00-04:00") == budget.deadline_timestamp(
        "2026-10-19T19:00:00+00:00"
    )


def test_deadline_check(monkeypatch):
    monkeypatch.delenv("SR_TRAIN_STOP_UTC", raising=False)
    budget.check_budget()
    value = "2026-10-19T19:00:00+00:00"
    stamp = budget.deadline_timestamp(value)
    monkeypatch.setenv("SR_TRAIN_STOP_UTC", value)
    monkeypatch.setattr(budget.time, "time", lambda: stamp - 1)
    budget.check_budget()
    monkeypatch.setattr(budget.time, "time", lambda: stamp)
    with pytest.raises(budget.TrainingBudgetEnded):
        budget.check_budget()


def test_training_and_validation_stop_before_model_or_optimizer_work(monkeypatch):
    from superres.train import train, validate

    monkeypatch.setenv("SR_TRAIN_STOP_UTC", "2020-01-01T00:00:00+00:00")
    with pytest.raises(budget.TrainingBudgetEnded):
        train({})
    with pytest.raises(budget.TrainingBudgetEnded):
        validate(None, [(None, None, None)], None)
