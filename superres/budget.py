"""Optional wall-clock training budget; checkpoints remain completed-epoch states."""

from datetime import datetime
import os
import time


class TrainingBudgetEnded(Exception):
    """Stop without promoting a partially trained epoch into a resumable checkpoint."""


def deadline_timestamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Training deadline must include its UTC offset")
    return stamp.timestamp()


def check_budget():
    value = os.environ.get("SR_TRAIN_STOP_UTC")
    if value and time.time() >= deadline_timestamp(value):
        raise TrainingBudgetEnded("Approved training allocation exhausted")
