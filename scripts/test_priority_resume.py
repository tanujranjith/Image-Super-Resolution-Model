"""No training or Torch imports: priority order, approval and deadline checks."""

import unittest

from overnight import training_allocation_end
from priority_resume import validate_approval
from scheduled_resume import timestamp


class PriorityResumeTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "explicit_user_approval": "resume training until 330 tmrw",
            "hard_stop_utc": "2026-10-23T19:30:00+00:00",
            "scales": [2, 4],
            "time_allocation": "sequential",
        }
        self.current = timestamp("2026-10-22T04:24:00+00:00")

    def test_explicit_approval_and_correct_order(self):
        validate_approval(self.plan, self.current)
        for changes in (
            {"scales": [4, 2]},
            {"explicit_user_approval": "remind me"},
            {"hard_stop_utc": "2026-10-24T19:30:00+00:00"},
            {"cancelled": True},
            {"launched_utc": "already"},
            {"time_allocation": "shared"},
        ):
            with self.assertRaises(ValueError):
                validate_approval(self.plan | changes, self.current)

    def test_no_start_after_window_expires(self):
        with self.assertRaises(TimeoutError):
            validate_approval(self.plan, timestamp(self.plan["hard_stop_utc"]) - 900)

    def test_2x_gets_full_training_window(self):
        self.assertEqual(training_allocation_end(10000, 900, 1000, 2, "sequential"), 9100)
        self.assertEqual(training_allocation_end(10000, 900, 7000, 1, "sequential"), 9100)
        self.assertEqual(training_allocation_end(10000, 900, 1000, 2, "shared"), 5050)
        with self.assertRaises(TimeoutError):
            training_allocation_end(10000, 900, 9100, 1, "sequential")

    def test_saturday_start_requires_date_confirmation_and_fresh_approval(self):
        plan = self.plan | {
            "explicit_user_approval": "start",
            "requested_window": "continue a run until 11am tmrw(saturday) does that make sense?",
            "date_confirmation": "today",
            "hard_stop_utc": "2026-10-24T15:00:00+00:00",
        }
        now = timestamp("2026-10-24T05:13:00+00:00")
        validate_approval(plan, now)
        for changes in (
            {"date_confirmation": None},
            {"explicit_user_approval": "stop"},
            {"hard_stop_utc": "2026-10-25T15:00:00+00:00"},
            {"requires_fresh_explicit_approval": True},
            {"cancelled": True},
        ):
            with self.assertRaises(ValueError):
                validate_approval(plan | changes, now)
        with self.assertRaises(TimeoutError):
            validate_approval(plan, timestamp(plan["hard_stop_utc"]) - 900)

    def test_sunday_morning_resume_has_fixed_11am_cutoff(self):
        plan = self.plan | {
            "explicit_user_approval": "continue untill 11am morning",
            "hard_stop_utc": "2026-10-25T15:00:00+00:00",
        }
        now = timestamp("2026-10-25T06:09:00+00:00")
        validate_approval(plan, now)
        for changes in (
            {"hard_stop_utc": "2026-10-26T15:00:00+00:00"},
            {"explicit_user_approval": "stop"},
            {"launched_utc": "already"},
            {"cancelled": True},
            {"scales": [4, 2]},
        ):
            with self.assertRaises(ValueError):
                validate_approval(plan | changes, now)
        with self.assertRaises(TimeoutError):
            validate_approval(plan, timestamp(plan["hard_stop_utc"]) - 900)


if __name__ == "__main__":
    unittest.main()
