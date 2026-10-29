"""CPU/std-library gate tests; never launch training or import Torch."""

import unittest

from extend_window import approved_deadline, eligibility, project_path
from scheduled_resume import timestamp


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"hard_stop_utc": "2026-10-20T19:30:00+00:00", "after_launched_utc": "original"}
        self.source = {"launched_utc": "original", "phase": "running"}
        self.now = timestamp("2026-10-20T11:00:00+00:00")

    def test_waits_for_exact_current_controller(self):
        self.assertEqual(eligibility(self.plan, self.source, self.now), "wait")
        self.assertEqual(
            eligibility(self.plan, self.source | {"launched_utc": "old"}, self.now), "wrong_source"
        )

    def test_requires_successful_both_scale_completion(self):
        self.assertEqual(
            eligibility(self.plan, self.source | {"phase": "failed"}, self.now),
            "source_not_complete",
        )
        source = self.source | {"phase": "complete", "completed_scales": [2, 4]}
        self.assertEqual(eligibility(self.plan, source, self.now), "start")
        self.assertEqual(
            eligibility(self.plan, source | {"completed_scales": [2]}, self.now),
            "source_not_complete",
        )

    def test_cancellation_duplicate_and_deadline(self):
        self.assertEqual(
            eligibility(self.plan | {"cancelled": True}, self.source, self.now), "cancelled"
        )
        self.assertEqual(
            eligibility(self.plan | {"launched_utc": "done"}, self.source, self.now),
            "already_launched",
        )
        self.assertEqual(
            eligibility(self.plan, self.source, timestamp(self.plan["hard_stop_utc"]) - 3300),
            "insufficient_time",
        )

    def test_deadline_extension_requires_matching_explicit_request(self):
        plan = self.plan | {
            "explicit_user_request": "Deadline has moved to 430",
            "hard_stop_utc": "2026-10-20T20:30:00+00:00",
        }
        self.assertTrue(approved_deadline(plan))
        self.assertFalse(approved_deadline(plan | {"explicit_user_request": "Get close to 330"}))
        self.assertFalse(approved_deadline(plan | {"hard_stop_utc": "2026-10-20T21:30:00+00:00"}))
        self.assertFalse(approved_deadline({}))

    def test_queued_extension_waits_for_own_source(self):
        plan = self.plan | {
            "hard_stop_utc": "2026-10-20T20:30:00+00:00",
            "after_launched_utc": "extension",
        }
        source = {"launched_utc": "extension", "phase": "running"}
        self.assertEqual(eligibility(plan, source, timestamp("2026-10-20T19:25:00+00:00")), "wait")
        self.assertEqual(
            eligibility(
                plan,
                source | {"phase": "complete", "completed_scales": [2, 4]},
                timestamp("2026-10-20T19:25:00+00:00"),
            ),
            "start",
        )

    def test_controller_paths_stay_in_project(self):
        with self.assertRaises(ValueError):
            project_path("../unrelated/status.json")

    def test_one_hour_approval_has_exact_bounded_duration(self):
        plan = {
            "explicit_user_request": "U can continue training for the next hour",
            "approved_utc": "2026-10-20T23:20:34+00:00",
            "hard_stop_utc": "2026-10-21T00:20:34+00:00",
        }
        self.assertTrue(approved_deadline(plan))
        self.assertFalse(approved_deadline(plan | {"hard_stop_utc": "2026-10-21T00:21:34+00:00"}))
        self.assertFalse(approved_deadline(plan | {"approved_utc": None}))

    def test_wednesday_deadline_requires_date_clarification(self):
        plan = {
            "explicit_user_request": "U can continue until 430 pm now",
            "deadline_confirmation": "Yeah, that's what I meant.",
            "confirmed_deadline_local": "Wednesday October 21, 2026, 4:30 PM America/Indianapolis",
            "hard_stop_utc": "2026-10-21T20:30:00+00:00",
        }
        self.assertTrue(approved_deadline(plan))
        self.assertFalse(approved_deadline(plan | {"deadline_confirmation": None}))
        self.assertFalse(approved_deadline(plan | {"hard_stop_utc": "2026-10-22T20:30:00+00:00"}))

    def test_monday_4pm_approval_has_fixed_date_and_time(self):
        plan = {
            "explicit_user_request": "start the next run and run it until 4pm today",
            "hard_stop_utc": "2026-10-26T20:00:00+00:00",
        }
        self.assertTrue(approved_deadline(plan))
        self.assertFalse(approved_deadline(plan | {"hard_stop_utc": "2026-10-27T20:00:00+00:00"}))
        self.assertFalse(approved_deadline(plan | {"explicit_user_request": "stop"}))


if __name__ == "__main__":
    unittest.main()
