"""Standard-library/PyYAML scheduler tests; never import Torch or launch jobs."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest import mock

import overnight
from overnight import checkpoint_for_launch
from scheduled_resume import (
    child_environment,
    load_plan,
    start_condition,
    supervise,
    timestamp,
    validate_start_approval,
)


class ScheduledResumeTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "start_at_utc": "2026-10-20T03:00:00+00:00",
            "hard_stop_utc": "2026-10-20T19:30:00+00:00",
            "max_lateness_seconds": 900,
        }
        self.start = timestamp(self.plan["start_at_utc"])

    def test_never_starts_early(self):
        self.assertEqual(start_condition(self.plan, self.start - 1), "wait")

    def test_starts_at_11_pm(self):
        self.assertEqual(start_condition(self.plan, self.start), "start")
        self.assertEqual(start_condition(self.plan, self.start + 900), "start")

    def test_refuses_missed_or_duplicate_start(self):
        self.assertEqual(start_condition(self.plan, self.start + 901), "missed_start")
        self.assertEqual(
            start_condition(self.plan | {"launched_utc": "recorded"}, self.start),
            "already_launched",
        )

    def test_cancelled_plan_never_launches(self):
        self.assertEqual(start_condition(self.plan | {"cancelled": True}, self.start), "cancelled")

    def test_fresh_approval_gate_blocks_start(self):
        plan = self.plan | {"requires_fresh_explicit_approval": True}
        self.assertEqual(start_condition(plan, self.start), "cancelled")
        with self.assertRaises(ValueError):
            validate_start_approval(plan)

    def test_explicit_start_now_approval(self):
        plan = self.plan | {
            "confirmed_start": "Start now",
            "explicit_start_approval": "Start the run",
            "explicit_start_approved_utc": self.plan["start_at_utc"],
        }
        validate_start_approval(plan)
        with self.assertRaises(ValueError):
            validate_start_approval(plan | {"explicit_start_approval": "remind me"})
        with self.assertRaises(ValueError):
            validate_start_approval(plan | {"start_at_utc": "2026-10-20T04:00:00+00:00"})

    def test_explicit_timezone_required(self):
        with self.assertRaises(ValueError):
            timestamp("2026-10-20T00:00:00")

    def test_unicode_approval_file_uses_explicit_utf8(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=project / "runs") as temporary:
            path = Path(temporary) / "plan.json"
            approved = {"confirmed_start": "Midnight tonight — Tue Oct 20, 12 AM"}
            path.write_text(json.dumps(approved, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(load_plan(path), approved)

    def test_old_cutoffs_and_hidden_cuda_removed_without_changing_parent(self):
        original = {
            "ML_HARD_STOP_UTC": "expired",
            "SR_TRAIN_STOP_UTC": "expired",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
        result = child_environment(original, self.plan["hard_stop_utc"])
        self.assertNotIn("SR_TRAIN_STOP_UTC", result)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", result)
        self.assertEqual(result["ML_HARD_STOP_UTC"], self.plan["hard_stop_utc"])
        self.assertEqual(original["ML_HARD_STOP_UTC"], "expired")
        self.assertEqual(original["SR_TRAIN_STOP_UTC"], "expired")
        self.assertEqual(result["OMP_NUM_THREADS"], "4")

    def test_deadline_blocks_late_launch(self):
        deadline = timestamp(self.plan["hard_stop_utc"])
        self.assertEqual(start_condition(self.plan, deadline - 30), "deadline_passed")
        self.assertEqual(start_condition(self.plan, deadline), "deadline_passed")

    def test_watchdog_stops_owned_child_before_deadline(self):
        child = mock.Mock()
        child.poll.return_value = None
        stop = mock.Mock()
        update = mock.Mock()
        sleep = mock.Mock()
        with self.assertRaises(TimeoutError):
            supervise(
                child, 100, update, clock=mock.Mock(side_effect=[69, 70]), sleep=sleep, stop=stop
            )
        stop.assert_called_once_with(child)
        sleep.assert_called_once_with(1)

    def test_watchdog_accepts_normal_completion(self):
        child = mock.Mock(returncode=0)
        child.poll.return_value = 0
        stop = mock.Mock()
        self.assertEqual(supervise(child, 100, mock.Mock(), clock=lambda: 1, stop=stop), 0)
        stop.assert_not_called()

    def test_resume_requires_checkpoint_and_fresh_launch_refuses_existing(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=project / "runs") as temporary:
            out = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                checkpoint_for_launch(out, True)
            self.assertIsNone(checkpoint_for_launch(out, False))
            (out / "last.pt").write_bytes(b"fixture; not a model")
            self.assertEqual(checkpoint_for_launch(out, True), out / "last.pt")
            with self.assertRaises(FileExistsError):
                checkpoint_for_launch(out, False)

    def test_resume_cli_passes_existing_checkpoint_to_training_without_launch(self):
        class ObservedBeforeLaunch(Exception):
            pass

        observed = []

        class FakeSupervisor:
            def __init__(self, directory):
                self.state = {}

            def update(self, **changes):
                self.state.update(changes)

            def run(self, phase, command):
                observed.append((phase, command))
                raise ObservedBeforeLaunch()

        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=project / "runs") as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/test.yaml").write_text(
                "scale: 2\noutput: runs/existing\nbatch_size: 12\n"
            )
            checkpoint = root / "runs/existing/last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fixture, never loaded")
            (root / "TRAINING_STATUS.json").write_text(json.dumps({}))
            with (
                mock.patch.object(overnight, "ROOT", root),
                mock.patch.object(overnight, "Supervisor", FakeSupervisor),
                mock.patch.object(
                    sys,
                    "argv",
                    ["overnight.py", "--resume-existing", "--configs", "configs/test.yaml"],
                ),
                mock.patch.dict(overnight.os.environ, {}, clear=True),
            ):
                with self.assertRaises(ObservedBeforeLaunch):
                    overnight.main()
            self.assertEqual(observed[0][0], "x2-training-1")
            self.assertEqual(observed[0][1][-2:], ["--resume", str(checkpoint)])


if __name__ == "__main__":
    unittest.main()
