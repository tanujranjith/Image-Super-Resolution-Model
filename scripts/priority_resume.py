"""Explicitly approved 2x-first checkpoint continuation with a hard deadline."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

from approved_run import save, supervise
from scheduled_resume import (
    child_environment,
    ensure_no_other_jobs,
    load_plan,
    timestamp,
    verify_resume_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def validate_approval(plan, current):
    previous_approval = (
        plan.get("explicit_user_approval") == "resume training until 330 tmrw"
        and plan.get("hard_stop_utc") == "2026-10-23T19:30:00+00:00"
    )
    saturday_approval = (
        plan.get("explicit_user_approval") == "start"
        and plan.get("requested_window")
        == "continue a run until 11am tmrw(saturday) does that make sense?"
        and plan.get("date_confirmation") == "today"
        and plan.get("hard_stop_utc") == "2026-10-24T15:00:00+00:00"
    )
    sunday_approval = (
        plan.get("explicit_user_approval") == "continue untill 11am morning"
        and plan.get("hard_stop_utc") == "2026-10-25T15:00:00+00:00"
    )
    if (
        not (previous_approval or saturday_approval or sunday_approval)
        or plan.get("scales") != [2, 4]
        or plan.get("time_allocation") != "sequential"
        or plan.get("cancelled")
        or plan.get("requires_fresh_explicit_approval")
        or plan.get("launched_utc")
    ):
        raise ValueError("Missing unused explicit 2x-first continuation approval")
    if current >= timestamp(plan["hard_stop_utc"]) - 900:
        raise TimeoutError("Approved continuation window has ended")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    path = args.plan.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Approval plan must remain within this project")
    plan = load_plan(path)
    validate_approval(plan, time.time())
    verify_resume_inputs(plan)
    lock = ROOT / "runs/priority-resume.pid"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = plan | {"phase": "preflight", "controller_pid": os.getpid(), "child_pid": None}

    def update(**changes):
        state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
        save(ROOT / "runs/priority-resume-current.json", state)

    try:
        update()
        ensure_no_other_jobs()
        plan["launched_utc"] = datetime.now(timezone.utc).isoformat()
        save(path, plan)
        directory = (
            ROOT
            / "runs"
            / ("priority-resume-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        )
        directory.mkdir()
        with (
            (directory / "stdout.log").open("w") as output,
            (directory / "stderr.log").open("w") as errors,
        ):
            child = subprocess.Popen(
                [
                    str(ROOT / ".venv/Scripts/python.exe"),
                    "scripts/overnight.py",
                    "--resume-existing",
                    "--configs",
                    "configs/edsr_large_x2.yaml",
                    "configs/edsr_large_x4.yaml",
                    "--time-allocation",
                    "sequential",
                    "--evaluation-reserve-seconds",
                    "900",
                    "--throughput-profile",
                    plan.get("throughput_profile", "high"),
                ],
                cwd=ROOT,
                env=child_environment(os.environ, plan["hard_stop_utc"]),
                stdout=output,
                stderr=errors,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            update(
                phase="running",
                child_pid=child.pid,
                launched_utc=plan["launched_utc"],
                stdout=str(directory / "stdout.log"),
                stderr=str(directory / "stderr.log"),
            )
            code = supervise(child, timestamp(plan["hard_stop_utc"]), update)
            if code:
                raise RuntimeError(f"Priority continuation supervisor exited with code {code}")
        final = load_plan(ROOT / "runs/overnight-current.json")
        completed = final.get("completed_scales")
        valid_completion = completed == [2, 4] or (
            completed == [2] and final.get("training_budget_exhausted")
        )
        if (
            final.get("phase") != "complete"
            or not valid_completion
            or timestamp(final["completed_utc"]) < timestamp(plan["launched_utc"])
        ):
            raise RuntimeError("Fresh evaluation/export completion for both scales missing")
        update(
            phase="complete",
            child_pid=None,
            completed_utc=final["completed_utc"],
            completed_scales=completed,
            unstarted_configs=final.get("unstarted_configs", []),
        )
    except TimeoutError as error:
        update(phase="stopped_at_deadline", child_pid=None, error=str(error))
    except Exception as error:
        update(phase="failed", error=str(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
