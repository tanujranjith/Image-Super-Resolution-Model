"""One-shot authorized x4 continuation after the current controller exits safely."""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from approved_run import save, supervise, verify
from scheduled_resume import (
    child_environment,
    ensure_no_other_jobs,
    load_plan,
    timestamp,
    verify_resume_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def approved_deadline(plan):
    if plan.get("explicit_user_request") == "U can continue until 430 pm now":
        return (
            plan.get("deadline_confirmation") == "Yeah, that's what I meant."
            and plan.get("confirmed_deadline_local")
            == "Wednesday October 21, 2026, 4:30 PM America/Indianapolis"
            and plan.get("hard_stop_utc") == "2026-10-21T20:30:00+00:00"
        )
    if plan.get("explicit_user_request") == "U can continue training for the next hour":
        try:
            return timestamp(plan["hard_stop_utc"]) - timestamp(plan["approved_utc"]) == 3600
        except (KeyError, TypeError, ValueError):
            return False
    approvals = {
        "Get close to 330": "2026-10-20T19:30:00+00:00",
        "Deadline has moved to 430": "2026-10-20T20:30:00+00:00",
        "start the next run and run it until 4pm today": "2026-10-26T20:00:00+00:00",
    }
    return (
        approvals.get(plan.get("explicit_user_request")) == plan.get("hard_stop_utc")
        and plan.get("explicit_user_request") in approvals
    )


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Controller paths must remain in this project")
    return path


def eligibility(plan, source, current):
    if plan.get("cancelled"):
        return "cancelled"
    if plan.get("launched_utc"):
        return "already_launched"
    # Avoid starting another epoch with less than 40 minutes before evaluation.
    if current >= timestamp(plan["hard_stop_utc"]) - 900 - 2400:
        return "insufficient_time"
    if source.get("launched_utc") != plan["after_launched_utc"]:
        return "wrong_source"
    if source.get("phase") == "running":
        return "wait"
    if source.get("phase") == "complete" and source.get("completed_scales") == [2, 4]:
        return "start"
    return "source_not_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    path = args.plan.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Plan must remain in this project")
    plan = load_plan(path)
    if (
        not approved_deadline(plan)
        or plan.get("original_start_approval") != "Start the run"
        or plan.get("scale") != 4
    ):
        raise ValueError("Missing explicit bounded continuation approval")
    lock = project_path(plan.get("controller_lock", "runs/extend-window.pid"))
    state_path = project_path(plan.get("status_file", "runs/extend-window-current.json"))
    source_path = project_path(plan.get("source_status", "runs/scheduled-sr-current.json"))
    source_lock = project_path(plan.get("source_lock", "runs/scheduled-sr.pid"))
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = plan | {
        "phase": "waiting_for_current_run",
        "controller_pid": os.getpid(),
        "child_pid": None,
    }

    def update(**changes):
        state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
        save(state_path, state)

    try:
        update()
        while True:
            plan = load_plan(path)
            if not approved_deadline(plan):
                raise ValueError("Deadline changed without matching explicit approval")
            source = load_plan(source_path)
            decision = eligibility(plan, source, time.time())
            if decision == "start":
                # Completion status is written just before the first controller
                # releases its lock. Wait for that release before duplicate check.
                if source_lock.exists():
                    time.sleep(2)
                    continue
                break
            if decision != "wait":
                update(phase="not_started_" + decision)
                return
            time.sleep(30)
        with (ROOT / "runs/edsr-large-x4-seed42/metrics.jsonl").open(encoding="utf-8") as stream:
            last = [json.loads(line) for line in deque(stream, maxlen=3) if line.strip()]
        if not last or last[-1].get("event") != "training_budget_ended":
            update(phase="not_started_training_finished_or_early_stopped")
            return
        parent_plan = load_plan(
            project_path(plan.get("approval_record", "runs/scheduled-sr-plan.json"))
        )
        if parent_plan.get("cancelled") or parent_plan.get("requires_fresh_explicit_approval"):
            update(phase="not_started_approval_revoked")
            return
        if "resume_input_sha256" in plan:
            verify_resume_inputs(plan)
        else:
            verify()
        ensure_no_other_jobs()
        plan["launched_utc"] = datetime.now(timezone.utc).isoformat()
        save(path, plan)
        directory = (
            ROOT / "runs" / ("extended-x4-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
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
                    "configs/edsr_large_x4.yaml",
                    "--evaluation-reserve-seconds",
                    "900",
                    "--throughput-profile",
                    plan.get("throughput_profile", "standard"),
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
                raise RuntimeError(f"Extension supervisor failed with code {code}")
        final = load_plan(ROOT / "runs/overnight-current.json")
        if (
            final.get("phase") != "complete"
            or final.get("completed_scales") != [4]
            or timestamp(final["completed_utc"]) < timestamp(plan["launched_utc"])
        ):
            raise RuntimeError("Fresh x4 evaluation/export verification missing")
        # x2 was already verified by the preceding successful controller.
        status_path = ROOT / "TRAINING_STATUS.json"
        status = load_plan(status_path)
        status["completed_scales"] = [2, 4]
        save(status_path, status)
        update(
            phase="complete",
            child_pid=None,
            completed_utc=final["completed_utc"],
            completed_scales=[2, 4],
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
