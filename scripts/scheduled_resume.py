"""Low-memory scheduled SR continuation. No Torch/GPU imports before launch."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from approved_run import save, supervise, verify

ROOT = Path(__file__).resolve().parents[1]


def load_plan(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Scheduled time requires an explicit timezone")
    return result.timestamp()


def validate_start_approval(plan):
    if plan.get("cancelled") or plan.get("requires_fresh_explicit_approval"):
        raise ValueError("Fresh explicit approval is required")
    if plan.get("confirmed_start") == "Start now":
        if plan.get("explicit_start_approval") != "Start the run":
            raise ValueError("Immediate launch requires explicit human approval")
        approved_start = plan["explicit_start_approved_utc"]
    elif plan.get("confirmed_start") == "11 PM tonight - Mon Oct 19":
        approved_start = "2026-10-20T03:00:00+00:00"
    else:
        raise ValueError("Missing confirmed start approval")
    if timestamp(plan["start_at_utc"]) != timestamp(approved_start):
        raise ValueError("Scheduled timestamp differs from the explicitly approved start")


def start_condition(plan, current):
    if plan.get("cancelled") or plan.get("requires_fresh_explicit_approval"):
        return "cancelled"
    if plan.get("launched_utc"):
        return "already_launched"
    if current >= timestamp(plan["hard_stop_utc"]) - 30:
        return "deadline_passed"
    start = timestamp(plan["start_at_utc"])
    if current < start:
        return "wait"
    if current - start > plan["max_lateness_seconds"]:
        return "missed_start"
    return "start"


def child_environment(original, hard_stop_utc):
    result = original | {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
    }
    # Replace stale budgets with the latest explicitly approved deadline.
    for key in ("ML_HARD_STOP_UTC", "SR_TRAIN_STOP_UTC", "CUDA_VISIBLE_DEVICES"):
        result.pop(key, None)
    result["ML_HARD_STOP_UTC"] = hard_stop_utc
    return result


def ensure_no_other_jobs():
    project = str(ROOT).replace("'", "''")
    command = (
        "$ErrorActionPreference='Stop'; @("
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
        f"-and $_.ProcessId -notin @({os.getpid()},{os.getppid()}) "
        f"-and $_.CommandLine -and $_.CommandLine.Contains('{project}') "
        "-and $_.CommandLine -match 'superres\\.train|overnight\\.py|approved_run\\.py|scheduled_resume\\.py|extend_window\\.py|priority_resume\\.py' "
        "} | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    )
    powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=True,
    )
    if result.stdout.strip() and json.loads(result.stdout):
        raise RuntimeError("Another SR job/controller is active; refusing duplicate launch")


def verify_resume_inputs(plan):
    verify()
    for name, expected in plan["resume_input_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Resume path escapes this project")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Approved resume input changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    if not plan_path.is_relative_to(ROOT):
        raise ValueError("Approval plan must remain in this project")
    plan = load_plan(plan_path)
    if (
        plan.get("confirmed_end") != "Up to 60 epochs each, with validation-based early stopping"
        or plan.get("scales") != [2, 4]
        or plan.get("hard_stop_utc") != "2026-10-20T19:30:00+00:00"
    ):
        raise ValueError("Missing explicit scheduled continuation approval")
    validate_start_approval(plan)
    deadline = timestamp(plan["hard_stop_utc"])
    verify_resume_inputs(plan)
    lock = ROOT / "runs/scheduled-sr.pid"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    state = {
        "phase": "waiting_for_scheduled_start",
        "controller_pid": os.getpid(),
        "child_pid": None,
        "start_at_utc": plan["start_at_utc"],
        "start_local": plan["start_local"],
        "approval_plan": str(plan_path),
        "stop_condition": plan["confirmed_end"],
        "hard_stop_utc": plan["hard_stop_utc"],
        "hard_stop_local": plan["hard_stop_local"],
        "evaluation_reserve_hours": 2,
        "training_started": False,
        "publication": "not_started",
    }

    def update(**changes):
        state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
        save(ROOT / "runs/scheduled-sr-current.json", state)

    try:
        update()
        while True:
            plan = load_plan(plan_path)
            decision = start_condition(plan, time.time())
            if decision == "start":
                break
            if decision != "wait":
                update(phase="not_started_" + decision)
                return
            time.sleep(max(0, min(30, timestamp(plan["start_at_utc"]) - time.time())))
            update()
        validate_start_approval(plan)
        verify_resume_inputs(plan)
        ensure_no_other_jobs()
        # Exclusive process lock plus a persistent one-shot receipt prevent
        # duplicate launches, including a later re-run of this plan.
        plan["launched_utc"] = datetime.now(timezone.utc).isoformat()
        save(plan_path, plan)
        directory = (
            ROOT / "runs" / ("scheduled-sr-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
        )
        directory.mkdir(parents=True)
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
                training_started=True,
                launched_utc=plan["launched_utc"],
                stdout=str(directory / "stdout.log"),
                stderr=str(directory / "stderr.log"),
            )
            supervise(child, deadline, update)
            if child.returncode:
                raise RuntimeError(
                    f"SR continuation supervisor failed with code {child.returncode}"
                )
        final = json.loads((ROOT / "runs/overnight-current.json").read_text())
        if (
            final.get("phase") != "complete"
            or final.get("completed_scales") != [2, 4]
            or timestamp(final["completed_utc"]) < timestamp(plan["launched_utc"])
        ):
            raise RuntimeError("Fresh completion and verified exports for both scales are missing")
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
