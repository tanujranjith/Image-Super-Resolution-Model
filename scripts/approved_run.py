"""Independent, explicitly approved SR-only controller with an optional hard stop."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
APPROVAL = "alr u can start next traiing run for the super resolution"


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def deadline_stamp(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Deadline must have an explicit timezone")
    return parsed.timestamp()


def supervise(child, deadline, update, clock=time.time, sleep=time.sleep, stop=None):
    next_update = 0
    while child.poll() is None:
        current = clock()
        if deadline is not None and current >= deadline - 30:
            if stop:
                stop(child)
            else:
                subprocess.run(
                    ["taskkill.exe", "/PID", str(child.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=20,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    check=True,
                )
                child.wait(timeout=20)
            raise TimeoutError("Approved SR hard-stop time reached; owned process tree stopped")
        if current >= next_update:
            update()
            next_update = current + 60
        sleep(1)
    return child.returncode


def verify():
    ready = json.loads((ROOT / "datasets/large/readiness.json").read_text())
    if ready["status"] != "ready_for_training_approval" or ready.get("user_stop_requested"):
        raise ValueError("Readiness or fresh training approval has not passed")
    files = (
        dict(ready["validated_source_hashes"])
        | ready["config_hashes"]
        | {
            "datasets/large/processed/training.json": ready["training_manifest_sha256"],
            "datasets/large/processed/lsdir-heldout.json": ready["heldout_manifest_sha256"],
            "datasets/processed/div2k.json": ready["original_div2k_manifest_sha256"],
        }
    )
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Readiness path escapes this project")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Validated input changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get("explicit_user_approval") != APPROVAL or plan.get("scales") != [2, 4]:
        raise ValueError("Missing explicit SR-only launch approval")
    deadline = deadline_stamp(plan.get("hard_stop_utc"))
    if deadline is None and plan.get("duration_choice") != "let_both_finish_explicit":
        raise ValueError("Removing the previous deadline needs explicit user authorization")
    if deadline is not None and time.time() >= deadline - 30:
        raise TimeoutError("Approved training window has already ended")
    verify()
    for scale in (2, 4):
        if (ROOT / f"runs/edsr-large-x{scale}-seed42/last.pt").exists():
            raise FileExistsError(
                "Existing experiment requires explicit resume, not a duplicate launch"
            )
    lock = ROOT / "runs/approved-sr.pid"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    directory = (
        ROOT / "runs" / ("approved-sr-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    )
    directory.mkdir(parents=True)
    state = plan | {
        "phase": "preflight",
        "controller_pid": os.getpid(),
        "child_pid": None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    def update(**changes):
        state.update(changes, updated_utc=datetime.now(timezone.utc).isoformat())
        save(directory / "status.json", state)
        save(ROOT / "runs/approved-sr-current.json", state)

    environment = os.environ | {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
    }
    for key in ("CUDA_VISIBLE_DEVICES", "SR_TRAIN_STOP_UTC", "ML_HARD_STOP_UTC"):
        environment.pop(key, None)
    if deadline is not None:
        environment["ML_HARD_STOP_UTC"] = plan["hard_stop_utc"]
    try:
        with (
            (directory / "stdout.log").open("w") as output,
            (directory / "stderr.log").open("w") as errors,
        ):
            child = subprocess.Popen(
                [
                    str(ROOT / ".venv/Scripts/python.exe"),
                    "scripts/overnight.py",
                    "--configs",
                    "configs/edsr_large_x2.yaml",
                    "configs/edsr_large_x4.yaml",
                ],
                cwd=ROOT,
                env=environment,
                stdout=output,
                stderr=errors,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            update(
                phase="super_resolution",
                child_pid=child.pid,
                stdout=str(directory / "stdout.log"),
                stderr=str(directory / "stderr.log"),
            )
            code = supervise(child, deadline, update)
            if code:
                raise RuntimeError(f"SR supervisor exited with code {code}; inspect {directory}")
        result = json.loads((ROOT / "runs/overnight-current.json").read_text())
        if result.get("phase") != "complete" or result.get("completed_scales") != [2, 4]:
            raise RuntimeError("Both scales have not completed evaluation/export")
        update(
            phase="complete",
            child_pid=None,
            completed_utc=datetime.now(timezone.utc).isoformat(),
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
