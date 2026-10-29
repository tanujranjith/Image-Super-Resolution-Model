"""Independent super-resolution overnight supervisor with an explicit conditional start gate.

No work happens on import. Launch only after the user approves training.
The supervisor uses stdlib/PyYAML only; model libraries live in sequential child processes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def tail(path, limit=65536):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def recovery(error, attempts, batch_size, has_checkpoint):
    """Never loop indefinitely or silently alter dataset labels/splits."""
    if attempts >= 3:
        return "stop"
    if ("out of memory" in error.lower() or "OutOfMemoryError" in error) and batch_size > 6:
        return "smaller_batch"
    network = ("URLError", "ConnectionError", "Connection reset", "timed out")
    if attempts == 1 and any(token in error for token in network):
        return "resume" if has_checkpoint else "retry"
    return "stop"


def checkpoint_for_launch(out, resume_existing):
    checkpoint = out / "last.pt"
    if resume_existing:
        if not checkpoint.is_file():
            raise FileNotFoundError("Explicit resume requires an existing last.pt")
        return checkpoint
    if checkpoint.exists():
        raise FileExistsError("An SR experiment already exists; explicit resume is required")
    return None


class Supervisor:
    def __init__(self, directory):
        self.directory = directory
        self.state = {
            "phase": "preflight",
            "started_utc": now(),
            "supervisor_pid": os.getpid(),
            "publication": "not_started",
        }
        self.update()

    def update(self, **changes):
        self.state.update(changes, updated_utc=now())
        save(self.directory / "status.json", self.state)
        save(
            ROOT / "runs/overnight-current.json",
            self.state | {"supervisor_directory": str(self.directory)},
        )

    def resources(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            record = {"utc": now(), "phase": self.state["phase"], "gpu_csv": result.stdout.strip()}
            with (self.directory / "resources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except (OSError, subprocess.TimeoutExpired):
            pass  # Monitoring failure must not abort a valid training process.

    def run(self, phase, arguments):
        stdout = self.directory / f"{phase}.stdout.log"
        stderr = self.directory / f"{phase}.stderr.log"
        environment = os.environ | {
            "PYTHONUNBUFFERED": "1",
            "TORCH_HOME": str(ROOT / "datasets/pretrained"),
            "GRADIO_ANALYTICS_ENABLED": "False",
            "MPLBACKEND": "Agg",
        }
        if self.state.get("cpu_threads"):
            environment.update(
                OMP_NUM_THREADS=str(self.state["cpu_threads"]),
                MKL_NUM_THREADS=str(self.state["cpu_threads"]),
            )
        with (
            stdout.open("w", encoding="utf-8") as output,
            stderr.open("w", encoding="utf-8") as errors,
        ):
            child = subprocess.Popen(
                [sys.executable, *arguments],
                cwd=ROOT,
                env=environment,
                stdout=output,
                stderr=errors,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            self.update(phase=phase, child_pid=child.pid, stdout=str(stdout), stderr=str(stderr))
            try:
                next_record = 0.0
                while child.poll() is None:
                    if time.monotonic() >= next_record:
                        self.resources()
                        self.update()
                        next_record = time.monotonic() + 60
                    time.sleep(5)
            except KeyboardInterrupt:
                child.terminate()
                child.wait()
                self.update(phase="interrupted", child_pid=None)
                raise
        self.update(child_pid=None, last_exit_code=child.returncode)
        return child.returncode, stderr


def eligibility(car, current, deadline=None, expected_experiment=None):
    """Wait for completed car evaluation/export; enforce an optional explicit deadline."""
    if deadline is not None and current >= deadline:
        return "deadline_passed"
    if expected_experiment:
        expected = os.path.normcase(os.path.abspath(expected_experiment))
        actual = (
            os.path.normcase(os.path.abspath(car["experiment"])) if car.get("experiment") else ""
        )
        if actual != expected and not actual.startswith(expected + "-smaller-batch"):
            return "wait"  # A previous completed run must not release a new queue.
    if car.get("phase") in ("failed", "interrupted"):
        return "car_failed"
    if car.get("phase") == "complete":
        completed = datetime.fromisoformat(car["completed_utc"])
        return "start" if deadline is None or completed < deadline else "deadline_passed"
    return "wait"


def training_allocation_end(hard_stop, reserve_seconds, current, remaining_scales, allocation):
    end = hard_stop - reserve_seconds
    if current >= end:
        raise TimeoutError("No training time remains before the evaluation reserve")
    if allocation == "sequential":
        return end
    return current + (end - current) / remaining_scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-car-status", type=Path)
    parser.add_argument("--after-car-experiment", type=Path)
    parser.add_argument("--deadline-utc")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Resume each configured experiment from its saved last.pt",
    )
    parser.add_argument(
        "--configs", nargs="+", default=["configs/edsr_x2.yaml", "configs/edsr_x4.yaml"]
    )
    parser.add_argument("--evaluation-reserve-seconds", type=int, default=7200)
    parser.add_argument("--throughput-profile", choices=("standard", "high"), default="standard")
    parser.add_argument("--time-allocation", choices=("shared", "sequential"), default="shared")
    args = parser.parse_args()
    if not 900 <= args.evaluation_reserve_seconds <= 7200:
        parser.error("Evaluation reserve must be between 900 and 7200 seconds")
    if args.deadline_utc and not args.after_car_status:
        parser.error("A deadline requires a car status path")
    if args.after_car_experiment and not args.after_car_status:
        parser.error("An expected car experiment requires a car status path")
    os.environ["SR_THROUGHPUT_PROFILE"] = args.throughput_profile
    directory = ROOT / "runs" / ("overnight-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    supervisor = Supervisor(directory)
    supervisor.update(
        cpu_threads=4,
        throughput_profile=args.throughput_profile,
        conditional_car_status=str(args.after_car_status) if args.after_car_status else None,
        expected_car_experiment=str(args.after_car_experiment.resolve())
        if args.after_car_experiment
        else None,
        deadline_utc=args.deadline_utc,
        scope="EDSR-style x2 then x4; sequential; no publication",
        resume_existing=args.resume_existing,
        evaluation_reserve_seconds=args.evaluation_reserve_seconds,
        time_allocation=args.time_allocation,
    )
    if args.after_car_status:
        deadline = datetime.fromisoformat(args.deadline_utc) if args.deadline_utc else None
        supervisor.update(phase="waiting_for_car")
        while True:
            try:
                car = json.loads(args.after_car_status.read_text())
            except (OSError, ValueError):
                car = {}
            decision = eligibility(
                car, datetime.now(timezone.utc), deadline, args.after_car_experiment
            )
            if decision == "start":
                supervisor.update(approved_start_utc=now(), car_completed_utc=car["completed_utc"])
                break
            if decision != "wait":
                supervisor.update(
                    phase="not_started_" + decision,
                    reason="Car run failed or the explicitly configured start deadline passed.",
                )
                return
            time.sleep(30)
    status_path = ROOT / "TRAINING_STATUS.json"
    project_status = json.loads(status_path.read_text())
    project_status.update(
        training="in_progress",
        approval_required=False,
        approval_scope="Explicitly user-approved local run; configured scales sequentially; optional car-completion gate and deadline recorded in supervisor status",
        supervisor_status=str(directory / "status.json"),
        completed_scales=[],
    )
    save(status_path, project_status)
    try:
        for index, config_name in enumerate(args.configs):
            if os.environ.get("ML_HARD_STOP_UTC"):
                hard_stop = datetime.fromisoformat(os.environ["ML_HARD_STOP_UTC"])
                if (
                    args.time_allocation == "sequential"
                    and time.time() >= hard_stop.timestamp() - args.evaluation_reserve_seconds
                ):
                    supervisor.update(
                        training_budget_exhausted=True,
                        unstarted_configs=args.configs[index:],
                    )
                    break
                allocated_end = training_allocation_end(
                    hard_stop.timestamp(),
                    args.evaluation_reserve_seconds,
                    time.time(),
                    len(args.configs) - index,
                    args.time_allocation,
                )
                os.environ["SR_TRAIN_STOP_UTC"] = datetime.fromtimestamp(
                    allocated_end, timezone.utc
                ).isoformat()
                supervisor.update(
                    training_cutoff_utc=os.environ["SR_TRAIN_STOP_UTC"],
                    hard_stop_utc=hard_stop.isoformat(),
                )
            cfg_path = (ROOT / config_name).resolve()
            cfg = yaml.safe_load(cfg_path.read_text())
            scale = cfg["scale"]
            out = ROOT / cfg["output"]
            resume = checkpoint_for_launch(out, args.resume_existing)
            for attempt in range(1, 4):
                supervisor.update(
                    scale=scale,
                    experiment=str(out),
                    config=str(cfg_path),
                    attempt=attempt,
                    resume_checkpoint=str(resume) if resume else None,
                )
                code, errors = supervisor.run(
                    f"x{scale}-training-{attempt}",
                    ["-m", "superres.train", "--config", str(cfg_path)]
                    + (["--resume", str(resume)] if resume else []),
                )
                if code == 0:
                    break
                action = recovery(
                    tail(errors), attempt, cfg["batch_size"], (out / "last.pt").exists()
                )
                if args.resume_existing and action == "smaller_batch":
                    action = "stop"  # Finishing an existing run never starts afresh.
                if action == "smaller_batch":
                    cfg = cfg | {
                        "batch_size": cfg["batch_size"] // 2,
                        "output": cfg["output"] + "-smaller-batch",
                    }
                    out = ROOT / cfg["output"]
                    cfg_path = directory / f"x{scale}-retry-{attempt}.yaml"
                    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
                    resume = None
                elif action in ("resume", "retry"):
                    resume = out / "last.pt" if action == "resume" else None
                else:
                    raise RuntimeError(
                        f"SR training stopped; checkpoints preserved. Inspect {errors}"
                    )
            else:
                raise RuntimeError("SR recovery attempts exhausted")
            best = out / "best.pt"
            for phase, command in [
                (
                    f"x{scale}-evaluation",
                    [
                        "-m",
                        "superres.evaluate",
                        "--checkpoint",
                        str(best),
                        "--output",
                        str(out / "evaluation-test"),
                    ],
                ),
                (f"x{scale}-export", ["-m", "superres.export", "--checkpoint", str(best)]),
            ]:
                code, errors = supervisor.run(phase, command)
                if code:
                    raise RuntimeError(f"{phase} stopped; inspect {errors}")
            for benchmark in cfg.get("benchmarks", []):
                phase = f"x{scale}-benchmark-{benchmark['name']}"
                code, errors = supervisor.run(
                    phase,
                    [
                        "-m",
                        "superres.evaluate",
                        "--checkpoint",
                        str(best),
                        "--manifest",
                        benchmark["manifest"],
                        "--benchmark",
                        "--output",
                        str(out / ("benchmark-" + benchmark["name"])),
                    ],
                )
                if code:
                    raise RuntimeError(f"{phase} stopped; inspect {errors}")
            manifest_path = ROOT / cfg.get("inference_manifest", cfg["manifest"])
            manifest = json.loads(manifest_path.read_text())
            row = next(r for r in manifest["pairs"] if r["split"] == "test" and r["scale"] == scale)
            input_path = (manifest_path.parent / manifest["root"] / row["lr"]).resolve()
            output_path = out / "held-out-inference.png"
            code, errors = supervisor.run(
                f"x{scale}-inference",
                [
                    "-m",
                    "superres.inference",
                    "--image",
                    str(input_path),
                    "--scale",
                    str(scale),
                    "--output",
                    str(output_path),
                    "--device",
                    "cuda",
                ],
            )
            if code:
                raise RuntimeError(f"Exported inference failed; inspect {errors}")
            from PIL import Image

            with Image.open(input_path) as original, Image.open(output_path) as result:
                if result.size != (original.width * scale, original.height * scale):
                    raise ValueError("Exported output dimensions do not match scale")
            project_status["completed_scales"].append(scale)
            project_status["trained_weights_present"] = True
            save(status_path, project_status)
        code, errors = supervisor.run("deployment-package", ["scripts/build_deployment.py"])
        if code:
            raise RuntimeError(f"Local package failed; inspect {errors}")
        project_status.update(training="complete")
        supervisor.update(
            phase="complete",
            completed_utc=now(),
            completed_scales=project_status["completed_scales"],
        )
    except Exception as error:
        project_status.update(training="stopped_with_error", last_error=str(error))
        supervisor.update(phase="failed", error=str(error))
        raise
    finally:
        save(status_path, project_status)


if __name__ == "__main__":
    main()
