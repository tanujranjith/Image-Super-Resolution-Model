"""Bounded real-image CPU loading comparison. Never calls training or a model."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from superres.common import config, digest, resolve, write_json  # noqa: E402
from superres.data import Pairs, loader  # noqa: E402


def peak_rss_mib():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Memory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    info = Memory()
    info.cb = ctypes.sizeof(info)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return info.PeakWorkingSetSize / 2**20


def run(cfg, images, mode):
    data = Pairs(
        cfg["manifest"], "train", cfg["scale"], cfg["patch_size"], repeats=1, seed=cfg["seed"]
    )
    selected = random.Random(72).sample(data.rows, min(images, len(data.rows)))
    data.rows = selected
    data.set_epoch(0)
    settings = cfg | {"sampling": "epoch-index-v1"}
    if mode == "serial":
        settings.update(decode_threads=0, prefetch_batches=0)
    stream = loader(data, settings, True, torch.Generator().manual_seed(41))
    started, count, checksums, scenes = time.perf_counter(), 0, [], []
    for low, high, names in stream:
        assert high.shape[-2:] == (low.shape[-2] * cfg["scale"], low.shape[-1] * cfg["scale"])
        count += len(low)
        # Determinism comparison without retaining input batches or decoded HRs.
        import hashlib

        checksums.append(hashlib.sha256(low.numpy().tobytes() + high.numpy().tobytes()).hexdigest())
        scenes.extend(names)
    seconds = time.perf_counter() - started
    return {
        "mode": mode,
        "seconds": seconds,
        "images": count,
        "images_per_second": count / seconds,
    }, (scenes, checksums)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--output", default="runs/loader-benchmark.json")
    args = parser.parse_args()
    if not 4 <= args.images <= 64:
        parser.error("Use 4–64 images to keep this preparation check bounded")
    torch.set_num_threads(2)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "training_executed": False,
        "gpu_work_executed": False,
        "protocol": "CPU-only real-image loading; 16-scene default; serial warmup then alternating serial/threaded repeats; OS cache is not flushed; no model or optimizer",
        "scales": {},
    }
    for scale in (2, 4):
        path = ROOT / f"configs/edsr_large_x{scale}.yaml"
        cfg = config(path)
        _, expected = run(cfg, args.images, "serial")
        measurements = []
        for mode in ("serial", "optimized", "optimized", "serial"):
            result, observed = run(cfg, args.images, mode)
            if observed != expected:
                raise ValueError("Parallel loader changed patch bytes or sampler order")
            measurements.append(result)
        medians = {
            mode: statistics.median(
                x["images_per_second"] for x in measurements if x["mode"] == mode
            )
            for mode in ("serial", "optimized")
        }
        report["scales"][str(scale)] = {
            "config_sha256": digest(path),
            "manifest_sha256": digest(resolve(cfg["manifest"])),
            "measurements": measurements,
            "median_images_per_second": medians,
            "loader_speed_ratio": medians["optimized"] / medians["serial"],
            "patch_bytes_and_order_identical": True,
        }
        print(json.dumps({"scale": scale, **medians}), flush=True)
    report["peak_process_working_set_mib"] = peak_rss_mib()
    write_json(resolve(args.output), report)
    print(
        json.dumps(
            {
                "report": args.output,
                "peak_process_working_set_mib": report["peak_process_working_set_mib"],
                "training_executed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
