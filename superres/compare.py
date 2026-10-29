"""Compare only actual evaluations with matching scale, condition, split and data."""

import argparse
import csv
from pathlib import Path

from .common import read_json


def compare(directories, output):
    rows, protocols = [], set()
    fields = (
        "architecture",
        "psnr_y",
        "ssim_y",
        "bicubic_psnr_y",
        "bicubic_ssim_y",
        "seconds",
        "model_parameters",
        "checkpoint_bytes",
        "peak_vram_mb",
    )
    for directory in directories:
        summary = read_json(Path(directory) / "metrics.json")["summary"]
        protocols.add(tuple(summary[k] for k in ("manifest_sha256", "scale", "condition", "split")))
        rows.append({"run": str(directory), **{k: summary.get(k) for k in fields}})
    if len(protocols) != 1:
        raise ValueError("Use the same data, scale, degradation condition and held-out split")
    with Path(output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="comparison.csv")
    args = parser.parse_args()
    compare(args.runs, args.output)
