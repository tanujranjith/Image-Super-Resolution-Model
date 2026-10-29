"""Validate DIV2K pair geometry and immutable scene splits, or reviewed RealSR CSV."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import os
from pathlib import Path

from PIL import Image

from .common import read_json, resolve, write_json


def inspect_image(path):
    payload = Path(path).read_bytes()
    with Image.open(io.BytesIO(payload)) as im:
        im.verify()
    with Image.open(io.BytesIO(payload)) as im:
        if im.mode != "RGB":
            raise ValueError(f"Expected RGB image, got {im.mode}: {path}")
        im.load()
        return im.size, hashlib.sha256(payload).hexdigest()


def geometry(path):
    return inspect_image(path)[0]


def validate_pairs(root, rows, output, condition):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not rows:
        raise ValueError("No pairs found")
    seen_hr, scene_split, audited, hr_cache = {}, {}, [], {}
    pair_ids = set()
    for index, r in enumerate(rows):
        key = (r["scene"], r["scale"])
        if key in pair_ids:
            raise ValueError(f"Duplicate scene/scale pair: {key}")
        pair_ids.add(key)
        if r["split"] not in ("train", "val", "test") or r["scale"] not in (2, 4):
            raise ValueError("Invalid split or scale")
        if r["scene"] in scene_split and scene_split[r["scene"]] != r["split"]:
            raise ValueError("Scene appears across splits")
        scene_split[r["scene"]] = r["split"]
        hr, lr = (root / r["hr"]).resolve(), (root / r["lr"]).resolve() if r.get("lr") else None
        if condition == "realsr-v3" and lr is None:
            raise ValueError(
                "RealSR requires actual LR observations; synthetic generation cannot replace them"
            )
        if not hr.is_relative_to(root) or (lr and not lr.is_relative_to(root)):
            raise ValueError("Pair path escapes dataset root")
        if hr not in hr_cache:
            hr_cache[hr] = inspect_image(hr)
        size, hr_sha = hr_cache[hr]
        if hr_sha in seen_hr and seen_hr[hr_sha] != r["scene"]:
            raise ValueError(
                "Identical HR appears under different scene IDs; resolve before splitting"
            )
        seen_hr[hr_sha] = r["scene"]
        if lr:
            low, lr_sha = inspect_image(lr)
            expected = (low[0] * r["scale"], low[1] * r["scale"])
            # Official DIV2K LR dimensions may correspond to modcropped HR.
            if expected != (size[0] - size[0] % r["scale"], size[1] - size[1] % r["scale"]):
                raise ValueError(
                    f"Misaligned pair geometry {r['scene']}: HR {size}, LR {low}, scale {r['scale']}"
                )
        if condition == "realsr-v3" and not r.get("alignment_reviewed"):
            raise ValueError(
                "RealSR geometry alone cannot establish alignment; reviewed alignment required"
            )
        audited.append(
            {
                **r,
                "hr_sha256": hr_sha,
                "lr_sha256": lr_sha if lr else None,
                "hr_size": list(size),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"Validated {index + 1}/{len(rows)} HR/LR pairs", flush=True)
    for scale in {r["scale"] for r in rows}:
        if {r["split"] for r in rows if r["scale"] == scale} != {"train", "val", "test"}:
            raise ValueError(f"Scale {scale} needs all three splits")
    write_json(
        output,
        {
            "schema": 1,
            "root": os.path.relpath(root, output.parent),
            "condition": condition,
            "pairs": audited,
            "counts": dict(Counter(f"x{r['scale']}/{r['split']}" for r in rows)),
        },
    )
    return len(audited)


def div2k(root, output, synthetic=False, allow_partial=False):
    root = Path(root)
    rows = []
    for section in ("train", "valid"):
        files = sorted((root / f"DIV2K_{section}_HR").glob("*.png"))
        expected = 800 if section == "train" else 100
        if not allow_partial and len(files) != expected:
            raise ValueError(f"Expected {expected} {section} HR files, found {len(files)}")
        for hr in files:
            n = int(hr.stem)
            if not (1 <= n <= 800 if section == "train" else 801 <= n <= 900):
                raise ValueError(f"Unexpected official DIV2K ID: {n}")
            # Official 100-image test HR is not public. Keep valid as test and reserve train IDs 751:800 for model selection.
            split = "test" if section == "valid" else "val" if n >= 751 else "train"
            for scale in (2, 4):
                lr = root / f"DIV2K_{section}_LR_bicubic/X{scale}/{hr.stem}x{scale}.png"
                if not synthetic and not lr.exists():
                    raise FileNotFoundError(f"Missing LR partner: {lr}")
                rows.append(
                    {
                        "scene": hr.stem,
                        "scale": scale,
                        "split": split,
                        "hr": hr.relative_to(root).as_posix(),
                        "lr": None if synthetic else lr.relative_to(root).as_posix(),
                    }
                )
    return validate_pairs(
        root, rows, output, "pillow-bicubic-synthetic" if synthetic else "div2k-official-bicubic"
    )


def derive_synthetic(source, output):
    """Reuse verified HR metadata without re-reading every PNG for an HR-only condition."""
    source, output = Path(source).resolve(), Path(output).resolve()
    info = read_json(source)
    if info["condition"] != "div2k-official-bicubic":
        raise ValueError(
            "Derive synthetic DIV2K experiments from a verified official paired manifest"
        )
    root = (source.parent / info["root"]).resolve()
    info["root"] = os.path.relpath(root, output.parent)
    info["condition"] = "pillow-bicubic-synthetic"
    info["source_manifest_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    info["pairs"] = [{**row, "lr": None, "lr_sha256": None} for row in info["pairs"]]
    write_json(output, info)
    return len(info["pairs"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="datasets/raw")
    p.add_argument("--output", default="datasets/processed/div2k.json")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--from-paired", help="Reuse validated paired-manifest HR metadata with --synthetic"
    )
    p.add_argument("--realsr-csv", help="CSV: scene,scale,split,hr,lr,alignment_reviewed")
    a = p.parse_args()
    if a.from_paired:
        if not a.synthetic or a.realsr_csv:
            p.error("--from-paired requires --synthetic and cannot be combined with RealSR")
        print(derive_synthetic(resolve(a.from_paired), resolve(a.output)))
    elif a.realsr_csv:
        with resolve(a.realsr_csv).open(encoding="utf-8", newline="") as f:
            rows = [
                {
                    **r,
                    "scale": int(r["scale"]),
                    "alignment_reviewed": r["alignment_reviewed"].lower() == "true",
                }
                for r in csv.DictReader(f)
            ]
        print(validate_pairs(resolve(a.root), rows, resolve(a.output), "realsr-v3"))
    else:
        print(div2k(resolve(a.root), resolve(a.output), a.synthetic))


if __name__ == "__main__":
    main()
