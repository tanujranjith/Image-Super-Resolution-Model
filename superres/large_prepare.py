"""Large HR-only corpus with scene-level duplicate isolation and protected benchmarks."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

from PIL import Image

from .common import digest, read_json, resolve, write_json
from .duplicates import group_duplicates
from .large_acquire import EXPECTED_HR, perceptual_hashes


def partition(groups, seed=42):
    """Protect original held-out scenes, then isolate official LSDIR validation."""
    main, heldout, excluded, reserve = [], [], [], []
    rank = {"train": 0, "val": 1, "test": 2}
    for group in groups:
        old = [r for r in group if r["source"] == "div2k"]
        new = [r for r in group if r["source"] == "lsdir-kaggle"]
        identifier = hashlib.sha256(
            "\n".join(sorted(r["path"] for r in group)).encode()
        ).hexdigest()
        if old:
            split = max((r["split"] for r in old), key=rank.get)
            main.extend(r | {"group": identifier} for r in old if r["split"] == split)
            excluded.extend(
                r | {"reason": "protected DIV2K duplicate group"}
                for r in old
                if r["split"] != split
            )
            excluded.extend(r | {"reason": "LSDIR group overlaps protected DIV2K"} for r in new)
            continue
        official_validation = any(r["official_split"] == "validation" for r in new)
        kept, seen = [], set()
        for row in sorted(new, key=lambda r: (r["official_split"] != "validation", r["path"])):
            if official_validation and row["official_split"] != "validation":
                excluded.append(row | {"reason": "official LSDIR validation duplicate"})
            elif row["sha256"] in seen:
                excluded.append(row | {"reason": "byte-identical duplicate"})
            else:
                seen.add(row["sha256"])
                kept.append(row | {"group": identifier})
        if official_validation:
            reserve.append(kept)
        else:
            main.extend(r | {"split": "train"} for r in kept)
    reserve.sort(key=lambda g: min(r["path"] for r in g))
    random.Random(seed).shuffle(reserve)
    boundary = len(reserve) // 2
    for i, group in enumerate(reserve):
        heldout.extend(r | {"split": "val" if i < boundary else "test"} for r in group)
    return main, heldout, excluded


def manifest_for(scenes, output, original_sha):
    pairs = []
    for row in scenes:
        for scale in (2, 4):
            pairs.append(
                {
                    "scene": row["scene"],
                    "source": row["source"],
                    "scale": scale,
                    "split": row["split"],
                    "hr": row["path"],
                    "lr": None,
                    "hr_sha256": row["sha256"],
                    "lr_sha256": None,
                    "hr_size": [row["width"], row["height"]],
                    "group": row["group"],
                }
            )
    result = {
        "schema": 1,
        "root": Path(os.path.relpath(resolve("datasets/raw"), Path(output).parent)).as_posix(),
        "condition": "pillow-bicubic-synthetic",
        "pairs": pairs,
        "counts": dict(Counter(f"x{r['scale']}/{r['split']}" for r in pairs)),
        "source_div2k_manifest_sha256": original_sha,
        "note": "HR-only training; Pillow bicubic is distinct from official MATLAB pairs and real camera degradation.",
    }
    write_json(output, result)
    return result


def prepare():
    folder = resolve("datasets/large")
    folder.mkdir(parents=True, exist_ok=True)
    journal = {}
    for line in (folder / "images.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
            journal[row["source_path"]] = row
        except json.JSONDecodeError:
            continue
    if len(journal) != EXPECTED_HR:
        raise ValueError(
            "Acquisition is incomplete; no partial training manifest will be generated"
        )
    source = resolve("datasets/processed/div2k.json")
    old = read_json(source)
    unique = {r["scene"]: r for r in old["pairs"]}
    raw = (source.parent / old["root"]).resolve()
    references = folder / "div2k-reference-hashes.json"
    cached = read_json(references) if references.exists() else {}
    old_rows = []
    for scene, row in sorted(unique.items()):
        path = (raw / row["hr"]).resolve()
        if not path.is_relative_to(resolve("datasets/raw")):
            raise ValueError("Original scene outside independent dataset root")
        info = cached.get(scene)
        stat = path.stat()
        if (
            not info
            or info.get("sha256") != row["hr_sha256"]
            or info.get("bytes") != stat.st_size
            or info.get("mtime_ns") != stat.st_mtime_ns
        ):
            if digest(path) != row["hr_sha256"]:
                raise ValueError("Original HR bytes changed")
            with Image.open(path) as im:
                image = im.convert("RGB")
                hashes = perceptual_hashes(image)
                info = {
                    "sha256": row["hr_sha256"],
                    "width": image.width,
                    "height": image.height,
                    "phash": hashes[0],
                    "phash_hflip": hashes[1],
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            cached[scene] = info
        old_rows.append(
            info
            | {
                "path": path.relative_to(resolve("datasets/raw")).as_posix(),
                "scene": "div2k-" + scene,
                "source": "div2k",
                "split": row["split"],
            }
        )
    write_json(references, cached)
    new = []
    for row in journal.values():
        if row["status"] != "verified":
            continue
        path = resolve("datasets/raw/LSDIR-kaggle") / row["path"]
        stat = path.stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError("Acquired HR cache identity changed; rerun acquisition to reverify")
        new.append(row | {"path": "LSDIR-kaggle/" + row["path"]})
    print(
        json.dumps(
            {"event": "duplicate_audit", "div2k_scenes": len(old_rows), "lsdir_scenes": len(new)}
        ),
        flush=True,
    )
    groups = group_duplicates(old_rows + new, 6)
    main, heldout, excluded = partition(groups)
    if not heldout or not any(r["source"] == "lsdir-kaggle" for r in main):
        raise ValueError("No usable large training/held-out data")
    group_splits = {}
    for row in main + heldout:
        group_splits.setdefault(row["group"], set()).add(row["split"])
    if any(len(v) > 1 for v in group_splits.values()):
        raise ValueError("Perceptual component spans multiple splits")
    output = folder / "processed"
    output.mkdir(parents=True, exist_ok=True)
    manifest_for(main, output / "training.json", digest(source))
    manifest_for(heldout, output / "lsdir-heldout.json", digest(source))
    write_json(output / "scenes.json", main + heldout)
    with (output / "quarantine.jsonl").open("w", encoding="utf-8") as stream:
        for row in excluded:
            stream.write(json.dumps(row) + "\n")
    report = {
        "phase": "data_prepared",
        "training_started": False,
        "kaggle_ref": "parampandya0310/hr-image-denoising-lsdir",
        "kaggle_version": 1,
        "acquired_scenes": len(journal),
        "verified_lsdir_scenes": len(new),
        "primary_splits": dict(Counter(r["split"] for r in main)),
        "primary_training_sources": dict(
            Counter(r["source"] for r in main if r["split"] == "train")
        ),
        "lsdir_heldout_splits": dict(Counter(r["split"] for r in heldout)),
        "distinct_retained_scenes": len(main) + len(heldout),
        "duplicate_groups": len(group_splits),
        "excluded_scenes": len(excluded),
        "exclusion_reasons": dict(Counter(r["reason"] for r in excluded)),
        "training_manifest_sha256": digest(output / "training.json"),
        "heldout_manifest_sha256": digest(output / "lsdir-heldout.json"),
        "original_div2k_manifest_sha256": digest(source),
        "raw_hr_bytes": sum(r["bytes"] for r in new),
        "condition": "pillow-bicubic-synthetic",
        "checks": {
            "whole_scene_groups_isolated": True,
            "official_lsdir_validation_not_used_for_training": True,
            "original_manifests_unmodified": True,
        },
        "limitations": "Exact/full/mirror perceptual hashes cannot prove absence of arbitrary crops or repeated scenes. "
        "New data is clean HR with synthetic bicubic LR, not real camera LR. "
        "Kaggle mirror license does not override original academic research terms. "
        "Mirror validation provenance is not independently authenticated; it is not an official hidden-test benchmark.",
    }
    write_json(output / "audit.json", report)
    write_json(folder / "status.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
