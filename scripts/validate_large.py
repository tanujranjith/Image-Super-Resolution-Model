"""CPU-only dataset/configuration readiness checks. Never performs optimization."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from superres.common import config, digest, read_json, resolve, write_json  # noqa: E402
from superres.duplicates import verify_split_isolation  # noqa: E402

FOLDER = ROOT / "datasets/large"


def validate_data():
    import torch
    from PIL import Image
    from superres.data import Pairs, modcrop, synthetic_patch
    from superres.models import build_model
    from torchvision.transforms.functional import to_tensor

    torch.set_num_threads(2)
    audit = read_json(FOLDER / "processed/audit.json")
    original = resolve("datasets/processed/div2k.json")
    if digest(original) != audit["original_div2k_manifest_sha256"]:
        raise ValueError("Original DIV2K manifest changed")
    scenes = read_json(FOLDER / "processed/scenes.json")
    seen_paths = set()
    for row in scenes:
        if row["path"] in seen_paths:
            raise ValueError("Repeated retained HR path")
        seen_paths.add(row["path"])
        path = (resolve("datasets/raw") / row["path"]).resolve()
        if not path.is_relative_to(resolve("datasets/raw")) or not path.is_file():
            raise ValueError("Missing or unsafe retained HR path")
        stat = path.stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError("Verified acquired file identity changed")
        if row["source"] == "lsdir-kaggle":
            if row["split"] == "train" and row["official_split"] != "train":
                raise ValueError("Official validation was used for training")
    verify_split_isolation(scenes)
    main_path = FOLDER / "processed/training.json"
    heldout_path = FOLDER / "processed/lsdir-heldout.json"
    if digest(main_path) != audit["training_manifest_sha256"]:
        raise ValueError("Main manifest changed")
    if digest(heldout_path) != audit["heldout_manifest_sha256"]:
        raise ValueError("Held-out manifest changed")
    manifests = [read_json(main_path), read_json(heldout_path)]
    observed = Counter()
    by_scene = {r["scene"]: r for r in scenes}
    for manifest in manifests:
        for row in manifest["pairs"]:
            source = by_scene[row["scene"]]
            assert row["scale"] in (2, 4) and row["lr"] is None
            assert row["hr_sha256"] == source["sha256"]
            assert row["hr"] == source["path"] and row["split"] == source["split"]
            observed[row["scene"], row["scale"]] += 1
    if len(observed) != len(scenes) * 2 or set(observed.values()) != {1}:
        raise ValueError("Expected exactly one x2 and one x4 row per HR scene")
    checks = []
    for scale in (2, 4):
        cfg_path = ROOT / f"configs/edsr_large_x{scale}.yaml"
        cfg = config(cfg_path)
        cfg.update(
            manifest_sha256=digest(main_path),
            benchmarks=[
                {
                    "name": "div2k-official",
                    "manifest": "datasets/processed/div2k.json",
                    "sha256": digest(original),
                },
                {
                    "name": "lsdir-heldout",
                    "manifest": "datasets/large/processed/lsdir-heldout.json",
                    "sha256": digest(heldout_path),
                },
            ],
        )
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        model = build_model(cfg).eval()
        examples = [
            next(r for r in scenes if r["source"] == "div2k" and r["split"] == "train"),
            next(r for r in scenes if r["source"] == "lsdir-kaggle" and r["split"] == "train"),
            next(r for r in scenes if r["source"] == "div2k" and r["split"] == "test"),
            next(r for r in scenes if r["source"] == "lsdir-kaggle" and r["split"] == "test"),
        ]
        for row in examples:
            with Image.open(resolve("datasets/raw") / row["path"]) as image:
                hr = modcrop(image.convert("RGB"), scale)
                low, target = synthetic_patch(hr, scale, 24, 0, 0)
            with torch.inference_mode():
                result = model(to_tensor(low).unsqueeze(0))
            assert result.shape == (1, 3, 24 * scale, 24 * scale)
            assert torch.isfinite(result).all() and target.size == (24 * scale, 24 * scale)
            checks.append({"scale": scale, "scene": row["scene"], "finite": True})
        dataset = Pairs(
            main_path, "train", scale, patch_size=cfg["patch_size"], repeats=cfg["repeats"]
        )
        low, target, _ = dataset[0]
        assert len(dataset) == audit["primary_splits"]["train"]
        assert target.shape[-2:] == (low.shape[-2] * scale, low.shape[-1] * scale)
        del model, dataset, low, target, result
    report = {
        "checks": {
            "independent_exact_and_phash_split_isolation": True,
            "acquired_file_size_and_mtime_unchanged_since_crc_sha_decode_checks": True,
            "official_validation_excluded_from_training": True,
            "both_scales_pair_geometry_and_scene_identity": True,
            "original_div2k_manifest_unchanged": True,
            "cpu_actual_hr_forward_passes": checks,
        },
        "training_started": False,
        "audit_sha256": digest(FOLDER / "processed/audit.json"),
    }
    write_json(FOLDER / "validation-data.json", report)
    print(json.dumps(report, indent=2))


def finalize():
    report = read_json(FOLDER / "validation-data.json")
    if report["audit_sha256"] != digest(FOLDER / "processed/audit.json"):
        raise ValueError("Data audit changed after validation")
    tree = ET.parse(FOLDER / "validation-tests.xml")
    suites = list(tree.getroot().iter("testsuite"))
    failures = sum(
        int(s.attrib.get("failures", 0)) + int(s.attrib.get("errors", 0)) for s in suites
    )
    if not suites or failures:
        raise ValueError("Tests have not passed")
    tests = sum(int(s.attrib["tests"]) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    paths = list((ROOT / "superres").glob("*.py")) + [
        ROOT / "scripts/overnight.py",
        ROOT / "scripts/train-large.ps1",
        ROOT / "scripts/validate_large.py",
        ROOT / "scripts/benchmark_loader.py",
        ROOT / "scripts/validate_loader.py",
        ROOT / "tests/test_loading.py",
    ]
    audit = read_json(FOLDER / "processed/audit.json")
    loader_report_path = ROOT / "runs/loader-validation.json"
    loading_report = read_json(loader_report_path) if loader_report_path.exists() else None
    if loading_report:
        if loading_report["training_executed"] or loading_report["optimizer_steps"] != 0:
            raise ValueError("Loader validation must not execute training")
        for check in loading_report["checks"]:
            if check["config_sha256"] != digest(
                ROOT / f"configs/edsr_large_x{check['scale']}.yaml"
            ):
                raise ValueError("Loader validation predates a config change")
            if check["manifest_sha256"] != digest(FOLDER / "processed/training.json"):
                raise ValueError("Loader validation predates a manifest change")
    previous = read_json(FOLDER / "readiness.json") if (FOLDER / "readiness.json").exists() else {}
    stopped = bool(
        previous.get("user_stop_requested")
        or previous.get("status") == "training_approval_revoked_by_user_stop"
    )
    approval_status = (
        "training_approval_revoked_by_user_stop" if stopped else "ready_for_training_approval"
    )
    readiness = {
        "status": approval_status,
        "user_stop_requested": stopped,
        "implementation_ready": True,
        "approval_required": True,
        "training_started": False,
        "tests_passed": tests - skipped,
        "tests_skipped": skipped,
        "training_manifest_sha256": digest(FOLDER / "processed/training.json"),
        "heldout_manifest_sha256": digest(FOLDER / "processed/lsdir-heldout.json"),
        "original_div2k_manifest_sha256": audit["original_div2k_manifest_sha256"],
        "config_hashes": {
            f"configs/edsr_large_x{s}.yaml": digest(ROOT / f"configs/edsr_large_x{s}.yaml")
            for s in (2, 4)
        },
        "validated_source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        "data_report": audit,
        "checks": report["checks"],
        "loading_validation": loading_report,
    }
    write_json(FOLDER / "readiness.json", readiness)
    status = read_json(ROOT / "TRAINING_STATUS.json")
    status.update(
        training="not_started",
        approval_required=True,
        trained_weights_present=False,
        tests_passed=tests - skipped,
        tests_skipped=skipped,
    )
    status["large_dataset"] = {
        "status": approval_status,
        "readiness": "datasets/large/readiness.json",
        "unique_training_hr": audit["primary_splits"]["train"],
        "training_started": False,
    }
    if "conditional_approval" in status:
        status["historical_conditional_approval"] = status.pop("conditional_approval") | {
            "outcome": "expired_unused"
        }
    status["future_training"] = (
        "Fresh approval required. User removed the previous 5 AM cutoff for future approved runs."
    )
    write_json(ROOT / "TRAINING_STATUS.json", status)
    print(
        json.dumps(
            {
                "status": readiness["status"],
                "tests_passed": tests - skipped,
                "tests_skipped": skipped,
                "training_started": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    finalize() if args.finalize else validate_data()
