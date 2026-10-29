"""CPU-only fixtures for acquisition, scene isolation and on-demand degradation."""

import io
import random
import sqlite3
import struct
import zipfile

import numpy as np
from PIL import Image, ImageOps
import pytest
import torch

from scripts.overnight import eligibility
from superres.common import config, digest, write_json
from superres.data import Pairs, degrade, synthetic_patch
from superres.duplicates import HammingIndex, group_duplicates, verify_split_isolation
from superres.evaluate import evaluation_manifest
from superres.large_acquire import (
    audit_bytes,
    build_index,
    intervals,
    member_role,
    unpack_member,
    zip64_values,
)
from superres.large_prepare import partition


def test_archive_roles_do_not_count_lr_variants_as_hr_scenes():
    assert member_role("shard-05/0030000/0030000.png") == "train"
    assert member_role("val1/val1/HR/val/0000001.png") == "validation"
    for scale in (2, 3, 4):
        assert member_role(f"val1/val1/X{scale}/val/0000001x{scale}.png") == "provided-lr"
    with pytest.raises(ValueError, match="Unexpected"):
        member_role("val1/val1/X2/val/0000001x4.png")


@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("point", [(0, 0), (1, 7), (14, 11), (40, 32), (64, 48), (64, 0), (0, 48)])
def test_context_crop_matches_full_image_bicubic(scale, point):
    hr = Image.fromarray(
        np.random.default_rng(7).integers(0, 256, (64 * scale, 80 * scale, 3), dtype=np.uint8)
    )
    x, y = point
    low, target = synthetic_patch(hr, scale, 16, x, y)
    expected = degrade(hr, scale).crop((x, y, x + 16, y + 16))
    np.testing.assert_array_equal(low, expected)
    np.testing.assert_array_equal(
        target, hr.crop((x * scale, y * scale, (x + 16) * scale, (y + 16) * scale))
    )


def test_context_crop_rejects_invalid_coordinates():
    with pytest.raises(ValueError, match="outside"):
        synthetic_patch(Image.new("RGB", (64, 64)), 4, 16, 1, 0)
    with pytest.raises(ValueError, match="modcropped"):
        synthetic_patch(Image.new("RGB", (65, 64)), 4, 8, 0, 0)


def test_hr_only_pair_has_deterministic_evaluation_and_aligned_training(tmp_path):
    image = Image.fromarray(
        np.random.default_rng(4).integers(0, 256, (260, 264, 3), dtype=np.uint8)
    )
    image.save(tmp_path / "hr.png")
    manifest = tmp_path / "manifest with spaces.json"
    write_json(
        manifest,
        {
            "root": ".",
            "condition": "pillow-bicubic-synthetic",
            "pairs": [
                {"scene": "fixture", "scale": 4, "hr": "hr.png", "lr": None, "split": s}
                for s in ("train", "test")
            ],
        },
    )
    random.seed(5)
    low, high, _ = Pairs(manifest, "train", 4, patch_size=32, repeats=1)[0]
    assert low.shape == (3, 32, 32) and high.shape == (3, 128, 128)
    evaluation = Pairs(manifest, "test", 4)
    torch.testing.assert_close(evaluation[0][0], evaluation[0][0])
    assert len(evaluation) == 1


def test_zip_member_crc_and_memoryview(tmp_path):
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("shard-00/0000001.png", b"verified pixels")
    payload = raw.getvalue()
    start = payload.index(b"PK\x01\x02")
    end = payload.index(b"PK\x05\x06")
    central = tmp_path / "central.bin"
    central.write_bytes(payload[start:end])
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    assert build_index(central, db) == 1
    row = db.execute("select * from entries").fetchone()
    assert unpack_member(row, memoryview(payload)) == b"verified pixels"
    damaged = bytearray(payload)
    damaged[start - 1] ^= 1
    with pytest.raises(zipfile.BadZipFile):
        unpack_member(row, damaged)
    with pytest.raises(ValueError, match="Short"):
        unpack_member(row, b"bad")


def test_zip64_sizes_and_range_coalescing():
    entry = [0] * 17
    entry[9] = entry[8] = entry[16] = 0xFFFFFFFF
    extra = struct.pack("<HHQQQ", 1, 24, 123, 100, 2**33)
    assert zip64_values(entry, extra) == (123, 100, 2**33, 0)
    with pytest.raises(ValueError, match="missing"):
        zip64_values(entry, b"")
    rows = [{"offset": n * 1000, "name": "a.png", "compressed": 500} for n in range(20)]
    groups = intervals(rows, maximum=10000)
    assert sum(len(group) for _, _, group in groups) == 20
    assert all(end - start <= 10000 for start, end, _ in groups)


def test_image_validation_and_mirror_hash():
    image = Image.fromarray(
        np.random.default_rng(9).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    )

    def packed(im):
        stream = io.BytesIO()
        im.save(stream, format="PNG")
        return stream.getvalue()

    info = audit_bytes(packed(image))
    mirror = audit_bytes(packed(ImageOps.mirror(image)))
    assert info["phash"] == mirror["phash_hflip"]
    assert info["phash_hflip"] == mirror["phash"]
    with pytest.raises(ValueError, match="size"):
        audit_bytes(packed(Image.new("RGB", (32, 32))))


def scene(name, source="lsdir-kaggle", split="train", sha=None, phash="aaaaaaaaaaaaaaaa"):
    row = {
        "path": name,
        "scene": name,
        "source": source,
        "sha256": sha or name,
        "phash": phash,
        "phash_hflip": phash,
    }
    row["split" if source == "div2k" else "official_split"] = split
    return row


def test_partition_protects_benchmarks_and_official_validation():
    groups = [
        [scene("old-test", "div2k", "test"), scene("new-overlap")],
        [scene("official", "lsdir-kaggle", "validation"), scene("leaky-train")],
        [scene("other-official", "lsdir-kaggle", "validation")],
        [scene("old-val", "div2k", "val"), scene("old-train", "div2k", "train")],
        [scene("new-train"), scene("exact-copy", sha="new-train")],
    ]
    main, held, excluded = partition(groups)
    assert {r["scene"] for r in main} == {"old-test", "old-val", "exact-copy"}
    assert {r["split"] for r in held} == {"val", "test"}
    assert all(r["official_split"] == "validation" for r in held)
    assert len(excluded) == 4
    assert partition(groups) == partition(groups)


def test_hamming_index_matches_bruteforce_and_groups_mirrors():
    rng = random.Random(5)
    values = [rng.getrandbits(64) for _ in range(200)]
    index = HammingIndex(6)
    for i, value in enumerate(values):
        index.add(value, i)
    for value in values[:15]:
        query = value ^ 31
        assert set(index.query(query)) == {
            i for i, v in enumerate(values) if (v ^ query).bit_count() <= 6
        }
    a = scene("a", phash="0123456789abcdef")
    b = scene("b", phash="ffffffffffffffff")
    b["phash_hflip"] = a["phash"]
    assert len(group_duplicates([a, b], 6)) == 1


def test_independent_split_audit_detects_mirrors_and_exact_duplicates():
    train = scene("train", "div2k", "train", phash="0123456789abcdef")
    held = scene("held", "div2k", "test", phash="ffffffffffffffff")
    verify_split_isolation([train, held], batch_size=1)
    held["phash_hflip"] = train["phash"]
    with pytest.raises(ValueError, match="perceptual"):
        verify_split_isolation([train, held], batch_size=1)
    held["phash_hflip"] = held["phash"]
    held["sha256"] = train["sha256"]
    with pytest.raises(ValueError, match="Byte-identical"):
        verify_split_isolation([train, held])


def test_pinned_external_benchmark_cannot_replace_training_manifest(tmp_path):
    training = tmp_path / "train.json"
    training.write_text("original")
    other = tmp_path / "test.json"
    other.write_text("held out")
    state = {
        "manifest_sha256": digest(training),
        "config": {
            "manifest": str(training),
            "benchmarks": [{"manifest": str(other), "sha256": digest(other)}],
        },
    }
    assert evaluation_manifest(state) == training
    with pytest.raises(ValueError, match="requires"):
        evaluation_manifest(state, other)
    assert evaluation_manifest(state, other, benchmark=True) == other
    other.write_text("changed")
    with pytest.raises(ValueError, match="requires"):
        evaluation_manifest(state, other, benchmark=True)
    training.write_text("changed")
    with pytest.raises(ValueError, match="Training"):
        evaluation_manifest(state)


def test_training_rejects_changed_prepared_manifest_before_creating_run(tmp_path):
    from superres.train import train

    manifest = tmp_path / "changed.json"
    manifest.write_text("changed")
    destination = tmp_path / "must not create"
    with pytest.raises(ValueError, match="Prepared manifest fingerprint changed"):
        train(
            {
                "architecture": "edsr",
                "seed": 42,
                "device": "cpu",
                "manifest": str(manifest),
                "manifest_sha256": "0" * 64,
                "output": str(destination),
            }
        )
    assert not destination.exists()


def test_gate_without_expired_one_night_deadline():
    from datetime import datetime, timezone

    current = datetime(2026, 10, 18, 12, tzinfo=timezone.utc)
    assert (
        eligibility({"phase": "complete", "completed_utc": current.isoformat()}, current) == "start"
    )
    assert eligibility({"phase": "training"}, current) == "wait"
    assert eligibility({"phase": "failed"}, current) == "car_failed"


def test_queue_cannot_accept_stale_completed_car_run(tmp_path):
    from datetime import datetime, timezone

    current = datetime.now(timezone.utc)
    expected = tmp_path / "expanded-run"
    car = {
        "phase": "complete",
        "completed_utc": current.isoformat(),
        "experiment": str(tmp_path / "old-run"),
    }
    assert eligibility(car, current, expected_experiment=expected) == "wait"
    car["experiment"] = str(expected)
    assert eligibility(car, current, expected_experiment=expected) == "start"
    car["experiment"] += "-smaller-batch"
    assert eligibility(car, current, expected_experiment=expected) == "start"


@pytest.mark.parametrize("scale", [2, 4])
def test_large_configs_have_single_repeat_and_independent_outputs(scale):
    cfg = config(f"configs/edsr_large_x{scale}.yaml")
    assert cfg["repeats"] == 1 and cfg["workers"] == 0 and cfg["scale"] == scale
    assert cfg["manifest"] == "datasets/large/processed/training.json"
    assert "large" in cfg["output"] and cfg["perceptual_weight"] == 0
