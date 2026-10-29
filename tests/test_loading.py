"""Loading-only checks: no optimizer, backward pass or trained artifact creation."""

import random
import threading

import numpy as np
from PIL import Image
import pytest
import torch

from superres.common import write_json
from superres.data import Pairs, degrade, loader, modcrop
from superres.prefetch import PrefetchLoader


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for i in range(6):
        image = Image.fromarray(
            np.random.default_rng(i).integers(0, 256, (140, 156, 3), dtype=np.uint8)
        )
        image.save(tmp_path / f"image {i}.png")
        for scale in (2, 4):
            low = degrade(image, scale)
            low.save(tmp_path / f"low {i} x{scale}.png")
            for split in ("train", "val"):
                rows.append(
                    {
                        "scene": f"scene-{i}",
                        "hr": f"image {i}.png",
                        "lr": None,
                        "scale": scale,
                        "split": split,
                    }
                )
    path = tmp_path / "manifest with spaces.json"
    write_json(path, {"root": ".", "condition": "pillow-bicubic-synthetic", "pairs": rows})
    return path


def settings(**updates):
    return {
        "batch_size": 2,
        "workers": 0,
        "decode_threads": 2,
        "prefetch_batches": 2,
        "sampling": "epoch-index-v1",
        "max_decode_pixels": 16000000,
    } | updates


def batches(path, scale=2, epoch=0, cfg=None, state=None, degradation="bicubic"):
    data = Pairs(path, "train", scale, patch_size=12, repeats=2, degradation=degradation)
    data.set_epoch(epoch)
    generator = torch.Generator().manual_seed(17)
    if state is not None:
        generator.set_state(state)
    result = list(loader(data, cfg or settings(), True, generator))
    assert data._decoder is None
    return result, generator.get_state()


def assert_batches_identical(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[2] == y[2]
        assert torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])


@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("condition", ["bicubic", "realistic-synthetic"])
def test_serial_threaded_pixels_order_and_sampler_state_match(manifest, scale, condition):
    serial, serial_state = batches(
        manifest, scale, cfg=settings(decode_threads=0, prefetch_batches=0), degradation=condition
    )
    threaded, threaded_state = batches(manifest, scale, degradation=condition)
    assert_batches_identical(serial, threaded)
    assert torch.equal(serial_state, threaded_state)


def test_epoch_boundary_resume_replays_same_patches_and_order(manifest):
    first, saved = batches(manifest)
    second, final = batches(manifest, epoch=1, state=saved)
    restored, restored_final = batches(manifest, epoch=1, state=saved.clone())
    assert_batches_identical(second, restored)
    assert torch.equal(final, restored_final)
    assert not torch.equal(first[0][0], second[0][0])


@pytest.mark.parametrize("scale", [2, 4])
def test_high_throughput_preserves_resume_pixels_order_and_rng(manifest, monkeypatch, scale):
    monkeypatch.setenv("SR_THROUGHPUT_PROFILE", "standard")
    _, saved = batches(manifest, scale)
    normal, final = batches(manifest, scale, epoch=1, state=saved.clone())
    cfg = settings()
    original = cfg.copy()
    monkeypatch.setenv("SR_THROUGHPUT_PROFILE", "high")
    high, high_final = batches(manifest, scale, epoch=1, state=saved.clone(), cfg=cfg)
    assert_batches_identical(normal, high)
    assert torch.equal(final, high_final)
    assert cfg == original
    data = Pairs(manifest, "train", scale, patch_size=12)
    stream = loader(data, cfg, True)
    assert data.decode_threads == 4 and stream.batches == 4
    validation = Pairs(manifest, "val", scale)
    assert not isinstance(loader(validation, cfg), PrefetchLoader)
    assert validation.decode_threads == 0


def test_local_sampling_does_not_consume_model_or_global_python_rng(manifest):
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    batches(manifest)
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_official_pairs_remain_aligned_and_identical(manifest):
    info = __import__("json").loads(manifest.read_text())
    for row in info["pairs"]:
        i = row["scene"].split("-")[-1]
        row["lr"] = f"low {i} x{row['scale']}.png"
    info["condition"] = "official-fixture"
    write_json(manifest, info)
    a, _ = batches(manifest, scale=4)
    b, _ = batches(manifest, scale=4, cfg=settings(decode_threads=0, prefetch_batches=0))
    assert_batches_identical(a, b)
    assert a[0][1].shape[-2:] == (48, 48)


def test_validation_is_full_resolution_and_never_thread_prefetched(manifest):
    data = Pairs(manifest, "val", 4)
    stream = loader(data, settings())
    assert not isinstance(stream, PrefetchLoader)
    assert data.decode_threads == 0
    low, high, _ = next(iter(stream))
    assert low.shape == (1, 3, 35, 39) and high.shape == (1, 3, 140, 156)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"workers": 2}, "workers=0"),
        ({"decode_threads": 8}, "decode_threads"),
        ({"prefetch_batches": -1}, "prefetch_batches"),
        ({"decode_threads": True}, "decode_threads"),
        ({"sampling": "global-rng"}, "epoch-index-v1"),
        ({"max_decode_pixels": 0}, "max_decode_pixels"),
    ],
)
def test_invalid_or_unbounded_settings_rejected(manifest, updates, match):
    with pytest.raises(ValueError, match=match):
        loader(Pairs(manifest, "train", 2), settings(**updates), True)


def test_decode_limit_checked_and_exceptions_release_threads(manifest):
    data = Pairs(manifest, "train", 2, patch_size=12)
    stream = loader(data, settings(max_decode_pixels=10), True)
    with pytest.raises(ValueError, match="pixel limit"):
        list(stream)
    assert data._decoder is None and not stream._active
    assert not any(t.name.startswith(("sr-decode", "sr-prefetch")) for t in threading.enumerate())


def test_early_consumer_exit_and_next_epoch_cleanup(manifest):
    data = Pairs(manifest, "train", 2, patch_size=12)
    stream = loader(data, settings(), True, torch.Generator().manual_seed(5))
    iterator = iter(stream)
    next(iterator)
    iterator.close()
    assert data._decoder is None and not stream._active
    assert not any(t.name.startswith(("sr-decode", "sr-prefetch")) for t in threading.enumerate())
    data.set_epoch(1)
    assert len(list(stream)) == len(stream)


def test_prefetch_overlaps_consumer_but_is_bounded():
    produced = []
    reached_bound = threading.Event()

    class Source:
        dataset = None

        def __init__(self):
            self.dataset = self

        def __iter__(self):
            for i in range(20):
                produced.append(i)
                if len(produced) == 3:
                    reached_bound.set()
                yield i

        def close(self):
            self.closed = True

    source = Source()
    iterator = iter(PrefetchLoader(source, 2))
    assert next(iterator) == 0
    assert reached_bound.wait(timeout=5)
    # Consumer holds one, with only two additional batches allowed ahead.
    assert len(produced) == 3
    iterator.close()
    assert source.closed


def test_decoder_concurrency_and_sampler_order_are_bounded(manifest, monkeypatch):
    data = Pairs(manifest, "train", 2)
    data.decode_threads = 2
    active = peak = 0
    lock = threading.Lock()
    together = threading.Barrier(2)

    def sample(_self, index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        together.wait(timeout=5)
        with lock:
            active -= 1
        return index

    monkeypatch.setattr(Pairs, "__getitem__", sample)
    try:
        assert data.__getitems__([3, 1, 2, 0]) == [3, 1, 2, 0]
        assert peak == 2
    finally:
        data.close()


def test_aligned_modcrop_reuses_buffer_and_metadata_drops_unused_rows(manifest):
    image = Image.new("RGB", (64, 64))
    assert modcrop(image, 4) is image
    data = Pairs(manifest, "val", 4)
    assert "pairs" not in data.info and len(data.rows) == 6


def test_readiness_revalidation_preserves_user_stop(tmp_path, monkeypatch):
    from scripts import validate_large
    from superres.common import digest, read_json

    folder = tmp_path / "datasets/large"
    (folder / "processed").mkdir(parents=True)
    audit = {"original_div2k_manifest_sha256": "original-fixture", "primary_splits": {"train": 6}}
    write_json(folder / "processed/audit.json", audit)
    write_json(
        folder / "validation-data.json",
        {"audit_sha256": digest(folder / "processed/audit.json"), "checks": {"fixture": True}},
    )
    write_json(
        folder / "readiness.json",
        {"status": "training_approval_revoked_by_user_stop", "user_stop_requested": True},
    )
    write_json(tmp_path / "TRAINING_STATUS.json", {"large_dataset": {}})
    (folder / "validation-tests.xml").write_text(
        '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0" /></testsuites>'
    )
    for name in (
        "datasets/large/processed/training.json",
        "datasets/large/processed/lsdir-heldout.json",
        "scripts/overnight.py",
        "scripts/train-large.ps1",
        "scripts/validate_large.py",
        "scripts/benchmark_loader.py",
        "scripts/validate_loader.py",
        "tests/test_loading.py",
        "configs/edsr_large_x2.yaml",
        "configs/edsr_large_x4.yaml",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    monkeypatch.setattr(validate_large, "ROOT", tmp_path)
    monkeypatch.setattr(validate_large, "FOLDER", folder)
    validate_large.finalize()
    ready = read_json(folder / "readiness.json")
    assert ready["status"] == "training_approval_revoked_by_user_stop"
    assert ready["approval_required"] and ready["user_stop_requested"]
    assert ready["implementation_ready"] and not ready["training_started"]
    assert read_json(tmp_path / "TRAINING_STATUS.json")["approval_required"]
