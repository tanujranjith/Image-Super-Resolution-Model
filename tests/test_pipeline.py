import math
import random
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from superres.acquire import extract
from superres.common import atomic_save, config, read_json, restore_rng, rng_state, seed_all
from superres.data import Pairs, degrade, loader
from superres.export import export
from superres.inference import Upscaler, tiled
from superres.metrics import psnr_ssim, psnr_y
from superres.models import build_model
from superres.prepare import derive_synthetic, div2k, validate_pairs
from superres.train import ReconstructionLoss


@pytest.mark.parametrize("architecture", ["compact", "edsr", "bicubic"])
@pytest.mark.parametrize("scale", [2, 4])
def test_dimensions_and_tiles(architecture, scale):
    model = build_model(
        {"architecture": architecture, "scale": scale, "features": 8, "blocks": 2}
    ).eval()
    x = torch.rand(1, 3, 29, 35)
    with torch.inference_mode():
        direct = model(x)
    tile = tiled(model, x, "cpu", tile_size=11)
    assert direct.shape == (1, 3, 29 * scale, 35 * scale)
    torch.testing.assert_close(tile, direct, rtol=1e-4, atol=2e-5)
    with pytest.raises(ValueError, match="Halo"):
        tiled(model, x, "cpu", 11, halo=0)


@pytest.mark.parametrize("name", ["edsr_x2", "edsr_x4", "compact_x2", "compact_x4", "realistic_x4"])
def test_configs(name):
    assert config(f"configs/{name}.yaml")["scale"] in (2, 4)


def fixture_pairs(tmp_path):
    rng = np.random.default_rng(9)
    root = tmp_path / "images with spaces"
    for section, ids in (("train", [1, 751]), ("valid", [801])):
        for n in ids:
            hr = Image.fromarray(rng.integers(0, 256, (80, 96, 3), dtype=np.uint8))
            path = root / f"DIV2K_{section}_HR/{n:04}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            hr.save(path)
            for scale in (2, 4):
                lr = degrade(hr, scale)
                path = root / f"DIV2K_{section}_LR_bicubic/X{scale}/{n:04}x{scale}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                lr.save(path)
    manifest = tmp_path / "manifest.json"
    div2k(root, manifest, allow_partial=True)
    return manifest


def test_pairs_patch_alignment_and_eval(tmp_path):
    manifest = fixture_pairs(tmp_path)
    seed_all(2)
    train = Pairs(manifest, "train", 2, patch_size=16, repeats=2)
    lr, hr, scene = train[0]
    assert lr.shape == (3, 16, 16) and hr.shape == (3, 32, 32) and scene == "0001"
    val = Pairs(manifest, "test", 4)
    first, second = val[0], val[0]
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert len(next(iter(loader(val, {"batch_size": 2, "workers": 0})))) == 3
    with pytest.raises(ValueError, match="HR-only"):
        Pairs(manifest, "train", 2, degradation="realistic-synthetic")


def test_missing_pair_fails(tmp_path):
    m = fixture_pairs(tmp_path)
    root = (m.parent / read_json(m)["root"]).resolve()
    (root / "DIV2K_train_LR_bicubic/X2/0001x2.png").unlink()
    with pytest.raises(FileNotFoundError):
        div2k(root, tmp_path / "bad.json", allow_partial=True)


def test_realsr_equal_dimensions_and_unreviewed_fail(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    Image.new("RGB", (64, 64)).save(root / "hr.png")
    Image.new("RGB", (64, 64)).save(root / "lr.png")
    rows = [{"scene": "1", "scale": 2, "split": "train", "hr": "hr.png", "lr": "lr.png"}]
    with pytest.raises(ValueError, match="geometry"):
        validate_pairs(root, rows, tmp_path / "bad.json", "realsr-v3")


def test_metrics_identity_and_known_mse():
    a = torch.full((1, 3, 40, 40), 0.5)
    p, s = psnr_ssim(a, a, 2)
    assert math.isinf(p) and abs(s - 1) < 1e-5
    b = a + 0.1
    p, _ = psnr_ssim(a, b, 0, y_channel=False)
    # Quantized .5 is128/255 and .6 is153/255: difference25/255.
    assert abs(p - (-20 * math.log10(25 / 255))) < 1e-4
    with pytest.raises(ValueError, match="too small"):
        psnr_ssim(a[..., :8, :8], a[..., :8, :8], 4)
    assert abs(psnr_y(a, b, 2) - psnr_ssim(a, b, 2)[0]) < 1e-6


def test_synthetic_conditions_recorded_separately(tmp_path):
    manifest = fixture_pairs(tmp_path)
    source = tmp_path / "synthetic.json"
    assert derive_synthetic(manifest, source) == 6
    assert read_json(source)["source_manifest_sha256"]
    manifest = source
    assert (
        Pairs(manifest, "train", 2, degradation="realistic-synthetic").condition
        == "realistic-synthetic"
    )
    assert Pairs(manifest, "train", 2).condition == "pillow-bicubic-synthetic"


def test_synthetic_degradation_deterministic():
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (80, 80, 3), dtype=np.uint8))
    a = degrade(image, 4, "realistic-synthetic", random.Random(12))
    b = degrade(image, 4, "realistic-synthetic", random.Random(12))
    assert a.size == (20, 20) and np.array_equal(a, b)


@pytest.mark.parametrize("kind", ["l1", "mse", "charbonnier"])
def test_losses_forward_only(kind):
    loss = ReconstructionLoss(kind)(torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8))
    assert torch.isfinite(loss) and loss > 0


def test_checkpoint_export_inference_fixture(tmp_path):
    # Random tensors exist only inside temporary test folder, never deployment artifacts.
    cfg = {"architecture": "compact", "scale": 2, "features": 8, "blocks": 2}
    model = build_model(cfg).eval()
    checkpoint = tmp_path / "fixture.pt"
    atomic_save(
        checkpoint,
        {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": 0,
            "condition": "synthetic-test-fixture",
            "manifest_sha256": "fixture",
        },
    )
    destination = tmp_path / "export"
    export(checkpoint, destination)
    upscaler = Upscaler(destination, "cpu")
    assert upscaler.upscale(Image.new("RGB", (24, 32))).size == (48, 64)
    seed_all(2)
    state = rng_state()
    expected = torch.rand(2)
    restore_rng(state)
    torch.testing.assert_close(torch.rand(2), expected)


def test_missing_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="not available"):
        Upscaler(tmp_path)


def test_zip_traversal(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract(path, tmp_path / "out")


def test_ui_constructs_and_missing_input():
    from app import build_app, enhance

    assert build_app().config["components"]
    with pytest.raises(Exception, match="Upload"):
        enhance(None, 2)


@pytest.mark.parametrize("scale", [2, 4])
def test_ui_missing_weights_uses_isolated_artifact_directory(tmp_path, monkeypatch, scale):
    # The repository now ships real exports. Never make this negative-path test
    # depend on their absence or on a model cached by another test.
    from app import enhance, upscaler

    monkeypatch.setenv("SR_ARTIFACTS", str(tmp_path / "missing-models"))
    upscaler.cache_clear()
    try:
        with pytest.raises(Exception, match="not available"):
            enhance(Image.new("RGB", (32, 32)), scale)
    finally:
        upscaler.cache_clear()


def test_cuda_forward_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = (
        build_model({"architecture": "edsr", "scale": 4, "features": 8, "blocks": 2}).cuda().eval()
    )
    with torch.inference_mode(), torch.autocast("cuda"):
        output = model(torch.rand(1, 3, 16, 16, device="cuda"))
    assert output.shape == (1, 3, 64, 64) and torch.isfinite(output).all()
