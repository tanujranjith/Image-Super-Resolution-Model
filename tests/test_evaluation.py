"""Temporary fixtures prove comparison generation, not trained restoration quality."""

import numpy as np
from PIL import Image

from superres.common import atomic_save, digest, write_json
from superres.data import degrade
from superres.evaluate import evaluate
from superres.models import build_model


def test_evaluation_outputs_real_comparisons_for_fixture(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    hr = Image.fromarray(np.random.default_rng(4).integers(0, 256, (48, 48, 3), dtype=np.uint8))
    hr.save(root / "hr.png")
    degrade(hr, 2).save(root / "lr.png")
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "root": "images",
            "condition": "temporary-test-fixture",
            "pairs": [
                {"scene": "fixture", "scale": 2, "split": "test", "hr": "hr.png", "lr": "lr.png"}
            ],
        },
    )
    cfg = {
        "architecture": "compact",
        "scale": 2,
        "features": 8,
        "blocks": 2,
        "manifest": str(manifest),
        "workers": 0,
        "batch_size": 1,
    }
    checkpoint = tmp_path / "random-fixture.pt"
    atomic_save(
        checkpoint,
        {
            "config": cfg,
            "model": build_model(cfg).state_dict(),
            "manifest_sha256": digest(manifest),
        },
    )
    output = tmp_path / "evaluation"
    summary = evaluate(checkpoint=checkpoint, output=output, tile_size=24, device_name="cpu")
    assert summary["images"] == 1 and summary["condition"] == "temporary-test-fixture"
    assert (output / "comparison-fixture.png").exists() and (output / "metrics.json").exists()
