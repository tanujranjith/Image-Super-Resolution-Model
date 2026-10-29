"""Four actual HR samples through the configured loader/models, CPU forward only."""

import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from superres.common import config, digest, resolve, write_json  # noqa: E402
from superres.data import Pairs, loader  # noqa: E402
from superres.models import build_model  # noqa: E402


def main():
    torch.set_num_threads(2)
    checks = []
    for scale in (2, 4):
        path = ROOT / f"configs/edsr_large_x{scale}.yaml"
        cfg = config(path)
        outputs = []
        for threaded in (False, True):
            dataset = Pairs(cfg["manifest"], "train", scale, cfg["patch_size"], repeats=1)
            dataset.rows = [
                next(row for row in dataset.rows if row["scene"].startswith(source))
                for source in ("div2k-", "lsdir-")
            ]
            dataset.set_epoch(5)
            settings = cfg | {"batch_size": 2}
            if not threaded:
                settings.update(decode_threads=0, prefetch_batches=0)
            result = list(loader(dataset, settings, True, torch.Generator().manual_seed(42)))
            assert len(result) == 1 and dataset._decoder is None
            outputs.append(result[0])
        assert outputs[0][2] == outputs[1][2]
        assert torch.equal(outputs[0][0], outputs[1][0])
        assert torch.equal(outputs[0][1], outputs[1][1])
        low, target, scenes = outputs[1]
        model = build_model(cfg).eval()
        with torch.inference_mode():
            prediction = model(low)
        assert prediction.shape == target.shape and torch.isfinite(prediction).all()
        checks.append(
            {
                "scale": scale,
                "scenes": list(scenes),
                "prediction_shape": list(prediction.shape),
                "finite_cpu_forward": True,
                "serial_threaded_patch_bytes_identical": True,
                "config_sha256": digest(path),
                "manifest_sha256": digest(resolve(cfg["manifest"])),
            }
        )
        del model, outputs, result, low, target, prediction, dataset
    report = {
        "training_executed": False,
        "optimizer_steps": 0,
        "gpu_work_executed": False,
        "checks": checks,
    }
    write_json(ROOT / "runs/loader-validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
