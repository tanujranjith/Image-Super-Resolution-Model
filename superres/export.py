"""Export tensors and provenance without network/account operations."""

import argparse

from safetensors.torch import save_file
import torch

from .common import digest, resolve, write_json


def export(checkpoint, destination=None):
    checkpoint = resolve(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    out = resolve(destination or f"artifacts/x{cfg['scale']}")
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in state["model"].items()}, str(out / "model.safetensors")
    )
    write_json(
        out / "config.json",
        {
            **cfg,
            "trained": True,
            "epoch": state["epoch"] + 1,
            "condition": state["condition"],
            "manifest_sha256": state["manifest_sha256"],
            "source_sha256": state.get("source_sha256"),
            "checkpoint_sha256": digest(checkpoint),
        },
    )
    write_json(
        out / "checksums.json",
        {
            "model.safetensors": digest(out / "model.safetensors"),
            "config.json": digest(out / "config.json"),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--destination")
    a = p.parse_args()
    export(a.checkpoint, a.destination)
