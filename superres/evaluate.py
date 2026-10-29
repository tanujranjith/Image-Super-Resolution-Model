"""Actual learned/bicubic metrics, visual comparisons and per-image errors."""

import argparse
import math
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import to_pil_image

from .common import choose_device, digest, resolve, write_json
from .data import Pairs, loader
from .inference import tiled
from .metrics import psnr_ssim, quantize
from .models import Bicubic, build_model


def evaluation_manifest(state, manifest=None, benchmark=False):
    """Keep training provenance fixed; permit only explicitly pinned benchmarks."""
    cfg = state["config"]
    training = resolve(cfg["manifest"]).resolve()
    if digest(training) != state["manifest_sha256"]:
        raise ValueError("Training dataset fingerprint changed")
    selected = resolve(manifest or training).resolve()
    actual = digest(selected)
    if selected == training:
        return selected
    approved = {
        resolve(item["manifest"]).resolve(): item["sha256"] for item in cfg.get("benchmarks", [])
    }
    if not benchmark or approved.get(selected) != actual:
        raise ValueError(
            "External benchmark requires --benchmark and a matching pinned configuration entry"
        )
    return selected


def comparison(lr, hr, sr, baseline, path):
    images = [to_pil_image(quantize(x[0])) for x in (baseline, sr, hr)]
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 3, height + 28), "#101827")
    draw = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate(
        zip(images, ["Bicubic baseline", "Neural reconstruction", "Ground truth"])
    ):
        canvas.paste(im, (width * i, 28))
        draw.text((width * i + 8, 8), title, fill="white")
    canvas.save(path)


@torch.inference_mode()
def evaluate(
    checkpoint=None,
    manifest=None,
    scale=None,
    split="test",
    output="runs/evaluation",
    use_lpips=False,
    tile_size=128,
    device_name="auto",
    benchmark=False,
):
    device = choose_device(device_name)
    if checkpoint:
        checkpoint = resolve(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = state["config"]
        manifest, scale = evaluation_manifest(state, manifest, benchmark), cfg["scale"]
        model = build_model(cfg).to(device).eval()
        model.load_state_dict(state["model"])
    else:
        if not manifest or scale not in (2, 4):
            raise ValueError("Provide checkpoint or manifest plus scale")
        cfg = {"degradation": "bicubic", "workers": 0, "batch_size": 1}
        model = None
    dataset = Pairs(
        manifest,
        split,
        scale,
        degradation=cfg.get("degradation", "bicubic"),
        seed=cfg.get("seed", 42),
    )
    baseline_model = Bicubic(scale).to(device)
    perceptual = None
    if use_lpips:
        import lpips

        # Explicit option may download pretrained AlexNet; never enabled by default.
        perceptual = lpips.LPIPS(net="alex").to(device).eval()
    out = resolve(output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (lr, hr, scene) in enumerate(loader(dataset, cfg)):
        if index == 0:
            tiled(model or baseline_model, lr, device, tile_size)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        sr = tiled(model or baseline_model, lr, device, tile_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
        baseline = tiled(baseline_model, lr, device, tile_size)
        psnr, ssim = psnr_ssim(sr, hr, scale)
        bp, bs = psnr_ssim(baseline, hr, scale)
        row = {
            "scene": scene[0],
            "psnr_y": psnr if math.isfinite(psnr) else None,
            "ssim_y": ssim,
            "exact_match": math.isinf(psnr),
            "bicubic_psnr_y": bp if math.isfinite(bp) else None,
            "bicubic_ssim_y": bs,
            "seconds": seconds,
            "peak_vram_mb": peak,
            "mae_rgb": (sr.clamp(0, 1) - hr).abs().mean().item(),
        }
        if perceptual:
            row["lpips_rgb"] = perceptual(
                quantize(sr).to(device) * 2 - 1, hr.to(device) * 2 - 1
            ).item()
        records.append(row)
        if index < 8:
            comparison(lr, hr, sr, baseline, out / f"comparison-{scene[0]}.png")
            error = (sr.clamp(0, 1) - hr).abs().mean(1, keepdim=True).expand(-1, 3, -1, -1)
            to_pil_image((error[0] * 5).clamp(0, 1)).save(out / f"error-5x-{scene[0]}.png")
    metrics = ["psnr_y", "ssim_y", "bicubic_psnr_y", "bicubic_ssim_y", "seconds", "mae_rgb"] + (
        ["lpips_rgb"] if perceptual else []
    )
    summary = {
        k: float(np.mean([r[k] for r in records]))
        if all(r[k] is not None for r in records)
        else None
        for k in metrics
    }
    summary.update(
        architecture=cfg["architecture"] if model else "bicubic",
        scale=scale,
        split=split,
        images=len(records),
        condition=dataset.condition,
        manifest_sha256=digest(resolve(manifest)),
        training_manifest_sha256=state["manifest_sha256"] if checkpoint else None,
        training_condition=state.get("condition") if checkpoint else None,
        external_benchmark=bool(
            checkpoint and resolve(manifest).resolve() != resolve(cfg["manifest"]).resolve()
        ),
        checkpoint_sha256=digest(checkpoint) if checkpoint else None,
        model_parameters=sum(p.numel() for p in model.parameters()) if model else 0,
        checkpoint_bytes=checkpoint.stat().st_size if checkpoint else 0,
        tile_size=tile_size,
        device=str(device),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        peak_vram_mb=max(r["peak_vram_mb"] for r in records) if device.type == "cuda" else None,
        protocol="8-bit quantized/clipped RGB; BT.601 limited-range Y; scale-pixel shave; valid 11x11 Gaussian SSIM sigma1.5; per-image arithmetic means; latency includes tile transfers, excludes decode/PNG",
    )
    write_json(
        out / "metrics.json",
        {
            "summary": summary,
            "images": records,
            "worst_psnr": sorted(records, key=lambda r: r["psnr_y"] or float("inf"))[:10],
        },
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint")
    p.add_argument("--manifest")
    p.add_argument("--scale", type=int, choices=[2, 4])
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--output", default="runs/evaluation")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--tile-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--benchmark", action="store_true", help="Allow a benchmark pinned in checkpoint config"
    )
    a = p.parse_args()
    print(
        evaluate(
            a.checkpoint,
            a.manifest,
            a.scale,
            a.split,
            a.output,
            a.lpips,
            a.tile_size,
            a.device,
            a.benchmark,
        )
    )
