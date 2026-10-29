"""Explicit local training, AMP, best/periodic checkpoints and epoch-level recovery."""

import argparse
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    atomic_save,
    choose_device,
    config,
    digest,
    log_event,
    resolve,
    restore_rng,
    rng_state,
    seed_all,
    source_digest,
    write_json,
)
from .data import Pairs, loader
from .inference import tiled
from .metrics import psnr_y
from .models import build_model
from .budget import TrainingBudgetEnded, check_budget


class ReconstructionLoss(nn.Module):
    def __init__(self, kind="l1", perceptual_weight=0.0):
        super().__init__()
        self.kind, self.weight = kind, perceptual_weight
        if kind not in ("l1", "mse", "charbonnier"):
            raise ValueError("Loss must be l1, mse or charbonnier")
        self.vgg = None
        if perceptual_weight:
            from torchvision.models import vgg16, VGG16_Weights

            self.vgg = (
                vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
                .features[:16]
                .eval()
                .requires_grad_(False)
            )
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, output, target):
        if self.kind == "l1":
            loss = F.l1_loss(output, target)
        elif self.kind == "mse":
            loss = F.mse_loss(output, target)
        else:
            loss = ((output - target).square() + 1e-6).sqrt().mean()
        if self.vgg is not None:
            self.vgg.eval()
            a = self.vgg((output - self.mean) / self.std)
            with torch.no_grad():
                b = self.vgg((target - self.mean) / self.std)
            loss = loss + self.weight * F.l1_loss(a, b)
        return loss


@torch.inference_mode()
def validate(model, batches, device, tile_size=128):
    scores = []
    for lr, hr, _ in batches:
        check_budget()
        prediction = tiled(model, lr, device, tile_size)
        psnr = psnr_y(prediction, hr, model.scale)
        if not torch.isfinite(torch.tensor(psnr)):
            raise FloatingPointError("Nonfinite validation metric")
        scores.append(psnr)
    return sum(scores) / len(scores)


def train(cfg, resume=None):
    check_budget()
    if cfg["architecture"] == "bicubic":
        raise ValueError("Bicubic is an evaluation baseline, not a trainable architecture")
    seed_all(cfg["seed"])
    device = choose_device(cfg.get("device", "auto"))
    manifest = resolve(cfg["manifest"])
    fingerprint = digest(manifest)
    if cfg.get("manifest_sha256") and fingerprint != cfg["manifest_sha256"]:
        raise ValueError("Prepared manifest fingerprint changed; training refused")
    code_sha256 = source_digest()
    out = resolve(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not resume:
        raise FileExistsError("Existing run; resume or choose another output")
    generator = torch.Generator().manual_seed(cfg["seed"])
    kwargs = dict(
        scale=cfg["scale"],
        patch_size=cfg["patch_size"],
        repeats=cfg.get("repeats", 20),
        degradation=cfg.get("degradation", "bicubic"),
        seed=cfg["seed"],
    )
    train_data = Pairs(manifest, "train", **kwargs)
    training = loader(train_data, cfg, True, generator)
    validation = loader(Pairs(manifest, "val", **kwargs), cfg)
    model = build_model(cfg).to(device)
    criterion = ReconstructionLoss(cfg.get("loss", "l1"), cfg.get("perceptual_weight", 0.0)).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg["epochs"], eta_min=cfg["lr"] / 100
    )
    amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, stale = 0, -float("inf"), 0
    if resume:
        state = torch.load(resolve(resume), map_location="cpu", weights_only=False)
        if state["manifest_sha256"] != fingerprint:
            raise ValueError("Manifest changed; resume refused")
        if state["config"] != cfg:
            raise ValueError(
                "Resume requires the identical resolved config; use a new experiment for changes"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, stale = state["epoch"] + 1, state["best"], state["stale"]
        restore_rng(state["rng"])
        generator.set_state(state["loader_rng"])
    write_json(out / "config.json", cfg)
    write_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "cpu_threads": torch.get_num_threads(),
            "decode_threads": train_data.decode_threads,
            "prefetch_batches": getattr(training, "batches", 0),
            "sampling": train_data.sampling,
            "condition": train_data.condition,
            "manifest_sha256": fingerprint,
            "source_sha256": code_sha256,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        },
    )
    try:
        for epoch in range(start, cfg["epochs"]):
            check_budget()
            train_data.set_epoch(epoch)
            model.train()
            started, total, count = time.perf_counter(), 0.0, 0
            data_wait, previous_step_end = 0.0, time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for step, (lr, hr, _) in enumerate(training, 1):
                data_wait += time.perf_counter() - previous_step_end
                check_budget()
                lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=amp):
                    loss = criterion(model(lr), hr)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Nonfinite loss; last completed epoch remains available"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(lr)
                count += len(lr)
                if step == 1 or step % cfg.get("log_every", 50) == 0:
                    log_event(
                        out / "metrics.jsonl",
                        event="train_progress",
                        epoch=epoch + 1,
                        step=step,
                        steps=len(training),
                        patches=count,
                        loss=total / count,
                        seconds=time.perf_counter() - started,
                        data_wait_seconds=data_wait,
                        peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20
                        if device.type == "cuda"
                        else None,
                    )
                previous_step_end = time.perf_counter()
            score = validate(model, validation, device, cfg.get("tile_size", 128))
            improved = score > best + cfg.get("min_delta", 0.001)
            best, stale = (score, 0) if improved else (best, stale + 1)
            scheduler.step()
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "config": cfg,
                "condition": train_data.condition,
                "manifest_sha256": fingerprint,
                "source_sha256": code_sha256,
                "rng": rng_state(),
                "loader_rng": generator.get_state(),
            }
            atomic_save(out / "last.pt", state)
            if improved:
                atomic_save(out / "best.pt", state)
            if (epoch + 1) % cfg.get("checkpoint_every", 5) == 0:
                atomic_save(out / f"epoch-{epoch + 1:03}.pt", state)
            log_event(
                out / "metrics.jsonl",
                epoch=epoch + 1,
                train_loss=total / count,
                val_psnr_y=score,
                lr=optimizer.param_groups[0]["lr"],
                seconds=time.perf_counter() - started,
                data_wait_seconds=data_wait,
                peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda"
                else None,
            )
            if stale >= cfg.get("patience", 10):
                break
    except TrainingBudgetEnded:
        if not (out / "best.pt").is_file() or not (out / "last.pt").is_file():
            raise RuntimeError(
                "Budget ended before a complete epoch; no trained checkpoint to evaluate"
            )
        log_event(
            out / "metrics.jsonl",
            event="training_budget_ended",
            reason="Use best.pt for evaluation; last.pt retains the last fully completed epoch. Incomplete epoch discarded.",
        )
        return
    except (KeyboardInterrupt, torch.cuda.OutOfMemoryError) as e:
        log_event(
            out / "metrics.jsonl",
            event="interrupted",
            reason=type(e).__name__,
            recovery="Replay incomplete epoch from last.pt. OOM: create new run with smaller batch/patch/tile size.",
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/edsr_x2.yaml")
    p.add_argument("--resume")
    a = p.parse_args()
    train(config(a.config), a.resume)


if __name__ == "__main__":
    main()
