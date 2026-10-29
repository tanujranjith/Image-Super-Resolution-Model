"""Atomic local artifacts and reproducibility, with no sibling dependencies."""

import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def config(path):
    cfg = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    required = {
        "architecture",
        "scale",
        "features",
        "blocks",
        "patch_size",
        "batch_size",
        "epochs",
        "lr",
        "seed",
        "manifest",
        "output",
    }
    if not isinstance(cfg, dict) or required - cfg.keys():
        raise ValueError(f"Missing configuration keys: {required - (cfg or {}).keys()}")
    if cfg["scale"] not in (2, 4):
        raise ValueError("Scale must be 2 or 4")
    for k in ("features", "blocks", "patch_size", "batch_size", "epochs"):
        if not isinstance(cfg[k], int) or cfg[k] < 1:
            raise ValueError(f"{k} must be positive")
    if cfg["lr"] <= 0 or cfg.get("loss", "l1") not in ("l1", "mse", "charbonnier"):
        raise ValueError("Invalid loss or learning rate")
    return cfg


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(_):
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state():
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(path, state):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_digest():
    h = hashlib.sha256()
    for path in sorted((ROOT / "superres").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def log_event(path, **event):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, allow_nan=False) + "\n")
    print(json.dumps(event), flush=True)


def choose_device(name="auto"):
    import torch

    name = (
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else "cpu"
        if name == "auto"
        else name
    )
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install the CUDA wheel or explicitly select CPU")
    return torch.device(name)
