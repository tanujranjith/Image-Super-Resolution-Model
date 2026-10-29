"""Aligned patch sampling and explicitly separate synthetic degradation conditions."""

import io
import os
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor

from .common import read_json, resolve, worker_seed
from .prefetch import PrefetchLoader


def modcrop(image, scale):
    w, h = image.size
    if w % scale == 0 and h % scale == 0:
        return image
    return image.crop((0, 0, w - w % scale, h - h % scale))


def degrade(image, scale, condition="bicubic", rng=None):
    """Pillow downsampling is explicitly NOT MATLAB/DIV2K official bicubic."""
    rng = rng or random.Random(0)
    image = modcrop(image, scale)
    if condition == "realistic-synthetic":
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.5)))
    elif condition != "bicubic":
        raise ValueError("Unknown synthetic degradation condition")
    low = image.resize((image.width // scale, image.height // scale), Image.Resampling.BICUBIC)
    if condition == "realistic-synthetic":
        a = np.asarray(low).astype(np.float32)
        noise = np.random.default_rng(rng.randrange(2**32)).normal(
            0, rng.uniform(0.0, 6.0), a.shape
        )
        low = Image.fromarray(np.uint8(np.clip(a + noise, 0, 255)))
        stream = io.BytesIO()
        low.save(stream, format="JPEG", quality=rng.randint(65, 95))
        stream.seek(0)
        with Image.open(stream) as compressed:
            low = compressed.convert("RGB").copy()
    return low


def synthetic_patch(image, scale, patch_size, x, y):
    """Downsample only a context-padded crop, preserving full-image pixel phase."""
    if image.width % scale or image.height % scale:
        raise ValueError("Synthetic crop requires modcropped HR")
    width, height = image.width // scale, image.height // scale
    if not (0 <= x <= width - patch_size and 0 <= y <= height - patch_size):
        raise ValueError("Synthetic crop outside HR image")
    left, top = max(0, x - 4), max(0, y - 4)
    right, bottom = min(width, x + patch_size + 4), min(height, y + patch_size + 4)
    with image.crop((left * scale, top * scale, right * scale, bottom * scale)) as context:
        with context.resize((right - left, bottom - top), Image.Resampling.BICUBIC) as low:
            return (
                low.crop((x - left, y - top, x - left + patch_size, y - top + patch_size)),
                image.crop(
                    (x * scale, y * scale, (x + patch_size) * scale, (y + patch_size) * scale)
                ),
            )


class Pairs(Dataset):
    def __init__(
        self, manifest, split, scale=2, patch_size=48, repeats=20, degradation="bicubic", seed=42
    ):
        info = read_json(resolve(manifest))
        self.root = (resolve(manifest).parent / info["root"]).resolve()
        self.rows = [r for r in info["pairs"] if r["split"] == split and r["scale"] == scale]
        # In particular, a 50-image validation set must not retain every training
        # row for both scales. Only retain this split/scale and small metadata.
        self.info = {key: value for key, value in info.items() if key != "pairs"}
        self.train, self.scale, self.patch_size = split == "train", scale, patch_size
        self.repeats = repeats if self.train else 1
        self.degradation, self.seed = degradation, seed
        self.epoch = 0
        self.sampling = "global-rng"
        self.decode_threads = 0
        self.max_decode_pixels = None
        self._decoder = None
        if not self.rows:
            raise ValueError(f"Empty split {split} x{scale}")
        if degradation != "bicubic" and any(r.get("lr") for r in self.rows):
            raise ValueError(
                "Synthetic real-world degradation requires an HR-only manifest, separate from official/RealSR pairs"
            )
        self.condition = (
            self.info["condition"]
            if all(r.get("lr") for r in self.rows)
            else ("pillow-bicubic-synthetic" if degradation == "bicubic" else "realistic-synthetic")
        )

    def __len__(self):
        return len(self.rows) * self.repeats

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def close(self):
        if self._decoder is not None:
            self._decoder.shutdown(wait=True, cancel_futures=True)
            self._decoder = None

    def __getitems__(self, indices):
        if not self.decode_threads:
            return [self[index] for index in indices]
        if self._decoder is None:
            self._decoder = ThreadPoolExecutor(
                max_workers=self.decode_threads, thread_name_prefix="sr-decode"
            )
        # map retains sampler order regardless of which decoding finishes first.
        return list(self._decoder.map(self.__getitem__, indices))

    def __getitem__(self, index):
        with ExitStack() as images:
            return self._sample(index, images)

    def _sample(self, index, images):
        def keep(image):
            images.callback(image.close)
            return image

        def open_rgb(path, transpose=False):
            image = images.enter_context(Image.open(path))
            if self.max_decode_pixels and image.width * image.height > self.max_decode_pixels:
                raise ValueError("Image exceeds configured decode pixel limit")
            if transpose and image.getexif().get(274, 1) not in (None, 1):
                image = keep(ImageOps.exif_transpose(image))
            # Reuse RGB PNG buffers instead of unconditionally making full-size
            # EXIF, RGB and modcrop copies before selecting a tiny patch.
            return image if image.mode == "RGB" else keep(image.convert("RGB"))

        rng = (
            random.Random(f"sr-epoch-index-v1:{self.seed}:{self.epoch}:{index}:{self.scale}")
            if self.train and self.sampling == "epoch-index-v1"
            else (random if self.train else random.Random(self.seed + index))
        )
        row = self.rows[index % len(self.rows)]
        hr_path = (self.root / row["hr"]).resolve()
        if not hr_path.is_relative_to(self.root):
            raise ValueError("HR path escapes dataset")
        hr = keep(modcrop(open_rgb(hr_path, transpose=True), self.scale))
        cropped = False
        if row.get("lr"):
            lr_path = (self.root / row["lr"]).resolve()
            if not lr_path.is_relative_to(self.root):
                raise ValueError("LR path escapes dataset")
            lr = open_rgb(lr_path)
        else:
            if self.train and self.degradation == "bicubic":
                p, s = self.patch_size, self.scale
                if min(hr.size) // s < p:
                    raise ValueError("LR image smaller than patch; lower patch_size")
                x, y = (
                    rng.randrange(hr.width // s - p + 1),
                    rng.randrange(hr.height // s - p + 1),
                )
                lr, hr = synthetic_patch(hr, s, p, x, y)
                keep(lr)
                keep(hr)
                cropped = True
            else:
                lr = keep(degrade(hr, self.scale, self.degradation, rng))
        if hr.size != (lr.width * self.scale, lr.height * self.scale):
            raise ValueError("Pair is not geometrically aligned")
        if self.train:
            if not cropped:
                p, s = self.patch_size, self.scale
                if min(lr.size) < p:
                    raise ValueError("LR image smaller than patch; lower patch_size")
                x, y = rng.randrange(lr.width - p + 1), rng.randrange(lr.height - p + 1)
                lr = keep(lr.crop((x, y, x + p, y + p)))
                hr = keep(hr.crop((x * s, y * s, (x + p) * s, (y + p) * s)))
            for transpose in (
                Image.Transpose.FLIP_LEFT_RIGHT,
                Image.Transpose.FLIP_TOP_BOTTOM,
                Image.Transpose.TRANSPOSE,
            ):
                if rng.random() < 0.5:
                    lr, hr = keep(lr.transpose(transpose)), keep(hr.transpose(transpose))
        return to_tensor(lr), to_tensor(hr), row["scene"]


def loader(dataset, cfg, train=False, generator=None):
    workers = cfg.get("workers", 2) if train else 0
    decode = cfg.get("decode_threads", 0) if train else 0
    prefetch = cfg.get("prefetch_batches", 0) if train else 0
    # Hardware-only overrides preserve the checkpoint config, sampler and patch
    # RNG. Validation remains serial and full resolution in either profile.
    if train and os.environ.get("SR_THROUGHPUT_PROFILE") == "high":
        decode, prefetch = 4, 4
    if type(decode) is not int or not 0 <= decode <= 4:
        raise ValueError("decode_threads must be an integer from 0 to 4")
    if type(prefetch) is not int or not 0 <= prefetch <= 4:
        raise ValueError("prefetch_batches must be an integer from 0 to 4")
    if workers and (decode or prefetch):
        raise ValueError("Bounded threaded loading requires workers=0")
    sampling = cfg.get("sampling", "global-rng")
    if sampling not in ("global-rng", "epoch-index-v1"):
        raise ValueError("Unsupported patch sampling scheme")
    if train and (decode or prefetch) and sampling != "epoch-index-v1":
        raise ValueError("Threaded loading requires epoch-index-v1 sampling")
    dataset.decode_threads = decode
    dataset.sampling = sampling
    dataset.max_decode_pixels = cfg.get("max_decode_pixels") if train else None
    if dataset.max_decode_pixels is not None and (
        type(dataset.max_decode_pixels) is not int or dataset.max_decode_pixels < 1
    ):
        raise ValueError("max_decode_pixels must be a positive integer")
    base = DataLoader(
        dataset,
        batch_size=cfg["batch_size"] if train else 1,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=worker_seed,
        persistent_workers=False,
        **({"prefetch_factor": 2} if workers else {}),
    )
    return PrefetchLoader(base, prefetch) if decode or prefetch else base
