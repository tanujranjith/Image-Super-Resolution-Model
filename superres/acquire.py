"""Official DIV2K archives. Safe extraction, resumable transfer, SHA256 receipts."""

import argparse
import os
from pathlib import Path
import shutil
import zipfile

from .common import digest, resolve, write_json

BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
ARCHIVES = ["DIV2K_train_HR.zip", "DIV2K_valid_HR.zip"] + [
    f"DIV2K_{split}_LR_bicubic_X{scale}.zip" for split in ("train", "valid") for scale in (2, 4)
]


def download(url, target, expected_sha256=None):
    import requests

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected_sha256 and digest(target) != expected_sha256:
            raise ValueError("Existing archive checksum mismatch")
        return target
    part = target.with_suffix(target.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    with requests.get(
        url, stream=True, timeout=(30, 120), headers={"Range": f"bytes={offset}-"} if offset else {}
    ) as r:
        r.raise_for_status()
        if offset and r.status_code != 206:
            offset = 0
        if offset and not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Inconsistent server range")
        if "text/html" in r.headers.get("Content-Type", ""):
            raise ValueError("Expected archive, received HTML")
        if shutil.disk_usage(target.parent).free < int(r.headers.get("Content-Length", 0)) + 2**30:
            raise OSError("Insufficient disk space")
        with part.open("ab" if offset else "wb") as f:
            for block in r.iter_content(1024 * 1024):
                f.write(block)
    sha = digest(part)
    if expected_sha256 and sha != expected_sha256:
        raise ValueError("Archive checksum mismatch")
    os.replace(part, target)
    write_json(
        target.with_suffix(".receipt.json"),
        {
            "url": url,
            "sha256": sha,
            "bytes": target.stat().st_size,
            "checksum_origin": "provided"
            if expected_sha256
            else "locally observed, not upstream-authenticated",
        },
    )
    return target


def extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for i in z.infolist():
            if (
                not (destination / i.filename).resolve().is_relative_to(destination)
                or (i.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("Unsafe archive path")
        if shutil.disk_usage(destination).free < sum(i.file_size for i in z.infolist()) + 2**30:
            raise OSError("Insufficient space to extract")
        # zipfile verifies member CRCs during extraction; a separate testzip pass would reread the full archive.
        z.extractall(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", nargs="+", choices=ARCHIVES, default=ARCHIVES)
    p.add_argument("--extract", action="store_true")
    a = p.parse_args()
    for name in a.archives:
        print(f"Downloading {name}", flush=True)
        path = download(BASE + name, resolve("datasets/downloads") / name)
        if a.extract:
            extract(path, resolve("datasets/raw"))


if __name__ == "__main__":
    main()
