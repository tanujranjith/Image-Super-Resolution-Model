"""Sequential SHA256/CRC extraction of the downloaded official archives, no Torch import."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

root = Path(__file__).resolve().parents[1]
destination = (root / "datasets/raw").resolve()
destination.mkdir(parents=True, exist_ok=True)
for archive in sorted((root / "datasets/downloads").glob("*.zip")):
    receipt_path = archive.with_suffix(".receipt.json")
    if not receipt_path.exists():
        h = hashlib.sha256()
        with archive.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(block)
        receipt_path.write_text(json.dumps({"url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/" + archive.name,
                                           "sha256": h.hexdigest(), "bytes": archive.stat().st_size,
                                           "checksum_origin": "locally observed; upstream digest unavailable"}, indent=2), encoding="utf-8")
    with zipfile.ZipFile(archive) as z:
        if shutil.disk_usage(destination).free < sum(i.file_size for i in z.infolist()) + 2**30:
            raise OSError("Insufficient free space")
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive member")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != info.file_size:
                with z.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    print(f"Verified/extracted {archive.name}", flush=True)
