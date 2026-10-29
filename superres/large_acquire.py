"""Kaggle-only streamed HR acquisition; pure CPU, bounded buffers, no training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import struct
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image, ImageOps
from .common import resolve, write_json

BASE = resolve("datasets/large/source-metadata")
SOURCE = "https://www.kaggle.com/api/v1/datasets/download/parampandya0310/hr-image-denoising-lsdir?datasetVersionNumber=1"
TOTAL_BYTES = 56846762573
EXPECTED_HR = 30250


def member_role(name):
    """Separate HR scenes from the validation archive's x2/x3/x4 copies."""
    source = PurePosixPath(name)
    parts = source.parts
    if (
        len(parts) == 3
        and parts[0] in {f"shard-{i:02}" for i in range(6)}
        and parts[1].isdigit()
        and source.stem.isdigit()
        and source.suffix == ".png"
    ):
        return "train"
    if len(parts) == 5 and parts[:2] == ("val1", "val1") and parts[3] == "val":
        if parts[2] == "HR" and source.stem.isdigit() and source.suffix == ".png":
            return "validation"
        if parts[2] in ("X2", "X3", "X4") and re.fullmatch(r"\d+x[234]", source.stem):
            if source.stem.endswith("x" + parts[2][-1]) and source.suffix == ".png":
                return "provided-lr"
    raise ValueError(f"Unexpected Kaggle PNG member layout: {name}")


def zip64_values(entry, extra):
    size, compressed, offset, disk = entry[9], entry[8], entry[16], entry[13]
    cursor = 0
    while cursor + 4 <= len(extra):
        kind, length = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        data = extra[cursor : cursor + length]
        if len(data) != length:
            raise ValueError("Truncated ZIP extra field")
        if kind == 1:
            pos = 0
            values = []
            for value, sentinel, width in (
                (size, 0xFFFFFFFF, 8),
                (compressed, 0xFFFFFFFF, 8),
                (offset, 0xFFFFFFFF, 8),
                (disk, 0xFFFF, 4),
            ):
                if value == sentinel:
                    if pos + width > len(data):
                        raise ValueError("Missing ZIP64 value")
                    value = int.from_bytes(data[pos : pos + width], "little")
                    pos += width
                values.append(value)
            return tuple(values)
        cursor += length
    if 0xFFFFFFFF in (size, compressed, offset) or disk == 0xFFFF:
        raise ValueError("ZIP64 extra field missing")
    return size, compressed, offset, disk


def build_index(central, database):
    database.execute(
        "CREATE TABLE IF NOT EXISTS entries (name TEXT PRIMARY KEY, disk INTEGER, "
        "offset INTEGER, compressed INTEGER, size INTEGER, crc INTEGER, flags INTEGER, "
        "method INTEGER, mtime INTEGER, mdate INTEGER, central BLOB)"
    )
    count = 0
    with Path(central).open("rb") as stream:
        while raw := stream.read(46):
            if len(raw) != 46:
                raise ValueError("Truncated central directory")
            entry = struct.unpack("<4s6H3L5H2L", raw)
            if entry[0] != b"PK\x01\x02":
                raise ValueError("Invalid ZIP central signature")
            name = stream.read(entry[10])
            extra = stream.read(entry[11])
            comment = stream.read(entry[12])
            if len(extra) != entry[11] or len(comment) != entry[12] or len(name) != entry[10]:
                raise ValueError("Truncated ZIP variable fields")
            text = name.decode("utf-8" if entry[3] & 2048 else "cp437")
            size, compressed, offset, disk = zip64_values(entry, extra)
            if disk != 0:
                raise ValueError("Expected single Kaggle ZIP, not split archive")
            database.execute(
                "INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    text,
                    disk,
                    offset,
                    compressed,
                    size,
                    entry[7],
                    entry[3],
                    entry[4],
                    entry[5],
                    entry[6],
                    raw + name + extra + comment,
                ),
            )
            count += 1
    database.commit()
    return count


def unpack_member(row, data):
    """Rewrap a single original member so standard zipfile validates its CRC."""
    if len(data) < 30:
        raise ValueError("Short ZIP local header")
    header = struct.unpack("<4s5H3L2H", data[:30])
    if header[0] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local header")
    size = 30 + header[-2] + header[-1] + row["compressed"]
    if size > len(data):
        raise ValueError("Insufficient bytes for member")
    local = bytes(data[:size])
    central = bytearray(row["central"])
    struct.pack_into("<H", central, 34, 0)
    struct.pack_into("<L", central, 42, 0)
    footer = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 1, 1, len(central), size, 0)
    with zipfile.ZipFile(io.BytesIO(local + central + footer)) as archive:
        payload = archive.read(row["name"])
    if len(payload) != row["size"]:
        raise ValueError("Uncompressed size mismatch")
    return payload


class RemoteArchive:
    def __init__(self, base=BASE):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        self.network_bytes = 0

    def span(self, start, length):
        end = start + length - 1
        if not 0 <= start <= end < TOTAL_BYTES:
            raise ValueError("Kaggle range outside expected archive")
        request = urllib.request.Request(
            SOURCE, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    wanted = f"bytes {start}-{end}/{TOTAL_BYTES}"
                    if response.status != 206 or response.headers.get("Content-Range") != wanted:
                        raise ValueError(
                            "Kaggle did not honor exact archive range/version; stopped"
                        )
                    data = response.read(length + 1)
                    if len(data) != length:
                        raise OSError("Truncated Kaggle archive range")
                self.network_bytes += len(data)
                with (self.base / "range-receipts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "source": SOURCE,
                                "start": start,
                                "end": end,
                                "sha256": hashlib.sha256(data).hexdigest(),
                            }
                        )
                        + "\n"
                    )
                return data
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                delay = min(60, max(2 ** (attempt + 1), int(error.headers.get("Retry-After", "0"))))
                time.sleep(delay)
            except (OSError, urllib.error.URLError):
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))

    def index(self):
        central = self.base / "lsdir-png-central.bin"
        if not central.exists():
            tail = self.span(TOTAL_BYTES - 65536, 65536)
            pos = tail.rfind(b"PK\x05\x06")
            if pos < 0:
                raise ValueError("Kaggle ZIP footer missing")
            footer = struct.unpack("<4s4H2LH", tail[pos : pos + 22])
            size, offset = footer[5], footer[6]
            if footer[1:3] != (0, 0):
                raise ValueError("Unexpected archive disks")
            if offset == 0xFFFFFFFF:
                pos64 = tail.rfind(b"PK\x06\x06")
                if pos64 < 0:
                    raise ValueError("ZIP64 footer missing")
                footer64 = struct.unpack("<4sQ2H2L4Q", tail[pos64 : pos64 + 56])
                size, offset = footer64[-2:]
            if size > 64 * 1024**2 or offset + size > TOTAL_BYTES:
                raise ValueError("Unexpected Kaggle central layout")
            temporary = central.with_suffix(".tmp")
            temporary.write_bytes(self.span(offset, size))
            temporary.replace(central)
        database = sqlite3.connect(self.base / "lsdir-index.sqlite")
        database.row_factory = sqlite3.Row
        ready = database.execute("SELECT name FROM sqlite_master WHERE name='entries'").fetchone()
        if not ready:
            build_index(central, database)
        return database


def intervals(rows, gap=1024**2, maximum=64 * 1024**2):
    groups = []
    for row in sorted(rows, key=lambda r: r["offset"]):
        start = row["offset"]
        end = start + 30 + len(row["name"].encode("utf-8")) + 4096 + row["compressed"]
        if groups and start - groups[-1][1] <= gap and end - groups[-1][0] <= maximum:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append(row)
        else:
            groups.append([start, end, [row]])
    return groups


def prefetch_intervals(groups, fetch):
    """One network worker; at most two ~64 MiB range buffers."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = None
        for i, group in enumerate(groups):
            if pending is None:
                start, end, _ = group
                pending = executor.submit(fetch, start, end - start)
            payload = pending.result()
            if i + 1 < len(groups):
                start, end, _ = groups[i + 1]
                pending = executor.submit(fetch, start, end - start)
            else:
                pending = None
            yield group, payload


def perceptual_hashes(image):
    """64-bit DCT pHash, including mirror; no additional scientific dependency."""
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    basis = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32)[None, :] + 0.5))
    result = []
    for view in (gray, ImageOps.mirror(gray)):
        low = basis @ np.asarray(view, dtype=np.float64) @ basis.T
        bits = (low > np.median(low)).reshape(-1)
        value = 0
        for bit in bits:
            value = (value << 1) | int(bit)
        result.append(f"{value:016x}")
    return result


def audit_bytes(data):
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 16_777_216 or min(image.size) < 256:
            raise ValueError("HR size outside 256px minimum / 16 megapixel memory guard")
        if image.mode not in ("RGB", "L"):
            raise ValueError(f"Expected opaque 8-bit natural image, got {image.mode}")
        image = ImageOps.exif_transpose(image).convert("RGB")
        hashes = perceptual_hashes(image)
        return {
            "width": image.width,
            "height": image.height,
            "sha256": hashlib.sha256(data).hexdigest(),
            "phash": hashes[0],
            "phash_hflip": hashes[1],
        }


def acquire():
    remote = RemoteArchive()
    db = remote.index()
    archive_rows = [
        dict(r) for r in db.execute("SELECT * FROM entries WHERE name LIKE '%.png' ORDER BY offset")
    ]
    if len(archive_rows) != 31000:
        raise ValueError("Kaggle version/layout differs from audited 31,000-PNG package")
    roles = [(row, member_role(row["name"])) for row in archive_rows]
    counts = {
        role: sum(kind == role for _, kind in roles)
        for role in ("train", "validation", "provided-lr")
    }
    if counts != {"train": 30000, "validation": 250, "provided-lr": 750}:
        raise ValueError("Kaggle HR/LR member counts changed")
    rows = [row for row, role in roles if role != "provided-lr"]
    destination = resolve("datasets/raw/LSDIR-kaggle")
    destination.mkdir(parents=True, exist_ok=True)
    journal = resolve("datasets/large/images.jsonl")
    prior = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                record = json.loads(line)
                prior[record["source_path"]] = record
            except json.JSONDecodeError:
                pass
    pending = []
    for row in rows:
        old = prior.get(row["name"])
        if old and old.get("status") == "excluded":
            continue
        if old:
            path = destination / old["path"]
            if path.is_file():
                stat = path.stat()
                if stat.st_size == old["bytes"] and stat.st_mtime_ns == old["mtime_ns"]:
                    continue  # Already CRC/SHA/decode verified, with unchanged local cache identity.
        pending.append(row)
    required = sum(r["size"] for r in pending)
    reserve = 64 * 1024**3
    if shutil.disk_usage(destination).free < required + reserve:
        raise OSError("Not enough free disk for remaining HR images plus 64 GiB reserve")
    groups = intervals(pending)
    print(
        json.dumps(
            {
                "event": "acquisition_plan",
                "source": "Kaggle",
                "selected_images": len(rows),
                "pending": len(pending),
                "hr_bytes": required,
                "requests": len(groups),
                "range_bytes": sum(b - a for a, b, _ in groups),
                "training_started": False,
            }
        ),
        flush=True,
    )
    if journal.exists() and journal.stat().st_size:
        with journal.open("rb") as stream:
            stream.seek(-1, 2)
            newline = stream.read(1) == b"\n"
        if not newline:
            with journal.open("a", encoding="utf-8") as stream:
                stream.write("\n")
    ready = len(rows) - len(pending)
    with journal.open("a", encoding="utf-8") as log:
        for number, ((start, end, group), buffer) in enumerate(
            prefetch_intervals(groups, remote.span), 1
        ):
            if resolve("datasets/large/PAUSE").exists():
                print("Acquisition paused at verified interval boundary; no training.", flush=True)
                return
            for row in group:
                pos = row["offset"] - start
                # A bounded member slice avoids copying the remaining 64 MiB for every photo.
                payload = unpack_member(row, memoryview(buffer)[pos:])
                source = PurePosixPath(row["name"])
                official = member_role(row["name"])
                local = Path(official) / (source.stem + ".png")
                record = {
                    "source_path": row["name"],
                    "scene": "lsdir-" + official + "-" + source.stem,
                    "path": local.as_posix(),
                    "source": "lsdir-kaggle",
                    "official_split": official,
                    "dataset_ref": "parampandya0310/hr-image-denoising-lsdir",
                    "dataset_version": 1,
                    "archive_crc32": row["crc"],
                }
                try:
                    record.update(audit_bytes(payload))
                except (OSError, ValueError) as error:
                    log.write(
                        json.dumps(record | {"status": "excluded", "reason": str(error)}) + "\n"
                    )
                    ready += 1
                    continue
                target = (destination / local).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("Image path escapes dataset root")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(payload)
                temporary.replace(target)
                record.update(
                    bytes=len(payload), mtime_ns=target.stat().st_mtime_ns, status="verified"
                )
                log.write(json.dumps(record) + "\n")
                ready += 1
            log.flush()
            write_json(
                resolve("datasets/large/status.json"),
                {
                    "phase": "acquiring",
                    "source": "Kaggle",
                    "processed": ready,
                    "total": len(rows),
                    "interval": number,
                    "intervals": len(groups),
                    "network_bytes_this_process": remote.network_bytes,
                    "training_started": False,
                },
            )
            if number == 1 or number % 10 == 0 or number == len(groups):
                print(
                    json.dumps(
                        {
                            "event": "acquisition_progress",
                            "processed": ready,
                            "total": len(rows),
                            "interval": number,
                            "intervals": len(groups),
                        }
                    ),
                    flush=True,
                )
    write_json(
        resolve("datasets/large/status.json"),
        {
            "phase": "acquired",
            "source": "Kaggle",
            "processed": ready,
            "total": EXPECTED_HR,
            "training_started": False,
            "source_png_members": 31000,
            "provided_lr_members_excluded": 750,
        },
    )
    print(
        json.dumps(
            {"event": "acquisition_complete", "processed": ready, "training_started": False}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    acquire()
