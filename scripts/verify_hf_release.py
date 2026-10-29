"""Verify public release downloads against local files without loading Torch."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMITS = {2: "27113b13bdef9246fcfa1f180124b62cdcd01107", 4: "488821e4ef22f2f88acddfae6e43989c8314661c"}


def main():
    receipt = {"verified_utc": datetime.now(timezone.utc).isoformat(), "models": []}
    for scale, commit in COMMITS.items():
        repo = f"Tanuj124/image-super-resolution-edsr-x{scale}"
        folder = ROOT / f"dist/hf-model-x{scale}"
        hashes = {}
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            with urlopen(f"https://huggingface.co/{repo}/resolve/{commit}/{path.name}", timeout=90) as response:
                remote = hashlib.sha256(response.read()).hexdigest()
            local = hashlib.sha256(path.read_bytes()).hexdigest()
            if remote != local:
                raise ValueError(f"Remote checksum mismatch: {repo}/{path.name}")
            hashes[path.name] = remote
        receipt["models"].append({"repo_id": repo, "commit": commit, "public_downloads_verified": hashes})
        print(f"Verified {repo}: {len(hashes)} public files match local release")
    (ROOT / "dist/hf-publication.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
