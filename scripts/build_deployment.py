"""Build a local Space source package; no account or network operations."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
out = root / "dist/space"
out.mkdir(parents=True, exist_ok=True)
for filename in ("app.py", "requirements.txt", "LICENSE"):
    shutil.copy2(root / filename, out / filename)
shutil.copy2(root / "requirements-space.txt", out / "requirements.txt")
shutil.copy2(root / "huggingface/space-README.md", out / "README.md")
shutil.copytree(root / "assets", out / "assets", dirs_exist_ok=True)
shutil.copytree(root / "superres", out / "superres", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(root / "artifacts", out / "artifacts", dirs_exist_ok=True)
shutil.make_archive(str(root / "dist/space-source"), "zip", out)
print(f"Prepared local Space at {out}. Training and publication remain pending approval.")
