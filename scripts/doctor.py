"""Read-only environment report. Does not train or download weights."""
import json
import platform
import shutil
import sys
from pathlib import Path

import torch
import torchvision

root = Path(__file__).resolve().parents[1]
report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
          "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
          "cuda_available": torch.cuda.is_available(), "free_disk_gib": round(shutil.disk_usage(root).free / 2**30, 2)}
if torch.cuda.is_available():
    report.update(gpu=torch.cuda.get_device_name(), vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                  compute_capability=torch.cuda.get_device_capability())
    a = torch.ones(8, 8, device="cuda")
    report["cuda_forward_ok"] = bool((a @ a == 8).all())
print(json.dumps(report, indent=2))
