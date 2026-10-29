"""Report generic runtime versions without local paths or hardware identifiers."""

import json
import platform

import torch
import torchvision

report = {
    "python": platform.python_version(),
    "platform": platform.system(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
print(json.dumps(report, indent=2))
