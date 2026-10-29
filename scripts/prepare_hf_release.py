"""Prepare allowlisted, image-free Hugging Face model release folders."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    for scale in (2, 4):
        out = ROOT / f"dist/hf-model-x{scale}"
        out.mkdir(parents=True, exist_ok=True)
        artifact = ROOT / f"artifacts/x{scale}"
        cfg = json.loads((artifact / "config.json").read_text())
        checksums = json.loads((artifact / "checksums.json").read_text())
        for name, expected in checksums.items():
            if name not in ("config.json", "model.safetensors"):
                raise ValueError("Unexpected artifact")
            assert hashlib.sha256((artifact / name).read_bytes()).hexdigest() == expected
            shutil.copy2(artifact / name, out / name)
        shutil.copy2(artifact / "checksums.json", out / "checksums.json")
        shutil.copy2(ROOT / "LICENSE", out / "LICENSE-CODE")
        shutil.copy2(ROOT / "requirements.txt", out / "requirements.txt")
        reports = {}
        for report in ("evaluation-test", "benchmark-div2k-official", "benchmark-lsdir-heldout"):
            reports[report] = json.loads((ROOT / f"runs/edsr-large-x{scale}-seed42/{report}/metrics.json").read_text())
            assert reports[report]["summary"]["checkpoint_sha256"] == cfg["checkpoint_sha256"]
        (out / "evaluation.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
        with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted((ROOT / "superres").glob("*.py")):
                z.write(path, path.relative_to(ROOT).as_posix())
            for path in sorted((ROOT / "configs").glob("*.yaml")):
                z.write(path, path.relative_to(ROOT).as_posix())
            for name in ("requirements.txt", "LICENSE"):
                z.write(ROOT / name, name)
        (out / "LICENSE-WEIGHTS.md").write_text('''# Research-use notice for trained weights

These weights are released for academic research evaluation. No permissive or
commercial-use license is asserted for the weights. MIT applies only to the
original source code (LICENSE-CODE), not to training images or trained weights.

Both upstream datasets state academic research use only and retain image
copyright with the original owners:
- DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
- LSDIR: https://ofsoundof.github.io/lsdir-data/

This notice does not replace upstream terms or establish that dataset terms
explicitly grant weight redistribution rights. No such explicit grant was found
in the reviewed dataset notices. Availability does not confer rights to the
source imagery or permission for commercial use. Obtain any additional rights
needed from the relevant rights holders. No dataset images are distributed.
''', encoding="utf-8")
        rows = []
        for name, report in reports.items():
            s = report["summary"]
            rows.append(f"| {name} | {s['images']} | {s['psnr_y']:.4f} | {s['ssim_y']:.6f} | {s['bicubic_psnr_y']:.4f} | {s['bicubic_ssim_y']:.6f} |")
        params = reports['evaluation-test']['summary']['model_parameters']
        card = f'''---
library_name: pytorch
pipeline_tag: image-to-image
tags: [super-resolution, image-restoration, edsr, research]
license: other
license_name: academic-research-use-notice
license_link: https://huggingface.co/Tanuj124/image-super-resolution-edsr-x{scale}/blob/main/LICENSE-WEIGHTS.md
---

# Compact EDSR Image Super-Resolution {scale}×

An independently implemented PyTorch reconstruction model trained locally on an
RTX 3060 12 GB. This is a learned {scale}× upscaler with {params:,} parameters.
Both scale experiments completed 60 epochs; this export is the validation-selected
checkpoint from **epoch {cfg['epoch']}**. Trained weights are in safetensors format.

## Architecture and training

EDSR-style network: 8 residual blocks, 48 feature channels, residual scaling 0.1,
RGB mean shift and pixel shuffle; no batch normalization, GAN or perceptual loss.
Adam, L1 reconstruction loss, initial learning rate 0.0002, cosine schedule,
mixed precision, gradient clipping 1.0, seed 42. LR patch size {cfg['patch_size']},
batch size {cfg['batch_size']}. Separate checkpoints are trained for each scale.
Validation selects weights using Y-PSNR; held-out test results are not used for selection.
The complete resolved configuration and checkpoint/manifest hashes are in config.json.

## Training data and provenance

30,749 unique HR training scenes: 750 DIV2K images (0001–0750) and 29,999
LSDIR images acquired from Kaggle
[parampandya0310/hr-image-denoising-lsdir version 1](https://www.kaggle.com/datasets/parampandya0310/hr-image-denoising-lsdir).
One exact duplicate was excluded. The Kaggle mirror's metadata does not override
the original dataset terms. Provided LR copies do not count as additional photos.
Training LR images use Pillow bicubic synthetic degradation; no RealSR training.
DIV2K 0751–0800 provides internal validation; 0801–0900 is the local held-out test.
The 250-image Kaggle LSDIR held-out subset is split into 125 validation and 125 test scenes.
Exact and full/mirror perceptual groups are isolated across splits; arbitrary crops
or repeated scenes cannot be ruled out exhaustively.

## Measured evaluation

All values below come from saved evaluation runs of this exported checkpoint.
The DIV2K test here is the publicly available official validation HR set, not
the unavailable challenge hidden-test HR. Kaggle LSDIR held-out provenance is
not independently authenticated; it is not an official hidden-test benchmark.

| Condition | Images | Y-PSNR dB | Y-SSIM | Bicubic dB | Bicubic SSIM |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Protocol: clamp/quantize to 8-bit RGB, BT.601 limited-range Y, shave {scale} pixels,
valid 11×11 Gaussian SSIM (sigma 1.5), per-image arithmetic means. Conditions use
different LR sources and should not be pooled. evaluation.json contains full
per-image metrics, matching checkpoint hashes, RTX 3060 latency and peak-memory
records. Those timings are GPU measurements, not CPU hosting promises. LPIPS was
not evaluated. No additional architecture-comparison result is claimed.

## Inference

Download the repository, install requirements.txt, and extract source.zip into
the same folder. From that folder:

```python
from pathlib import Path
from PIL import Image
from superres.inference import Upscaler

model = Upscaler(Path('.').resolve(), device='auto')
with Image.open('input.png') as image:
    output = model.upscale(image, tile_size=128)
output.save('output.png')
```

CPU and CUDA inference are supported. Install the appropriate PyTorch build for
your machine. Tiled inference uses a convolution halo and assembles output on CPU;
outputs above 16 megapixels are rejected to limit host memory use. No remote API
or Transformers AutoModel integration is implied. source.zip includes original
model, inference, training, acquisition/preparation and evaluation code plus configs.
Training datasets and optimizer checkpoints are not included.

## Intended use, limits and rights

Academic image-restoration research and educational evaluation. Learned details
are estimates and can be wrong, especially in text, faces, plates, noise and JPEG
artifacts. Bicubic training does not establish performance on real camera degradation.
Do not use reconstructed details as forensic, medical or identity evidence.

Read [the weight research-use notice](LICENSE-WEIGHTS.md) before use. Source code is
MIT; a permissive license for trained weights is not asserted. The dataset notices
do not explicitly establish redistribution rights for trained weights. Source
images and sample reconstructions from those images are not included in this release.

## References

- Agustsson and Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study*, CVPR Workshops 2017. [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
- Li et al., *LSDIR: A Large Scale Dataset for Image Restoration*, CVPR Workshops 2023. [LSDIR](https://ofsoundof.github.io/lsdir-data/).
- Lim et al., *Enhanced Deep Residual Networks for Single Image Super-Resolution*, CVPR Workshops 2017. This repository uses an original compact EDSR-style implementation, not authors' pretrained weights.
'''
        (out / "README.md").write_text(card, encoding="utf-8")
        print(json.dumps({"scale": scale, "folder": str(out), "files": [p.name for p in out.iterdir() if p.is_file()], "export_epoch": cfg["epoch"]}))


if __name__ == "__main__":
    main()
