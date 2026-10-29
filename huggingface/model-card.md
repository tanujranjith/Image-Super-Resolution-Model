---
library_name: pytorch
pipeline_tag: image-to-image
tags: [super-resolution, image-restoration, research]
license: other
license_name: pending-weight-license-review
---

# Local Image Super-Resolution — SCALE PENDING

**No approved training or measured benchmark performance yet.**

## Architecture and training
Original compact residual or EDSR-style reconstruction; exact scale/features/blocks/mean shift recorded in config.json. Independent x2/x4 checkpoints, RGB [0,1], pixel shuffle, no batch normalization or GAN. Adam, reconstruction loss, cosine LR, AMP and early stopping on validation Y-PSNR. Optional perceptual features off by default. Fill resolved config, epochs, seed, GPU and runtime only from actual run artifacts.

## Data provenance
DIV2K official paired PNGs, academic research only; cite Agustsson/Timofte CVPRW2017 and review image/derived-weight rights. Record source URLs, archive SHA256 receipts, manifest hash and exclusions. Train IDs0001:0750, internal validation0751:0800, local test official validation0801:0900. Official challenge test HR is unavailable. Synthetic Pillow and RealSR experiments remain separately labeled; no implicit merging.

For the separately configured large-data experiments, training additionally uses a Kaggle LSDIR subset from `parampandya0310/hr-image-denoising-lsdir` version 1. Cite Li et al., *LSDIR*, CVPRW2023. The archive contains 30,000 training and 250 validation HR photos before filtering. Fill actual retained counts from `datasets/large/processed/audit.json`; do not confuse x2/x4 manifest rows or sampled patches with unique images. Source terms are academic-research-only despite the mirror's broader metadata label; publication rights remain pending. Whole-scene exact/full/mirror perceptual components isolate held-out scenes across both corpora. New training uses Pillow bicubic; original provided DIV2K pairs and Kaggle LSDIR held-out are separate benchmarks. The Kaggle validation mirror is not an authenticated official hidden-test benchmark. See `datasets/large/README.md` for provenance limitations and filtering rules.

## Evaluation
Pending real runs. Report bicubic/compact/EDSR results for matching scales and manifest, quantized limited-range Y-PSNR, scale shave, 11x11 valid Gaussian SSIM, optional RGB LPIPS, per-image averaging, warmed tiled latency, tile size, peak VRAM and parameter count. Link actual errors/comparisons. Never insert literature results or synthetic fixture metrics as model performance.

## Intended use and limitations
Educational reconstruction research. Neural outputs estimate missing details; small text, faces and plates can be inaccurate. Real camera images differ from bicubic training. Not appropriate for forensic, medical or factual-detail recovery. Output limit16MP; CPU inference may be slow. No remote inference API wrapper.

## License
MIT source code only. Weight license pending source-rights review. DIV2K image rights remain with original owners and data is academic-research-only. RealSR rights are not assumed. No dataset images or untrained example outputs are distributed.

