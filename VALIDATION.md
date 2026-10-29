# Technical validation — 2026-10-15

**Status: implemented and validated for the first approved local training run. Training and publication have not started.**

- 26 unit/integration tests passed, 0 failed, 0 skipped. JUnit evidence: `validation-tests.xml`.
- Imports/bytecode compilation, Ruff checks, editable package installation and pip dependency checks passed.
- Local Gradio HTTP startup and `/config` both returned 200; server closed afterward. SDK pinned to 5.49.1.
- Windows PowerShell setup, train, resume, download and validation scripts parsed successfully.
- Coverage: Compact residual / EDSR-style / bicubic at 2x and 4x, tiled/full output equivalence, pair geometry, deterministic degradations, separate synthetic conditions, PSNR/SSIM formulas, fixture evaluation/comparison outputs, export/load, CUDA AMP forward and honest missing weights.
- CPU and CUDA fixture forwards passed. Fixture checkpoints and safetensors loaded successfully. No optimizer updates or substantive training were executed.

## Actual prepared data

900 public HR scenes, 1800 official LR/HR pairs. Each scale: 750 train, 50 validation, 100 held-out local test scenes. Official challenge test HR is unavailable. Separate synthetic manifest prepared.

Prepared manifest SHA256: `184255cb609a3de216c7a5d0ccbe947982ea16a9ed1d8571fb5626d225769569`. See `datasets/PROVENANCE.json` and the detailed ignored local manifests/audits under `datasets/processed/`. Split isolation checks passed.

## Verified environment and resource choices

Python 3.12.14, PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126, CUDA runtime 12.6; NVIDIA GeForce RTX 3060, 12 GiB VRAM, SM 8.6. Ryzen 7 5700G and approximately 16 GB system RAM. Windows 11. D: is a SATA HDD per the user's hardware evidence.

Each project owns its environment, dependencies, Git repository, configs, data, checkpoints and deployment source package. Training defaults use zero DataLoader workers to avoid duplicating Torch memory, require CUDA, use AMP, and record resolved config, source fingerprint, dataset fingerprint, RNG and optimizer/scheduler/scaler state. Data work ran sequentially; redundant reads were reduced and obsolete partial extraction was removed. No power settings changed.

## Explicit limits of pre-training validation

Actual model quality, calibration/rejection quality, real training throughput and backward-pass peak VRAM, semantic Grad-CAM usefulness, convergence and interrupted real-run recovery remain to be measured after approval. Only temporary fixtures generated evaluation numbers/plots during tests; these are not portfolio performance results and are not deployment weights.

Optional LPIPS/VGG perceptual weights were not downloaded or exercised. Secondary dataset access, reviewed Cars196 mappings / RealSR alignment and source-rights review remain separate from the ready primary-data runs. Linux/hosted Space execution and account eligibility are not claimed to be verified; local SDK startup and CPU source packages are prepared. No account publication or paid cloud operation occurred.

Upstream Pillow/matplotlib/pyparsing deprecation warnings were non-blocking with the pinned versions. No unresolved local implementation blocker remains.

## Selected dark interface update

Three targeted UI checks passed per project, including two new presentation regression tests. Final helper checks passed again after lazy model imports. Desktop/mobile browser comparisons and primary pre-training interactions passed; see `design-qa.md` and `design/` evidence. CPU Space source packages include licensed local assets and passed integrity checks. Neither training nor publication started.
