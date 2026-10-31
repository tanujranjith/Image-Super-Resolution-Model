# Image Super-Resolution Model

An independent local 2×/4× neural reconstruction project with compact residual and EDSR-style networks, original PyTorch training/inference code, official DIV2K pair validation and measured evaluation. **The recorded run status reports 60 completed training epochs per scale. Trained weights, configurations, checksums and saved evaluation reports are included in this repository.**

The interface follows the selected dark workbench design: mint controls, an input tool panel and a large draggable comparison canvas. Run `app.py` and open http://127.0.0.1:7861. Upload an image and select 2× or 4× to generate a learned reconstruction and download a PNG. Changing the image or scale clears old results. Model imports are lazy to keep the preview light. See [design QA](design-qa.md) and [asset provenance](assets/README.md).

## Included models

| Scale | Selected epoch | Parameters | Weights | Saved evaluation |
|---|---:|---:|---|---|
| 2× | 59 | 439,107 | [model.safetensors](artifacts/x2/model.safetensors) | [evaluation.json](artifacts/x2/evaluation.json) |
| 4× | 57 | 522,243 | [model.safetensors](artifacts/x4/model.safetensors) | [evaluation.json](artifacts/x4/evaluation.json) |

Each scale uses its validation-selected checkpoint from the completed 60-epoch run. The exports total about 3.9 MB and are stored directly in Git; Git LFS is not required. See [artifact details](artifacts/README.md) and the existing [research-use notice for weights](artifacts/LICENSE-WEIGHTS.md). MIT applies to source code. Datasets, the local Python environment and optimizer checkpoints are excluded.

## Setup and use

**Recorded training provenance:** the included model configurations reference the large-data manifest documented as 30,749 unique HR scenes from DIV2K and the additional Kaggle LSDIR corpus in [datasets/large/README.md](datasets/large/README.md). The exported metadata pins the manifest hash; the underlying local corpus and full run logs are not included for independent reproduction. Large configs use one repeat and separate run directories; the older DIV2K-only defaults below remain reproducible. No dataset download or training is needed to use the included weights. No additional training starts without approval.

Python 3.12 with a project-owned `.venv`; reference machine Windows 11 and RTX 3060 12 GiB.

```powershell
.\scripts\setup.ps1 -Python "C:\path\to\python.exe"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/doctor.py
.\.venv\Scripts\python.exe app.py
```

See [datasets/README.md](datasets/README.md) for data rights, full pairing rules, split conventions and RealSR limitations. The Gradio demo offers upload, scale selection, comparison slider, dimensions, timing and PNG download. It loads exported real weights once per scale and explains missing weights instead of pretending bicubic is learned reconstruction.

## Training — separate approval required

For the prepared large Kaggle corpus, use `scripts/train-large.ps1 -Background` only after explicit approval. It checks the readiness report and launches large x2/x4 EDSR runs sequentially. The commands below reproduce the smaller original DIV2K-only experiment. Large-data resource estimates, recovery and benchmark commands are in [datasets/large/README.md](datasets/large/README.md).

```powershell
.\.venv\Scripts\python.exe -m superres.acquire --extract
.\.venv\Scripts\python.exe -m superres.prepare
.\scripts\train.ps1 -Config configs/edsr_x2.yaml -Background
.\scripts\resume.ps1 -Config configs/edsr_x2.yaml -Checkpoint runs/edsr-x2-seed42/last.pt -Background
```

Separate configs/checkpoints for each scale; no automatic multi-run launch. Default EDSR-style network: 8 residual blocks, 48 features, residual scale 0.1, mean shift, pixel shuffle, no batch normalization. Compact baseline: 4 residual blocks, 32 features and learned residual over bicubic. Bicubic is evaluated only, never presented as a learned model. Adam, L1/Charbonnier/MSE choices, cosine decay, AMP, clipping and early stopping on validation Y-PSNR. Optional VGG feature loss is off by default; enabling downloads ImageNet weights and requires a separately labeled perceptual experiment. No GAN losses. Save best/last every epoch and periodic snapshots every five epochs.

RTX 3060 planning defaults: EDSR x2 LR48 patches/batch12; x4 LR32/batch8. Expect roughly 2–6 GiB VRAM plus display allocation; actual backward-pass peaks are unmeasured pending approval. Lower batch/patch sizes if needed and use a new run directory. Tiled validation/inference uses a conservative LR halo covering all convolutions, crops tile edges and assembles on CPU. GPU memory depends on tile size rather than full output size; output capped at 16 MP (approximately 192 MB float RGB, plus display/encoding copies in host RAM).

For a future reproduction of the smaller DIV2K-only defaults, budget **6–18 hours per scale** for 60 epochs as a provisional estimate, strongly dependent on PNG decode and full-image validation; measure first approved epoch before planning overnight completion. There are 15,000 sampled training patches per epoch under the default repeat count. This is a planning estimate, not a measurement of the completed large-data runs. Budget 20–35 GB for archives/extracted pairs and several GB for experiment history. Windows uses zero workers by default to avoid duplicating Python/Torch memory. Increase to two workers only if free RAM and decoding measurements justify it. Background launch records PID/stdout/stderr and survives the coding agent. Checkpoints recover the last completed epoch including optimizer/scheduler/scaler/RNG state; incomplete epochs replay. Keep PC awake manually; scripts never change power settings.

## Evaluate and export

```powershell
.\.venv\Scripts\python.exe -m superres.evaluate --manifest datasets/processed/div2k.json --scale 2 --output runs/bicubic-x2
.\.venv\Scripts\python.exe -m superres.evaluate --checkpoint runs/edsr-x2-seed42/best.pt --output runs/edsr-x2-test
.\.venv\Scripts\python.exe -m superres.export --checkpoint runs/edsr-x2-seed42/best.pt
```

Repeat explicitly for x4 after approved training. `--lpips` requires requirements-optional.txt and explicitly downloads the LPIPS/AlexNet weights; it is not enabled by tests or defaults. Evaluation reports per-image/arithmetic-mean Y-PSNR, SSIM, bicubic baseline, RGB MAE, warm tiled latency, peak allocated VRAM, parameters/size, genuine side-by-side comparisons and error heatmaps. Quantize/clamp outputs to 8-bit RGB, convert to BT.601 limited-range Y, shave scale pixels, SSIM uses 11×11 Gaussian sigma1.5 valid windows. LPIPS uses RGB [-1,1] without luminance conversion. Exact PSNR infinity is represented as null with an exact-match flag. Different metric/downsample conventions are not interchangeable with published benchmarks.

Learned detail is a plausible estimate of the original. Synthetic bicubic training generalizes poorly to some real noise, compression, text, faces and plates. Do not treat reconstruction as evidence. See [EXPERIMENTS.md](EXPERIMENTS.md), [VALIDATION.md](VALIDATION.md) and [publication guide](huggingface/PUBLISH.md). MIT covers source code; source images and trained weights need separate rights review.



Reference training configs explicitly require CUDA and fail fast if the GPU is unavailable. Use device: cpu only for an intentionally configured CPU experiment. Spaces use the separate CPU-wheel requirements-space.txt, copied automatically by scripts/build_deployment.py; local environments retain CUDA wheels.
