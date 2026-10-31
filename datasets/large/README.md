# Large Kaggle HR corpus

## Current artifact status

The repository now includes x2 and x4 trained exports and saved evaluation reports in [artifacts/](../../artifacts/README.md). Both exported configurations reference `datasets/large/processed/training.json`, manifest SHA-256 `6b8726365559e9f72b8e944e6b59464b16e1799d852e3a379b1006907f36afa9`, with selected checkpoint epochs 59 (x2) and 57 (x4). The run status records completion of 60 epochs per scale; the selected epoch is not the total run length.

The preparation/readiness information below describes the earlier data-preparation and launch workflow. In particular, `training_started: false` in historical readiness/preparation records must not be read as proof that the included exports do not exist. Those records are also not permission to launch another run. Full local run logs and optimizer checkpoints are not included, so repository inspection alone cannot independently reproduce the complete training history or revalidate the local corpus. Any new training still requires explicit approval.

## Source and rights

Prepared counts on October 17, 2026, after removing one exact duplicate:

| Partition | HR scenes |
|---|---:|
| Training: Kaggle LSDIR | 29,999 |
| Training: original DIV2K | 750 |
| **Total training** | **30,749** |
| DIV2K synthetic validation / test | 50 / 100 |
| Kaggle LSDIR held-out validation / test | 125 / 125 |

This is about **41×** the original 750-scene training set. x2/x4 share the same scenes; their manifest rows are not additional photographs. The 30,250 acquired Kaggle HR files occupy **56,640,253,584 bytes (52.75 GiB)**, including the quarantined duplicate. Full readiness is recorded separately in `readiness.json`; these are dataset counts, not model-performance claims.

All new image bytes come from [Kaggle: parampandya0310/hr-image-denoising-lsdir](https://www.kaggle.com/datasets/parampandya0310/hr-image-denoising-lsdir), version 1. Its inspected ZIP contains **30,000 training HR PNGs and 250 validation HR PNGs**, about **56.64 GB (52.75 GiB)** uncompressed. This is a subset of LSDIR, not its complete 84,991-image training set. Actual retained counts come from `processed/audit.json` after size checks, decoding and duplicate filtering.

The [original LSDIR project](https://ofsoundof.github.io/lsdir-data/) describes manually quality-checked Flickr HR photos for restoration. Attribution: Yawei Li and coauthors, *LSDIR: A Large Scale Dataset for Image Restoration*, CVPR Workshops 2023. The Kaggle listing uses a CC BY 4.0 metadata label, but the original project explicitly limits the dataset to **academic research purposes**, with image copyrights retained by their owners. The mirror label does not establish broader rights. Keep raw images local; do not package them as demo examples or redistribute them. Public weight/demo licensing needs separate review before publication.

The mirror's validation directories contain 250 HR scenes plus 750 provided x2/x3/x4 LR copies. Only the 250 HR scenes are selected; LR copies are never counted as additional photos. This matches the original project's advertised public validation count, but the mirror's image provenance has not been independently authenticated. Report it as **Kaggle LSDIR held-out**, not the official hidden LSDIR test benchmark.

## Acquisition and integrity

From this project's directory, acquire or resume preparation:

```powershell
.\.venv\Scripts\python.exe -m superres.large_acquire
.\.venv\Scripts\python.exe -m superres.large_prepare
.\.venv\Scripts\python.exe scripts/validate_large.py
```

The downloader reads the pinned Kaggle ZIP64 central directory and bounded HTTP ranges. The server must honor the exact byte range and archive size. Standard `zipfile` checks each member's CRC before PNG decode and SHA-256/pHash computation. A single network worker prefetches one range while the main process verifies/writes images. Buffers are about 64 MiB each, plus one decoded image. No full ZIP, precomputed LR copies or corpus-sized RAM cache is retained. Remaining image storage plus a 64 GiB free-space reserve is required. TLS verification remains enabled.

`images.jsonl` records version, original member path/CRC, image SHA-256, dimensions, full/mirrored perceptual hashes, local size and modification time. Resume skips previously verified files whose size/time are unchanged; changed files are downloaded and fully reverified. This efficient cache check does not rehash all 57 GB on every run and cannot detect deliberate same-size/same-time tampering. Delete a specific journal receipt to force reacquisition. Partial trailing journal lines are recovered at restart. Create `datasets/large/PAUSE` to stop at a verified interval boundary; remove it to resume. Never run duplicate acquisition jobs.

Accept opaque RGB/grayscale PNGs with minimum dimension 256 and at most 16 megapixels. Corrupt files, unsupported modes and unsafe geometry are excluded and counted. These bounds support the configured patches and limit decode memory.

## Splits

Whole HR scenes determine splits before making patches or either scale. SHA-256 plus full/mirrored 64-bit DCT pHash, Hamming radius 6, form transitive duplicate components across existing DIV2K and new photos.

- `processed/training.json`: filtered DIV2K training plus retained Kaggle LSDIR training. Validation/test use protected DIV2K held-out scenes with Pillow-generated LR. Held-out splits take priority if an old component conflicts; conflicting training scenes are excluded from this new manifest.
- `processed/lsdir-heldout.json`: mirror-validation components split deterministically about half validation/half test. Components overlapping mirror training retain validation members only.
- Original `datasets/processed/div2k.json` and image files remain unchanged. Its provided MATLAB-bicubic pairs form a separately named benchmark after training.

Both scales and all variants of a component stay in one split. Exact new copies are removed, cross-corpus matches quarantined. `processed/audit.json` records counts, exclusions and fingerprints. An independent checker searches all retained full/mirror hashes across splits. These hashes cannot establish absence of arbitrary crops or different photographs of the same scene.

## Training condition and resources

Clean HR photos become LR/HR pairs through scale-aligned **Pillow bicubic** downsampling. Training selects a random patch coordinate, downsamples a context-padded HR crop, and removes the context; tests verify exact agreement with cropping full-image downsampling. LR/HR augmentations are joint. Evaluation is deterministic, without training augmentation. Pillow bicubic, provided DIV2K MATLAB bicubic, realistic synthetic noise/JPEG and RealSR camera pairs are distinct conditions. This corpus contains no real camera LR/HR pairs.

`configs/edsr_large_x2.yaml` and `edsr_large_x4.yaml`: separate EDSR-style networks, 48 features/8 blocks, L1/Adam, AMP, cosine scheduling, validation early stopping, periodic checkpoints and optimizer/scaler/RNG recovery. One repeat gives one random patch per training scene per epoch, roughly 30,000 patches versus the old 15,000-patch budget. More scenes do not automatically establish better restoration. Saved evaluation results for the included exports are in `artifacts/x2/evaluation.json` and `artifacts/x4/evaluation.json`; future-run quality and timing are not guaranteed by these historical results.

Windows uses zero DataLoader workers and four CPU math threads. x2: LR48/batch12; x4: LR32/batch8. Plan roughly 2–6 GiB GPU memory; measure actual backward peaks after approval. A provisional **12–36 hours per scale** accounts for HDD decoding, twice the sample budget and full-image validation. This is not a benchmark or overnight-completion promise. Replace it with first-epoch measurements after approval. Additional benchmarks run once after training. Scales run sequentially.

One HR copy supports both scales. Keep at least 64 GiB free beyond the data, plus a few GB for checkpoints/reports. No power settings, paid services or cloud jobs are changed.

## Validation and future launch

After preparation, validate actual crops on CPU, run tests and finalize readiness:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.\.venv\Scripts\python.exe scripts/validate_large.py
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp datasets/large/test-temp --junitxml datasets/large/validation-tests.xml
.\.venv\Scripts\python.exe scripts/validate_large.py --finalize
```

The validator pins dataset/benchmark fingerprints, verifies split/file identities and performs CPU forward passes on real crops. No optimization occurs. Random test weights are never deployed. `readiness.json` is generated only after passing checks; changes to code/config/manifests require revalidation.

**Only after explicit approval**, in a fresh PowerShell session without `CUDA_VISIBLE_DEVICES=-1`:

```powershell
.\scripts\train-large.ps1 -Background
# Alternatively queue after an approved expanded-car run, without a 5 AM cutoff:
.\scripts\train-large.ps1 -Background -AfterCarStatus '..\What Car Is This\runs\overnight-current.json' -AfterCarExperiment '..\What Car Is This\runs\expanded-efficientnet_b0-seed43'
```

Use one command, not both. The launcher checks fingerprints and duplicate supervisors. The optional car gate waits for training/evaluation/export completion of the explicitly named experiment. A stale completed v1 status cannot release the expanded-run queue. The supervisor evaluates synthetic DIV2K, provided DIV2K and Kaggle LSDIR held-out separately, then verifies exported inference. Monitor `runs/overnight-current.json` and named logs. Recover a stopped scale with `scripts/resume.ps1`, the identical large config and its `last.pt`; explicitly evaluate/export that scale before continuing. Incomplete epochs replay. Existing experiments are never implicitly overwritten. A reminder does not invoke training. Readiness indicates readiness **for approval**, not approval to run.
