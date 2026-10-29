# Bounded image loading for the large super-resolution runs

Prepared October 18, 2026. **Training remains stopped and requires fresh explicit
approval.** Changing loading code or preparing configs does not authorize a run.

Both `configs/edsr_large_x2.yaml` and `configs/edsr_large_x4.yaml` now use:

```yaml
workers: 0
decode_threads: 2
prefetch_batches: 2
sampling: epoch-index-v1
max_decode_pixels: 16000000
```

Two threads decode and prepare independent images concurrently. One producer
prepares future CPU batches while the training thread can run the current batch
on the GPU. The prefetch window holds at most two future batches, including any
being prepared; it does not read another epoch or cache the corpus. Worker
processes remain disabled to avoid duplicating Python/Torch memory on Windows.
Pinned host memory and nonblocking device copies are retained when CUDA is used.

The implementation also reuses already-aligned RGB buffers instead of making
unconditional full-resolution EXIF/RGB/modcrop copies. Validation datasets retain
only their split and scale, rather than a second copy of the whole manifest.
Large decoded images are closed when their patches have been converted to
tensors. At two decoding threads, no more than two image-processing tasks run
at once. Patch batches are small, but decoded PNGs and Torch still consume RAM:
this is bounded buffering, not a guarantee of a specific total working set.

## Reproducibility and evaluation

Each training patch gets its own random generator keyed by seed, epoch, dataset
index and scale. Thread completion order cannot change the crop or augmentation.
Sampler order and its dedicated Torch generator are preserved. The trainer sets
the epoch before creating that epoch's iterator, and saves RNG state after the
iterator drains. This supports the existing completed-epoch checkpoint recovery.

This is the explicit `epoch-index-v1` sampling scheme. Older configs retain the
legacy global-RNG behavior when threading/prefetching is disabled. Existing
checkpoints still require the identical resolved config; do not silently change
sampling schemes during a resumed experiment.

Validation/test images stay full resolution, deterministic, single-process and
unprefetched. Synthetic Pillow bicubic, official paired bicubic, and real-world
degradation conditions remain distinct. Source image bytes, splits, scale,
architectures, reconstruction losses and dataset hashes have not been changed.

The training log now records cumulative `data_wait_seconds` per epoch. It measures
time waiting for the loader, including startup, rather than pretending to measure
GPU utilization. Actual GPU utilization and end-to-end training speed must be
measured during a separately approved training run.

## Validation and bounded profiling

The first 19 loading-only cases passed: identical serial/threaded patches and
sample order at both scales, synthetic/paired conditions, epoch-boundary replay,
global RNG isolation, bounded concurrency/prefetch, early-exit/error cleanup,
pixel-limit checks and unchanged full-resolution validation.

Extended checks are deferred while the user is actively using the memory-limited
PC. The new readiness-stop preservation regression, complete suite, actual-image
forward checks and throughput comparison must pass before updating readiness
fingerprints and launching training. Existing user-stop gates remain in place.

When the PC is idle, the following are **non-training** checks, run sequentially
from this project only. These commands do not authorize subsequent training:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
.\.venv\Scripts\python.exe -m pytest -q --basetemp=datasets/large/test-temp-loading-all --junitxml=datasets/large/validation-tests.xml
.\.venv\Scripts\python.exe scripts/validate_loader.py
.\.venv\Scripts\python.exe scripts/benchmark_loader.py --images 16
.\.venv\Scripts\python.exe scripts/validate_large.py --finalize
```

`validate_loader.py` uses only two existing scenes per scale, CPU inference and
zero optimizer steps. `benchmark_loader.py` is CPU loading only, with a small
fixed subset, warmup and alternating serial/threaded repeats. It checks identical
patch bytes/order and records process peak working set on Windows. The OS cache
is not flushed, so its results are a bounded warm-cache comparison, not promised
training speedups. It never calls a model or an optimizer. No benchmark results
have been claimed yet.

Finalization preserves `user_stop_requested` and the revoked-approval status.
Only fresh human training approval may clear that gate. For future tuning, try
four decoder threads only after measuring RAM and HDD contention; do not infer
that more threads or a fuller CPU meter guarantees higher throughput. Keep the
two-thread preset for the first validation. No disk-sized caches or LR copies
are created.

## Approved launch — October 19, 2026

The deferred checks subsequently passed: 80 CPU tests, one CUDA-only skip,
configured-model actual-image CPU forwards at both scales and identical serial/
threaded patches. The 16-scene warm-cache loader benchmark measured approximately
1.71x (2x) and 1.60x (4x) versus serial mode, with 640 MiB peak process working
set. See `runs/loader-benchmark.json`; these are not GPU training speedups.

Following fresh explicit human approval, the SR-only controller launched 2x
training, with 4x queued. The previously authorized Monday 3 PM local cutoff
remains in force pending an explicit change. Early full-corpus shuffled HDD
reads still cause roughly 90% loader wait and low GPU utilization. Checkpoints,
evaluation/export and cutoff enforcement are managed locally by
`scripts/approved_run.py` and `scripts/overnight.py`. Scheduled chat monitoring
was unavailable in the current toolset. See `runs/launch-verification.json`.
