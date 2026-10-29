# Conditional overnight approval — October 17, 2026

The user permits this project to start only after the car run completes and the SR start can occur **before 5 AM America/Indianapolis on October 17, 2026 (09:00 UTC)**. This is a one-night authorization, not a recurring daily permission.

The standalone `scripts/overnight.py --after-car-status <car-status.json> --deadline-utc 2026-10-17T09:00:00+00:00` gate waits using only small JSON reads. It does not import Torch or reserve GPU memory while waiting. If the deadline passes, or car training fails, it records a non-start reason and exits. Pure tests cover complete/running/failed car states, the exact deadline and later dates.

If eligible, it trains the configured EDSR-style **2× then 4×** models sequentially, each to completion/validation early stopping. After each scale it evaluates against bicubic on the held-out split, exports genuine weights and checks output dimensions on a held-out LR image. CPU threads are limited to four, DataLoader worker processes remain zero, and periodic checkpoints/provenance are retained. OOM retries use smaller batches in separately recorded experiment directories, with a fixed retry limit.

The status file is `runs/overnight-current.json`; per-stage logs and GPU telemetry live under its supervisor directory. The waiting state is not an actual training start. No extra architectures, perceptual-weight download, cloud compute or publication is approved.
