# Data sources and preparation

[Official DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) provides RGB PNGs: 800 training HR images, 100 validation HR images and corresponding MATLAB-generated bicubic LR pairs. The official 100 test HR images are withheld; this project never pretends to evaluate on them.

**License:** official page states academic research only; image copyrights remain with original owners. Do not redistribute images or use them as public Space examples by default. Public model/demo use and derived-weight terms need rights review before publication.

```powershell
.\.venv\Scripts\python.exe -m superres.acquire --extract
.\.venv\Scripts\python.exe -m superres.prepare
```

The downloader fetches six official HTTPS ZIP archives: train/valid HR and train/valid bicubic LR X2/X4. Transfers resume from `.part` files, save local SHA256 receipts, check ZIP CRC and extraction safety and enforce disk-space safeguards. Upstream SHA256 is not published; locally observed hashes establish repeatability only.

The manifest requires all 800/100 HR files, pairs by exact numeric IDs, checks RGB decode and scale geometry, records HR/LR checksums and rejects duplicated HR under distinct scene IDs. Same scenes at different scales remain in the same split. Protocol: training IDs 0001–0750, internal validation 0751–0800, **held-out test = official validation IDs 0801–0900**. This is a local protocol, not official challenge test. For comparisons publish this exact distinction and manifest hash. A scene is never patch-split across train/test.

An HR-only manifest creates separate locally degraded experiments:

```powershell
.\.venv\Scripts\python.exe -m superres.prepare --synthetic --output datasets/processed/div2k-synthetic.json
```

Pillow bicubic is not MATLAB bicubic. `realistic-synthetic` adds Gaussian blur, noise and JPEG; it is a controlled synthetic condition, not a claim to match actual camera noise. Paired training uses official LR; the degradation switch refuses to overwrite real or official LR. Patches align on LR coordinates, augmentations are applied jointly, tensors are RGB [0,1]. Evaluation uses full images, no randomness for provided pairs, and deterministic seeded synthesis for HR-only experiments.

Secondary: [RealSR authors' repository](https://github.com/csjcai/RealSR), Cai et al. ICCV 2019. Version 1: 234 scenes, 204/30 train/test; versions 2/3: 559 scenes, 459/100 train/test. V1/V2 HR and LR have equal dimensions. Use **version 3**, whose HR/LR sizes differ. The authors describe coarse alignment, central crop, fine alignment and rejection of misaligned pairs. Geometry checks do not prove alignment. No explicit dataset reuse license was found in the authors' repository; contact the owners / verify terms before use or redistribution. Drive/Baidu links may require interactive access; this project does not bypass it.

Manually obtain and align lawful RealSR v3 pairs, retain original scene/camera grouping, reserve validation scenes inside official train, preserve official test scenes, and create a reviewed CSV:

```csv
scene,scale,split,hr,lr,alignment_reviewed
Canon-001,2,train,Canon/HR/001.png,Canon/LR/001.png,true
```

`python -m superres.prepare --root datasets/raw/RealSR-v3 --realsr-csv datasets/realsr-pairs.csv --output datasets/processed/realsr-v3.json` requires all three splits per scale and reviewed alignment. Keep camera/scene variants in the same split. V1/V2 same-size files fail; do not silently resize to satisfy geometry. Train/evaluate separately from DIV2K. Manual alignment and verified RealSR rights are external prerequisites, not generated outputs.

Cite Agustsson & Timofte, *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study*, and Timofte et al., *Methods and Results*, CVPR Workshops 2017. For RealSR cite Cai et al., *Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model*, ICCV 2019.

Efficient HR-only setup after paired validation: python -m superres.prepare --synthetic --from-paired datasets/processed/div2k.json --output datasets/processed/div2k-synthetic.json. This reuses verified HR checksums and scene assignments rather than rereading every PNG. The synthetic manifest stores its source-manifest hash; generated degradations remain separately labeled at training/evaluation time.
