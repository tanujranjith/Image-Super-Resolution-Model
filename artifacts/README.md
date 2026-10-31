# Trained model exports

The recorded run status reports that both EDSR-style scale experiments completed 60 epochs. The included `x2/` and `x4/` folders contain the validation-selected exports from epochs 59 and 57, respectively.

Each folder includes:

- `model.safetensors`: trained inference weights.
- `config.json`: resolved configuration, selected epoch and checkpoint/manifest hashes.
- `checksums.json`: SHA-256 hashes verified when the model loads.
- `evaluation.json`: saved per-image metrics and summaries for the matching checkpoint under three evaluation conditions.

Each network has 8 residual blocks and 48 feature channels. The 2× model has 439,107 parameters; the 4× model has 522,243. These are trained exports, not random test fixtures. Dataset images and optimizer checkpoints under `runs/` are excluded.

Read [LICENSE-WEIGHTS.md](LICENSE-WEIGHTS.md) for the existing research-use notice. The repository's [MIT license](../LICENSE) applies to original source code, not to training images or trained weights.

## Evidence and limits

Both exported configurations identify the large training manifest and its SHA-256, selected checkpoint epoch, checkpoint SHA-256, and Pillow-bicubic synthetic training condition. Each evaluation file contains separate `evaluation-test`, `benchmark-div2k-official`, and `benchmark-lsdir-heldout` reports. Their checkpoint and training-manifest hashes match the corresponding exported configuration. Do not combine these conditions into a single benchmark or call the local held-out split the official hidden challenge test.

A file checksum verifies integrity against the supplied manifest, not training provenance, performance reproducibility, or redistribution permission. The full run logs, source training data and optimizer checkpoints are not included. The 60-epoch completion claim comes from the recorded run status, while exported configurations directly identify selected epochs 59 and 57.

`VALIDATION.md` is explicitly a historical pre-training record. `TRAINING_STATUS.json` and `datasets/large/readiness.json` retain older nested preparation/approval fields; they are not a coherent current launch authorization and must not trigger a new run. No launch/gate records, model bytes or checksums were changed in this documentation reconciliation.

The research-use notice is not an independent clearance of source-image or weight redistribution rights. That review remains separate.
