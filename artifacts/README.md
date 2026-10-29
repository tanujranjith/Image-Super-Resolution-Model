# Trained model exports

Both EDSR-style scale experiments completed 60 epochs. The included `x2/` and `x4/` folders contain the validation-selected exports from epochs 59 and 57, respectively.

Each folder includes:

- `model.safetensors`: trained inference weights.
- `config.json`: resolved configuration, selected epoch and checkpoint/manifest hashes.
- `checksums.json`: SHA-256 hashes verified when the model loads.
- `evaluation.json`: saved per-image metrics and summaries for the matching checkpoint under three evaluation conditions.

Each network has 8 residual blocks and 48 feature channels. The 2× model has 439,107 parameters; the 4× model has 522,243. These are trained exports, not random test fixtures. Dataset images and optimizer checkpoints under `runs/` are excluded.

Read [LICENSE-WEIGHTS.md](LICENSE-WEIGHTS.md) for the existing research-use notice. The repository's [MIT license](../LICENSE) applies to original source code, not to training images or trained weights.
