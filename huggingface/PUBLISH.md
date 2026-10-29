# Deployment preparation — no publication performed

1. Train only after separate approval, evaluate each scale on held-out pairs, inspect real comparisons and complete cards with observed results.
2. Export best x2 and x4 checkpoints with `python -m superres.export --checkpoint ...`.
3. Prepare independent model repos (or scale subfolders): exported safetensors/config/checksums, actual evaluation JSON, completed card renamed README.md. Keep optimizer checkpoints private/local.
4. Space files: app.py, superres/, requirements.txt, LICENSE, artifacts/x2 and artifacts/x4. Copy space-README.md to root README.md. Git LFS for weights. Do not upload datasets, test fixtures or generated outputs masquerading as benchmark examples.
5. Clean Python3.12 install with the pinned requirements. CPU Torch on Spaces; CUDA install is a local Windows setup choice. Test upload/scale/errors/PNG download and before-after slider with actual weights. CPU Basic Spaces may need smaller images/tiles and have slow inference.
6. Review DIV2K academic-only terms, original LSDIR academic-research terms (not just the Kaggle mirror's CC BY metadata) and RealSR rights before choosing weight/usage license. Record actual retained counts, pinned Kaggle version and distinct benchmark conditions for large-data runs. The Kaggle validation subset is not an authenticated official hidden-test benchmark. Use owned example photos. Ask for explicit account-publication approval before creating/pushing repositories or Spaces.

GitHub: initialize this folder separately, exclude .venv/raw data/runs, and record source provenance and validation report. No dependency on the car project.

Run scripts/build_deployment.py to prepare dist/space-source.zip with CPU-specific requirements for Linux Spaces. This does not create or upload any repository. Windows CUDA lockfiles are environment records, not Space requirements.

Check current hosting/account eligibility before publication: https://huggingface.co/docs/hub/spaces-overview#hardware-resources . CPU Basic has no hourly hardware charge, but current documentation describes plan requirements for creating compute Spaces and a limited ZeroGPU exception. No paid subscription, hosted GPU or account eligibility is assumed or configured. The entry point binds externally when SPACE_ID is present and stays on loopback locally.

