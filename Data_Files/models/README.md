# Integrated Model Staging Area

This folder is a stable, repo-local place to stage ML models for discovery/integration.

## What’s here

- `external_candidates/`: symlinks to existing checkpoint artifacts in the repo (kept in their original locations).
- `pytorch/`: canonical symlink names intended for `python/penta_core/ml/model_registry.py` auto-discovery (`*.pt` / `*.pth`).

## Notes

- Many `.pt` files are **not** guaranteed to be TorchScript; `python/penta_core/ml/inference.py` currently uses `torch.jit.load()`. Verify each file before relying on it in runtime inference.
- RTNeural JSON models used by the C++ pipeline are wired via `models/*.json` symlinks (see `docs/DISCOVERY/INTEGRATION_SUMMARY.md`).

