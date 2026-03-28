# JEPA lib — modules and files to implement

## Auto-install on open

Dependencies for this lib are installed when the workspace opens (task **Resolve workspace** runs `uv sync`). Ensure `workbench.task.allowAutomaticTasks` is `on` so deps are applied automatically.

## Modules / files to create or implement

| Status   | Module / file        | Description |
|---------|----------------------|-------------|
| Stub    | `jepa/encoder.py`    | Context and target encoders (e.g. CNN/Transformer); optional stop-gradient on target. |
| Stub    | `jepa/predictor.py`  | Predictor: context_embed -> predicted_target_embed. |
| Stub    | `jepa/loss.py`       | Loss (e.g. VICReg-style or MSE); optional masking. |
| To add  | `jepa/config.py`     | Config dataclass or TypedDict for encoder/predictor dims, masking, etc. |
| To add  | `jepa/training.py`   | Training step / loop (optimizer step, logging). |
| To add  | `jepa/dataloader.py` | Data loading and augmentation for context/target views. |
| To add  | `tests/test_encoder.py` | Unit tests for encoder forward shape and grad. |
| To add  | `tests/test_loss.py`    | Unit tests for loss value and backward. |

## Dependencies to add (in `pyproject.toml` when needed)

- `torch` (already in pyproject)
- `numpy` (already in pyproject)
- Optional: `einops`, `timm`, or domain-specific (e.g. `librosa` for audio JEPA).

## Todos (implementation order)

1. [ ] Implement `encode_context` / `encode_target` in `encoder.py` (and optionally a small `Encoder` class).
2. [ ] Implement `predict_target` in `predictor.py`.
3. [ ] Implement `jepa_loss` in `loss.py`.
4. [ ] Add `config.py` and wire config into encoder/predictor/loss.
5. [ ] Add `training.py` and a minimal training script or entrypoint.
6. [ ] Add `dataloader.py` and dataset adapters for your modality (audio/vision).
7. [ ] Add tests and run `uv run pytest libs/jepa/tests -v`.
