# jepa

Stubs for implementing **JEPA** (Joint-Embedding Predictive Architecture): encoder, predictor, and loss for self-supervised learning (e.g. audio, vision). See `TODO.md` for modules/files to implement and dependencies.

Training data and checkpoints: set `DATASETS_PATH` (e.g. to `/Volumes/Sean's SSD/Datasets`) in env or app config; see [docs/DATASETS_AND_TRAINING.md](../../docs/DATASETS_AND_TRAINING.md).

## Build / resolve

From the **workspace root**:

- Resolve deps and build: `uv sync`
- Run tests: `uv run pytest libs/jepa/tests -v`

## Auto-apply on open

Opening the workspace runs **Resolve workspace** (`uv sync`), which installs this lib’s dependencies. See the top-level README for how to enable automatic tasks.
