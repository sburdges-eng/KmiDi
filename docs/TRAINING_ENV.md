# Training Environment — JEPA and Current Builds

One env for **JEPA training** (configs in `configs/jepa_*.yaml`) and **current builds** (penta_core/ml/training, KmiDi_CANON/training). Datasets and checkpoints live under `~/Datasets` and `~/Models` (DATA LAW). Envs live in `~/envs` (ENV LAW).

## Env: kmidi-training

Create **outside** the repo (never inside repo):

```bash
export MAMBA_ROOT_PREFIX=~/envs/mamba   # if not already in .zshrc
micromamba create -n kmidi-training python=3.11
micromamba activate kmidi-training
```

### Dependencies (JEPA + current)

- **Core:** `pip install torch numpy pyyaml tqdm`
- **Audio / JEPA audio:** `pip install librosa soundfile` (mel spectrograms, loading)
- **Current penta_core training:** same as Brain; install from repo if needed: `pip install -e .` from project root (or install deps only).
- **GPU (optional):** Install CUDA-enabled PyTorch when ready; start with CPU to validate pipeline.

Install in one go (CPU PyTorch first):

```bash
micromamba activate kmidi-training
pip install torch numpy pyyaml tqdm librosa soundfile
# From repo root, if package is installable:
# pip install -e ".[training]"   # or whatever extra is defined
```

## Local learning paths

| Purpose            | Path                          | Notes |
|--------------------|--------------------------------|--------|
| JEPA datasets      | `~/Datasets/kmidi_jepa/`       | Manifests + audio/midi/specto (or symlinks) |
| Other training data | `~/Datasets/kmidi_learning/` | Optional; current-build datasets |
| Checkpoints        | `~/Models/checkpoints/`        | Per-run; e.g. `~/Models/checkpoints/exp_001_jepa_midi` |
| Run manifests      | Repo `experiments/exp_NNN_*/manifest_run_*.json` | Traceability |

Create dirs once: run `./scripts/setup_training_env.sh` from repo root (creates `~/Datasets/kmidi_jepa`, `~/Datasets/kmidi_learning`, `~/Models/checkpoints`). **Cloud GPU:** See [CLOUD_TRAINING.md](CLOUD_TRAINING.md) for Lambda Labs / RunPod; use tmux and WebDataset shards.

## JEPA datasets

- **Manifest:** `~/Datasets/kmidi_jepa/manifests/aligned.jsonl` (see `data/manifests/README.md` for format).
- **Acquire or create:** See `docs/JEPA_DATASETS.md`. Use `scripts/create_jepa_manifest_stub.py` to generate an empty or minimal manifest for testing.

## Current builds (non-JEPA)

- **penta_core:** `KmiDi_CANON/brain/penta_core/ml/training/` (augmentation, evaluation).
- **KmiDi_CANON/training:** `KmiDi_CANON/training/models/`, `utils/audio.py` (audio classifier, etc.).
- Point dataset and checkpoint paths in experiment configs to `~/Datasets/kmidi_learning` and `~/Models/checkpoints/<exp_name>`.

## Run training (tmux)

Long runs use tmux (see `docs/ENV_AND_TMUX.md`):

```bash
tmux new -s kmidi
# or ./scripts/tmux_kmidi.sh
micromamba activate kmidi-training
cd /path/to/KmiDi\ MIDI\ Companion
# Override manifest if needed: export KMI_DI_JEPA_MANIFEST=~/Datasets/kmidi_jepa/manifests/aligned.jsonl
python -m KmiDi_CANON.training.train_jepa --config configs/jepa_midi.yaml   # when implemented
```

Checkpoint path in config or CLI should be `~/Models/checkpoints/exp_NNN_description`.
