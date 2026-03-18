# Local training setup (16 GB macOS or free tier)

Run training on your machine before using cloud GPU. Optimized for 16 GB RAM (e.g. MacBook Air/Pro) or free alternatives (Colab, Kaggle).

## Prerequisites

- **Base dev:** `./scripts/dev-setup.sh` (Python 3.11+, `pip install -e .`, npm if needed).
- **PyTorch:** Install with MPS support on Apple Silicon for faster local runs: `pip install torch`.
- **Data:** Small datasets under `data/audio` and `data/midi`, or set `JEPA_AUDIO_DIR` / `JEPA_MIDI_DIR`. For governance, use `~/Datasets` and `~/Models/checkpoints` (see docs/DATA_AND_TRAINING.md if present).
- **TUI (optional):** `pip install rich` or `pip install -e ".[tui]"` for the local training UI. Run from **iTerm2** or **WezTerm** for the best experience (progress, speed, model metrics).

## 16 GB RAM tips

- Use **small batch size** (4–8). The local JEPA config uses batch 4.
- **Close other apps** to free memory.
- Prefer **MPS** on Apple Silicon; use **CPU** if you hit OOM or don’t have MPS.
- Start with **one model** (e.g. Chord-JEPA only) to validate the pipeline.

## Quick start: JEPA (local)

1. Create data dirs and add a few files (or symlinks):
   ```bash
   mkdir -p data/audio data/midi
   # Add some WAV/MP3 to data/audio and MIDI to data/midi (or use JEPA_AUDIO_DIR / JEPA_MIDI_DIR)
   ```

2. **TUI (recommended)** – Run in iTerm2 or WezTerm for a live UI (progress bar, speed, loss, model uniqueness):
   ```bash
   ./scripts/run_local_training.sh
   ./scripts/run_local_training.sh --config config/jepa_training_local_mac.yaml --model chord_jepa
   ```
   Open in a new window: `./scripts/run_local_training.sh --new-window wezterm` or `--new-window iterm`.  
   Requires: `pip install rich` (or `pip install -e ".[tui]"`).

3. **Plain script** – Run JEPA without the TUI:
   ```bash
   python scripts/train_jepa_local.py --config config/jepa_training_local_mac.yaml
   python scripts/train_jepa_local.py --config config/jepa_training_local_mac.yaml --model chord_jepa
   ```

4. Checkpoints go to `checkpoints/audio_jepa` and `checkpoints/chord_jepa` (or `JEPA_CHECKPOINT_DIR`).

## Integrated Kelly training (MPS/CPU)

For DynamicsEngine / GroovePredictor (small models, good on Mac). From repo root:

```bash
python scripts/training/train_integrated.py --model dynamics_engine --device mps --samples 500
python scripts/training/train_integrated.py --model groove_predictor --device mps --samples 500
```

Use `--device cpu` if MPS is unavailable or you hit memory issues.

## Free alternatives (no local GPU)

- **Google Colab:** Free tier GPU; upload repo or clone, install deps, run the same scripts with a smaller batch (e.g. 8). Good for trying cloud-style runs before paying for Lambda/AWS.
- **Kaggle Notebooks:** Free GPU hours; similar workflow: clone repo, `pip install -e .`, run `train_jepa_local.py` or integrated training with a small config.

Use the same configs (e.g. `config/jepa_training_local_mac.yaml`) and point data/checkpoints to the environment’s paths.

## Env (optional)

- `JEPA_AUDIO_DIR` – directory of audio files (default: `data/audio`).
- `JEPA_MIDI_DIR` – directory of MIDI files (default: `data/midi`).
- `JEPA_CHECKPOINT_DIR` – where to save checkpoints (default: `checkpoints`).

For training env vars (paths, W&B): `source scripts/load-env.sh training` and edit `env/.env.training` if needed (see [ENVIRONMENT.md](ENVIRONMENT.md)).

## Next: cloud training

Once local runs work, move to cloud for larger models and data:

- [docs/CLOUD_DEV_SETUP.md](CLOUD_DEV_SETUP.md) – one-time cloud setup (Lambda, SageMaker, EC2).
- [docs/ml/CLOUD_TRAINING_GUIDE.md](ml/CLOUD_TRAINING_GUIDE.md) – run and monitor cloud jobs.
