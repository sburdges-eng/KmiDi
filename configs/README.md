# Configs

Training and run configuration files. Reference from experiments via `configs/exp_NNN_config.yaml` (or similar). Checkpoints and logs go to `~/Models/checkpoints`, not here.

## JEPA training configs

- `jepa_audio.yaml` — Audio JEPA (A-JEPA style, mel spectrogram patches)
- `jepa_midi.yaml` — MIDI JEPA (piano-roll)
- `jepa_specto.yaml` — Spectocloud JEPA
- `jepa_multimodal.yaml` — Cross-modal JEPA (audio + MIDI + Spectocloud)

Manifest path `data/manifests/aligned.jsonl` is relative to run dir; use symlink to `~/Datasets` or override in experiment config.

## Cloud training

- `cloud_training.yaml` — Cloud GPU config (mixed precision, num_workers=8, pin_memory, auto_resume). Use with `scripts/cloud_train.sh`. See [docs/CLOUD_TRAINING.md](docs/CLOUD_TRAINING.md).

## Experiment config template

- `model_exp_template.yaml` — Template for AI model experiment configs. Copy to `configs/exp_NNN_short_name.yaml`; set paths to `~/Datasets` and `~/Models/checkpoints/<exp_name>`. See [docs/AI_MODEL_STRUCTURES.md](../docs/AI_MODEL_STRUCTURES.md).
