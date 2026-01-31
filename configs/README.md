# Configs

Training and run configuration files. Reference from experiments via `configs/exp_NNN_config.yaml` (or similar). Checkpoints and logs go to `~/Models/checkpoints`, not here.

## JEPA training configs

- `jepa_audio.yaml` — Audio JEPA (A-JEPA style, mel spectrogram patches)
- `jepa_midi.yaml` — MIDI JEPA (piano-roll)
- `jepa_specto.yaml` — Spectocloud JEPA
- `jepa_multimodal.yaml` — Cross-modal JEPA (audio + MIDI + Spectocloud)

Manifest path `data/manifests/aligned.jsonl` is relative to run dir; use symlink to `~/Datasets` or override in experiment config.
