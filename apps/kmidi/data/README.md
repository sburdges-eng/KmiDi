# KmiDi data layout

- **Audio:** `data/audio/` — Raw or processed audio (e.g. train/val splits). See `config/datasets.toml` for paths.
- **MIDI:** `data/midi/` — MIDI files if the app uses them.
- **JEPA (optional):** When using libs/jepa for training, add features or manifest paths under `config/datasets.toml` → `[jepa]` and document here.

Do not commit large audio/MIDI; `data/` is ignored at repo root.
