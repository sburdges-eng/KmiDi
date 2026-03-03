# configs/

This directory contains **committed** training and model configuration files (YAML / JSON).

## Purpose

| What goes here | What does NOT go here |
|----------------|-----------------------|
| Hyperparameter YAML/JSON files | Dataset files or audio |
| Model architecture configs | Model weights / checkpoints |
| Tokenizer / vocab configs (small) | Experiment outputs or logs |
| Environment-variable templates | Anything generated at run-time |

## Conventions

- **File naming:** `<model_or_task>_<variant>.yaml` (e.g. `wavjepa_emotion_v2.yaml`)
- **Every training run** must reference a config file in this directory via the
  `run_manifest.yaml` `reproducibility.config_file` field.
- **Environment variables** (`KELLY_AUDIO_DATA_ROOT`, etc.) are expanded at load-time
  by `configs/config_loader.py` — use `${VAR_NAME:-default}` syntax in YAML files.
- **Do not hardcode** absolute paths like `/Users/…` or `/Volumes/…`.
  Use env-var substitution or relative paths anchored to `~/Datasets` / `~/Models`.

## Loader usage

```python
from configs.config_loader import load_config

cfg = load_config("configs/wavjepa_emotion_v2.yaml")
data_root = cfg["data"]["root"]   # already expanded from ${KELLY_AUDIO_DATA_ROOT}
```

## See also

- `docs/DATA_AND_TRAINING.md` — full dataset / model / training governance rules
- `docs/ENVIRONMENT.md` — all supported environment variables
