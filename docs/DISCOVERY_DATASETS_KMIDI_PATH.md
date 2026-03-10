# Discovery: Datasets path and by_source/kmidi

**Date:** 2026-03-10  
**Plan:** KmiDi folders discovery — local drive scan

## Canonical dataset root (confirmed)

The repo uses a **single dataset root** resolved in this order:

1. **KMIDI_DATASETS_PATH** (env)
2. **AUDIO_DATA_ROOT** (env)
3. **config `dataset_root`** (in experiment/config YAML)
4. **~/Datasets** (fallback)

References:

- [experiments/exp_002_wavjepa_emotion/dataset.py](experiments/exp_002_wavjepa_emotion/dataset.py): `resolve_data_root()` and docstring "Root = KMIDI_DATASETS_PATH | AUDIO_DATA_ROOT | ~/Datasets"; subpaths `emotions/ravdess`, `emotions/cremad`.
- [experiments/exp_002_wavjepa_emotion/config.yaml](experiments/exp_002_wavjepa_emotion/config.yaml): comment "Data: root = KMIDI_DATASETS_PATH | AUDIO_DATA_ROOT | config dataset_root | ~/Datasets".
- [configs/storage.py](configs/storage.py): KELLY_AUDIO_DATA_ROOT / KELLY_SSD_PATH; standard subdirs include `raw`, `raw/emotions`, `raw/melodies`, etc.
- [scripts/utilities/prepare_datasets.py](scripts/utilities/prepare_datasets.py): uses `KMIDI_DATA_ROOT/Datasets` when set (writes to that tree).
- [docs/SSD_WORKDIR_STRUCTURE.md](docs/SSD_WORKDIR_STRUCTURE.md): `KMIDI_DATASETS_PATH` = `$KMIDI_DATA_ROOT/Datasets/COLD_STORAGE/kmidi_datasets` when using external SSD.

So the **canonical root** for dataset resolution in code is: **KMIDI_DATASETS_PATH or AUDIO_DATA_ROOT or ~/Datasets** (with KMIDI_DATA_ROOT/Datasets used by prepare_datasets for writing when set).

## by_source/kmidi — not referenced in repo

- **Path:** `~/Datasets/by_source/kmidi` (or `$KMIDI_DATASETS_PATH/by_source/kmidi` if root is overridden).
- **Contents (discovery):** `audio` subfolder, `.gitkeep`.
- **Grep result:** No references to `by_source` or `by_source/kmidi` in the workspace. No code or config points at this path.
- **Conclusion:** `by_source/kmidi` is an **optional, organizational** location for KmiDi-sourced audio. It is **not** the canonical path used by training or scripts. Experiments and prepare_datasets use the same root with subpaths like `raw/emotions`, `raw/melodies`, and (per SSD doc) `COLD_STORAGE/kmidi_datasets`. If you want scripts to use `by_source/kmidi`, you would either:
  - Place/link content under the existing expected subpaths (e.g. under `raw/` or a documented subdir), or
  - Add explicit support for `by_source/kmidi` in config or dataset loaders and document it.

## Recommendation

- **Canonical path:** Document in one place (e.g. ENVIRONMENT.md or DATA_AND_TRAINING.md) that dataset root is `KMIDI_DATASETS_PATH` | `AUDIO_DATA_ROOT` | `~/Datasets`, and that standard subpaths are as in configs/storage.py and exp_002.
- **by_source/kmidi:** Treat as optional dataset source; document under "Optional dataset locations" or "Discovery" that `~/Datasets/by_source/kmidi` exists and can hold KmiDi audio but is not currently wired into the repo.
