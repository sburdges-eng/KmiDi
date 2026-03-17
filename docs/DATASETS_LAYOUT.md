# Datasets layout (canonical)

Canonical layout for the KmiDi dataset volume so repo scripts and acquisition scaffolding cooperate with existing structure.

## Env and root

- **Dataset root:** Set `KMIDI_DATASETS_PATH` to the **Datasets** directory that contains `by_source/`, `by_domain/`, `Experiments/`, etc. Example: `/Volumes/Sean's SSD/Datasets`.
- **Alternative:** `KMIDI_DATA_ROOT` (used by `prepare_datasets.py` and `download_all_datasets.sh`) can point to the volume root; then `KMIDI_DATA_ROOT/Datasets` is used as the dataset root. For this layout, `KMIDI_DATASETS_PATH` should equal that (or be set explicitly to the same path).
- **Models/checkpoints:** `KELLY_MODELS_PATH` (or `~/Models`) — not under Datasets.

## Top-level structure (under dataset root)

| Path | Purpose |
|------|--------|
| `by_source/` | One directory per **source** (e.g. kmidi, manual, subwooder, totali). Acquisition scaffolding and source_manifest write new downloads here. |
| `by_source/scripts/` | Scripts that live on the volume; repo scripts may invoke or complement these. |
| `by_source/kmidi/` | KmiDi-specific datasets: downloads (archives), raw, processed, consolidated, training_tests, midi_companion. |
| `by_domain/` | Domain- or task-oriented organization. |
| `Experiments/` | Experiment-specific data. |
| `_index/`, `00_catalog/` | Catalog and index. |
| `staging/`, `saved/`, `by_run/` | Staging, saved runs, run-keyed data. |
| `20_audio_music/`, `30_audio_speech/`, `40_multimodal/`, etc. | Domain-numbered buckets. |
| `50_cache/`, `70_quarantine/`, `90_archive_readonly/` | Cache, quarantine, archive. |
| `Library/`, `_FORENSIC_READONLY_KMIDI/` | Library and read-only forensic copy. |

## by_source convention

- **Per-source dir:** `by_source/<source_name>/` (e.g. `by_source/kmidi/`).
- **Standard subdirs** (used by existing kmidi layout and by acquisition):
  - `downloads/` — Raw fetched archives (e.g. `.zip`, `.tar.gz`) before unpacking.
  - `raw/` — Unpacked or raw assets.
  - `processed/` — Prepared/processed outputs.
  - `consolidated/` — Consolidated datasets (if used).
- **Acquisition scaffolding** (from `config/source_manifest.yaml`) writes **new** external-source downloads to:
  - `by_source/<source_item>/downloads/`
  where `source_item` is the manifest slug (e.g. `emotion_labeled_music_corpora`, `new_midi_datasets_models`). This complements existing `by_source/kmidi/downloads/` and does not replace it.
- **prepare_datasets.py** uses its own `output_dir` (e.g. `emotions/ravdess`, `grooves/groove_midi`) under the same dataset root; those paths may live at top level or, if configured, under a source. No change required to prepare_datasets for acquisition to work.

## How repo scripts use this layout

- **download_all_datasets.sh** — Sets `KMIDI_DATA_ROOT` (or uses env), runs `prepare_datasets.py --dataset all --download`. Writes under the root chosen by `prepare_datasets` (e.g. `KMIDI_DATA_ROOT/Datasets` or `AUDIO_DATA_ROOT`).
- **prepare_datasets.py** — Resolves root via `AUDIO_DATA_ROOT` / `KMIDI_DATA_ROOT`; writes to `output_dir` per dataset (emotions/, grooves/, etc.).
- **Acquisition from source_manifest** — `scripts/acquire/acquire_from_manifest.py` reads `config/source_manifest.yaml`; for items with `storage_env_var: KMIDI_DATASETS_PATH`, resolves path to `$KMIDI_DATASETS_PATH/by_source/<source_item>/downloads`. For `KELLY_MODELS_PATH`, uses `proposed_storage_path` under that env. Does not overwrite existing by_source/kmidi layout.
- **make_jepa_manifest.py** — Reads audio/MIDI from paths configured or passed in; Lhotse manifests can reference files under this volume (e.g. under by_source/kmidi/processed or by_domain).

## Paths with spaces

Volume names like `Sean's SSD` are valid. Use quoted paths in shell; in Python use `pathlib.Path` so no extra quoting is needed.

## See also

- `config/source_manifest.yaml` — External source items and storage_env_var / proposed_storage_path.
- `docs/DATA_AND_TRAINING.md` — DATA LAW, checkpoint paths, reproducibility.
- `docs/ENVIRONMENT.md` — Env vars and loading.
