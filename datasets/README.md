# Datasets (pointer only — data not in repo)

Dataset files are **local only** and are not committed to the repository.

- **Audio data:** Use `datasets/audio/` on disk for migrated or local audio (e.g. from `scripts/migrate_data_into_kmidi_compile.py`). Configure paths via `.env`:
  - `KELLY_AUDIO_DATA_ROOT` — primary audio data root
  - `AUDIO_DATA_ROOT` — alternate audio root
- **Docs:** See `docs/EXTERNAL_PATH_REFERENCES.md` and `MIGRATION_COMPLETE.md` for paths and migration.

Do not upload large dataset contents; this directory is only a pointer to where data lives locally.
