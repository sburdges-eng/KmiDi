# Acquisition scripts

Scripts that fetch or resolve paths for external assets from `config/source_manifest.yaml`. They cooperate with the existing dataset layout (see `docs/DATASETS_LAYOUT.md`).

- **acquire_from_manifest.py** — List manifest items, resolve storage paths, or **download** adopted items. Use `--list` and `--resolve-paths` to inspect; use `--download` to fetch (only entries with `adoption_decision: adopted` and a valid `primary_url`). Items with `license: UNKNOWN` are skipped unless `--accept-unknown-license`. No download until URLs and adoption are set in the manifest.
- **verify_manifest_status.py** — Report verification/adoption status and optional URL reachability (Phase 5; read-only).

Existing volume layout (complemented, not replaced):

- `KMIDI_DATASETS_PATH` → dataset root containing `by_source/`, `by_domain/`, etc.
- Dataset downloads from manifest → `by_source/<source_item>/downloads/`
- Models → `KELLY_MODELS_PATH/<proposed_storage_path>`

## Begin downloads

1. Set in `config/source_manifest.yaml`: `adoption_decision: adopted`, `primary_url: <https://...>`, and ensure `storage_env_var` and `proposed_storage_path` (or dataset layout) are set so paths resolve.
2. Optionally run `python scripts/acquire/verify_manifest_status.py --check-urls` to confirm URLs are reachable.
3. Run:
   - `python scripts/acquire/acquire_from_manifest.py --download --dry-run` — show what would be downloaded.
   - `python scripts/acquire/acquire_from_manifest.py --download` — download (set `KMIDI_DATASETS_PATH` or `KELLY_MODELS_PATH` so storage paths exist).
4. For items with `license: UNKNOWN` you must pass `--accept-unknown-license` to allow download (red flag per SOURCE_INTEGRATION_PLAN).
