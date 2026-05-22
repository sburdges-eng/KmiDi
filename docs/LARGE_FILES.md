# Large files and repo hygiene

Summary of files that were too large for practical use in the repo and how they're handled.

## Removed from git tracking (still present on disk)

| Path | Size | Integral? | Action |
|------|------|-----------|--------|
| `engine/intent_ir/binaries/kmidi_brain-aarch64-apple-darwin` | ~307 MB | No — build artifact | Build via PyInstaller + `build_v1.sh`; copy to `engine/intent_ir/binaries/kmidi_brain-<triple>`. Now in `.gitignore`. |
| `docs/EXTERNAL_DRIVE_AUDIO_COMPLETE_LIST.txt` | ~38 MB | No — audit output | Regenerate from SSD/external drive if needed. Now in `.gitignore`. |
| `docs/EXTERNAL_DRIVE_AUDIO_FULL_LIST.txt` | ~37 MB | No — audit output | Same. Now in `.gitignore`. |

## Still tracked (smaller)

- `docs/EXTERNAL_DRIVE_AUDIO_ADDITIONAL_LIST.txt` (~224 KB), `docs/EXTERNAL_DRIVE_AUDIO_DUPLICATES.txt` (~438 KB) — kept; under 1 MB.
- `docs/EXTERNAL_DRIVE_AUDIO_FILES.md`, `docs/EXTERNAL_DRIVE_DATASET_SCRIPTS.md` — small docs; integral for reference.

## Not in repo (build/cache/tooling)

These are already ignored or live outside the repo; they can be large on disk but are not committed:

- `engine/intent_ir/target/`, `build/`, `node_modules/`, `npm-cache/`
- `.tools/` — JDK/Maven and other tooling: **1,878 files** are currently tracked (including large `.jmod` files). Consider moving to "install via script" and adding `.tools/` to `.gitignore` if you want to shrink the repo further; otherwise leave as-is for a self-contained toolchain.

## KmiDi_FINAL and checkpoints

- `KmiDi_FINAL/` (reference/legacy tree) and `checkpoints/` are ignored or partially ignored; ML model weights and large JSON/models there should not be committed. See `.gitignore` and `docs/DATASETS_LAYOUT.md`.
