# External Path References (pre-deletion audit)

This document lists files inside **KmiDi-compile** that reference paths outside this folder. Use it before deleting sibling folders (AUDIO_MIDI_DATA, CANONICAL_REBUILD, FINAL_KMIDI, kelly-project, etc.).

**Generated for:** Keeping only `KmiDi-compile/` and removing other project data at the workspace root.

---

## 1. Hardcoded absolute paths (code/config)

| File | Line / context | Path / snippet |
|------|----------------|----------------|
| `music_brain/emotion/audio_emotion_classifier.py` | ~218 | `Path("/Users/seanburdges/RECOVERY_OPS/sbdrive/ml-training-suite/models/checkpoints")` |
| `scripts/kelly_song_example_fixed.py` | ~109, ~118, ~163 | `/Users/seanburdges/Downloads/kelly_outputs/` |
| `scripts/train_model.py` | ~84 | `AUDIO_DATA_ROOT = Path("/Volumes/Extreme SSD/kelly-audio-data")` |
| `scripts/train.py` | ~595 | `os.environ.get("KELLY_AUDIO_DATA_ROOT", "/Volumes/sbdrive/audio/datasets")` |
| `scripts/safe_extended_training.py` | ~172 | `env["KELLY_AUDIO_DATA_ROOT"] = "/Volumes/sbdrive/audio/datasets"` |
| `scripts/prepare_datasets.py` | ~49–51 | `/Volumes/sbdrive/kmidi_audio_data`, `/Volumes/Extreme SSD/...`, `/Users/seanburdges/BASIC STRUCTURE FOR miDiKompanion` |
| `scripts/parallel_train.py` | ~65, ~115, ~151, ~157 | `/Volumes/sbdrive/audio/datasets` (default + docs) |
| `scripts/dataset_loaders.py` | ~14, ~556 | `data_dir="/Volumes/sbdrive/audio/datasets/lakh_midi"`, `--data-dir` default |
| `scripts/ai_training_orchestrator.py` | ~43 | `KELLY_AUDIO_DATA_ROOT` default `/Volumes/sbdrive/audio/datasets` |
| `training/cache_audio_manifest.py` | ~85–86 | default cache `/Volumes/sbdrive/kmidi_audio_cache` |
| `scripts/upload_to_b2.sh` | ~20 | `LOCAL_DATA="/Volumes/sbdrive/audio/datasets"` |
| `scripts/ssh_server_for_codespaces.sh` | ~22 | `LOCAL_DATA="/Volumes/sbdrive/audio/datasets"` |
| `scripts/setup_ssh_tunnel.sh` | ~25 | `LOCAL_DATA="/Volumes/sbdrive/audio/datasets"` |
| `scripts/setup_ml_env.sh` | ~186 | `AUDIO_DATA_ROOT="/Volumes/Extreme SSD"` |
| `scripts/local_train.sh` | ~13 | `VOL="/Volumes/sbdrive"` |
| `scripts/bulk_download_data.py` | ~37, ~139 | `/Volumes/Extreme SSD/kelly-audio-data` |
| `penta_core/ml/datasets/audio_downloader.py` | ~4, ~60, ~440 | `/Volumes/Extreme SSD/kelly-audio-data` |
| `penta_core/ml/datasets/__init__.py` | ~63, ~71, ~101 | Darwin default `/Volumes/Extreme SSD`, `KELLY_AUDIO_DATA_ROOT` |
| `scripts/push_one_by_one.sh` | ~2 | `cd /Users/seanburdges/Documents/GitHub/iDAW` |
| `scripts/overwrite-local-kmidi.sh` | ~2–5 | `/Users/seanburdges/Desktop/KmiDi`, `KmiDi-remote` |
| `cache_audio_manifest.py` (root) | ~33 | `/Volumes/sbdrive/audio_cache` |

---

## 2. Documentation / examples only (safe to delete sibling folders)

These mention `~/Music`, `~/Downloads`, `/Users/seanburdges/...`, or “refer to …” and are docs/examples, not runtime data dependencies.

| Location | Snippet / purpose |
|----------|--------------------|
| `vault/Sample_Library_Index.md` | `~/Music/Samples/` |
| `vault/Groove_Template_Library.md` | `~/Music-Brain/groove-library/...` |
| `vault/Audio_Feel_Extractor.md` | `~/Music/References`, `~/Music/HipHop`, etc. |
| `vault/Audio_Cataloger_Setup.md` | `~/Music-Brain/audio-cataloger`, `~/Music/Samples` |
| `tools/kb_analyzer/README.md` | `--repo-dir ../../` |
| `tools/TODO.md`, `tools/ROADMAP.md` | Refer to `/Users/seanburdges/Desktop/final kel` |
| `tests/.../TODO.md`, `tests/.../ROADMAP.md` | Same “final kel” reference |
| `src_penta-core/`, `music_brain/`, `mobile/`, `iOS/`, `iDAW_Core/`, `docs/.../TODO.md`, `ROADMAP.md` | Same “final kel” reference |
| `tools/audio_cataloger/audio_cataloger.py` | `%(prog)s scan ~/Music/Samples` (help text) |
| `scripts/audio_cataloger.py` | Same (help text) |
| `scripts/daiw_menubar.py` | “Outputs to ~/Music/DAiW_Output/” (docstring) |
| `docs/summaries/SAMPLE_LIBRARY_COMPLETE.md` | `~/Music/Samples/`, `~/Music/Audio Music Apps/...` |
| `docs/summaries/MVP_COMPLETE.md` (and copy) | `~/Music/AudioVault/`, `~/Music/Audio Music Apps/...` |
| `docs/music_brain/DAIW_INTEGRATION.md` | `cp ~/Downloads/...` |
| `docs/music_brain/Audio Feel Extractor.md` | `~/Music/References`, etc. |
| `docs/music_brain/AUDIO_ANALYZER_TOOLS.md` | `daiw batch-analyze ~/Music/` |
| `docs/integrations/DAW_INTEGRATION.md` | `~/Music/Ableton/...` |
| `docs/integrations/DAIW_INTEGRATION.md` | `~/Downloads/...` |
| `docs/daw_integration/TEMPLATES_OVERVIEW.md` | `~/Music/Audio Music Apps/...`, `~/Music/Ableton/...` |
| `docs/ai_setup/DAiW_GPT_Instructions.md` | `~/Music/DAiW_Output` |
| `docs/PROPOSAL_SUMMARY.md` | `/Users/seanburdges/Desktop/...`, `/Users/seanburdges/Applications/...` |

---

## 3. Relative paths (inside KmiDi-compile only)

These use `../../` or similar but stay under the repo; no dependency on sibling project folders.

- `tests/unit/test_tier1_midi.py`, `test_tier1_audio.py`: `sys.path.insert(0, ... '../../')` → project root.
- `penta_core/phases/phase1_infrastructure.py`: `root_dir ... join(..., '../../../..')` → project root.
- `docs/mobile/IOS_AUDIO_UNIT.md`, `ANDROID_AAP.md`: `../../../src_penta-core/...` → build paths inside repo.

---

## 4. What to do before deleting sibling data

1. **Run the dependency checker**
   ```bash
   cd "KmiDi-compile"
   python scripts/check_external_dependencies.py
   python scripts/check_external_dependencies.py --json > docs/external_deps_report.json  # optional
   python scripts/check_external_dependencies.py --fix-env   # suggested env vars
   ```

2. **Set env defaults to a path inside KmiDi-compile** (or another drive you keep):
   - `KELLY_AUDIO_DATA_ROOT` → e.g. `KmiDi-compile/datasets/audio`
   - `AUDIO_DATA_ROOT` → same, if you use it.

3. **If you want to keep any of “AUDIO_MIDI_DATA”**: move/copy that data into `KmiDi-compile/datasets/` (or `KmiDi-compile/kmidi_audio_data/`) and point `KELLY_AUDIO_DATA_ROOT` / `AUDIO_DATA_ROOT` there. Use the migration script in `scripts/migrate_data_into_kmidi_compile.md` (or the accompanying script) to do that move.

4. **Checkpoints** in `music_brain/emotion/audio_emotion_classifier.py`: the in-repo path `KmiDi-compile/models/checkpoints` is already searched first. Optionally set `KMIDI_CHECKPOINTS_DIR` to override; the legacy RECOVERY_OPS path is still tried last.

After that, it is **safe to delete** all project data outside `KmiDi-compile/` as long as you do not rely on those external paths at runtime.
