# Discovery: xcode/KmiDi vs Dev/KmiDi (workspace)

**Date:** 2026-03-10  
**Plan:** KmiDi folders discovery — local drive scan

## Git state

| Clone | Branch | Status |
|-------|--------|--------|
| **xcode/KmiDi** | `kmidi-companion-dev` | Staged: adds (KmiDi_CANON/README.md, configs, scripts), deletes (GOOGLE KELLY INFO TEST UPLOADS .mid files) |
| **Dev/KmiDi** (workspace) | `juce-dirty-fa319` | Only modified: `external/JUCE` (submodule dirty) |

Both use same remote: `origin` → `https://github.com/sburdges-eng/KmiDi.git`.

## Files/dirs only in xcode/KmiDi (candidates for pull-in)

### Top-level docs (not in workspace)

- `CLEANUP_GUIDE.md`
- `DISTRIBUTE.md`
- `INSTALL_ALL.md`
- `README.macos-metal.md`
- `TODO.md`

### Top-level dirs (not in workspace)

- `KmiDi_CANON/` — brain, body, training, ui; has README, api_server.py, train_emotion_optimized.py, train_jepa.py
- `GOOGLE KELLY INFO/` — test uploads (some .mid files staged for delete in xcode)
- `Documents/` — Dev_Notes, Kelly_Business, cursor_* docs
- `Music/` — AudioVault (Python scripts), Kelly_Song_Project (LYRICS, generate_midi.py), iDAW_Output
- `ML_TRAINED_MODELS/` — ml/ (cli, device_picker)
- `docker/`
- `.vscode/`
- `.cursorignore`

### configs (xcode has extra YAMLs)

- `configs/exp_cad_intent_training.yaml` (staged add in xcode)
- `configs/exp_m2_m4_emotion.yaml`
- `configs/jepa_audio.yaml`, `jepa_midi.yaml`, `jepa_multimodal.yaml`, `jepa_specto.yaml`
- `configs/model_exp_template.yaml`
- `configs/cloud_training.yaml`

### docs (xcode has extra)

- `docs/CLOUD_AWS.md`, `CLOUD_SETUP_REFERENCE.md`, `CLOUD_TRAINING.md`
- `docs/CONTRACTS.md`, `DATA_AND_TRAINING.md`, `DEBUG_INTEGRATION_GUIDE.md`
- `docs/DEVELOPMENT_ROADMAP_FORENSIC.md`, `ENV_AND_TMUX.md`, `FUNCTION_INDEX_README.md`
- `docs/GIT_RESTORE_PATHWAYS.md`, `INCOMPLETE_MODULES_LAST_KNOWN_PATHS.md`, `INTEGRATION_MAP.md`
- `docs/ISSUES_LIST.md`, `JEPA_DATASETS.md`, `PROJECT_ROADMAP.md`, `PROJECT_ROADMAP_REIMPLEMENTATION.md`
- `docs/PROJECT_STRUCTURE_AND_DEV_WORK_DIR.md`, `SESSION_COMPLETION_2026-01-31.md`, `SPECTOCLOUD.md`, `TRAINING_ENV.md`
- `docs/.index`

### Other notable (xcode)

- `lariat-bible/` — desktop_app (DOCX, vendor, build_standalone), scripts, tests
- `run_brain.py` — xcode version longer (5324 B, Feb 21) vs workspace (4372 B, Mar 7); workspace may have been trimmed/refactored
- `data/manifests/README.md`
- `scripts/`: acquire_maestro.py, acquire_real_data.py, build_function_index.py, check_stub_creep.py, create_jepa_manifest_stub.py, prepare_webdataset_shards.py, plus staged download_ultradata_math_* and run_ultradata_math_download.sh

## Files/dirs only in Dev/KmiDi (workspace)

- AGENTS.md, BUILD.md, QUICK_START.md, README.md
- CMake/ monorepo layout (CMakeLists.txt, cmake/, external/JUCE, etc.)
- .github/ (workflows, ISSUE_TEMPLATE, PULL_REQUEST_TEMPLATE, agents, copilot-instructions)
- config/, configs/config_loader.py, configs/storage.py, configs/train_remi_bpe_30k.json
- apps/, adapters, bindings, bridges, cloud_training, common, datasets, etc.
- docs/: BOOT.md (differs), plus AI_CONTROL_LAYER, API, ARCHITECTURE, BREAK_FIX_RUN_LOG, CANONICAL_FOLDER_STRUCTURE, CHECK_EXTERNAL_FILES, CLAP_DESIGN_NOTE, DEVELOPMENT.md, ENVIRONMENT.md, SSD_WORKDIR_STRUCTURE, etc.
- KmiDi_FINAL, KmiDi_PROJECT, KmiDi_TRAINING, music_brain (Python package), bootstrap.sh, etc.

## Suggested follow-up

- **Merge/copy:** If consolidating, consider bringing xcode-only docs (CLEANUP_GUIDE, DISTRIBUTE, INSTALL_ALL, README.macos-metal, TODO) and selected configs/docs (e.g. CLOUD_*, JEPA_*, PROJECT_ROADMAP) into workspace under `docs/` or `configs/` with a single import commit.
- **KmiDi_CANON:** **Superseded by workspace layout.** Canonical source for brain/body/training/ui is the Dev/KmiDi monorepo: `music_brain`, `apps/`, KmiDi_* trees. KmiDi_CANON (xcode layout) is legacy; workspace does not use it. See DISCOVERY_RUN_BRAIN_DIFF.md.
- **run_brain.py:** Diff the two versions to see if workspace dropped logic or refactored into other modules.
