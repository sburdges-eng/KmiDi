# xcode/KmiDi doc import

One-time import of docs that existed only in the xcode clone (see [DISCOVERY_XCODE_KMIDI_VS_WORKSPACE.md](DISCOVERY_XCODE_KMIDI_VS_WORKSPACE.md)).

**Source:** `/Users/seanburdges/Dev/xcode/KmiDi` (branch `kmidi-companion-dev`).

| File | Purpose |
|------|---------|
| CLEANUP_GUIDE.md | Cleanup guidance |
| DISTRIBUTE.md | Distribution notes |
| INSTALL_ALL.md | Install instructions |
| README.macos-metal.md | macOS Metal readme |
| TODO.md | Task list |

### Optional: not yet imported (copy from xcode if needed)

**configs/** (xcode has; workspace may have different versions):

- `configs/exp_cad_intent_training.yaml`, `exp_m2_m4_emotion.yaml`
- `configs/jepa_audio.yaml`, `jepa_midi.yaml`, `jepa_multimodal.yaml`, `jepa_specto.yaml`
- `configs/model_exp_template.yaml`, `configs/cloud_training.yaml`

**docs/** (xcode-only or different):

- `docs/CLOUD_AWS.md`, `CLOUD_SETUP_REFERENCE.md`, `CLOUD_TRAINING.md`
- `docs/CONTRACTS.md`, `DEBUG_INTEGRATION_GUIDE.md`, `DEVELOPMENT_ROADMAP_FORENSIC.md`
- `docs/ENV_AND_TMUX.md`, `FUNCTION_INDEX_README.md`, `GIT_RESTORE_PATHWAYS.md`
- `docs/INCOMPLETE_MODULES_LAST_KNOWN_PATHS.md`, `INTEGRATION_MAP.md`, `ISSUES_LIST.md`
- `docs/JEPA_DATASETS.md`, `PROJECT_ROADMAP.md`, `PROJECT_ROADMAP_REIMPLEMENTATION.md`
- `docs/PROJECT_STRUCTURE_AND_DEV_WORK_DIR.md`, `SESSION_COMPLETION_2026-01-31.md`, `SPECTOCLOUD.md`, `TRAINING_ENV.md`

Copy into workspace `configs/` or `docs/` (or under `docs/xcode-import/`) after comparing with existing files to avoid overwriting.
