# Transfer Candidates from /Volumes

Date: 2026-02-13
Scope scanned: `/Volumes/KmiDi-external`

## Summary
- Strong transfer sources found for: training scaffolds, plugin scaffolds, model artifacts, MIDI corpora, and generated schemas.
- Recommended approach: transfer code/config/templates first, then selectively ingest datasets/checkpoints with provenance tracking.

## Priority 1 (Transfer Now)

### 1) ML training scaffold (code + config)
- Source:
  - `/Volumes/KmiDi-external/ml-training-suiteEXTERNAL/src/`
  - `/Volumes/KmiDi-external/ml-training-suiteEXTERNAL/scripts/`
  - `/Volumes/KmiDi-external/ml-training-suiteEXTERNAL/configs/config.yaml`
  - `/Volumes/KmiDi-external/ml-training-suiteEXTERNAL/requirements.txt`
- Why:
  - Already structured for dataset loading, training loop, preprocess/inference scripts.
  - `config.yaml` has concrete audio preprocessing and split/training settings.
- Suggested destination:
  - `ml/training/shared/bootstrap_from_ml_training_suite/`
  - `scripts/legacy-import/ml-training-suite/`
- Notes:
  - Refactor from classifier-centric naming to symbolic/embedding/audio-render tasks.

### 2) JUCE plugin scaffold/code fragments
- Source:
  - `/Volumes/KmiDi-external/_sortedEXTERNAL/CPP_JUCE/My Mac/Downloads/plugin-update/plugin/PluginProcessor.cpp`
  - `/Volumes/KmiDi-external/_sortedEXTERNAL/CPP_JUCE/My Mac/Downloads/plugin-update/plugin/PluginProcessor.h`
  - `/Volumes/KmiDi-external/_sortedEXTERNAL/CPP_JUCE/My Mac/Downloads/plugin-update/plugin/PluginEditor.cpp`
  - `/Volumes/KmiDi-external/_sortedEXTERNAL/CPP_JUCE/My Mac/Downloads/plugin-update/plugin/PluginEditor.h`
  - `/Volumes/KmiDi-external/_sortedEXTERNAL/CPP_JUCE/My Mac/Downloads/plugin-update/engine/IntentPipeline.cpp`
  - `/Volumes/KmiDi-external/_sortedEXTERNAL/CPP_JUCE/My Mac/Downloads/plugin-update/engine/IntentPipeline.h`
- Why:
  - Directly relevant AU/VST3 component patterns and intent-pipeline bridge stubs.
- Suggested destination:
  - `apps/plugin-juce/legacy-reference/plugin-update/`
- Notes:
  - Treat as reference-first import, then isolate reusable abstractions.

### 3) Existing model registry schemas
- Source examples:
  - `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/KmiDi_FINAL/ml/models/registry.schema.json`
  - `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/ML Kelly Training/backup/models/registry.schema.json`
- Why:
  - Useful for defining internal model artifact/version registry.
- Suggested destination:
  - `schemas/model-registry.legacy.schema.json`
- Notes:
  - Multiple duplicates exist; normalize to one canonical base.

## Priority 2 (Transfer Selectively)

### 4) MIDI corpora for symbolic modeling
- Source:
  - `/Volumes/KmiDi-external/DatasetsEXTERNAL/midi/` (296 `.mid` files counted)
  - `/Volumes/KmiDi-external/DevEXTERNAL/_FORENSIC_READONLY_KMIDI/**/examples/midi/` (multiple groove/idaw examples)
- Why:
  - Immediate data for symbolic baseline and constraint testing.
- Suggested destination:
  - `ml/data/manifests/sources/midi/`
  - `ml/data/raw/midi_staging/` (if you create this path)
- Notes:
  - Prefer manifest-based references before bulk copying.

### 5) ONNX/PT model artifacts (for baseline inference comparison)
- Source examples:
  - `/Volumes/KmiDi-external/_sortedEXTERNAL/ML_Datasets/My Mac/Desktop/KmiDi-remote/models/*.onnx`
  - `/Volumes/KmiDi-external/_sortedEXTERNAL/ML_Datasets/My Mac/Desktop/KmiDi-remote/checkpoints/**/*.pt`
  - `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/**/checkpoints/**/*.pt`
- Why:
  - Useful as teacher/baseline benchmarks while rebuilding fully-owned models.
- Suggested destination:
  - `ml/inference/runtimes/legacy_models/`
  - `ops/runs/imported-model-artifacts.md`
- Notes:
  - Keep provenance + training origin metadata for each artifact.

### 6) Tauri-generated desktop schemas
- Source examples:
  - `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src-tauri/gen/schemas/desktop-schema.json`
  - `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src-tauri/gen/schemas/macOS-schema.json`
- Why:
  - Good reference for desktop app config schema patterns.
- Suggested destination:
  - `schemas/reference/tauri-desktop-schema.json`
  - `schemas/reference/tauri-macos-schema.json`

## Priority 3 (Hold / Review Before Use)

### 7) Large raw audio datasets
- Source:
  - `/Volumes/KmiDi-external/audioEXTERNAL/` (~93,457 audio files counted)
  - Includes `m4singer` instructions and emotion datasets.
- Why hold:
  - Licensing and dataset policy vary; ingestion should be policy-gated.
- Suggested action:
  - Create dataset manifest + license acceptance checklist before import.

### 8) Personal/creative song docs
- Source:
  - `/Volumes/KmiDi-external/Kelly_Song_ProjectEXTERNAL/README.md`
  - `/Volumes/KmiDi-external/Kelly_When_I_Found_You_2024EXTERNAL/README.md`
- Why hold:
  - Better as prompt/evaluation fixtures, not training core by default.
- Suggested destination:
  - `docs/product/reference-song-briefs/` (optional)

## Transfer Mapping (Quick Start)
- `ml-training-suiteEXTERNAL/src/*` -> `ml/training/shared/bootstrap_from_ml_training_suite/`
- `ml-training-suiteEXTERNAL/scripts/*` -> `scripts/legacy-import/ml-training-suite/`
- `plugin-update/*` -> `apps/plugin-juce/legacy-reference/plugin-update/`
- `registry.schema.json` -> `schemas/model-registry.legacy.schema.json`
- selected `*.mid` -> `ml/data/raw/midi_staging/` (manifest-driven)
- selected `*.onnx` and `*.pt` -> `ml/inference/runtimes/legacy_models/`

## Suggested Next Step
- Execute a controlled import pass for Priority 1 only, with:
  - dedupe by checksum
  - no overwrite of existing files
  - generated import log in `ops/runs/2026-02-13-transfer-pass-01.md`
