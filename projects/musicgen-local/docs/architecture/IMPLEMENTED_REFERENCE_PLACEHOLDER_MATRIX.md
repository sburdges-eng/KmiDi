# Implemented vs Reference vs Placeholder Matrix

## Summary
This matrix captures current system status for `musicgen-local`, including what is actively implemented, what is imported/reference-only, and what is placeholder scaffolding.

Status values:
- `implemented`: executable logic exists in current local project stack
- `reference-imported`: logic exists as imported legacy/reference code and is not yet integrated into current production pipeline
- `placeholder`: scaffold/documentation stubs only

## Features

| Capability | Status | How It Works | Where | Limitations |
|---|---|---|---|---|
| Project structure and local workflow | implemented | Local-first monorepo layout with ops/docs/scripts conventions | `/Volumes/KmiDi-external/musicgen-local/README.md` | No orchestration service wiring yet |
| Music graph schema v0.1 | implemented | JSON schema validates metadata/global/structure/tracks/generation sections | `/Volumes/KmiDi-external/musicgen-local/schemas/music-graph.schema.json` | No runtime compiler/validator wiring in services yet |
| Model registry schema (legacy) | reference-imported | Legacy registry schema defines model metadata/training/export/integration fields | `/Volumes/KmiDi-external/musicgen-local/schemas/model-registry.legacy.schema.json` | Not yet bound to active registry runtime |
| Data provenance and licensing governance | implemented | Source roots tracked with licensing status and gate policy | `/Volumes/KmiDi-external/musicgen-local/docs/ml/DATA_PROVENANCE.md` | Policy enforcement is script-based, not service-enforced |
| License evidence index | implemented | Curated evidence paths for each imported source root | `/Volumes/KmiDi-external/musicgen-local/docs/ml/LICENSE_EVIDENCE_INDEX.md` | Evidence is documentary only (not legal adjudication) |
| Approval checklist workflow | implemented | Per-source owner/date/use/commercial/training fields drive gate | `/Volumes/KmiDi-external/musicgen-local/docs/ml/APPROVAL_CHECKLIST.md` | Manual updates required |
| Rulebreak reference crosswalk | implemented | Canonical mapping from music-theory rulebreak types to source interfaces/docs | `/Volumes/KmiDi-external/musicgen-local/docs/architecture/RULEBREAK_REFERENCE_INDEX.md` | Reference only; no local rulebreak engine integrated |

## Engines

| Engine | Status | How It Works | Where | Limitations |
|---|---|---|---|---|
| C++ IntentPipeline | reference-imported | Emotion->mode/tempo/rulebreak orchestration surface | `/Volumes/KmiDi-external/musicgen-local/apps/plugin-juce/legacy-reference/plugin-update/engine/IntentPipeline.h` | Not compiled/integrated in current local app pipeline |
| C++ RuleBreakEngine surfaces | reference-imported | Source repo defines category-specific rulebreak generation by emotion | `/Volumes/KmiDi-external/KmiDi_MASTER_VAULTEXTERNAL/KmiDi/src/src/engine/RuleBreakEngine.h` | Outside `musicgen-local`; used as reference source |
| JUCE PluginProcessor scheduler | reference-imported | Host-sync beat scheduling for generated MIDI notes | `/Volumes/KmiDi-external/musicgen-local/apps/plugin-juce/legacy-reference/plugin-update/plugin/PluginProcessor.cpp` | Reference code only; no built plugin artifact in this stack |
| PyTorch audio classifier/ResNet | reference-imported | CNN/ResNet mel-spectrogram classification models | `/Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/models/audio_classifier.py` | Task-specific to classification, not full music generation |
| LoRA adaptation utilities | reference-imported | LoRA layer wrapping and weight merge/load/save helpers | `/Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/models/lora.py` | Not yet wired into symbolic/JEPA local training entrypoints |
| Trainer loop | reference-imported | Train/val loop, scheduler, early stop, checkpointing, TensorBoard | `/Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/training/trainer.py` | Not yet connected to production training scripts |
| Audio dataset pipeline | reference-imported | Directory-backed audio loading, mel extraction, dataloader splits | `/Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite/data/dataset.py` | No local training job currently invokes it |

## Pipelines

| Pipeline | Status | How It Works | Where | Limitations |
|---|---|---|---|---|
| Training gate validation | implemented | Parses approval table; requires Owner+Date+Allowed Uses+Commercial YES+Training YES per source | `/Volumes/KmiDi-external/musicgen-local/scripts/check_training_gate.sh` | Markdown-table parsing contract must remain stable |
| Gate open/close/status | implemented | Opens/closes provenance gate line and reports validation status | `/Volumes/KmiDi-external/musicgen-local/scripts/open_training_gate.sh`, `/Volumes/KmiDi-external/musicgen-local/scripts/close_training_gate.sh`, `/Volumes/KmiDi-external/musicgen-local/scripts/gate_status.sh` | Document line text is string-matched |
| Source intake + dedupe | implemented | Manifested source capture, SHA256 generation, dedupe maps for MIDI/model artifacts | `/Volumes/KmiDi-external/musicgen-local/ml/data/manifests/sources/` and run logs in `/Volumes/KmiDi-external/musicgen-local/ops/runs/` | Intake currently one-time/manual by transfer passes |
| Active manifest generation | implemented | Filters source manifests by approved roots in checklist | `/Volumes/KmiDi-external/musicgen-local/scripts/build_training_manifests.sh` | Root-prefix matching assumes consistent absolute paths |
| Train entrypoint preflight | implemented | Train scripts enforce gate check before any training call | `/Volumes/KmiDi-external/musicgen-local/scripts/train-symbolic.sh`, `/Volumes/KmiDi-external/musicgen-local/scripts/train-jepa.sh` | Training payload still placeholder echo |
| Eval gate | placeholder | Stub script for future metric thresholds and release gating | `/Volumes/KmiDi-external/musicgen-local/scripts/eval-gate.sh` | No metric logic implemented |
| Local stack runner/package | placeholder | Stub scripts for runtime stack and plugin packaging | `/Volumes/KmiDi-external/musicgen-local/scripts/run-local-stack.sh`, `/Volumes/KmiDi-external/musicgen-local/scripts/package-plugin.sh` | No executable stack/package behavior yet |

## Outputs

| Output | Status | How It Works | Where | Current State |
|---|---|---|---|---|
| Staged MIDI corpus | implemented | Manifest-driven copy + dedupe into staging | `/Volumes/KmiDi-external/musicgen-local/ml/data/raw/midi_staging` | 311 files |
| Imported model artifacts | implemented | Curated `.pt/.onnx` ingest into runtime artifact folder | `/Volumes/KmiDi-external/musicgen-local/ml/inference/runtimes/legacy_models` | 47 files |
| Active approved manifests | implemented | Derived from checklist approvals via manifest builder | `/Volumes/KmiDi-external/musicgen-local/ml/data/manifests/active/midi-approved.txt`, `/Volumes/KmiDi-external/musicgen-local/ml/data/manifests/active/models-approved.txt` | 326 MIDI entries, 47 model entries |
| Transfer/audit run logs | implemented | Pass-by-pass operational record of imports, gate setup, and status checks | `/Volumes/KmiDi-external/musicgen-local/ops/runs` | Passes 01-11 present |

## Model Integrations

| Integration | Status | How It Works | Where | Limitations |
|---|---|---|---|---|
| Legacy ONNX/PT artifacts | reference-imported | Imported for baseline compatibility/reference | `/Volumes/KmiDi-external/musicgen-local/ml/inference/runtimes/legacy_models` | Provenance approved for current workflow; runtime serving not wired |
| Legacy model registry schema | reference-imported | Schema available for future registry normalization | `/Volumes/KmiDi-external/musicgen-local/schemas/model-registry.legacy.schema.json` | No live registry file/service contract yet |
| Training scaffold modules | reference-imported | Imported code supports model/data/trainer primitives | `/Volumes/KmiDi-external/musicgen-local/ml/training/shared/bootstrap_from_ml_training_suite` | Needs integration into symbolic/JEPA/audio-render job definitions |

## Not Implemented Yet

| Area | Status | Notes |
|---|---|---|
| Sidecar runtime logic | placeholder | `/Volumes/KmiDi-external/musicgen-local/apps/sidecar-engine` has no active implementation yet |
| Generation orchestrator API path | placeholder | `/Volumes/KmiDi-external/musicgen-local/services/generation-orchestrator` not implemented |
| Prompt->graph->MIDI/audio e2e pipeline | placeholder | Schemas/docs exist, execution path not wired |
| Production plugin build/integration | placeholder | JUCE code is reference import; no integrated build chain in this project |

## Rulebreak Positioning
- Rulebreak behavior in this project is currently `reference-imported` and documentation-driven.
- Canonical reference mapping is maintained in `RULEBREAK_REFERENCE_INDEX.md`.
- Local production integration is pending future sidecar/orchestrator implementation.
