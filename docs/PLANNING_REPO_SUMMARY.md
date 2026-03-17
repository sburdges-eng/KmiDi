# KmiDi Repo — Planning Summary

Structured summary for planning: layout, domain touchpoints, data/config locations, single source of truth, and gaps.

---

## 1. Layout — Top-level directory structure

| Area | Purpose |
|------|---------|
| **src/** | React (Vite) app: components, hooks, types. UI, intent builder, emotion wheel. |
| **src-tauri/** | Tauri 2 + Rust: native host, FFI bridge to C++, build.rs, generated intent.rs. |
| **music_brain/** | Python FastAPI backend and engine API: api.py, `/generate`, session, generative, jepa, emotion, groove, harmony, penta_core, engine_api, etc. |
| **shared_schemas/** | Single source of truth for intent: `CompleteSongIntentRequest.json`, `CompleteSongIntent.json`. Synced to TS/Rust via `sync_entities.py`. |
| **scripts/** | Dev/setup: `sync_entities.py`, `dev-setup.sh`, `build-full-stack.sh`, `build_v1.sh`, env (load-env.sh, setup-env.sh, validate-env.sh). Data/training: `make_jepa_manifest.py`, `train_jepa_local.py`, `launch_jepa_sagemaker.py`, `prepare_datasets.py`, `download_all_datasets.sh`, `download_musicnet_aria2.sh`, `build_manifests.py`, `package_dataset.py`. MCP, migrations, CI. |
| **tests/** | Python tests (pytest). No `--timeout` (pytest-timeout not installed). |
| **engine/** | C++ engine: IntentPipeline, CoreBridge, AdapterRegistry, StateMachineConductor; CMake subdir. |
| **include/** | C++ public headers (e.g. for KellyCore/KellyFFI). |
| **rt_harness/** | Headless RT callback harness (BUILD_RT_HARNESS); P50/P90/P99 stats. |
| **cmake/** | CMake helpers and project config. |
| **config/** | Training/model config YAML: `jepa_training.yaml`, `models.yaml`, emotion/harmony/groove/dynamics configs, `dataset_manifest_schema.json`, env examples (sagemaker, ec2-s3). |
| **configs/** | Alternate config root (e.g. storage, experiments); see DATA_AND_TRAINING. |
| **docs/** | DEVELOPMENT.md, ENVIRONMENT.md, FULL_STACK_BUILD.md, DATA_AND_TRAINING.md, SAGEMAKER_SETUP.md, BUILD.md refs, research/, specs/, audit/, discovery notes. |
| **experiments/** | Isolated experiment code: e.g. `exp_002_wavjepa_emotion`, `perch_remi_pipeline`; lightweight summaries, not weights. |
| **external/** | JUCE 8 submodule (required for C++/plugins/FFI). |
| **build/** | CMake out-of-tree build output (e.g. KellyFFI, KellyCore, KellyFFIBenchmark). |
| **training/** | Training entrypoints and shared training code. |
| **apps/, libs/, bindings/, bridges/** | Additional app/lib and bridge code. |
| **KmiDi_FINAL/, legacy/, frontend/** | Legacy / alternate UI and integration trees. |

**Data flow (AGENTS.md):** React → invoke()/events → Tauri → Rust FFI → KellyFFI (C ABI) → KellyCore (C++).  
**API flow:** React → HTTP → Music Brain API (`/generate`, `/docs`).

---

## 2. Domain touchpoints — Existing references

| Topic | Where | Summary |
|-------|--------|---------|
| **Open song generation** | music_brain (api, session/generator, tier1/midi_generator, structure, etc.) | HTTP `/generate` and intent-driven generation; no explicit “open song generation” product name. |
| **MidiTok** | docs/REMI_BPE_TOKENIZATION.md, docs/research/MULTIMODAL_IMPLEMENTATIONS_PLAN.md, experiments/perch_remi_pipeline/ (remi_bpe_demo.py, make_fixtures.py) | REMI + BPE tokenization; Maestro-REMI-bpe20k; `pip install miditok`. |
| **Autotroph** | — | **No references.** |
| **Tokenization** | Intent.ts / CompleteSongIntentRequest (rhythmic “quantization/swing”); REMI-BPE (MidiTok); REMI_BPE_TOKENIZATION.md; MULTIMODAL_IMPLEMENTATIONS_PLAN. | UI/API: quantization = swing/feel. Symbolic: REMI-BPE pipeline. |
| **KellyBrain** | docs/DEVELOPMENT.md (KellyBrain.h/.cpp, useKellyBrain.ts), AGENTS.md | Documented as main AI interface in C++ and React hook; paths may be aspirational vs current engine layout. |
| **Brain–Body** | — | **No references.** |
| **kellyharness** | — | **No references.** RT harness is `rt_harness` + BUILD_RT_HARNESS (headless RT callback runner). |
| **JEPA** | SAGEMAKER_SETUP.md, config/jepa_training.yaml, scripts/make_jepa_manifest.py, scripts/train_jepa_local.py, scripts/launch_jepa_sagemaker.py, music_brain/jepa/, pyproject.toml [jepa] (lhotse, soundfile) | Audio-JEPA and Chord-JEPA; SageMaker training; Lhotse manifests for MAESTRO-like audio+MIDI. |
| **WavJEPA** | docs/WAVJEPA_KMIDI_TASKS.md, docs/WAVJEPA_LATENT_PIPELINE.md, docs/wavjepa_*.md, docs/UMP_JEPA_EXPRESSIVE_CONDITIONING.md, experiments/exp_002_wavjepa_emotion | Frozen WavJEPA front-end; emotion separability (CREMA-D, RAVDESS); latent pipeline design. |
| **Transcriber** | docs/mt3-transcription-baseline.md | MT3 as audio→symbolic transcription baseline; latent→token head. |
| **MoE** | — | **No references.** |
| **MIDI datasets** | make_jepa_manifest.py (MAESTRO-like audio+MIDI), DATA_AND_TRAINING (maestro, groove_midi), prepare_datasets, JEPA config (data/audio, data/midi) | MAESTRO, groove_midi, local/S3; Lhotse RecordingSet/SupervisionSet/CutSet. |
| **Emotion corpora** | exp_002_wavjepa_emotion (CREMA-D, RAVDESS); DATA_AND_TRAINING; DISCOVERY_DATASETS_KMIDI_PATH (emotions/ravdess, emotions/cremad) | Emotion datasets under dataset root; WavJEPA emotion separability experiment. |
| **Evaluation / benchmarks** | KellyFFIBenchmark (CMake, FFI integration); DEVELOPMENT.md “Performance benchmarking”; wavjepa_baselines_results.md; research (RESEARCH_AI_EVAL_GUARDRAILS, etc.) | FFI benchmark; WavJEPA baselines; research eval guardrails. No single evaluation harness doc. |
| **Apple Silicon** | docs/apple-silicon-low-latency.md | Low-latency tuning (Audio Workgroup, QoS, buffer sweep, ANE vs GPU). |
| **XPC / UDS** | — | **No references.** |
| **Label Studio** | — | **No references.** |
| **Lhotse** | scripts/make_jepa_manifest.py, pyproject.toml [jepa] | JEPA manifests: RecordingSet, SupervisionSet, CutSet; optional `pip install -e ".[jepa]"`. |
| **Core ML** | — | **No references.** |
| **Quantization** | shared_schemas/CompleteSongIntentRequest.json, src/types/Intent.ts | Rhythmic feel (quantization/swing) in intent schema only; no model quantization references. |

---

## 3. Data / config locations (external and in-repo)

| Purpose | Location | Notes |
|---------|----------|--------|
| **Dataset root** | Resolution order: `KMIDI_DATASETS_PATH` → `AUDIO_DATA_ROOT` → config `dataset_root` → `~/Datasets` | DATA_AND_TRAINING.md, DISCOVERY_DATASETS_KMIDI_PATH.md. Optional: `~/Datasets/by_source/kmidi` (not wired in code). |
| **Models / checkpoints** | `~/Models` (can override with `KELLY_MODEL_ROOT`). Checkpoints: `~/Models/checkpoints/<experiment_name>/` | ENVIRONMENT.md: `KELLY_MODELS_PATH` (default `./models`) for C++; `PYTHON_MODEL_PATH`, `CHECKPOINT_PATH`. |
| **Training data** | `TRAINING_DATA_PATH` (default `./data/training`); JEPA: `data/audio`, `data/midi` or `JEPA_AUDIO_DIR`, `JEPA_MIDI_DIR` | train_jepa_local.py; prepare_datasets writes under `KMIDI_DATA_ROOT/Datasets` when set. |
| **Configs (training/model)** | `config/` (YAML: jepa_training, models, emotion, harmony, groove, etc.); `configs/` for storage/experiments | DATA_AND_TRAINING: configs committed; experiments in experiments/. |
| **Experiments** | `experiments/` (e.g. exp_002_wavjepa_emotion, perch_remi_pipeline) | Lightweight summaries only; no weights in repo. |
| **Env files** | `.env`, `.env.development`, `.env.production`, `env/.env.*`, `.env.local` (git-ignored) | scripts/load-env.sh, validate-env.sh, setup-env.sh. |
| **External SSD** | `KMIDI_DATA_ROOT` = e.g. `/Volumes/KmiDi-external`; under it: Datasets/, build/, Models/ | docs/SSD_WORKDIR_STRUCTURE.md, DISCOVERY_SSD_WHEN_MOUNTED. |
| **Cache** | `KMIDI_CACHE_ROOT` (e.g. on external volume) | Build/deps; see workspace rules. |

---

## 4. Single source of truth — Key files and docs

| Role | File(s) |
|------|---------|
| **Intent schema (UI–engine contract)** | `shared_schemas/CompleteSongIntentRequest.json` (source); synced to `src/types/Intent.ts`, `src-tauri/src/generated/intent.rs`, and Python validation via `scripts/sync_entities.py`. |
| **Agent/dev context** | `AGENTS.md` (canonical project overview, layout, services, build, gotchas, API minimal example). |
| **Build (C++ / CMake / Tauri)** | `BUILD.md`; `CMakeLists.txt`; `docs/FULL_STACK_BUILD.md`. |
| **Environment** | `docs/ENVIRONMENT.md` (vars, paths, categories, loading); `.env.example`; `env/*.example`. |
| **Data and training governance** | `docs/DATA_AND_TRAINING.md` (DATA LAW, dataset root, model/checkpoint paths, run manifest, reproducibility). |
| **API contract** | `music_brain/api.py` (GenerateRequest, EmotionalIntent; structure/instruments format); engine boundary uses `music_brain/engine_api/schema.py` (CompleteSongIntentRequest) separately. |
| **Dataset manifest schema** | `config/dataset_manifest_schema.json` (DatasetManifest 2.0). |
| **Migration manifest** | `migration_manifest.yaml` (optional schema/migration versioning; minimal content). |

---

## 5. Download scripts, manifests, and data-prep pipelines

| Asset | Location | Purpose |
|-------|----------|---------|
| **sync_entities.py** | scripts/ | Sync shared_schemas → Intent.ts, intent.rs, Python validation. |
| **make_jepa_manifest.py** | scripts/ | Lhotse JEPA manifests (RecordingSet, SupervisionSet, CutSet) for MAESTRO-like audio+MIDI; sha1_audio/sha1_midi, window/stride; output `manifests/` + manifest_args.json. |
| **download_all_datasets.sh** | scripts/ | Calls `prepare_datasets.py --dataset all --download`; uses `KMIDI_DATA_ROOT` (default repo `datasets/`). |
| **download_musicnet_aria2.sh** | scripts/ | MusicNet download via aria2. |
| **prepare_datasets.py** | scripts/utilities/ | Dataset prep; uses `KMIDI_DATA_ROOT/Datasets` when set. |
| **build_manifests.py** | scripts/ | Build manifest artifacts. |
| **package_dataset.py** | scripts/ | Package dataset (e.g. for S3 or distribution). |
| **JEPA manifest output** | `manifests/` (default from make_jepa_manifest.py) | recordings.jsonl, supervisions.jsonl, cuts.jsonl, cuts_with_hashes.jsonl, manifest_args.json. |
| **SageMaker** | scripts/launch_jepa_sagemaker.py, sagemaker_train.py, sagemaker_entrypoint.py | Submit jobs; training reads S3 audio/midi, writes to `/opt/ml/model`. |
| **setup-workspace** | — | **Not found** in repo; workspace rules mention `scripts/setup-workspace.sh` for dataset symlinks when using external SSD. |

---

## 6. Gaps (for planning)

- **Autotroph, Brain–Body, kellyharness, MoE, Label Studio, Core ML, XPC, UDS:** No current references; add if adopting.
- **“Open song generation”:** No dedicated product/feature name; generation is under `/generate` and session/generative code.
- **setup-workspace.sh:** Referenced in workspace rules (dataset symlinks) but not present in repo; add or document alternative.
- **Evaluation/benchmarks:** Scattered (KellyFFIBenchmark, WavJEPA baselines, research); no single evaluation harness or benchmark manifest.
- **Model quantization:** Only “quantization” in intent is rhythmic (swing); no ML model quantization (e.g. ONNX/Core ML) references.
- **KellyBrain:** Documented in DEVELOPMENT.md as C++/React; confirm whether engine paths (e.g. engine/KellyBrain.h) exist in current tree or are legacy/aspirational.
- **Single dataset-root doc:** DATA_AND_TRAINING and DISCOVERY_DATASETS_KMIDI_PATH define resolution; consider one canonical subsection in ENVIRONMENT.md or AGENTS.md.

---

*Generated for planning; update as repo and docs change.*
