# Issues Report

## Scope
- Reviewed runtime-critical sources in `KmiDi_PROJECT`, `music_brain`, and training code under `KmiDi_TRAINING`.
- External/third-party code (for example `KmiDi_PROJECT/external`) and large datasets were not deeply reviewed.

## Findings

### Blockers
1) Missing module import prevents the orchestrator from starting.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py:13` imports `music_brain.tier1.midi_pipeline_wrapper`, but `music_brain/tier1/` does not exist in the repo.
- Impact: `python -m mcp_workstation` and any orchestration flow will fail at import time.

2) `get_workstation()` is incompatible with the current `Orchestrator` signature.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py:18-35` requires `llm_model_path`.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py:227-229` exposes `get_workstation()` with no required args.
- `KmiDi_PROJECT/source/python/mcp_workstation/cli.py:166` and `KmiDi_PROJECT/source/python/mcp_workstation/server.py:365` call `get_workstation()` with no arguments.
- Impact: CLI/server paths will raise `TypeError` before doing any work.

### High
3) CLI/server call a proposal/task API that does not exist on `Orchestrator`.
- `KmiDi_PROJECT/source/python/mcp_workstation/cli.py:168-258` calls methods like `get_status`, `submit_proposal`, `get_phase_progress`.
- `KmiDi_PROJECT/source/python/mcp_workstation/server.py:369-458` calls the same proposal/task API surface.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py` only defines `execute_workflow` and lock helpers.
- Impact: even if the constructor mismatch is fixed, these calls will raise `AttributeError` at runtime.

4) Image/audio generation paths are effectively stubbed in normal flows.
- `KmiDi_PROJECT/source/python/mcp_workstation/llm_reasoning_engine.py:70-73` constructs `ImageGenerationEngine`/`AudioGenerationEngine` but never loads models.
- `KmiDi_PROJECT/source/python/mcp_workstation/image_generation_engine.py:82-96` returns a stub unless `_load_pipeline()` has been called.
- `KmiDi_PROJECT/source/python/mcp_workstation/audio_generation_engine.py:62-75` returns a stub unless `_load_model()` has been called.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py:30-35` never calls `_load_pipeline()` or `_load_model()`.
- Impact: image/audio generation will always return placeholder data unless a caller manually loads models.

5) Distributed training is not initialized but DDP is enabled.
- `KmiDi_TRAINING/training/training/train_integrated.py:528-531` wraps the model in `DistributedDataParallel`, but there is no `dist.init_process_group(...)` anywhere in the file.
- Impact: enabling `distributed=True` will raise at runtime because the default process group is not initialized.

6) DDP device index assumes CUDA and will fail on CPU/MPS devices.
- `KmiDi_TRAINING/training/training/train_integrated.py:528-533` uses `device.index` for `device_ids` and `output_device`.
- Impact: on CPU or MPS, `device.index` is `None`, which causes DDP initialization errors.

7) Spectocloud ONNX export likely fails because the model returns a dict.
- `KmiDi_TRAINING/training/training/cuda_session/train_spectocloud.py:96-131` returns a dict from `PointCloudDecoder`.
- `KmiDi_TRAINING/training/training/cuda_session/export_models.py:48-63` calls `torch.onnx.export` with named outputs but the model forward returns a dict, which ONNX export does not accept as-is.
- Impact: export script will error or produce invalid ONNX output unless the model output is converted to a tuple/tensors.

### Medium
8) Audio generation is permanently disabled by a hardcoded flag.
- `KmiDi_PROJECT/source/python/mcp_workstation/audio_generation_engine.py:7-12` comments out the audiocraft imports and sets `AUDIOCRAFT_AVAILABLE = False` unconditionally.
- Impact: `AudioGenerationEngine` never transitions out of stub mode even if audiocraft is installed.

9) `music_brain` is used as a package but has no `__init__.py` at its root.
- `KmiDi_PROJECT/source/python/kmidi_gui/core/preset.py:12-13` imports `music_brain.session.intent_schema`.
- `music_brain/` lacks an `__init__.py` file, making it a namespace package and potentially breaking tooling/packaging assumptions.
- Impact: imports may fail depending on the runtime packaging or how PYTHONPATH is configured.

10) MIDI generator config special tokens and vocab sizes do not match the tokenizer or loss masking.
- `KmiDi_TRAINING/training/training/cuda_session/midi_generator_training_config.yaml:42-105` defines `vocab_size: 512` and special tokens `pad: 0`, `bos: 1`, `eos: 2`, `bar: 3`.
- `KmiDi_TRAINING/training/training/cuda_session/train_midi_generator.py:45-57` hardcodes `pad_token=384`, `bos_token=385`, `eos_token=386`, `bar_token=387`.
- `KmiDi_TRAINING/training/training/cuda_session/train_midi_generator.py:621-666` masks padding via `ignore_index=384`.
- Impact: if training data follows the YAML token IDs, padding will be treated as real tokens and loss/accuracy will be wrong.

11) Spectocloud config advertises multi-GPU training, but the script is single-GPU only.
- `KmiDi_TRAINING/training/training/cuda_session/spectocloud_training_config.yaml:18-25` sets `num_gpus: 4`.
- `KmiDi_TRAINING/training/training/cuda_session/train_spectocloud.py:658-672` always selects a single device and does not initialize DDP.
- Impact: expected multi-GPU speedups will not be realized; large batch settings in the config may OOM on a single GPU.

12) Legacy training script uses dummy data instead of real datasets.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/scripts/train.py:568-571` explicitly calls `create_dummy_dataloaders` with a TODO to replace.
- Impact: training results and exported models are not based on real data.

13) Backup training pipeline silently substitutes random noise when audio is missing or torchaudio is unavailable.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/scripts/train_model.py:230-264` falls back to random mel spectrograms without warning when audio cannot be loaded.
- Impact: models can appear to train but learn from synthetic noise, masking data/IO failures.

14) Dataset preparation scripts hardcode a machine-specific external SSD path.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/scripts/prepare_datasets.py:44-45` and `KmiDi_TRAINING/training/ML Kelly Training/backup/scripts/train_model.py:83-85` use `/Volumes/Extreme SSD/kelly-audio-data`.
- Impact: out-of-the-box dataset prep and training fail on machines without the same mount point.

15) Evaluation fails when dataloaders provide inputs without targets.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/training/evaluation.py:401-421` only calls `metrics.update(...)` if targets are present.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/training/evaluation.py:67-95` assumes non-empty prediction/target buffers and uses `np.concatenate`, which throws on empty lists.
- Impact: evaluation will crash on inference-only loaders that yield inputs without labels.

16) Cross-validation always uses classification loss, even for regression tasks.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/training/evaluation.py:492-507` uses `torch.nn.CrossEntropyLoss()` unconditionally.
- Impact: regression models will train with the wrong loss during cross-validation.

17) Multi-head attention positional encoding can fail on longer sequences.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/training/architectures.py:206-243` slices a fixed-size positional encoding without handling `seq_len > max_len`.
- Impact: sequences longer than `max_len` will raise a shape error when adding the positional encoding.

18) MIDI feature extraction treats tick counts as seconds.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/datasets/midi_features.py:179-199` accumulates `msg.time` directly into `current_time` and stores `onset`/`duration` without converting ticks to seconds.
- Downstream groove/tempo features use these values as seconds.
- Impact: timing-derived features (duration, onset_times, groove metrics) are incorrect for files with non-default tempo or ticks-per-beat.

19) CMake source list includes duplicate OSC source entries.
- `KmiDi_PROJECT/source/cpp/src/CMakeLists.txt:23-30` lists `osc/OSCClient.cpp` twice.
- Impact: can lead to duplicate object compilation and linker errors or redundant build steps.

### Low
20) Tauri HTTP bridge has no timeouts for local API requests.
- `KmiDi_PROJECT/source/frontend/src-tauri/src/bridge/musicbrain.rs:7-75` uses `reqwest::Client::new()` and `.send().await?` without a timeout.
- Impact: UI commands can hang indefinitely if the local service is down or unresponsive.

21) Dataset downloader uses network requests without timeouts.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/datasets/audio_downloader.py:120-233` calls `requests.get(...)` without a timeout.
- Impact: dataset downloads can hang indefinitely on network stalls.

22) Registry manifest validation is silently skipped when `jsonschema` is missing.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/model_registry.py:19-27` sets `jsonschema = None` on import failure and validation is skipped without warning.
- Impact: invalid registry manifests can be accepted without any signal.

23) Review artifacts claim critical issues are fixed, but current backup scripts still include hardcoded paths and dummy data.
- `KmiDi_TRAINING/outputs/output/review/COMPREHENSIVE_PROJECT_REVIEW.md:1-90` and `KmiDi_TRAINING/outputs/output/review/FINAL_STATUS_REMAINING_ISSUES.md:1-90` state hardcoded paths are removed and only stylistic issues remain.
- Current backup scripts still use `/Volumes/Extreme SSD/kelly-audio-data` and dummy datasets.
- Impact: review reports are stale and can mislead validation/QA.

24) Frontend hook bypasses Tauri invoke and hardcodes local API URLs without timeouts.
- `KmiDi_PROJECT/source/cpp/src/hooks/useMusicBrain.ts:94-170` uses `fetch('http://127.0.0.1:8000/...')` directly for config and render calls.
- Impact: frontend calls fail when the local API is not running and can hang without a timeout; also ignores any Tauri-side base URL configuration.

25) Platform/tooling roadmaps reference a local workstation path.
- `KmiDi_PROJECT/source/frontend/iOS/ROADMAP.md:2`, `KmiDi_PROJECT/source/frontend/iOS/TODO.md:2`, `KmiDi_PROJECT/source/frontend/mobile/ROADMAP.md:2`, `KmiDi_PROJECT/source/frontend/mobile/TODO.md:2`, `KmiDi_PROJECT/tools/ROADMAP.md:2`, and `KmiDi_PROJECT/tools/TODO.md:2` reference `/Users/seanburdges/Desktop/final kel`.
- Impact: documentation is not portable and can mislead contributors who do not have that path.

### Build Notes (Non-blocking)
- JUCE macOS 15 deprecation warnings during `KellyTests` build (CoreVideo/CoreText).
- Missing `WrapVulkanHeaders` and `pybind11` are reported by CMake; builds still succeed without them.
- `KellyPlugin_VST3` logs missing runtime data directories and falls back to embedded defaults.
