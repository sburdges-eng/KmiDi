# Issues Report

## Scope
- Focused on runtime-critical sources in `KmiDi_PROJECT/source/python/mcp_workstation`, `music_brain`, and `KmiDi_PROJECT/source/frontend/src-tauri`.
- External/third-party code (for example `KmiDi_PROJECT/external`) and large training/output datasets were not deeply reviewed.

## Findings

### Resolved (Workstation/Tauri)
- Added a `Workstation` facade so CLI/server methods (`get_status`, proposals, phase, C++ plan, debug) are available without errors.
- `get_workstation()` now uses a singleton with optional `llm_model_path`, matching CLI/server usage.
- Image/audio engines now lazy-load pipelines/models on first use, avoiding permanent stub mode.
- Audiocraft import is no longer hard-disabled; it is enabled when installed.
- `music_brain` package initialization is present.
- Tauri HTTP bridge now uses a reqwest client with a 10s timeout.

## Training Findings (KmiDi_TRAINING)

### High
8) Distributed training is not initialized but DDP is enabled.
- `KmiDi_TRAINING/training/training/train_integrated.py:528-531` wraps the model in `DistributedDataParallel`, but there is no `dist.init_process_group(...)` anywhere in the file.
- Impact: enabling `distributed=True` will raise at runtime because the default process group is not initialized.

9) DDP device index assumes CUDA and will fail on CPU/MPS devices.
- `KmiDi_TRAINING/training/training/train_integrated.py:528-533` uses `device.index` for `device_ids` and `output_device`.
- Impact: on CPU or MPS, `device.index` is `None`, which causes DDP initialization errors.

10) Spectocloud ONNX export likely fails because the model returns a dict.
- `KmiDi_TRAINING/training/training/cuda_session/train_spectocloud.py:96-131` returns a dict from `PointCloudDecoder`.
- `KmiDi_TRAINING/training/training/cuda_session/export_models.py:48-63` calls `torch.onnx.export` with named outputs but the model forward returns a dict, which ONNX export does not accept as-is.
- Impact: export script will error or produce invalid ONNX output unless the model output is converted to a tuple/tensors.

### Medium
11) MIDI generator config special tokens and vocab sizes do not match the tokenizer or loss masking.
- `KmiDi_TRAINING/training/training/cuda_session/midi_generator_training_config.yaml:42-105` defines `vocab_size: 512` and special tokens `pad: 0`, `bos: 1`, `eos: 2`, `bar: 3`.
- `KmiDi_TRAINING/training/training/cuda_session/train_midi_generator.py:45-57` hardcodes `pad_token=384`, `bos_token=385`, `eos_token=386`, `bar_token=387`.
- `KmiDi_TRAINING/training/training/cuda_session/train_midi_generator.py:621-666` masks padding via `ignore_index=384`.
- Impact: if training data follows the YAML token IDs, padding will be treated as real tokens and loss/accuracy will be wrong.

12) Spectocloud config advertises multi-GPU training, but the script is single-GPU only.
- `KmiDi_TRAINING/training/training/cuda_session/spectocloud_training_config.yaml:18-25` sets `num_gpus: 4`.
- `KmiDi_TRAINING/training/training/cuda_session/train_spectocloud.py:658-672` always selects a single device and does not initialize DDP.
- Impact: expected multi-GPU speedups will not be realized; large batch settings in the config may OOM on a single GPU.

13) Legacy training script uses dummy data instead of real datasets.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/scripts/train.py:568-571` explicitly calls `create_dummy_dataloaders` with a TODO to replace.
- Impact: training results and exported models are not based on real data.

14) Backup training pipeline silently substitutes random noise when audio is missing or torchaudio is unavailable.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/scripts/train_model.py:230-264` falls back to random mel spectrograms without warning when audio cannot be loaded.
- Impact: models can appear to train but learn from synthetic noise, masking data/IO failures.

15) Dataset preparation scripts hardcode a machine-specific external SSD path.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/scripts/prepare_datasets.py:44-45` and `KmiDi_TRAINING/training/ML Kelly Training/backup/scripts/train_model.py:83-85` use `/Volumes/Extreme SSD/kelly-audio-data`.
- Impact: out-of-the-box dataset prep and training fail on machines without the same mount point.

### Low
16) Dataset downloader uses network requests without timeouts.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/datasets/audio_downloader.py:120-233` calls `requests.get(...)` without a timeout.
- Impact: dataset downloads can hang indefinitely on network stalls.

17) Registry manifest validation is silently skipped when `jsonschema` is missing.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/model_registry.py:19-27` sets `jsonschema = None` on import failure and validation is skipped without warning.
- Impact: invalid registry manifests can be accepted without any signal.

### Medium
18) Evaluation fails when dataloaders provide inputs without targets.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/training/evaluation.py:401-421` only calls `metrics.update(...)` if targets are present.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/training/evaluation.py:67-95` assumes non-empty prediction/target buffers and uses `np.concatenate`, which throws on empty lists.
- Impact: evaluation will crash on inference-only loaders that yield inputs without labels.

19) Cross-validation always uses classification loss, even for regression tasks.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/training/evaluation.py:492-507` uses `torch.nn.CrossEntropyLoss()` unconditionally.
- Impact: regression models will train with the wrong loss during cross-validation.

20) Multi-head attention positional encoding can fail on longer sequences.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/training/architectures.py:206-243` slices a fixed-size positional encoding without handling `seq_len > max_len`.
- Impact: sequences longer than `max_len` will raise a shape error when adding the positional encoding.

### Low
21) Review artifacts claim critical issues are fixed, but current backup scripts still include hardcoded paths and dummy data.
- `KmiDi_TRAINING/outputs/output/review/COMPREHENSIVE_PROJECT_REVIEW.md:1-90` and `KmiDi_TRAINING/outputs/output/review/FINAL_STATUS_REMAINING_ISSUES.md:1-90` state hardcoded paths are removed and only stylistic issues remain.
- Current backup scripts still use `/Volumes/Extreme SSD/kelly-audio-data` and dummy datasets.
- Impact: review reports are stale and can mislead validation/QA.

### Build Notes (Non-blocking)
- JUCE macOS 15 deprecation warnings during `KellyTests` build (CoreVideo/CoreText).
- Missing `WrapVulkanHeaders` and `pybind11` are reported by CMake; builds still succeed without them.
- `KellyPlugin_VST3` logs missing runtime data directories and falls back to embedded defaults.
