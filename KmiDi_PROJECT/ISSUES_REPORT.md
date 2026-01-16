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

31) Core inference module is missing, breaking enhanced and async inference paths.
- `python/penta_core/ml/inference_enhanced.py:19-33` imports `penta_core.ml.inference`, but there is no `python/penta_core/ml/inference.py` in the repo, so `HAS_BASE_INFERENCE` is always false.
- `python/penta_core/ml/async_inference.py:27` imports `InferenceResult` from `penta_core.ml.inference` without a try/except; the module import raises immediately.
- Impact: enhanced inference cannot load any model, and async inference fails to import at all.

32) Training transformer models reference `torch` without importing it.
- `python/penta_core/ml/training_orchestrator.py:422-434` and `python/penta_core/ml/training_orchestrator.py:475-487` use `torch.arange(...)` but only import `torch.nn` in those helper methods.
- Impact: creating harmony or melody models will throw `NameError` at runtime when the forward pass runs.

33) Training uses classification loss for regression targets.
- `python/penta_core/ml/training_orchestrator.py:324-340` sets `CrossEntropyLoss` globally for all tasks.
- `python/penta_core/ml/training_orchestrator.py:563-612` generates float targets for groove/dynamics tasks.
- Impact: training will error or optimize the wrong objective for regression models.

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

20) ONNX inference compile error due to duplicate session declaration.
- `KmiDi_PROJECT/source/cpp/src/ml/ONNXInference.cpp:82-94` declares `Session* session` twice in the same scope.
- Impact: build fails when `ENABLE_ONNX_RUNTIME` is enabled.

21) AdaptiveGenerator ignores learned preferences.
- `KmiDi_PROJECT/source/cpp/src/engine/AdaptiveGenerator.cpp:26-53` returns the input intent unchanged, with TODOs for preference application.
- Impact: enabling adaptive mode has no effect, so personalization is effectively disabled.

22) Onset detection is stubbed and never computes spectral flux.
- `KmiDi_PROJECT/source/cpp/src/groove/OnsetDetector.cpp:18-57` leaves processing methods as no-ops and always clears detection state.
- Impact: any features depending on onset detection will be non-functional.

23) Harmony history APIs are placeholders that return only current state.
- `KmiDi_PROJECT/source/cpp/src/harmony/HarmonyEngine.cpp:86-104` returns `{currentChord_}` and `{currentScale_}` with TODOs for history tracking.
- Impact: consumers expecting history will see only the latest snapshot.

24) Master EQ processing is a pass-through stub.
- `KmiDi_PROJECT/source/cpp/src/plugin/MasterEQProcessor.cpp:33-78` does not apply any filtering and leaves TODOs for biquad processing.
- Impact: EQ controls appear to work but do not affect audio output.

34) Training orchestrator always trains on dummy data.
- `python/penta_core/ml/training_orchestrator.py:606-607` calls `_create_dummy_dataloader` for both train/val splits.
- Impact: training metrics are not based on real datasets and cannot produce usable models.

35) Training export step logs success but does not write artifacts.
- `python/penta_core/ml/training_orchestrator.py:1152-1173` only logs export paths and never creates `export_dir` or saves a model.
- Impact: runs report successful exports even though no files are produced.

74) WAV reader mis-parses the fmt chunk and shifts all header fields.
- `KmiDi_PROJECT/source/cpp/src/audio/AudioFile.cpp:75-112` reads `header.fmtSize` from the stream even though the fmt chunk size was already read into `chunkSize`.
- This shifts the read offset so `audioFormat`, `numChannels`, and `sampleRate` are decoded from the wrong bytes.
- Impact: standard WAV files can be rejected or decoded with invalid metadata.

75) Project JSON loading discards track details and replaces them with placeholders.
- `KmiDi_PROJECT/source/cpp/src/project/ProjectFile.cpp:171-233` only counts `{}` blocks and creates default `Track` entries.
- Track properties (`type`, `midiEvents`, `audioFile`, volume/pan, etc.) are ignored on load.
- Impact: reloading a saved project loses all track-specific data.

### Low
25) Tauri HTTP bridge has no timeouts for local API requests.
- `KmiDi_PROJECT/source/frontend/src-tauri/src/bridge/musicbrain.rs:7-75` uses `reqwest::Client::new()` and `.send().await?` without a timeout.
- Impact: UI commands can hang indefinitely if the local service is down or unresponsive.

26) Dataset downloader uses network requests without timeouts.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/datasets/audio_downloader.py:120-233` calls `requests.get(...)` without a timeout.
- Impact: dataset downloads can hang indefinitely on network stalls.

27) Registry manifest validation is silently skipped when `jsonschema` is missing.
- `KmiDi_TRAINING/training/ML Kelly Training/backup/python/penta_core/ml/model_registry.py:19-27` sets `jsonschema = None` on import failure and validation is skipped without warning.
- Impact: invalid registry manifests can be accepted without any signal.

28) Review artifacts claim critical issues are fixed, but current backup scripts still include hardcoded paths and dummy data.
- `KmiDi_TRAINING/outputs/output/review/COMPREHENSIVE_PROJECT_REVIEW.md:1-90` and `KmiDi_TRAINING/outputs/output/review/FINAL_STATUS_REMAINING_ISSUES.md:1-90` state hardcoded paths are removed and only stylistic issues remain.
- Current backup scripts still use `/Volumes/Extreme SSD/kelly-audio-data` and dummy datasets.
- Impact: review reports are stale and can mislead validation/QA.

29) Frontend hook bypasses Tauri invoke and hardcodes local API URLs without timeouts.
- `KmiDi_PROJECT/source/cpp/src/hooks/useMusicBrain.ts:94-170` uses `fetch('http://127.0.0.1:8000/...')` directly for config and render calls.
- Impact: frontend calls fail when the local API is not running and can hang without a timeout; also ignores any Tauri-side base URL configuration.

30) Platform/tooling roadmaps reference a local workstation path.
- `KmiDi_PROJECT/source/frontend/iOS/ROADMAP.md:2`, `KmiDi_PROJECT/source/frontend/iOS/TODO.md:2`, `KmiDi_PROJECT/source/frontend/mobile/ROADMAP.md:2`, `KmiDi_PROJECT/source/frontend/mobile/TODO.md:2`, `KmiDi_PROJECT/tools/ROADMAP.md:2`, and `KmiDi_PROJECT/tools/TODO.md:2` reference `/Users/seanburdges/Desktop/final kel`.
- Impact: documentation is not portable and can mislead contributors who do not have that path.

76) Project JSON writing does not escape string values.
- `KmiDi_PROJECT/source/cpp/src/project/ProjectFile.cpp:52-120` writes metadata and track names directly into JSON strings.
- Names or authors containing quotes, backslashes, or newlines will produce invalid JSON.
- Impact: project files can become unreadable with common user input.

77) Spectral analyzer ignores requested frame size and can write past the magnitude buffer.
- `KmiDi_PROJECT/source/cpp/src/audio/SpectralAnalyzer.cpp:64-105` builds a `magnitude` vector sized `frameSize / 2 + 1`.
- `KmiDi_PROJECT/source/cpp/src/audio/SpectralAnalyzer.cpp:149-194` `computeFFT()` uses `fftSize_` for bin count, writing `fftSize_ / 2 + 1` elements even when `frameSize != fftSize_`.
- Impact: STFT calls can corrupt memory or produce garbage spectra unless `frameSize == fftSize_`.

78) FFT size is not validated as a power of two.
- `KmiDi_PROJECT/source/cpp/src/audio/SpectralAnalyzer.cpp:9-17` constructs `juce::dsp::FFT` with `std::log2(fftSize)` but never checks that `fftSize` is a power of two.
- `juce::dsp::FFT` expects an integer order; non-power-of-two sizes round down and mismatch buffer sizes.
- Impact: non-power-of-two inputs can yield incorrect transforms or undefined behavior.

79) OSC JSON request strings are built without escaping user text.
- `KmiDi_PROJECT/source/cpp/src/bridge/OSCBridge.cpp:71-103` formats JSON using `R"({"text":"%s", ...})"` with raw `text`.
- If `text` contains quotes, backslashes, or newlines, the JSON becomes invalid or truncated.
- Impact: OSC requests fail for common user inputs and can break downstream parsing.

80) MIDI stem rendering hardcodes tempo to 120 BPM.
- `KmiDi_PROJECT/source/cpp/src/export/StemExporter.cpp:132-195` uses `bpm = 120.0` for tick-to-seconds conversion and duration estimation.
- The project tempo is never consulted when rendering MIDI.
- Impact: exported stems are off-tempo relative to the project, even when MIDI is otherwise correct.

81) OSCBridge request tracking is not thread-safe.
- `KmiDi_PROJECT/source/cpp/src/bridge/OSCBridge.cpp:40-188` mutates `pendingRequests_` and `nextMessageId_` from request methods.
- `KmiDi_PROJECT/source/cpp/src/bridge/OSCBridge.cpp:219-340` reads and erases `pendingRequests_` from the OSC receiver callback thread.
- Impact: concurrent access can race and corrupt the map, leading to crashes or dropped callbacks.

82) CacheManager setters update shared state without locking.
- `KmiDi_PROJECT/source/cpp/src/bridge/CacheManager.h:69-79` `setTTL` and `setMaxSize` mutate `ttlMs_`/`maxSize_` without acquiring `mutex_`.
- Other methods assume those fields are stable under the lock.
- Impact: concurrent calls can cause data races and undefined behavior.

83) MIDI file builder drops rhythm and drum groove tracks.
- `KmiDi_PROJECT/source/cpp/src/midi/MidiBuilder.cpp:20-110` only writes chords, melody, bass, counter-melody, pad, strings, and fills.
- `GeneratedMidi` also includes `rhythm` and `drumGroove`, but they are never serialized in `buildMidiFile` or `buildMidiBuffer`.
- Impact: exported MIDI from this builder silently omits percussion/rhythm layers.

84) MIDI file builder divides by zero when tempo is unset.
- `KmiDi_PROJECT/source/cpp/src/midi/MidiBuilder.cpp:33-41` computes `microsecondsPerBeat` from `midi.bpm` without guarding for `0.0f`.
- Impact: if `GeneratedMidi.bpm` is zero (or uninitialized), tempo meta events and timing become invalid.

85) MIDI quantization can divide by zero.
- `KmiDi_PROJECT/source/cpp/include/daiw/midi/MidiSequence.h:104-118` computes `((timestamp + gridSize / 2) / gridSize)` without checking `gridSize`.
- Impact: calling `quantize(0)` or passing an invalid grid size crashes or yields undefined timing.

86) MIDI note pairing drops overlapping notes on the same channel/pitch.
- `KmiDi_PROJECT/source/cpp/src/midi/MidiSequence.cpp:15-60` stores active notes in a map keyed by `channel*128 + note`.
- A second note-on for the same key overwrites the first, so stacked notes lose their original start and duration.
- Impact: sequences with repeated/overlapping notes (common in MIDI) collapse into incorrect note events.

87) Humanizer can generate negative timestamps.
- `KmiDi_PROJECT/source/cpp/src/midi/humanizer.cpp:52-75` applies random timing offsets without clamping `startTick` to `>= 0`.
- Impact: downbeats near tick 0 can move into negative time and produce invalid MIDI timing.

88) MIDI buffer builder can divide by zero when BPM is invalid.
- `KmiDi_PROJECT/source/cpp/src/midi/MidiBuilder.cpp:85-90` computes `samplesPerBeat = (sampleRate * 60.0) / bpm` without guarding against `bpm <= 0`.
- Impact: passing a zero/negative BPM will crash or generate invalid timing.

89) F0 extractor interpolation can divide by zero.
- `KmiDi_PROJECT/source/cpp/src/audio/F0Extractor.cpp:198-214` computes `offset = (y2 - y0) / (2 * (2*y1 - y0 - y2))` without checking for a zero denominator.
- Impact: flat CMNDF regions can yield NaN/inf offsets, destabilizing pitch detection.

90) Voice synthesizer does not validate BPM before time conversion.
- `KmiDi_PROJECT/source/cpp/src/voice/VoiceSynthesizer.cpp:477-481` converts beats to samples using `60.0 / bpm_` without guarding for `bpm_ <= 0`.
- `KmiDi_PROJECT/source/cpp/src/voice/VoiceSynthesizer.cpp:640-643` allows any BPM value to be set.
- Impact: invalid BPM values produce divide-by-zero or invalid timing during synthesis.

91) Drum humanization returns unsorted events after timing jitter.
- `KmiDi_PROJECT/source/python/music_brain/groove/groove_engine.py:224-340` applies per-event timing jitter but never sorts the result by `start_tick`.
- `KmiDi_PROJECT/source/python/music_brain/tier1/midi_pipeline_wrapper.py:238-285` assumes events are in chronological order and converts to MIDI delta times.
- Impact: later events can appear before earlier ones, causing collapsed timing when delta times are clamped to zero.

92) MCP TODO storage uses `fcntl`, which is unavailable on Windows.
- `KmiDi_PROJECT/source/python/mcp_todo/storage.py:9-52` imports and relies on `fcntl` for file locking.
- Impact: the MCP TODO server crashes at import time on Windows, preventing cross-platform use.

93) Undo/redo actions are no-ops in the GUI history system.
- `KmiDi_PROJECT/source/python/kmidi_gui/core/history.py:57-170` action `apply()`/`undo()` methods only log and never mutate state.
- `KmiDi_PROJECT/source/python/kmidi_gui/core/history.py:223-287` `undo()`/`redo()` report success even though nothing changes.
- Impact: users see “undo/redo” succeed without any changes being applied.

94) GUI MIDI export can silently write empty files.
- `KmiDi_PROJECT/source/python/kmidi_gui/core/export.py:50-77` falls back to creating an empty file when `midi_data` lacks a source path.
- Impact: exports appear to succeed but produce unusable MIDI files.

95) Penta-core integration is a non-functional stub.
- `KmiDi_PROJECT/source/python/music_brain/integrations/penta_core.py:60-220` marks all core methods as placeholders and never connects or sends data.
- `connect()` always leaves `_connected = False`, and every send method raises `ConnectionError`.
- Impact: any penta-core integration path is dead code; calls will fail even with a valid endpoint.

96) AI specialization summary is a stub and does not reflect actual capabilities.
- `KmiDi_PROJECT/source/python/music_brain/ai_specializations.py:1-24` defines a minimal enum and prints hardcoded task names.
- Impact: CLI status output can mislead users into thinking specialized tooling exists when it does not.

97) Audio texture generator writes empty files.
- `KmiDi_PROJECT/source/python/music_brain/generators/audio_texture.py:23-40` writes a zero-byte WAV as a placeholder.
- Impact: generated textures are unusable while the pipeline reports success.

98) Voice rendering pipeline emits empty audio files.
- `KmiDi_PROJECT/source/python/music_brain/tier1/voice_pipeline.py:43-116` writes an empty output file and returns metadata only.
- Impact: vocal generation appears to succeed but produces silent files.

99) Guide voice synthesis divides by zero when tempo is invalid.
- `KmiDi_PROJECT/source/python/music_brain/voice/synthesizer.py:126-132` computes `beat_duration = 60.0 / tempo_bpm` without guarding `tempo_bpm <= 0`.
- Impact: invalid tempo values crash synthesis or produce invalid timing.

100) Guide voice synthesis crashes when melody is empty.
- `KmiDi_PROJECT/source/python/music_brain/voice/synthesizer.py:146-171` assumes `melody_midi` has at least one note and indexes `melody_midi[-1]`.
- Impact: empty melody inputs raise `IndexError` rather than returning a safe fallback.

101) Neural backend never uses DiffSinger even when available.
- `KmiDi_PROJECT/source/python/music_brain/voice/neural_backend.py:94-129` checks for DiffSinger but `_try_diffsinger()` always returns `False`.
- Impact: high-quality DiffSinger path is effectively disabled; the backend falls back to ONNX or no synthesis.

102) `music_brain.metrics` requires `lldb` and appears unrelated to runtime metrics.
- `KmiDi_PROJECT/source/python/music_brain/metrics.py:1-122` imports `lldb` and defines LLDB formatter helpers.
- The module is not referenced elsewhere, but importing it in non-LLDB environments raises `ImportError`.
- Impact: accidental imports crash in typical runtime environments; file seems out of place for production use.

103) Orchestrator context class is missing `@dataclass`, so construction fails.
- `KmiDi_PROJECT/source/python/music_brain/orchestrator/interfaces.py:246-318` defines `ExecutionContext` with `field(...)` but does not decorate it as a dataclass.
- `KmiDi_PROJECT/source/python/music_brain/orchestrator/orchestrator.py:118-137` instantiates `ExecutionContext(...)` with constructor args.
- Impact: orchestrator execution raises `TypeError` because `ExecutionContext` has no generated `__init__`.

104) Orchestrator cancellation and shutdown are no-ops.
- `KmiDi_PROJECT/source/python/music_brain/orchestrator/orchestrator.py:540-571` `cancel_execution()` only logs and returns True without cancelling the running task.
- `KmiDi_PROJECT/source/python/music_brain/orchestrator/orchestrator.py:579-583` `__aexit__` is `pass`, leaving running executions intact.
- Impact: calling cancel/cleanup does not stop work, leading to leaked tasks and misleading status.

105) WebSocket server cannot be stopped cleanly.
- `KmiDi_PROJECT/source/python/music_brain/agents/websocket_api.py:214-226` `start()` blocks forever on `await asyncio.Future()` with no cancellation hook.
- `KmiDi_PROJECT/source/python/music_brain/agents/websocket_api.py:242-268` `stop()` closes sockets but does not unblock `start()`, so background threads remain alive.
- Impact: stop/shutdown leaves the server coroutine hanging and leaks the background thread.

106) Audio diffusion reports success but returns silence when models are unavailable.
- `KmiDi_PROJECT/source/python/music_brain/generative/audio_diffusion.py:108-164` sets `_is_loaded = True` even if model loading failed.
- `KmiDi_PROJECT/source/python/music_brain/generative/audio_diffusion.py:242-268` returns zeroed audio when `_pipeline`/`_model` is missing.
- Impact: generation appears successful while producing silent audio with no error signal.

107) Callback removal can raise exceptions when the callback is missing.
- `KmiDi_PROJECT/source/python/music_brain/agents/unified_hub.py:1094-1098` calls `list.remove` without guarding for missing callbacks.
- `KmiDi_PROJECT/source/python/music_brain/agents/ableton_bridge.py:177-181` does the same in `AbletonOSCBridge.off`.
- Impact: attempting to remove a callback twice (or after errors) raises `ValueError` and can break cleanup flows.

108) Intent bridge imports a missing rule-break enum.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_bridge.py:14-21` imports `MelodyRuleBreak`.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_schema.py` does not define `MelodyRuleBreak`.
- Impact: importing the intent bridge raises `ImportError`, breaking C++ ↔ Python intent processing.

109) Intent bridge converts the wrong result schema and returns defaults.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_bridge.py:47-78` calls `IntentProcessor.process_intent()` and passes its output to `_convert_to_cpp_format`.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_processor.py:708-724` returns a dict with `harmony/groove/arrangement/production/intent_summary`, not top-level `key`, `mode`, `tempo`, or `chords`.
- `_convert_to_cpp_format` therefore falls back to defaults (`C`, `major`, `120`, empty chords).
- Impact: C++ side receives generic defaults regardless of intent content, so generation is disconnected from input.

110) Rule-break suggestions return empty justifications.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_bridge.py:119-137` looks up `RULE_BREAKING_EFFECTS[rule_break]["justification"]`.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_schema.py` defines `description/effect/use_when/example_emotions` but no `justification` key.
- Impact: suggested rule-breaks include empty justification text even when data exists.

111) Emotion-conditioned generator never loads the emotion encoder.
- `KmiDi_PROJECT/source/python/music_brain/generative/emotion_conditioned.py:140-162` imports `load_model` from `penta_core.ml.inference`, but that function does not exist.
- The import fails and is swallowed, leaving `_emotion_encoder` unset.
- Impact: emotion embeddings are never computed, so generation ignores any learned emotion encoder.

36) Adaptive batch sizing is computed but never applied.
- `python/penta_core/ml/inference_batching.py:311-329` adjusts `_current_batch_size`, but `process_batch` only uses `config.max_batch_size` and never references `_current_batch_size`.
- Impact: the adaptive batch size logic has no effect on throughput/latency tradeoffs.

37) Mixer MIDI routing never stores per-channel MIDI buffers.
- `KmiDi_PROJECT/source/cpp/src/ui/MixerConsolePanel.cpp:556-567` only routes MIDI if `channelIndex < channelMidi_.size()`.
- `KmiDi_PROJECT/source/cpp/src/ui/MixerConsolePanel.cpp:309-329` adds channels but never inserts entries into `channelMidi_`.
- Impact: `routeMIDIToChannel` drops MIDI for normal channel indices, so `getMixedOutput()` remains empty.

38) Removing channels leaves stale MIDI/automation data and index mismatches.
- `KmiDi_PROJECT/source/cpp/src/ui/MixerConsolePanel.cpp:321-329` erases channel UI and instruments only.
- `KmiDi_PROJECT/source/cpp/src/ui/MixerConsolePanel.h:267-270` keeps `channelMidi_` and `automation_` keyed by channel index.
- Impact: deleting a channel leaves old MIDI/automation entries and shifts indices, so future routing/snapshots can reference the wrong data.

39) Groove benchmark asserts on onset output but detector is a stub.
- `KmiDi_PROJECT/benchmarks/groove_latency.cpp:26-44` requires `analysis.currentTempo > 0` and `analysis.onsetPositions.size() > 0`.
- `KmiDi_PROJECT/source/cpp/src/groove/OnsetDetector.cpp:18-57` is a no-op implementation.
- Impact: `groove_latency` benchmark will fail consistently, blocking benchmark runs/CI.

40) Master EQ UI draws a placeholder response curve rather than the real filter response.
- `KmiDi_PROJECT/source/cpp/src/ui/MasterEQComponent.cpp:271-320` uses a simplified approximation and TODOs instead of the actual biquad response.
- Impact: the EQ visualization does not match audio processing, so users see misleading curves.

41) Applying AI EQ suggestions is a no-op.
- `KmiDi_PROJECT/source/cpp/src/ui/MasterEQComponent.cpp:223-241` shows an alert instead of mapping the suggested curve to parameters.
- Impact: “Apply Suggested Curve” appears to work but does not change EQ settings.

42) MIDI editor humanize action is unimplemented.
- `KmiDi_PROJECT/source/cpp/src/ui/MidiEditor.cpp:360-373` leaves `humanizeSelected` as a placeholder with no edits applied.
- Impact: humanize controls do nothing, so users cannot add timing/velocity variation.

43) Suggestion overlay is rendered with placeholders rather than real suggestion data.
- `KmiDi_PROJECT/source/cpp/src/ui/InteractiveCustomizationPanel.cpp:96-146` uses hardcoded drawing paths and a `if (false)` placeholder instead of suggestion data.
- Impact: enabling suggestions shows dummy markers rather than actual recommendation output.

44) Health status report runs checks twice per call.
- `python/penta_core/ml/health.py:357-394` calls `run_all_checks()` and then `get_overall_status()`, which runs all checks again.
- Impact: health checks execute twice per request, doubling overhead and side effects.

45) Async inference queue full handling never triggers.
- `python/penta_core/ml/async_inference.py:132-170` awaits `self._queue.put(...)` and catches `asyncio.QueueFull`, but `Queue.put` blocks instead of raising.
- Impact: callers can hang indefinitely when the queue is full; the intended backpressure error path is dead code.

46) TimelinePanel is declared but has no implementation.
- `KmiDi_PROJECT/source/cpp/src/ui/MixerConsolePanel.h:477-520` declares `TimelinePanel` methods, but there is no corresponding `TimelinePanel` implementation in the codebase.
- Impact: any translation unit that instantiates or links `TimelinePanel` will fail at link time.

47) Monitoring alert logic never detects unhealthy checks.
- `python/penta_core/ml/monitoring.py:234-252` looks for `check_data.get("current_status")`.
- `python/penta_core/ml/health.py:379-394` populates `checks` with `HealthCheckResult.to_dict()` entries that use the key `status`, not `current_status`.
- Impact: per-check unhealthy alerts never trigger even when checks fail.

48) Suggestion overlay button callbacks can target the wrong card after a dismissal.
- `KmiDi_PROJECT/source/cpp/src/ui/SuggestionOverlay.cpp:55-123` captures `cardIndex` at creation time.
- `KmiDi_PROJECT/source/cpp/src/ui/SuggestionOverlay.cpp:170-190` erases `cards_` entries, shifting indices without updating callbacks.
- Impact: after any card is dismissed, Apply/Dismiss/Expand actions can act on the wrong suggestion or do nothing.

49) Suggestion overlay confidence bars never render any fill.
- `KmiDi_PROJECT/source/cpp/src/ui/SuggestionOverlay.cpp:80-120` creates `confidenceBar` components but never sets a colour or paint routine.
- Impact: confidence bars are invisible, so confidence UI is missing even though labels render.

50) AI generation dialog returns immediately without waiting for user input.
- `KmiDi_PROJECT/source/cpp/src/ui/AIGenerationDialog.cpp:149-177` uses `launchAsync()` plus `enterModalState()` and then returns the cached request immediately.
- `KmiDi_PROJECT/source/cpp/src/ui/AIGenerationDialog.cpp:153-176` reads `dialogPtr` after ownership is transferred; the dialog content can be destroyed when the window closes, so the pointer can be stale.
- Impact: callers receive default requests before the user interacts, or risk a use-after-free if the dialog is destroyed.

51) Workstation track visualization code is never invoked.
- `KmiDi_PROJECT/source/cpp/src/ui/WorkstationPanel.cpp:161-186` only paints the header and track count.
- `KmiDi_PROJECT/source/cpp/src/ui/WorkstationPanel.cpp:233-320` defines `paintTrack`/`paintTrackContent`, but they are never called and `trackList_` is a plain `juce::Component` with no custom paint.
- Impact: track note visuals never render; only the control widgets appear.

52) Side panel input controls are never initialized or added to the UI.
- `KmiDi_PROJECT/source/cpp/src/ui/SidePanel.cpp:8-24` only sets up the label; `input_` and `intensity_` are never configured or added via `addAndMakeVisible`.
- `KmiDi_PROJECT/source/cpp/src/ui/SidePanel.h:34-47` exposes `getInputEditor()`/`getIntensitySlider()` and returns state from these controls.
- Impact: the side panel shows only a label and returns default/empty values, so Side A/B inputs never reach callers.

53) AI service initialization short-circuits and never initializes components.
- `python/penta_core/ml/ai_service.py:328-356` sets `_initialized = True` during construction.
- `python/penta_core/ml/ai_service.py:358-360` exits early in `initialize()` when `_initialized` is already true.
- Impact: `AIService.initialize()` returns success without initializing the model/inference/training services.

54) Score entry playback never produces MIDI data.
- `KmiDi_PROJECT/source/cpp/src/ui/ScoreEntryPanel.cpp:392-405` returns an empty `juce::MidiBuffer` and leaves conversion as a comment.
- `KmiDi_PROJECT/source/cpp/src/ui/ScoreEntryPanel.cpp:407-438` `playFromStart()` calls `toMidiBuffer()` and `playFromCursor()` is empty.
- Impact: playback/export from the score entry panel does nothing even when notes exist.

55) Score entry view uses HTML entity strings for clef glyphs.
- `KmiDi_PROJECT/source/cpp/src/ui/ScoreEntryPanel.cpp:455-468` sets `clefSymbol` to strings like `"& #x1d11e;"` (with a space).
- Impact: JUCE draws the literal text instead of the intended clef glyph, so clefs render incorrectly.

56) Lyric line highlighting can mark the wrong section.
- `KmiDi_PROJECT/source/cpp/src/ui/LyricDisplay.cpp:36-86` resets `lineIndex` to 0 for every section but compares it to a single `currentLineIndex`.
- `KmiDi_PROJECT/source/cpp/src/ui/LyricDisplay.cpp:122-157` `getCurrentLineIndex()` returns an index local to one section.
- Impact: when `currentLineIndex` is 0 (or any small value), the same line number in every section can be highlighted simultaneously.

57) Syllable highlighting is never computed.
- `KmiDi_PROJECT/source/cpp/src/ui/LyricDisplay.cpp:160-177` `getCurrentSyllableIndex()` always returns -1.
- Impact: syllable highlighting never activates even when `highlightSyllables_` is true.

58) Piano roll preview ignores most tracks.
- `KmiDi_PROJECT/source/cpp/src/ui/PianoRollPreview.cpp:19-60` derives time/pitch ranges only from `midi.melody` and `midi.bass`.
- `KmiDi_PROJECT/source/cpp/src/ui/PianoRollPreview.cpp:98-141` only draws melody and bass notes.
- Impact: counter-melody, pad, strings, fills, and chord tracks never appear in the preview.

59) IntegrationManager reconnection settings are unused.
- `python/penta_core/ml/integration_manager.py:62-76` defines `_reconnect_enabled`, `_reconnect_interval`, and `_max_reconnect_attempts`.
- No method uses these settings or starts a reconnection loop.
- Impact: integrations never auto-reconnect despite the configuration flags.

60) TooltipComponent never sets its display text.
- `KmiDi_PROJECT/source/cpp/src/ui/TooltipComponent.cpp:8-20` `showTooltip()` only calls `target->setHelpText` and does not update `tooltipText_`.
- `KmiDi_PROJECT/source/cpp/src/ui/TooltipComponent.cpp:30-43` paints `tooltipText_`, which remains empty.
- Impact: the custom tooltip component renders blank even when invoked.

61) TrainingInferenceBridge cannot deploy older versions because the registry overwrites by name.
- `python/penta_core/ml/model_registry.py:96-140` stores models in `_models` by name, overwriting prior versions.
- `python/penta_core/ml/training_inference_bridge.py:126-188` tracks multiple versions and searches the registry list for a matching `model_name` and `version`.
- Impact: once a new version is registered, older versions are no longer discoverable, so rollback or explicit version deployment fails.

62) Cassette view ignores tape position state.
- `KmiDi_PROJECT/source/cpp/src/ui/CassetteView.cpp:104-132` draws tape reels with only `animationPhase_`.
- `KmiDi_PROJECT/source/cpp/src/ui/CassetteView.cpp:160-169` stores `tapePosition_` but never uses it in rendering.
- Impact: UI cannot reflect tape progress/position, even when state is updated.

63) Vocal control panel never labels the voice type selector.
- `KmiDi_PROJECT/source/cpp/src/ui/VocalControlPanel.cpp:41-77` lays out `voiceTypeLabel_` but never sets its text.
- Impact: the voice type selector appears without a label, reducing UI clarity and accessibility.

64) IntegrationManager health check can crash when health support is unavailable.
- `python/penta_core/ml/integration_manager.py:186-215` uses `HealthStatus` unconditionally in `check_health()`.
- When `HAS_HEALTH` is false, `HealthStatus` is not imported and this method raises `NameError`.
- Impact: calling `check_health()` without the health module installed crashes instead of returning a safe status.

65) TrainingInferenceBridge subscriptions are ineffective because no training events are published.
- `python/penta_core/ml/training_inference_bridge.py:55-78` subscribes to `training.completed` and `training.failed`.
- `python/penta_core/ml/training_orchestrator.py` never publishes these events.
- Impact: auto-registration and auto-deploy flows never trigger after training.

66) ScoreEntryPanel pitch mapping is hardcoded.
- `KmiDi_PROJECT/source/cpp/src/ui/ScoreEntryPanel.cpp:548-560` returns constant values from `staffPositionToPitch` and `pitchToStaffPosition`.
- Impact: note placement and hit testing are incorrect; score input cannot map screen positions to pitches.

67) ScoreEntryPanel quick entry parsing is unimplemented.
- `KmiDi_PROJECT/source/cpp/src/ui/ScoreEntryPanel.cpp:566-571` leaves `parseQuickEntry` empty.
- Impact: the quick entry input does nothing despite the UI exposure.

68) Async inference shutdown can hang with queued requests.
- `python/penta_core/ml/async_inference.py:131-140` sets `_running = False` and then awaits `queue.join()`.
- Worker loops exit when `_running` is false and will not drain remaining queue items, so `join()` can block forever.
- Impact: stopping the async inference engine can deadlock when the queue is non-empty.

69) Async inference latency stats are never updated.
- `python/penta_core/ml/async_inference.py:110-112` initializes `average_latency_ms`.
- No code updates this value after processing requests.
- Impact: monitoring metrics report 0.0ms average latency even under load.

70) Model pool can exceed memory limits when a single model is too large.
- `python/penta_core/ml/model_pool.py:172-179` evicts models if `current_memory + memory_estimate` exceeds the cap but does not re-check or abort.
- When `memory_estimate` exceeds `max_memory_mb`, eviction cannot reduce below the limit and the model is still loaded.
- Impact: memory cap is not enforced for large models, risking OOM.

71) Phase status references MCP TODO server; implementation exists.
- `KmiDi_PROJECT/source/python/mcp_workstation/phases.py:27-79` lists “MCP TODO server” as a Phase 1 milestone and marks `p1_mcp` as `COMPLETED`.
- The MCP TODO server is implemented under `KmiDi_PROJECT/source/python/mcp_todo` (CLI, MCP server, HTTP server).
- Impact: the earlier “missing implementation” note was incorrect; keep the status if completion is based on code availability.

72) LLM intent parsing crashes on unexpected fields.
- `KmiDi_PROJECT/source/python/mcp_workstation/llm_reasoning_engine.py:101-103` calls `StructuredIntent(**intent_dict)` without filtering keys.
- The exception handler does not catch `TypeError`, so extra keys from the LLM response raise and abort parsing.
- Impact: minor prompt drift or model updates can break intent parsing entirely.

73) Orchestrator saves final intent without MIDI/image/audio results.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py:153-276` attaches `midi_plan`, `generated_image_data`, and `generated_audio_data` to `CompleteSongIntent`.
- `music_brain/session/intent_schema.py:454-538` `to_dict()` ignores these fields, so `save()` drops them.
- Impact: `final_intent.json` omits the generated outputs, making post-run inspection misleading.

74) Batched inference can produce misaligned inputs when requests have missing keys.
- `python/penta_core/ml/inference_batching.py:232-251` builds each batched input by stacking only non-None arrays, dropping requests that omit a key.
- This yields per-key batch sizes that no longer match the original request count and can misalign inputs across keys or trigger shape errors in the backend.
- Impact: batched inference can mix inputs from different requests or fail unpredictably when optional inputs are omitted.

75) AI service never initializes its components due to premature `_initialized` flag.
- `python/penta_core/ml/ai_service.py:285-317` sets `_initialized = True` inside `__init__`, then `initialize()` exits early when `_initialized` is true.
- `get_ai_service()` relies on `initialize()` to wire model registry/health/resources, so none of the services are actually initialized.
- Impact: inference/training/health service dependencies silently remain uninitialized; status APIs report ready while internals are not set up.

76) Async inference singleton crashes on import due to missing threading import.
- `python/penta_core/ml/async_inference.py:489` declares `_async_engine_lock = threading.Lock()` but `threading` is never imported.
- Impact: importing the module raises `NameError` before any inference can be used.

77) Plugin discovery registers non-importable module paths.
- `python/penta_core/ml/plugin_system.py:121-158` discovers plugins from arbitrary files but stores `module_path` as the file stem.
- `_load_plugin_class` later calls `importlib.import_module(module_path)` which fails unless that module is on `sys.path`, so reloading/discovery results are not reusable.
- Impact: plugins discovered from directories cannot be loaded later, breaking plugin persistence.

78) Training orchestration fallback leaves a no-op TrainActor when Ray is unavailable.
- `KmiDi_TRAINING/training/training/train_integrated.py:808-815` defines an empty `TrainActor` when `ray` is missing.
- Any code that expects `TrainActor.train()` will crash at runtime with `AttributeError`.
- Impact: distributed training paths fail silently in environments without Ray.

79) Training subprocess uses hard-coded /workspaces paths.
- `KmiDi_TRAINING/training/training/train_integrated.py:832-840` calls a fixed venv/python path and script location under `/workspaces/KmiDi/...`.
- Impact: training fails outside that specific environment layout (local machines, CI, or different repo roots).

80) Monitoring summary ignores requested time range.
- `python/penta_core/ml/monitoring.py:158-188` accepts `time_range_minutes` but always returns full-history `metric.get_stats()` without filtering by timestamp.
- Impact: monitoring dashboards cannot request time-windowed stats; results are misleading for “last N minutes” views.

81) Spectocloud training never supervises the model’s `properties` output.
- `KmiDi_TRAINING/training/training/cuda_session/train_spectocloud.py:176-212` predicts `positions`, `colors`, and `properties`, but the dataset only returns positions/colors.
- Training loss in `train_epoch`/`evaluate` (`KmiDi_TRAINING/training/training/cuda_session/train_spectocloud.py:520-666`) ignores `properties` entirely.
- Impact: `properties` learns arbitrary values, making downstream visuals (size/glow/depth) unreliable.

82) Model registry overwrites previous versions, breaking rollback and multi-version deploys.
- `python/penta_core/ml/training_inference_bridge.py:139-186` registers trained models with `name=model_name`.
- `python/penta_core/ml/model_registry.py:108-142` uses `name` as the dict key, so each new version replaces the prior entry.
- Impact: previous versions are lost in the registry; deploy/rollback cannot locate older versions even if `version` fields exist.

83) GPU resource quotas only track the first detected GPU and ignore per-device limits.
- `python/penta_core/ml/resource_manager.py:92-128` sets a single `ResourceType.GPU_MEMORY` quota when the first GPU is seen and skips subsequent devices.
- Allocations are not tied to a device, so multi-GPU systems can over-allocate or misreport usage.
- Impact: resource enforcement is inaccurate on multi-GPU hosts, leading to OOMs or misleading monitoring.

84) Melody training crashes due to target shape mismatch in dummy data path.
- `python/penta_core/ml/training_orchestrator.py:568-586` creates `ModelTask.MELODY_GENERATION` dummy targets as shape `(num_samples,)`.
- `python/penta_core/ml/training_orchestrator.py:488-518` melody model outputs `(batch, seq, vocab)`, so `CrossEntropyLoss` in `train()` (`python/penta_core/ml/training_orchestrator.py:632-676`) receives incompatible shapes.
- Impact: `queue_standard_models()` or any melody training run using the built-in trainer fails with a runtime shape error.

85) Voice synth helper hard-depends on LLDB and ships stub methods.
- `KmiDi_PROJECT/source/python/music_brain/voice/synth.py:1-45` imports `lldb` unconditionally and leaves `make_children()`/`update()` as `pass`.
- Importing `music_brain.voice.synth` fails on systems without LLDB (non-debug environments), and the class cannot actually build child values.
- Impact: voice synth tooling crashes at import time outside LLDB and the provider is non-functional.

86) Image generation engine never downloads real model weights.
- `KmiDi_PROJECT/source/python/mcp_workstation/image_generation_engine.py:23-71` “downloads” by creating a dummy marker file in `model_dir`.
- `_load_pipeline` then calls `StableDiffusionPipeline.from_pretrained(self.model_dir)` which expects a real model directory, so it will fail or load invalid data.
- Impact: image generation always falls back to stubbed output even when diffusers is installed.

87) Audio generation engine never uses MusicGen output.
- `KmiDi_PROJECT/source/python/mcp_workstation/audio_generation_engine.py:94-120` returns simulated base64 strings after `time.sleep()` instead of calling the loaded `MusicGen` model.
- Even with audiocraft installed, `_generate()` never invokes `self.model` or `audio_write`.
- Impact: audio generation reports “completed” but produces no real audio output.

88) Image pipeline forces fp16 even on CPU fallback.
- `KmiDi_PROJECT/source/python/mcp_workstation/image_generation_engine.py:55-74` always passes `torch_dtype=torch.float16` to `StableDiffusionPipeline.from_pretrained`.
- When MPS is unavailable and the pipeline runs on CPU, fp16 weights are typically unsupported and can raise runtime errors.
- Impact: image generation can fail on CPU-only systems even if diffusers is installed.

89) Async hub never emits `hub.stopped` because the event bus is shut down first.
- `KmiDi_PROJECT/source/python/music_brain/agents/async_hub.py:334-339` calls `self.events.shutdown()` and then emits `"hub.stopped"`.
- `KmiDi_PROJECT/source/python/music_brain/agents/events.py:293-308` drops events when `_running` is false, so the final event is ignored.
- Impact: listeners never receive the shutdown notification and may leak resources or hang waiting.

90) EventBus `wait` parameter is ignored, making emit_sync blocking.
- `KmiDi_PROJECT/source/python/music_brain/agents/events.py:271-308` accepts `wait` but `emit_event()` ignores it and always awaits handlers.
- `emit_sync()` claims to be non-blocking but ends up waiting for all handlers when it calls `emit(..., wait=False)`.
- Impact: synchronous callers can block unexpectedly on slow handlers, defeating the non-blocking API contract.

91) WebSocket auth token is accepted but never enforced.
- `KmiDi_PROJECT/source/python/music_brain/agents/websocket_api.py:132-152` stores `auth_token`, but no request path checks or header validation use it.
- Impact: all WebSocket commands are unauthenticated even when a token is configured.

92) WebSocket server thread never exits after stop().
- `KmiDi_PROJECT/source/python/music_brain/agents/websocket_api.py:208-218` runs `await asyncio.Future()` forever inside `start()`.
- `stop()` closes the server but never cancels the pending Future, so `start()` never returns and the background thread stays alive.
- Impact: repeated start/stop leaks threads and prevents clean shutdown.

93) Melody ML generator assumes registry models are loaded but never loads them.
- `KmiDi_PROJECT/source/python/music_brain/session/ml_melody_generator.py:193-216` calls `get_model("melodytransformer")` without first loading the registry manifest.
- `penta_core.ml` only exposes `load_registry_manifest` but `_load_ml_model` never calls it, so `get_model` returns None unless something else pre-populates the registry.
- Impact: ML melody generation always falls back to rule-based output in normal runs.

### Build Notes (Non-blocking)
- JUCE macOS 15 deprecation warnings during `KellyTests` build (CoreVideo/CoreText).
- Missing `WrapVulkanHeaders` and `pybind11` are reported by CMake; builds still succeed without them.
- `KellyPlugin_VST3` logs missing runtime data directories and falls back to embedded defaults.

94) CompleteSongIntent coerces `vulnerability_scale` to float, breaking enum validation.
- `music_brain/session/intent_schema.py:442-481` defines `vulnerability_scale` as a float and converts it with `float(...)` before assigning to `SongIntent.vulnerability_scale`.
- `SongIntent.vulnerability_scale` is intended to be an enum-like string ("Low"/"Medium"/"High"), and `validate_intent()` only validates when the field is a string.
- `CompleteSongIntent.from_dict()` passes strings like "Medium", but the `float()` conversion fails and defaults to 0.5, so the string value is lost.
- Impact: vulnerability scale is silently coerced to a numeric value, bypassing validation and breaking downstream enum-based logic.

95) IntentProcessor drops provided intent data when building CompleteSongIntent.
- `KmiDi_PROJECT/source/python/music_brain/orchestrator/processors/intent.py:287-318` calls `CompleteSongIntent(...)` with `song_root=SongRoot(...)`, `song_intent=SongIntent(...)`, and `technical_constraints=TechnicalConstraints(...)`.
- `music_brain/session/intent_schema.py:422-481` defines a custom `CompleteSongIntent.__init__` that does not accept those keyword arguments; it only accepts flattened fields and ignores `**kwargs`.
- As a result, the constructed `CompleteSongIntent` uses default empty values, so `validate_intent()` and downstream logic operate on blank intent data.
- Impact: validation results and affect mapping suggestions are computed against defaults instead of the user-provided intent.

96) Emotion API `create_intent()` builds CompleteSongIntent with ignored keyword args.
- `KmiDi_PROJECT/source/python/music_brain/emotion_api.py:559-591` passes `song_root=SongRoot(...)`, `song_intent=SongIntent(...)`, `technical_constraints=TechnicalConstraints(...)`, and `system_directive=SystemDirective(...)` into `CompleteSongIntent(...)`.
- `music_brain/session/intent_schema.py:422-481` defines a custom initializer that does not accept those parameters and ignores `**kwargs`, so the constructed intent falls back to empty defaults.
- Impact: `create_intent()` returns intents missing the caller-provided fields, leading to empty/invalid intent data in downstream generation.

97) CLI import references missing RuleBreakingCategory enum.
- `KmiDi_PROJECT/source/python/music_brain/cli.py:17` imports `RuleBreakingCategory` from `music_brain.session.intent_schema` with a comment “Assuming this exists”.
- The intent schema only defines `HarmonyRuleBreak`, `RhythmRuleBreak`, `ArrangementRuleBreak`, and `ProductionRuleBreak`, so the import raises `ImportError` on CLI startup.
- Impact: the CLI fails to run before any command handling, even for unrelated subcommands.

98) AdaptiveGenerator never applies preferences even when enabled.
- `KmiDi_PROJECT/source/cpp/src/engine/AdaptiveGenerator.cpp:34-55` calls `adaptIntent()` when adaptive mode and preference tracking are enabled, but `adaptIntent()` just returns the input unchanged.
- `getPreferredAdjustments()` always returns an empty map and there is no PreferenceTracker query.
- Impact: adaptive generation paths silently behave the same as non-adaptive paths, so learned preferences are never reflected.

99) MusicTheoryBridge lesson plan fallback is disabled, always returns error without Python.
- `KmiDi_PROJECT/source/cpp/src/bridge/MusicTheoryBridge.cpp:303-339` comments out the C++ fallback path with a `FIXME` and returns `{"error":"Lesson plan not available"}`.
- When the Python bridge is unavailable (`PYTHON_AVAILABLE` false or `createLessonPlanFunc_` null), there is no functional fallback.
- Impact: `createLessonPlan()` always fails on native-only builds, blocking lesson plan generation.

100) MIDI transformer defines rotary embeddings but never applies them.
- `KmiDi_TRAINING/training/training/cuda_session/train_midi_generator.py:170-233` instantiates `RotaryEmbedding`, but `EmotionMIDITransformer.forward()` never applies it to attention queries/keys.
- Impact: training runs without positional encoding despite intended rotary setup, degrading sequence modeling and making the rotary module unused.

101) Self-attention uses both `attn_mask` and `is_causal=True`, which can error on newer PyTorch.
- `KmiDi_TRAINING/training/training/cuda_session/train_midi_generator.py:227-235` builds a causal `attn_mask` and passes it to `nn.MultiheadAttention` while also setting `is_causal=True`.
- In recent PyTorch versions, `attn_mask` and `is_causal` are mutually exclusive and can raise a runtime error.
- Impact: training can crash at runtime depending on the PyTorch version.

102) Autoregressive generate() breaks for batch size > 1.
- `KmiDi_TRAINING/training/training/cuda_session/train_midi_generator.py:333-340` uses `next_token.item()` to check EOS, which only works for a single-element tensor.
- For batch generation, this raises `ValueError` or only checks the first sample, preventing multi-sample generation.
- Impact: batched inference fails or terminates incorrectly.

103) MIDI generator ONNX export uses hard-coded token range that can exceed vocab size.
- `KmiDi_TRAINING/training/training/cuda_session/export_models.py:86-92` builds `dummy_input_ids` with `torch.randint(0, 388, ...)` regardless of the model’s configured `vocab_size`.
- If the checkpoint was trained with a smaller vocab, the dummy IDs exceed embedding bounds and ONNX export fails.
- Impact: export crashes for non-default vocab sizes.

104) Spectocloud ONNX export ignores configured mel/bin dimensions.
- `KmiDi_TRAINING/training/training/cuda_session/export_models.py:50-58` creates `dummy_spectrogram` with fixed shape `(1, 128, 64)`.
- If the config uses a different `n_mels` or time-frame length, the exported graph input shape doesn’t match the trained model.
- Impact: ONNX export can fail or produce a model with incorrect input dimensions.

105) Ray Tune integration never reports metrics, so tuning results are invalid.
- `KmiDi_TRAINING/training/training/train_integrated.py:842-861` runs `tune.run(lambda config: train_subprocess(config), ...)`.
- `train_subprocess()` just launches a subprocess and never calls `tune.report`, so trials emit no metrics and Tune cannot select a best config.
- Impact: hyperparameter tuning either errors or yields meaningless results.

106) TrainableTuner.step() returns only `{"done": True}` without metrics.
- `KmiDi_TRAINING/training/training/train_integrated.py:760-772` defines a Ray Tune `Trainable` but `step()` returns no loss/accuracy metrics.
- Impact: Ray Tune has nothing to optimize; best-config selection is undefined.

107) Hyperparameter tuning subprocess uses hard-coded /workspaces paths.
- `KmiDi_TRAINING/training/training/train_integrated.py:813-824` calls `/workspaces/KmiDi/.venv/bin/python` and `/workspaces/KmiDi/training/train_integrated.py`.
- These paths are environment-specific and will fail outside that workspace layout.
- Impact: tuning fails on local machines or CI runners with different paths.

108) Audio cache script hardcodes default output to a specific external volume.
- `KmiDi_TRAINING/training/training/cache_audio_manifest.py:72-79` defaults `--out-dir` to `/Volumes/sbdrive/kmidi_audio_cache`.
- This path is machine-specific and fails on systems without that volume mounted.
- Impact: the script fails or writes to a non-existent path unless the user overrides the default.

109) Flat note names are mis-parsed in `_get_note_index`, causing wrong pitch classes.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_processor.py:56-64` normalizes notes by `note.replace('b', '#').upper()`, so `Bb` becomes `B#`.
- The flat-to-sharp fallback map expects `BB`, so flats like `Bb`, `Eb`, `Ab` never match and default to index 0.
- Impact: progressions generated with flat keys resolve to the wrong root notes.

110) Intent bridge instantiates IntentProcessor without required intent argument and calls missing method.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_bridge.py:20-49` initializes `IntentProcessor()` with no arguments, but `IntentProcessor.__init__` requires a `CompleteSongIntent`.
- `process_intent()` then calls `_intent_processor.process_intent(intent)`, yet `IntentProcessor` exposes `generate_all()` and no `process_intent` method.
- Impact: C++ bridge calls raise `TypeError`/`AttributeError`, so intent processing fails before returning any result.

111) Intent bridge imports non-existent MelodyRuleBreak enum.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_bridge.py:12-19` imports `MelodyRuleBreak`, which is not defined in `music_brain.session.intent_schema`.
- Impact: importing `intent_bridge` raises `ImportError`, preventing the Python bridge from loading.

112) MCP intent template builder drops placeholder fields due to CompleteSongIntent constructor mismatch.
- `KmiDi_PROJECT/source/python/music_brain/session/intent.py:18-55` builds a `CompleteSongIntent` using `song_root=SongRoot(...)`, `song_intent=SongIntent(...)`, and `technical_constraints=TechnicalConstraints(...)`.
- `music_brain/session/intent_schema.py:422-481` ignores those keyword arguments, so the returned template omits the placeholder values.
- Impact: `daiw.intent.create_template` yields an empty/default intent instead of the intended guidance fields.

113) OnsetDetector is a stub that never detects onsets.
- `KmiDi_PROJECT/source/cpp/src/groove/OnsetDetector.cpp:13-56` leaves `process()` empty and always sets `onsetDetected_ = false`.
- `computeSpectralFlux()` and `detectPeaks()` are unimplemented.
- Impact: any groove/tempo features relying on onsets will never trigger or update.

114) `suggest_progression()` fails for flat keys.
- `KmiDi_PROJECT/source/python/music_brain/session/generator.py:377-385` only resolves `key_num` if the key is in `NOTE_NAMES` (sharp-only list) and falls back to 0 otherwise.
- Flat keys like `Bb`, `Eb`, `Db` map to C, producing incorrect chord outputs.
- Impact: progression suggestions are wrong for any flat key inputs.

115) diagnose_progression() ignores flat key hints and defaults to C.
- `KmiDi_PROJECT/source/python/music_brain/structure/progression.py:279-289` parses `key_name` from the user hint and does `NOTE_NAMES.index(key_name)`.
- `NOTE_NAMES` contains sharps only, so hints like `Bb major` or `Eb minor` raise `ValueError` and fall back to C.
- Impact: analysis and suggestions are wrong for flat key inputs even when the user provides the correct key.

116) Intent bridge format conversion expects keys that `process_intent()` never returns.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_bridge.py:152-171` maps `result.get("key")`, `result.get("tempo")`, etc.
- `music_brain/session/intent_processor.process_intent()` returns a dict with `harmony`, `groove`, `arrangement`, `production`, and `intent_summary` keys, not flat `key/tempo` fields.
- Impact: even if the intent bridge calls the correct processor, it still emits default/fallback values instead of real musical parameters.

117) Suggested rule-break justifications are always empty.
- `KmiDi_PROJECT/source/python/music_brain/session/intent_bridge.py:128-142` reads `RULE_BREAKING_EFFECTS[rule_break].get("justification")`.
- The rule database only defines `description`, `effect`, and `use_when`, so `justification` is never present.
- Impact: `get_suggested_rule_breaks()` returns blank justifications for every suggestion.

118) Flat bass notes in slash chords are dropped during parsing.
- `KmiDi_PROJECT/source/python/music_brain/structure/progression.py:120-129` leaves `bass` unnormalized when parsing slash chords.
- `KmiDi_PROJECT/source/python/music_brain/structure/chord.py:214-223` then looks up `parsed.bass` in `NOTE_NAMES` (sharp-only list), so flats like `Bb` fail and bass is set to None.
- Impact: slash chords with flat bass (e.g., `F/Bb`) lose their bass note and render incorrectly.

119) Audio waveform analysis crashes on silent inputs when computing dynamic range.
- `KmiDi_PROJECT/source/python/music_brain/audio/analyzer.py:233-236` filters RMS dB values with `rms_db > -60` and calls `np.min(...)` without checking for an empty array.
- For silent or near-silent audio, the filtered array is empty and `np.min` raises `ValueError`.
- Impact: `analyze_waveform()` fails on quiet/silent clips instead of returning a valid analysis.

120) IntentProcessor keyword matching is case-sensitive.
- `KmiDi_PROJECT/source/cpp/src/core/intent_processor.cpp:7-19` searches for "loss", "grief", "anger", etc. without normalizing the input.
- Inputs like "Grief" or "Loss" in different casing won’t match and fall through to the default emotion.
- Impact: emotion classification is inconsistent and misses obvious keywords depending on user capitalization.

121) Chord key-mode estimation misclassifies major chords as minor.
- `KmiDi_PROJECT/source/python/music_brain/audio/chord_detection.py:329-333` counts minor chords using `if 'min' in c.quality or 'm' in c.chord_name`.
- Chord names like `Cmaj7` include "m", so they are counted as minor.
- Impact: key mode is biased toward minor and can be incorrect for major-heavy progressions.

122) Audio feel analysis crashes on silent inputs when computing dynamic range.
- `KmiDi_PROJECT/source/python/music_brain/audio/feel.py:96-99` calls `np.min(rms_db[rms_db > -60])` without guarding for an empty slice.
- For silent/near-silent audio, the filtered array is empty and raises `ValueError`.
- Impact: `analyze_feel()` fails on quiet audio files instead of returning a valid analysis.

123) Fitbit streaming can never start because initialization is a stub.
- `KmiDi_PROJECT/source/cpp/src/biometric/BiometricInput.cpp:290-320` sets `fitbitInitialized_ = false` and returns false in `initializeFitbit()`.
- `startStreaming()` only enables Fitbit when `fitbitInitialized_` is true, so the Fitbit path is permanently disabled.
- Impact: Fitbit integration is non-functional even with a valid access token.

124) Section detection ignores time signature denominator when computing bars.
- `KmiDi_PROJECT/source/python/music_brain/structure/sections.py:232-233` sets `ticks_per_bar = ppq * time_sig[0]`.
- For signatures like 6/8 or 3/8, the denominator changes beat length; bar length should scale by `4/denominator`.
- Impact: bar boundaries and section lengths are incorrect for non‑4/4 MIDI files.

125) Drum analyzer crashes on unknown drum pitches.
- `KmiDi_PROJECT/source/python/music_brain/groove/drum_analysis.py:112-115` builds `hihat_notes` with `get_drum_category(n.pitch).startswith('hihat')`.
- `get_drum_category()` can return None for unmapped pitches, so `.startswith()` raises `AttributeError`.
- Impact: analysis fails when MIDI uses drum notes outside the known map.

126) MIDI duration calculation ignores tempo changes.
- `KmiDi_PROJECT/source/python/music_brain/utils/midi_io.py:72-103` records only the last `set_tempo` value and computes `duration_seconds` using a single `seconds_per_tick`.
- For files with multiple tempo changes, the reported duration is incorrect.
- Impact: MIDI metadata (duration_seconds) is wrong for tempo‑mapped files.

127) Bridge API key detection fails on lowercase key names.
- `KmiDi_PROJECT/source/python/music_brain/orchestrator/bridge_api.py:547-555` searches for patterns like `" C major"` and `" Cm "` without normalizing the pattern for lowercase input.
- `text_prompt.lower()` is used only in one branch, but the search string still contains uppercase `k`, so inputs like "c minor" or "f major" are missed.
- Impact: key detection silently falls back to C for lowercase prompts.

128) Orchestrator cancellation is a no-op.
- `KmiDi_PROJECT/source/python/music_brain/orchestrator/orchestrator.py:552-573` logs a cancellation request but never cancels or signals the running task.
- Impact: callers believe executions are cancelled, but processing continues and resources are not freed.

129) CUDA device “free memory” is misreported.
- `KmiDi_PROJECT/source/python/penta_core/ml/gpu_utils.py:70-88` sets `memory_free_mb = memory_total - torch.cuda.memory_reserved(i)`.
- `memory_reserved` is the current process reservation, not global free VRAM, so the reported value is incorrect (often near total on fresh processes).
- Impact: device selection and diagnostics can pick GPUs based on misleading free-memory figures.

130) Chord progression detection can divide by zero with small window sizes.
- `KmiDi_PROJECT/source/python/music_brain/audio/chord_detection.py:252-259` computes `frames_per_window = int(self.window_size * sr / self.hop_length)` and then uses it as a divisor.
- If `window_size` is smaller than `hop_length / sr`, `frames_per_window` becomes 0 and `chroma.shape[1] // frames_per_window` raises `ZeroDivisionError`.
- Impact: chord detection crashes for short window configurations.

131) Model pool pre-warm functionality is unreachable because a boolean field shadows the method.
- `python/penta_core/ml/model_pool.py:58-76` sets `self.pre_warm = pre_warm` in `__init__`.
- The class also defines a `pre_warm()` method at `python/penta_core/ml/model_pool.py:202-211`, so instances resolve `self.pre_warm` to the boolean and the method is not callable.
- Impact: calling `pool.pre_warm(...)` raises `TypeError: 'bool' object is not callable`, so pre-warming cannot be triggered.

132) Test input generation ignores single-dimension input shapes.
- `python/penta_core/ml/training_inference_bridge.py:362-371` builds a shape using `shape = [1] + list(model_info.input_shape[1:])`.
- When `input_shape` is a single dimension (e.g., `[128]` from the registry manifest), this produces `[1]` instead of `[1, 128]`.
- Impact: validation inference uses the wrong input shape and can fail for models expecting a feature dimension.

133) TorchScript models are registered as PyTorch and lose backend specificity.
- `python/penta_core/ml/model_registry.py:336-352` maps `"torchscript"` to `ModelBackend.PYTORCH`.
- `ModelBackend.TORCHSCRIPT` is never returned, so TorchScript files are indistinguishable from standard PyTorch entries.
- Impact: backend-specific handling for TorchScript can never be selected via the registry.

134) Async inference module raises NameError on import.
- `python/penta_core/ml/async_inference.py:336-339` references `threading.Lock()` but `threading` is never imported in this module.
- This triggers `NameError: name 'threading' is not defined` when the module is imported.
- Impact: async inference singleton cannot be initialized; importing the module fails.

135) Adaptive batch sizing is computed but never applied.
- `python/penta_core/ml/inference_batching.py:59-76` stores `_current_batch_size` and updates it in `_update_stats()`.
- `process_batch()` always uses `self.config.max_batch_size` and never uses `_current_batch_size`.
- Impact: adaptive batching has no effect; batch size never changes based on latency.

136) Batch queue waiting blocks producers because the lock is held while sleeping.
- `python/penta_core/ml/inference_batching.py:118-132` calls `time.sleep(0.001)` inside `with self._lock:`.
- While sleeping, `add_request()` cannot acquire the lock to enqueue new work, so the batch cannot reach the minimum size.
- Impact: batching can stall or underfill when the queue is empty, and throughput suffers.

137) Resource manager only tracks GPU quota for the first detected device.
- `python/penta_core/ml/resource_manager.py:88-123` initializes `ResourceType.GPU_MEMORY` only once and ignores subsequent devices.
- Multi‑GPU systems share a single quota based on the first device, and per‑device limits are not represented.
- Impact: GPU allocation tracking is incorrect on multi‑GPU hosts and can under/over‑allocate.

138) Metrics summary ignores requested time range.
- `python/penta_core/ml/monitoring.py:150-175` computes `cutoff_time` but never filters metric samples by timestamp.
- The summary always reflects all historical data, regardless of the requested `time_range_minutes`.
- Impact: dashboards and alerts can misrepresent recent system health.

139) AIService initialization is permanently skipped.
- `python/penta_core/ml/ai_service.py:226-254` sets `self._initialized = True` inside `__init__`.
- `initialize()` immediately returns `True` when `_initialized` is already true, so components never call their `initialize()` methods.
- Impact: model discovery, inference, and training services stay in `INITIALIZING` state despite `get_ai_service()` reporting success.

140) Integration health checks can crash when HealthStatus is unavailable.
- `python/penta_core/ml/integration_manager.py:24-33` only defines `HealthStatus` if the health module imports.
- `check_health()` always references `HealthStatus` (`python/penta_core/ml/integration_manager.py:201-221`) without guarding `HAS_HEALTH`.
- Impact: if the health module is missing, calling `check_health()` raises `NameError` instead of returning a status.

141) `penta_core.ml` package import can fail because core modules are missing.
- `python/penta_core/ml/__init__.py:37-83` imports `.inference`, `.chord_predictor`, `.style_transfer`, and `.gpu_utils` without try/except.
- Those modules are not present under `python/penta_core/ml`, so importing `penta_core.ml` raises `ModuleNotFoundError`.
- Impact: the package cannot be imported at all, blocking downstream usage.

142) Training orchestrator installs signal handlers unconditionally.
- `python/penta_core/ml/training_orchestrator.py:676-679` calls `signal.signal()` inside `TrainingOrchestrator.__init__()`.
- In non‑main threads (GUI/worker contexts), `signal.signal()` raises `ValueError`.
- Impact: constructing `TrainingOrchestrator` can crash in threaded or embedded environments.

143) Melody training uses dummy data incompatible with the melody model.
- `python/penta_core/ml/training_orchestrator.py:561-595` falls back to generic dummy data for `MELODY_GENERATION`, producing float tensors shaped `(num_samples, 128)`.
- `MelodyTransformer` expects integer token sequences and uses an embedding layer (`python/penta_core/ml/training_orchestrator.py:447-471`), so the dummy data will error.
- Impact: melody training crashes immediately when using the built‑in dummy dataset.

144) Registered trained models point to files that are never saved.
- `python/penta_core/ml/training_orchestrator.py:1112-1133` registers a `.pt` file in the registry.
- `_export_model()` only logs paths and never writes weights, and training never saves model checkpoints to `.pt`.
- Impact: registry entries reference non‑existent model files, so downstream loading fails.
