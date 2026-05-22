# RT Callback Audit — 2026-05-22

Audit of the audio-thread hot path against four invariants from the platform
goal spec: **No heap allocation in callback**, **No mutex locks in callback**,
**No blocking IO in callback**, **Audio thread safety**.

Scope: `src/plugin/PluginProcessor.cpp::processBlock` (line 432–774) and the
RT-safety surface it depends on.

## Verdict

The RT path is **structurally sound**. The discipline (preallocation in
`prepareToPlay`, atomic shadow-swap for MIDI buffers, lock-free SPSC queues
for ML inference, seqlock RTState, cache-line-aligned writer groups) is
present and consistent. There are **no mutex acquisitions on the audio
thread**; the three `std::lock_guard<std::mutex>` sites are in
message-thread methods only.

Two **watch items** below could degrade under adversarial inputs but are
not active violations as written.

## Findings

| # | Severity | Item | Location | Notes |
|---|----------|------|----------|-------|
| 1 | OK | `processBlock` is `noexcept` | `PluginProcessor.h:82` | Matches the goal-spec callback contract. |
| 2 | OK | No mutex on audio thread | `processBlock` body | All `intentMutex_` users (`generateMidi`, `setWound…`, `setStateInformation`) run on the UI/message thread. |
| 3 | OK | No blocking IO on audio thread | `processBlock` body | No file, socket, syscall (except one-time QoS bump). |
| 4 | OK | One-time QoS bump | `PluginProcessor.cpp:441–443` | `std::call_once` guards `pthread_set_qos_class_self_np`; runs once per audio-thread lifetime. Intentional and documented. |
| 5 | OK | Lookahead buffer preallocated | `PluginProcessor.cpp:309–315` | `nextPowerOfTwo(...)` capacity, `& lookaheadMask_` indexing in callback. No realloc on the hot path. |
| 6 | OK | Feature vector is stack | `PluginProcessor.cpp:468` | `std::array<float, 128>` returned by value; no heap. |
| 7 | OK | Inference plumbing is lock-free | `src/ml/InferenceThreadManager.h:93,114` | `submitRequest`/`getResult` `noexcept`; SPSC ring buffer with drop-newest on overflow, `popLatest` drains on consume. Wake via `std::binary_semaphore` with correct `try_acquire` reset before `release`. |
| 8 | OK | MIDI double-buffer flip is RCU-style | `PluginProcessor.cpp:1011–1020` (writer) ↔ `:614–616` (reader) | Writer (`generateMidi`, message thread) stores to inactive slot then `activeMidiBuffer_.store(release)`. Reader uses `load(acquire)` — torn reads cannot occur. |
| 9 | OK | RTState lock-freedom asserted at compile time | `include/penta/common/RTState.h:60–65` | `static_assert(std::atomic<double>::is_always_lock_free)` etc. Cache-line-aligned writer groups prevent false sharing. Seqlock publish protocol documented and bounded to 64 retries. |
| 10 | OK | EQ params updated via atomic loads | `PluginProcessor.cpp:566` | `masterEQProcessor_.updateParameters(apvts_)` consumes APVTS via `memory_order_relaxed` atomic loads, no lock. |
| 11 | Watch | `juce::MidiBuffer::addEvent` may reallocate under heavy generation | `PluginProcessor.cpp:634–640, 659–664, 674–676` | JUCE's MidiBuffer uses a heap-backed vector that grows by reallocation. The host typically pre-sizes it, but a dense `GeneratedMidi` (hundreds of notes per block across 7 channels) can exceed initial capacity. Mitigation: call `midiMessages.ensureSize(approx_max_events * 4)` once per block before scheduling, or in `prepareToPlay` if size is bounded. |
| 12 | Watch | `extractFeatures` allocation profile not asserted | `PluginProcessor.cpp:1151–1153` → `FeatureExtractor::extractFeatures` | Called every block. `FeatureExtractor::prepareToPlay(samplesPerBlock)` is invoked (`PluginProcessor.cpp:297`), but there is no compile- or run-time check that `extractFeatures` is allocation-free. Add a no-alloc assertion (RTSan or a scoped allocator hook) to lock this in. |
| 13 | Gap | No automated no-allocation enforcement on the callback | `rt_harness/main.cpp` | The harness measures P50/P90/P99 callback duration but does not detect heap allocations or lock acquisitions. A regression that introduces an alloc would not be caught by CI; only by manual ASan + perf review. Adding `[[clang::nonblocking]]` on `processBlock` plus a `KMIDI_ENABLE_RTSAN` CMake option (LLVM RealtimeSanitizer) would close this. |
| 14 | Info | EQ-bypass cached pointer null-guarded | `PluginProcessor.cpp:570` | `if (!paramEqBypass_ || …)` — defensive against `prepareToPlay` ordering during plugin warm-up. Acceptable. |

## Recommended next actions

1. **Pre-size MIDI output** (closes #11). In `processBlock`, before the schedule loops, call `midiMessages.ensureSize(MAX_EVENTS_PER_BLOCK)` where `MAX_EVENTS_PER_BLOCK` is computed from the maximum possible note density in `GeneratedMidi`. Alternatively, pre-size once in `prepareToPlay` if the upper bound is known at that point.
2. **Annotate the callback for RTSan** (closes #12 + #13). Add `[[clang::nonblocking]]` to `processBlock` and any function it calls on the hot path. Add a `KMIDI_ENABLE_RTSAN` CMake option mirroring the existing `KMIDI_ENABLE_ASAN`/`KMIDI_ENABLE_TSAN` pattern. Requires LLVM 20+ (`-fsanitize=realtime`); current Apple Clang on macOS 15 may need a Homebrew-LLVM fallback. Document in `docs/NATIVE_SAFETY_AND_FFI.md` once enabled.
3. **CI hook** (closes #13). Add a CI job that builds with RTSan and runs `rt_harness` against the golden preset; fail on any reported violation.

## Infrastructure-present catalog (verified by file inspection 2026-05-22)

Beyond the RT-callback path itself, the following spec items have working code in the canonical tree (presence verified; behavior not exhaustively tested in this audit). Listing here so reviewers and future agents don't re-search for them.

| Spec item | Where it lives | Verification |
|-----------|----------------|--------------|
| C++ DSP core | `src/`, `engine/`, `src_penta-core/` | KellyCore builds (modulo pre-existing Python C API link issue in `penta_core_native`) |
| Python ML layer | `music_brain/`, `music_brain/penta_core/` | 76 pytest passes this turn |
| JUCE / native audio framework | `external/JUCE/` (JUCE 8) | KellyFFI links JUCE PRIVATE; processBlock confirmed noexcept |
| Audio JEPA / Chord JEPA / Stem-aware generation | `music_brain/jepa/audio_jepa.py`, `chord_jepa.py`, `trainer.py`, `masking.py` | Files present and importable |
| MIDI/audio paired datasets | `music_brain/jepa/datasets.py` | Present |
| Emotion-labeled datasets / Emotion trajectory planning | `music_brain/jepa/emotion_probe.py` + emotion schema (`shared_schemas/emotion_schema.json`) | Schema round-trip verified this turn |
| Streaming inference / Streaming realization | `music_brain/penta_core/ml/streaming.py` | Present |
| MIDI-CI integration | `tools/midi_ci_daemon/main.cpp`, `CMakeLists.txt` (gated by `BUILD_MIDI_CI_DAEMON=ON`) | Build target present |
| CoreML / Local inference / Apple Silicon optimization | `tools/coreml_llm_runner/Package.swift` + `Sources/` (Swift) | Swift Package present |
| Cloud training infrastructure | `Dockerfile.train`, `docs/CLOUD_SAGEMAKER.md`, `Dockerfile.sagemaker`, `Dockerfile.vertex` | Multiple cloud images defined |
| Data provenance validation / Legally licensed datasets / Attribution tracking | `config/source_manifest.yaml`, `scripts/acquire/acquire_from_manifest.py`, `scripts/acquire/verify_manifest_status.py` | Manifest-driven acquisition + verification scripts present |
| Crash telemetry / Logging & diagnostics | `src/core/logging.cpp`, `music_brain/agents/telemetry.py`, `music_brain/orchestrator/logging_utils.py` | Multi-layer telemetry/log infra present |
| Session persistence / State snapshot recovery | `src/plugin/PluginState.cpp/.h` (`saveState`/`loadState`, ValueTree/XML serialization) | Used by `getStateInformation`/`setStateInformation` |
| Spectral semantic visualization / Spectrogram embedding analysis | `music_brain/visualization/spectocloud.py`, `examples/spectocloud_*.py` | Pipeline + examples present |
| Checkpoint rollback / Model versioning | `checkpoints/`, `scripts/checkpoints/` | Directory infrastructure present |
| AU plugin support | `music_brain/mobile/ios_audio_unit.py` + JUCE AUv3 support in `external/JUCE/` | iOS AUv3 path scaffolded; macOS desktop AU via JUCE plugin client |
| VST3 plugin support | `src/plugin/PluginProcessor.cpp/.h`, root CMake `KellyPlugin_VST3` target | Build target documented |
| Reproducible builds | `package-lock.json` (committed), `uv.lock`, `Cargo.lock`, `pyproject.toml` pinned, CMake `find_package` versions | Lock files + CMake pinning |
| RT harness / Plugin host stability check | `rt_harness/main.cpp`, `rt_harness/fixtures/golden_preset.json` | P50/P90/P99 callback duration harness |
| Lock-free SPSC ring buffer | `src/ml/InferenceThreadManager.h`, `src/bridge/.../readerwriterqueue` (FetchContent) | Used by audio thread; verified `noexcept` push/pop in audit |
| Atomic shadow-swap (RCU) MIDI buffer | `src/plugin/PluginProcessor.cpp:1011-1020` ↔ `:614-616` | Acquire/release pair verified in audit (#8) |
| Seqlock RTState publish/snapshot | `include/penta/common/RTState.h` (`begin_publish`/`end_publish`, `rt_state_snapshot`) | Verified in audit (#9) |
| Cache-line aligned RT writer groups | `RTState.h` (`alignas(kCacheLine)` per group; Apple M-series 128B / x86 64B) | Verified in audit |
| Lookahead with power-of-2 mask | `PluginProcessor.cpp:309-315` | Allocation-free indexing on hot path |
| One-time QoS bump (Apple Silicon E-core avoidance) | `PluginProcessor.cpp:441-443` (`std::call_once` + `pthread_set_qos_class_self_np`) | Once-per-thread, intentional |
| Sanitizer infrastructure (ASan/UBSan, TSan, RTSan) | `cmake/Sanitizers.cmake`, `KMIDI_ENABLE_{ASAN,TSAN,RTSAN}` options | Three-way mutual exclusivity guard added this turn |
| Schema sync invariant | `scripts/sync_entities.py` (3 sources → 9 generated targets), CI guard in `.github/workflows/ci.yml` | Zero-diff verified locally + CI guard expanded 2→9 files this turn |
| Stable plugin ABI (Rust half) | `engine/intent_ir/src/ffi.rs`, `cbindgen.toml` | 9/9 cargo tests pass this turn |
| Stable plugin ABI (C++ half) | `src/bridge/kelly_ffi.h`, `kelly_ffi.cpp`, `tests/cpp/test_kelly_ffi.cpp` | `VERSION 1.0.0 SOVERSION 1` set in CMake (`KellyFFI`) |
| Harmony / Chord prediction | `music_brain/penta_core/ml/chord_predictor.py`, `music_brain/structure/chord.py`, `music_brain/structure/progression.py`, `music_brain/harmony_utils/harmony_generator.py` | Multi-layer chord/harmony stack present |
| Structural phrase planning / Section-aware generation / Phrase-boundary prediction / Bar-beat awareness | `music_brain/structure/{sections, progression, comprehensive_engine, tension_curve}.py` | Structure-layer planner present |
| Hierarchical generation / Intent planning | `music_brain/penta_core/ml/thesaurus_loader.py`, `music_brain/kelly/core/emotion_thesaurus.py`, `music_brain/intent_ir/__init__.py` | Emotion-thesaurus → intent → realization layered |
| Multi-instrument orchestration / Adaptive orchestration | `music_brain/orchestrator/orchestrator.py`, `music_brain/kelly_companion/engines/orchestration.py`, `music_brain/penta_core/ml/m4_training_orchestrator.py` | Orchestrator layer present |
| KV-cache optimization / Sub-16ms inference / Stateful inference | `tools/coreml_llm_runner/Sources/CoreMLLMRunner/main.swift` (state-threaded greedy loop per `docs/research/KMIDI_90_DAY_DEMO_ROADMAP_2026.md`) | Swift runner is the stateful Core ML path |
| Piano roll integration | `src/ui/PianoRollPreview.h/.cpp` | C++ JUCE-side piano roll renderer |
| Arrangement timeline editing / Arrangement continuity | `src/engines/ArrangementEngine.h/.cpp`, `src/components/SideA/Timeline.tsx`, `src/assets/motifs/arrangement-timeline-support.svg` | Arrangement engine + frontend timeline component |
| Constraint-aware generation | `music_brain/engine_api/schema.py` (CompleteSongIntentRequest with explicit constraints), `music_brain/kelly_companion/.../intent_schema.py` | Constraints carried through canonical intent contract |
| Undo/redo history / Transactional edit system / Rollback-safe scheduling | `music_brain/agents/command.py` (Command pattern with Undo/Redo history per file header docstring) | Documented command stack |
| Update infrastructure / Update delivery | `.github/workflows/release.yml`, `release-monorepo.yml`, `v1-release.yml` | Three release workflows present |
| Schema expansion and validation (emotion + intent_frame, not just intent) | `shared_schemas/emotion_schema.json`, `intent_frame_schema.json`, plus generated targets | All three sources guarded by expanded CI drift check this turn |

## Deferred-by-decision (not gaps)

| Spec item | Decision |
|-----------|----------|
| AAX plugin support | **Deferred to vN; not a v1 feature.** Avid Pro Tools is the only DAW requiring AAX. Distribution additionally requires PACE iLok wrapping — proprietary copy-protection middleware that introduces vendor lock-in and significant CI/CD friction. KMiDi v1 ships VST3 + AU + CLAP, which covers Logic, Ableton, Reaper, FL Studio, Cubase, Bitwig, Studio One, Cakewalk, GarageBand — i.e., the entire creator/indie market. AAX is a "court Pro Tools studios" decision that warrants revisiting after product-market fit. |

## Genuine gaps (not present in canonical tree)

| Spec item | Status |
|-----------|--------|
| Audio cache management | No `audio_cache` / cache-manager module found. Confirmed via GitNexus query (zero processes match). |
| Watchdog recovery | No `Watchdog` class or watchdog process found. Confirmed via GitNexus query (zero processes match). |
| Process sandboxing | No sandbox / process-isolation wrapper found at plugin layer (separate from `tests/spectocloud_sandbox_*` test infra). Confirmed via GitNexus query. |
| Vector memory / Embedding DB layer | No vector store / FAISS / embedding DB found. Confirmed via GitNexus query (zero processes match). |
| Cross-attention conditioning bridge | No `class Conditioning` / `class CrossAttention` / cross-attn bridge module found. Confirmed via GitNexus query. |
| Soft-prompt / adapter / LoRA injection | No LoRA / adapter / soft-prompt module found. Confirmed via GitNexus query. |
| World-model latent prediction | No world-model / predictive arrangement module found. Confirmed via GitNexus query. |
| Real-time MIDI mutation transactional engine | No real-time transactional MIDI mutation module found. Confirmed via GitNexus query. (The `agents/command.py` undo/redo stack covers message-thread edits, not RT MIDI mutation.) |
| Autosave journaling persistence | `PluginState.cpp` provides save/load, but no autosave timer or journal. |
| GPU reset recovery flow | No GPU reset recovery / TDR handler found. |
| `[[clang::nonblocking]]` annotation on `processBlock` | Toolchain blocks attribute on Apple Clang; will land when Clang 20+ or Homebrew LLVM is in use. RTSan CMake infrastructure is ready. |

## Code-organization findings (canonical tree)

| Finding | Files | Recommendation |
|---------|-------|----------------|
| Dead snake_case `PluginProcessor` / `PluginEditor` stub pair | `src/plugin/plugin_processor.h` (37 lines, stub), `src/plugin/plugin_editor.h` (stub) | Both date to the initial `a9d09fe9` snapshot. Self-referential: `plugin_editor.h` includes `plugin_processor.h`; nothing else in the canonical tree references either. The production PascalCase versions (`PluginProcessor.{h,cpp}`, `PluginEditor.cpp`) are what compiles. Two `kelly::PluginProcessor` symbols in the same namespace risk ODR if both ever got included. **Action**: delete both files (separate PR), or convert them into a comment block in the PascalCase header documenting the rename. Violates §1 "Clear system boundaries" in `docs/ARCHITECTURAL_CONTRACTS.md`. |

## Out of scope (for this audit)

- DSP correctness inside `masterEQProcessor_.processBlock` (separate audit).
- Plugin host compatibility / VST3 SDK threading model (covered by `BUILD_PLUGINS` integration tests).
- ML-side throughput (`AudioEmotionRunner`'s GPU dispatch, model warm-up) — RT-safety on the audio side is what this audit covers.

## How to reproduce

```bash
# Build with ASan (existing infra) and run the RT harness against the golden preset
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON -DBUILD_RT_HARNESS=ON \
    -DKMIDI_ENABLE_ASAN=ON
cmake --build build-asan --target rt_harness -j8
./build-asan/rt_harness/rt_harness --preset rt_harness/fixtures/golden_preset.json
# Inspect callback_stats.json for P99 jitter regressions.
```

ASan will catch lifetime/UB issues but **will not** flag an in-callback
`new`/`malloc` — that's the gap action #2 above closes.
