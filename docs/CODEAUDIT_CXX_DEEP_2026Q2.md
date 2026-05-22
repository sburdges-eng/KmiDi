# KmiDi C++ Audit — Deep Dive (2026 Q2)

**Date:** 2026-04-07
**Scope:** All first-party C++ in `/Users/seanburdges/Dev/KmiDi/` (~1,088 files across `src/`, `src_penta-core/`, `include/`, `engine/`, `bindings/`, `c/`, `plugins/`, `interfaces/`, `adapters/`, `KmiDi_PROJECT/`, `xcode/KmiDi/KmiDi_CANON/`, `tools/midi_ci_daemon/`, `common/`, `legacy/`, `final_kel/`)
**Method:** 5 parallel deep-dive subagent passes (RT/audio core, DSP/ML/voice, FFI/plugins/ABI, KmiDi_PROJECT/legacy, xcode CANON+UI) + main-thread structural pass over the build graph
**Effort:** MAX (Opus 4.6 deepest reasoning)
**Findings:** ~140 across 8 sections

**Relationship to prior `docs/CODE_REVIEW_REPORT.md`:** That review explicitly states (lines 92-93) that it skipped "src/engine/* (C++ beyond Wound/IntentResult): No full line-by-line; FFI and headers only" and "src/plugin/*, rt_harness, src_penta-core: Not inspected for audio-thread/realtime in this pass". This audit fills exactly that gap. The two are complementary; both should remain in `docs/`.

---

## Executive summary

The codebase is functionally green only because most of the dangerous code lives in **orphan trees that the active build graph never touches**. The `src/` mirror that is shipped contains a smaller but more dangerous subset of the same problems: silent ODR with diverged content, `noexcept`-lying RT functions, raw Python/JUCE bridges with no GIL discipline, and a CMake `file(GLOB_RECURSE) + EXCLUDE_REGEX` filter that has effectively become a mutable reference list nobody owns.

**Three dominant root causes** account for ~70% of the critical findings. Fix any one and ~10–15 high-severity items collapse:

1. **The "consolidation that wasn't"** — `src/harmony/`, `src/groove/`, `src/osc/` were supposedly merged into `src_penta-core/` but the originals were never deleted. `diff -q` confirms 4/5 harmony files, 4/4 groove files, and 5/5 osc files have **actively diverged in body**. Both copies compile into separate static libs which both link into `bindings/penta_core_native` — strict ODR with non-equal definitions. Whichever copy the dynamic linker resolves first determines runtime semantics. Plus 4 more `kelly::*` types and 4 more `kelly::IntentPipeline`/`KellyBrain`/`Wound`/`InferenceRequest` classes have parallel definitions in different headers under the same namespace.

2. **The Python ↔ JUCE allocator-mismatch trap is set** — `bindings/penta_core_native` links JUCE PUBLIC into the Python module (`bindings/CMakeLists.txt:15-20`); the moment any Tauri/ctypes path also loads `KellyFFI.dylib` into the same Python interpreter, two JUCE copies share one address space. The `KellyFFI` target documents this exact failure mode at `CMakeLists.txt:480-483`, then **defeats its own protection** by linking `KellyCore` PUBLIC at line 484 — and `KellyCore` PUBLIC-links JUCE at lines 294-308. Plus the Python C API itself is called without `PyGILState_Ensure()` from arbitrary C++ threads across the entire `src/bridge/` layer (zero GIL hits across all bridge files via grep). This is documented UB.

3. **`noexcept` is decoration, not contract** — Public RT methods declared `noexcept` allocate, lock, throw, or call into JUCE: `HarmonyEngine::suggestVoiceLeading` returns `std::vector<Note>`; `VoiceLeading::findOptimalVoicing` same; `PRROTEngine::processAudioSegment` does `std::string + std::to_string(...)` 8+ times per call; `SpectralAnalyzer::computeFFT` allocates a 16 KB `std::vector<float>` per FFT inside a `noexcept` caller; `PhonemeSegmenter` same; `RTMessageQueue::push` copies an `OSCMessage` containing `std::string` + `std::vector<variant>`; `InputValidation`, `AudioValidator`, `RTNeuralProcessor::process` all `noexcept`-allocate. Each is one allocation-failure away from `std::terminate` mid-audio-callback.

**Build status disclaimer:** Memory says "build green 227/227, Phases 1a/1b/2 done." Main-thread verification: that 227/227 figure is **Python tests via pytest**. The C++ test target `KellyTests` references **four source files that do not exist on disk** (`tests/cpp/test_emotion_engine.cpp`, `test_midi_pipeline.cpp`, `test_chord_diagnostics.cpp`, `test_ml_pipeline.cpp`). Anyone enabling `BUILD_TESTS=ON` hits an immediate configure-time failure. The default build is green only because the C++ test path is OFF by default.

---

## SECTION 1 — Critical Issues (UB, crashes, corruption, RT violations)

> Convention: `[file:line]` references are absolute paths under `/Users/seanburdges/Dev/KmiDi/`. "Cited:" tag identifies the audit source: A=RT/audio core, B=DSP/ML/voice, C=FFI/plugins/ABI, D=KmiDi_PROJECT+legacy, E=xcode CANON+UI, M=main-thread.

### C-1. Diverged ODR across `harmony/`, `groove/`, `osc/` — silent semantic drift in shipped binaries
**[`src/harmony/{VoiceLeading,HarmonyEngine,ChordAnalyzer,ChordAnalyzerSIMD}.cpp` ↔ `src_penta-core/harmony/{same}.cpp`]** — Cited: M, B
Symptom: `diff -q` confirms 4/5 harmony files, 4/4 groove files, 5/5 osc files differ in body. `src/harmony/HarmonyEngine.cpp` is 149 lines; `src_penta-core/harmony/HarmonyEngine.cpp` is 160 lines with **different `updateChordAnalysis()` semantics**: src/ deduplicates by `(root, quality, pitchClass)` equality vs the latest history slot; penta/ deduplicates by `(root, quality, confidence > 0.7)` against `currentChord_`. Same input, different chord history populated. `getChordHistory()` returns `{}` in src/ and `{currentChord_}` in penta/ when history is empty — different shape, breaks `if (history.empty())` callers.
Root cause: The "merge VoiceLeading into penta" memory item was halfway done. `src/harmony/*.cpp` files were never deleted. Both copies are picked up by `file(GLOB_RECURSE KELLY_CORE_SOURCES src/*.cpp)` (root `CMakeLists.txt:245`) and `file(GLOB_RECURSE PENTA_CORE_SOURCES …)` (`src_penta-core/CMakeLists.txt:50`). `KELLY_CORE_SOURCES` does NOT exclude `/src/harmony/`, `/src/groove/`, `/src/osc/`. Both libraries are linked into `bindings/penta_core_native` (`bindings/CMakeLists.txt:15-19`).
Risk: **strict ODR violation with non-equal definitions** — runtime UB. Linker picks one copy at load time per symbol. Symbol resolution order is platform/version dependent. Bug fixes in one tree don't propagate. Tests in one harness can pass while the other path silently fails.

### C-2. `penta::harmony::VoiceLeading` is missing 4 method definitions in `src_penta-core/`
**[`src_penta-core/harmony/VoiceLeading.cpp`]** — Cited: B
Symptom: `include/penta/harmony/VoiceLeading.h` declares `analyze(...)`, `voiceProgression(...)`, `calculateSmoothness(...)`, and `static invertVoicing(...)`. `src/harmony/VoiceLeading.cpp:207,254,273,295` defines all four. `src_penta-core/harmony/VoiceLeading.cpp` defines **none**.
Root cause: The merge from `kelly::VoiceLeadingEngine` into penta was applied to the shared header and to the `src/` copy but never to the penta copy. `penta_core` static lib has 4 declared-but-undefined symbols.
Risk: **undefined-symbol link error the moment any client of `penta_core` calls one of those four methods.** Currently masked because `bindings/harmony_bindings.cpp` does not yet expose them and no test in `tests/penta_core/` exercises them.

### C-3. Duplicate `analyzeSIMD` symbol on ARM/Apple Silicon
**[`src_penta-core/harmony/ChordAnalyzer.cpp:188-192` + `ChordAnalyzerSIMD.cpp:164`]** — Cited: B
Symptom: When `__AVX2__` is undefined (Apple Silicon, ARM), `ChordAnalyzer.cpp`'s `#ifndef __AVX2__` block defines `scoreAgainstTemplateSIMD`, `findBestMatchSIMD`, AND `analyzeSIMD`. `ChordAnalyzerSIMD.cpp` defines the first two inside its own `#ifndef __AVX2__` block (lines 144-161) AND defines `analyzeSIMD` **unconditionally** at line 164. Result: each symbol has TWO definitions in `penta_core` on ARM.
Root cause: Comment in `ChordAnalyzer.cpp:165-167` says "AVX2 versions live exclusively in ChordAnalyzerSIMD.cpp" but the author defensively kept scalar fallbacks in BOTH files. The unconditional `analyzeSIMD` at SIMD.cpp:164 is the same regardless of arch.
Risk: link error on ARM build (which is the dev machine per memory). The "build green 227/227" claim implies CI is x86 and never exercises this path; ARM build is broken.

### C-4. `KellyTests` references 4 nonexistent test files — `BUILD_TESTS=ON` is broken
**[`CMakeLists.txt:570-575`]** — Cited: M
Symptom: `add_executable(KellyTests tests/cpp/test_emotion_engine.cpp tests/cpp/test_midi_pipeline.cpp tests/cpp/test_chord_diagnostics.cpp tests/cpp/test_ml_pipeline.cpp)`. None of those four files exist in `tests/cpp/` (verified by direct stat). Anyone running `cmake -DBUILD_TESTS=ON` hits a configure or build error.
Root cause: Source files were renamed/deleted; CMakeLists never updated. The "227/227 build green" memory item is **Python tests via pytest**, not C++.
Risk: C++ test infrastructure is non-functional. There is no automated regression coverage for any KellyCore code path.

### C-5. `BiometricInput.cpp` AND `BiometricInput.mm` both compiled into KellyCore — duplicate symbols
**[`src/biometric/BiometricInput.cpp` + `src/biometric/BiometricInput.mm`]** — Cited: C
Symptom: Both files define the entire `kelly::BiometricInput` class (constructors, dtor, `processBiometricData`, `addToHistory`, `heartRateToArousal`). Root `CMakeLists.txt:245-248` globs `src/*.cpp` AND `src/*.mm` with no exclusion for the duplicate. Both TUs land in `KellyCore`.
Root cause: When the `.mm` ObjC++ variant was added for HealthKit, the `.cpp` was not removed and the `EXCLUDE_REGEX` list was not updated. Either link fails with duplicate symbols, or one copy silently wins and ODR-violates every caller. Additionally, the `.cpp` dtor deletes `void* healthKitBridge_` cast to `kelly::biometric::HealthKitBridge*`, but `BiometricInput.h:149` forward-declares `HealthKitBridge` as a *nested class* of `BiometricInput`, in a different namespace from the actual implementation type.
Risk: link failure or ODR UB; type-confusion in delete (wrong vtable).

### C-6. Python C API called without GIL from C++ threads
**[`src/bridge/PythonBridgeBase.cpp:128`, `OrchestratorBridge.cpp:104-137,150-153`, `IntentBridge.cpp:79`, `ContextBridge.cpp:75-76`, `EngineIntelligenceBridge.cpp:76-82`, `StateBridge.cpp`, `PreferenceBridge.cpp`, `SuggestionBridge.cpp`; same pattern in `xcode/KmiDi/KmiDi_CANON/body/bridge/OrchestratorBridge.cpp:104-137,150-153`]** — Cited: C, E
Symptom: Every `PyTuple_New`, `PyObject_CallObject`, `PyUnicode_AsUTF8`, `PyObject_Str`, `Py_DECREF` is invoked from arbitrary C++ caller threads. `OrchestratorBridge::executePipelineAsync` spawns `std::thread([this,...]() { executePipeline(...); }).detach();` — the worker thread has *never* held the GIL.
Root cause: `PythonBridgeBase` was structured around `Py_Initialize()` (which acquires the GIL on the calling thread, then releases on return) but no subsequent function pairs Python calls with `PyGILState_Ensure()/PyGILState_Release()`. `grep -r PyGILState_Ensure src/bridge/` returns **zero** hits across all bridge files. `OrchestratorBridge` additionally captures `this` raw in the detached thread with no shutdown synchronization (UAF on `~OrchestratorBridge`).
Risk: interpreter heap corruption, refcount underflow, hard crashes, UAF on bridge destruction.

### C-7. `OrchestratorBridge` detached thread captures `this` raw with no shutdown synchronization
**[`src/bridge/OrchestratorBridge.cpp:150-153` + `xcode/KmiDi/KmiDi_CANON/body/bridge/OrchestratorBridge.cpp:150-153`]** — Cited: C, E
Symptom: `std::thread([this, ...]).detach();` runs `executePipeline` on a background thread that outlives the bridge object if `~OrchestratorBridge` runs before the thread completes. There's no `std::atomic<int> active_workers_` + drain; no `weak_ptr` indirection; no `std::jthread` or stop_token. The pruning of `asyncThreads_` (line 162) erases threads while surviving threads still hold references to the lambda capture.
Risk: UAF on `this->executePipeline(...)` and on member `executePipelineFunc_` after `~OrchestratorBridge`.

### C-8. `IntentBridge`/`ContextBridge`/`OrchestratorBridge` etc. — Unchecked `PyTuple_SetItem` with potentially-NULL argument
**[`src/bridge/IntentBridge.cpp:79` + `ContextBridge.cpp:75-76` + `OrchestratorBridge.cpp:120-124` + `EngineIntelligenceBridge.cpp:76-82` + others]** — Cited: C
Symptom: `PyUnicode_FromString(...)` may return NULL (alloc failure or non-UTF-8). Result passed straight to `PyTuple_SetItem`, which writes NULL into a tuple slot. The next `PyObject_CallObject` then dereferences NULL inside the interpreter.

### C-9. `PluginProcessor.cpp` ships hardcoded developer absolute path
**[`src/plugin/PluginProcessor.cpp:346,358`]** — Cited: C
Symptom: `juce::File("/Users/seanburdges/Dev/KmiDi/models/audio_jepa_v01.onnx")` and `.../emotion_probe_v01.onnx` literal in `prepareToPlay()`. CLAUDE.md line 104 explicitly forbids this: "Hardcoded `/Users/<name>/...` paths in source are prohibited."
Root cause: Developer-machine fallback never replaced with `KELLY_MODEL_ROOT` env var.
Risk: Plugin shipped on any other machine silently fails to load models. Dev-machine username leaked in shipped plugin binary.

### C-10. `engine_rt.cpp` declares C ABI symbols but is never compiled
**[`engine/include/engine.h:103-121` ↔ `engine/src/engine_rt.cpp:22-87` ↔ `engine/CMakeLists.txt:8-10`]** — Cited: A, D
Symptom: `kmidi_engine_get_state()` and `kmidi_engine_push_param()` are declared in the public C header. Implementations live in `engine_rt.cpp`. The `engine` static-lib target only lists `src/engine.c`. The file is also marked C++ but the `engine` target declares `LANGUAGES C` only.
Risk: undefined-symbol link error for any caller of those functions. Plus the file contains a process-global `static moodycamel::ReaderWriterQueue g_rtParamQueue(256)` and `static penta::RTState g_rtState` — multi-instance plugin hosts (VST3, AU) routinely instantiate >1 engine; if the file ever *is* linked, all instances alias the same state.

### C-11. `engine_rt.cpp` reader implements seqlock but no writer exists
**[`engine/src/engine_rt.cpp:32-57`]** — Cited: A, D
Symptom: `kmidi_engine_get_state` implements the *reader* side of a seqlock pattern (retry on odd `seq1` or mismatched `seq2`). Nothing in the partition increments `g_rtState.sequence` before/after writes. Reader returns `seq1 == seq2 == 0`, `(seq1 & 1) == 0`, loop exits on first iteration, snapshot is full of zeroes forever.
Risk: API lying about RT state — readers always see zero. Also: `try_dequeue` on `g_rtParamQueue` never called → queue fills silently → all parameter updates dropped.

### C-12. `IntentFrame` `#pragma pack(push, 1)` produces unaligned 8-byte fields
**[`include/kmidi/IntentIR.h:179-188`]** — Cited: A
Symptom: `IntentFrame` is force-packed to 1 byte. Contains `uint64_t intent_id`, `uint64_t session_id`, etc. Taking the address of any field gives an unaligned pointer; using it where alignment is assumed (`std::atomic`, SIMD load) is UB. ARMv7 and sanitizers will trap.
Risk: UB on load/store on strict-alignment targets, slow unaligned access on x86, dangerous if ever atomicized, UBSan trap.

### C-13. `daiw::AudioStream` references non-existent ring buffer API
**[`include/daiw/audio_io.hpp:277,281,292`]** — Cited: A
Symptom: `RingBuffer<float> buffer_;` — but `daiw::RingBuffer` is `template<typename T, size_t Capacity> class RingBuffer`. There is no single-parameter overload. Also calls `buffer_.available()`/`space()` — actual API is `available_read()`/`available_write()`.
Risk: Dead-on-arrival header. Will explode the moment anyone reaches for `daiw::audio_io::AudioStream`.

### C-14. `daiw::MPSCQueue::size_approx` wrong post-wrap
**[`include/daiw/lock_free_queue.hpp:107`]** — Cited: A
Symptom: `(tail - head) & MASK` computes signed-wrap-around incorrectly because head/tail are stored MASKED in this class (line 75: `head_.store((head + 1) & MASK)`), unlike `RingBuffer` which stores unmasked. Two ring-buffer conventions in the same namespace; wrong one used here.
Risk: Queue full appears empty / vice versa around wraparound. Producer overwrites consumer's slot → silent data loss + UB if consumer is reading the slot.

### C-15. `daiw::LockFreeQueue` is a misnamed mutex queue with per-element heap allocation
**[`include/daiw/memory.hpp:188-218` + `src/core/memory.cpp:88-117`]** — Cited: A
Symptom: Class is named `LockFreeQueue` and documented "MPSC lock-free queue for event passing" but every push/pop takes `std::lock_guard<std::mutex>`. `push()` does `new QueueNode<T>(item)`, `pop()` does `delete tail_`. RT-fatal on both counts.

### C-16. `RTMemoryPool` ABA hazard in lock-free pool
**[`src_penta-core/common/RTMemoryPool.cpp:30-56`]** — Cited: B
Symptom: Treiber stack with bare `compare_exchange_weak` on `freeList_` head pointer carries no version tag. Standard ABA: T1 reads `head=X`, `next=Y`, is preempted; T2 allocates X, allocates Y, deallocates X (now `head=X` again, but `X->next=Z`); T1 resumes, CAS succeeds, `freeList_=Y`, but Y is currently owned by T2.
Risk: Two threads receive the same `void*` → silent memory corruption inside any subsystem using the pool.

### C-17. `daiw::memory_pool::release` ABA window in CAS loop
**[`include/daiw/memory_pool.hpp:84-105`]** — Cited: A
Symptom: 32-bit generation tag protects head pointer (good), but `release` writes `slots_[index].next.store(...)` BEFORE the CAS, so a concurrent acquirer can pull `index` off the free list before this thread's CAS lands. The released slot's `next` may then be stale.
Risk: Race between concurrent release and acquire that re-uses the same slot — broken free list, lost slots, potential UAF.

### C-18. `MidiBuilder::buildMidiFile` divide-by-zero on fractional or sub-1 BPM
**[`src/midi/MidiBuilder.cpp:40` + `src/midi/MidiExporter.cpp:380`]** — Cited: D
Symptom: `int microsecondsPerBeat = MIDI_MICROSECONDS_PER_MINUTE / static_cast<int>(midi.bpm);`. Truncates fractional BPM (120.7 → 120). For `0 < midi.bpm < 1`, the cast to `int` → 0 → integer divide-by-zero → SIGFPE.
Risk: SIGFPE on any caller supplying fractional/tiny BPM.

### C-19. `MidiGenerator::generate()` returns pointer to `static thread_local` storage
**[`src/midi/MidiGenerator.cpp:54-65` + `xcode/.../body/midi/MidiGenerator.cpp:54-65`]** — Cited: D, E
Symptom: `static thread_local ArrangementOutput storedArrangement; storedArrangement = arrangement; arrangementPtr = &storedArrangement; ... result.arrangement = arrangementPtr;` — `GeneratedMidi::arrangement` holds a pointer into a thread-local static. Next call to `generate()` on the same thread silently rewrites the referent. Moving `GeneratedMidi` to a different thread and dereferencing `arrangement` is UB.
Risk: Audio thread reads `generatedMidi_.arrangement` while message thread regenerates → silent data corruption + cross-thread UAF + UB.

### C-20. `AffectUMP` UB converting `std::round(norm * 0xFFFFFFFF)` to `uint32_t`
**[`src/midi/AffectUMP.cpp:19-20`]** — Cited: D
Symptom: `auto v = static_cast<uint32_t>(std::round(norm * 0xFFFFFFFF)); return (v <= 0xFFFFFFFF) ? v : 0xFFFFFFFF;` — `float` cannot represent `0xFFFFFFFF = 4294967295` exactly; nearest representable float is `4294967296.0f`. When `norm == 1.0f`, conversion is UB.
Risk: UB; clamp guard is unreachable.

### C-21. ODR — two `kelly::Wound` / `EmotionNode` / `IntentResult` / `RuleBreakType` types in the same namespace
**[`src/engine/IntentProcessor.h:34-126` ↔ `src/common/KellyTypes.h`]** — Cited: A
Symptom: Two different definitions of these types in the same `kelly` namespace. `IntentProcessor.h` (via `Kelly.h`) defines a `Wound` with fields `description, intensity, source, timestamp, context, triggers`. `KellyTypes.h` (via `KellyBrain.h`) defines a different `Wound` with `urgency, expression, primaryEmotion, desire`. Same applies to `EmotionNode`, `IntentResult`, `RuleBreakType`.
Root cause: `KellyBrain.cpp:5-12` confesses the trick — it aliases `KellyTypesWound = Wound`, etc., then redefines them by including `Types.h` after the alias. Works in that single TU but produces two object-layout-distinct types under the same mangled name.
Risk: Linker picks one or the other arbitrarily. **Almost certainly the worst single bug in the partition.**

### C-22. ODR — two `kelly::IntentPipeline` classes with different APIs and members
**[`engine/IntentPipeline.h:12-42` ↔ `src/engine/IntentPipeline.h:29-109`]** — Cited: A, D
Symptom: Top-level `engine/IntentPipeline.h` (no non-static data members, 5 private decls) vs active `src/engine/IntentPipeline.h` (`thesaurus_`, `woundProcessor_`, `ruleBreakEngine_` members). Same `kelly::IntentPipeline` class.
Risk: Currently masked because nothing in the active build graph includes the dead path. A single misrouted `#include "engine/IntentPipeline.h"` produces ODR with corrupted vtables.

### C-23. `daiw::AudioBuffer` ODR — header version vs `src/dsp/audio_buffer.cpp` version
**[`include/daiw/audio_io.hpp:125-228` ↔ `src/dsp/audio_buffer.cpp`]** — Cited: A
Two distinct definitions of `daiw::AudioBuffer` in the same namespace.

### C-24. `kelly::InferenceRequest`/`InferenceResult` defined in two headers
**[`src/ml/InferenceThreadManager.h:15-34` ↔ `src/ml/InferenceRequest.h:12-32`]** — Cited: B

### C-25. `HarmonyEngine::activeNotes_` OOB on pitch ≥ 128
**[`src_penta-core/harmony/HarmonyEngine.cpp:24-27` + `src/harmony/HarmonyEngine.cpp:25-28`]** — Cited: B, M
Symptom: `activeNotes_[note.pitch] = note.velocity` and `activeNotes_[note.pitch] = 0`. `Note::pitch` is `uint8_t` (0-255 per the type, "0-127" per the comment), but `activeNotes_` is `std::array<uint8_t, 128>`. Any malformed MIDI input with `pitch >= 128` writes 1-128 bytes past the end. RT-side `processNotes` is `noexcept`, so OOB-induced trap → terminate.
Risk: Memory corruption, UB. In a plugin context where untrusted hosts forward arbitrary MIDI, **possible RCE primitive**.

### C-26. `HarmonyEngine::suggestVoiceLeading` modulo by zero on empty chord
**[`xcode/KmiDi/KmiDi_CANON/body/music_theory/harmony/HarmonyEngine.cpp:490,497`]** — Cited: E
Symptom: `fromChord.notes[i % fromChord.notes.size()]` — if either notes vector is empty, `i % 0` is UB (typically SIGFPE).

### C-27. `RhythmEngine::detectTimeSignature` divide-by-zero on degenerate input
**[`xcode/KmiDi/KmiDi_CANON/body/music_theory/rhythm/RhythmEngine.cpp:54,1245,1265-1267`]** — Cited: E

### C-28. `MelodyEngine` OOB on empty/`numNotes < 0`
**[`xcode/KmiDi/KmiDi_CANON/body/engines/MelodyEngine.cpp:231,312`]** — Cited: E

### C-29. `F0Extractor` parabolic interpolation NaN propagates through `static_cast<int>`
**[`src/audio/F0Extractor.cpp:209-214`]** — Cited: A
Symptom: `offset = (y2 - y0) / (2.0f * (2.0f * y1 - y0 - y2))` — denominator zero whenever `2*y1 == y0+y2`. Result `±inf` or `NaN` (0/0). `std::clamp(NaN, ...)` returns NaN. Then `static_cast<int>(std::round(NaN))` is UB.

### C-30. `AudioFile.cpp` WAV reader trusts hostile chunk sizes
**[`src/audio/AudioFile.cpp:87,117,152`]** — Cited: A
Symptom: Reads 32-bit chunk lengths from input, calls `file.seekg(chunkSize, std::ios::cur)` and `data_.resize(totalSamples * header.numChannels)` without bounds-checking against actual file size.
Risk: OOM-abort/DoS on malformed input.

### C-31. `daiw/midi.hpp` signed shift overflow in `read_uint32_be`
**[`include/daiw/midi.hpp:580-585`]** — Cited: A
Symptom: `(data[0] << 24) | ...` — `data[0]` is `uint8_t`, integer-promoted to `int`. If `data[0] >= 0x80`, `data[0] << 24` shifts a 1 into the sign bit of a signed 32-bit int — implementation-defined / UB.

### C-32. PRROT FFT path heap allocation in noexcept
**[`src/prrot/SpectralAnalyzer.cpp:333` + `src/prrot/PhonemeSegmenter.cpp:583`]** — Cited: B
Symptom: `std::vector<float> fft_data(kFFTSize * 2, 0.0f);` per call, called from RT-marked `noexcept` paths. The pre-allocated `fft_real_`/`fft_imag_` are sized `kFFTSize`, not `kFFTSize*2`.
Risk: Heap allocation in noexcept audio path → terminate-on-OOM, RT violation, priority inversion.

### C-33. `PRROTEngine::processAudioSegment` 8+ heap allocations per call from noexcept
**[`src/prrot/PRROTEngine.cpp:62-87,128-131`]** — Cited: B
Symptom: `noexcept` audio function constructs `std::string` via `+` and `std::to_string(...)` for every `logRT` call (8+ in this single function), plus returns `std::vector` from `analyzePhonemes`, `detectBreathMarkers`, `shapeMidiNotes`, plus `pitch_targets.push_back(...)`, plus envelope `clear/reserve/push_back`.

### C-34. `RTLogger::getLogger` non-thread-safe singleton init
**[`xcode/KmiDi/KmiDi_CANON/body/common/RTLogger.cpp:7,95-100`]** — Cited: E
Symptom: `static RTLogger* g_logger = nullptr; if (!g_logger) { g_logger = new RTLogger(); g_logger->start(); }` — classic non-thread-safe lazy init.

### C-35. `RTLogger::getLogger` (penta) lazy thread spawn from audio thread
**[`src_penta-core/common/RTLogger.cpp:94-99`]** — Cited: B
Symptom: First call to `penta::getLogger()` lazily executes `instance.start()` under `std::call_once`, which constructs a `std::thread`. Multi-millisecond stall on first audio block.

### C-36. `BiometricInput` destructor deletes hardware bridges before stopping streaming thread
**[`src/biometric/BiometricInput.cpp:17-30`]** — Cited: C, E
Symptom: dtor body runs `delete fitbitBridge_` then later `if (streamingActive_) stopStreaming();`. Streaming thread is still alive while `fitbitBridge_` is deleted. Plus `streamingActive_` is non-atomic `bool`, written by streamingLoop, read by dtor.

### C-37. `VoiceProcessor::queuePhonemes` racing audio thread vector reassignment
**[`src/VoiceProcessor.cpp:419 ↔ :351`]** — Cited: C
Symptom: `processBlock()` (line 325) reads `phonemeQueue_[currentPhonemeIndex_]` from the audio thread without any lock. `queuePhonemes()` (line 419) does `phonemeQueue_ = phonemes;` which destroys/reallocates the vector storage, possibly mid-read.
Risk: UAF, crash, data race UB.

### C-38. JUCE Component dtor missing `setLookAndFeel(nullptr)` and `stopTimer()` (multiple sites)
**[`xcode/KmiDi/KmiDi_CANON/body/ui/{EmotionWorkstation,LyricDisplay,WorkstationPanel,theory/MusicTheoryWorkstation}.{h,cpp}`]** — Cited: E
Symptom: Multiple `juce::Component`+`juce::Timer` subclasses register `setLookAndFeel(&lookAndFeel_)` and have `~Class() override = default`. Defaulted dtor does NOT call `stopTimer()` or `setLookAndFeel(nullptr)`.
Note: xcode tree is orphan source (see C-44), but the same files exist in `src/ui/` (which IS built) with the same names — almost certainly the same bugs apply. **High-priority follow-up: audit `src/ui/EmotionWorkstation.cpp` etc. for the same patterns.**

### C-39. JUCE file-chooser callbacks capture `this` raw — UAF on close
**[`xcode/KmiDi/KmiDi_CANON/body/plugin/PluginEditor.cpp:556,719,800` + `PluginIRInspector.cpp:40`]** — Cited: E

### C-40. `ScoreEntryPanel` member declaration order causes UAF on shutdown
**[`xcode/KmiDi/KmiDi_CANON/body/ui/ScoreEntryPanel.h:301-302` + `cpp:54`]** — Cited: E

### C-41. `WavetableSynth` non-atomic `shared_ptr` swap during audio playback
**[`xcode/KmiDi/KmiDi_CANON/body/WavetableSynth.cpp:613-622`]** — Cited: E

### C-42. `GlottalSource::process` function-local static differentiator state
**[`xcode/KmiDi/KmiDi_CANON/body/VoiceProcessor.cpp:99-101`]** — Cited: E
Symptom: `static float lastOutput = 0.0f;` inside `GlottalSource::process()` — every voice writes to and reads from the same float.

### C-43. `KellyBrain` pimpl with destructor `= default` in header
**[`src/engine/KellyBrain.h:39` + `KellyBrain.cpp:168`]** — Cited: A
Symptom: Header forward-declares `IntentPipeline` and holds `std::unique_ptr<IntentPipeline>`. Destructor is `= default` IN THE HEADER. IFNDR — undefined behavior on destruction in any TU that creates a temporary `KellyBrain` without including `IntentPipeline.h`.

### C-44. xcode/KmiDi/KmiDi_CANON is orphan source — 406 files of dead code, NOT built
**[`/Users/seanburdges/Dev/KmiDi/xcode/KmiDi/KmiDi_CANON/`]** — Cited: E
Evidence: (1) `find xcode -name "*.xcodeproj" -o -name "*.pbxproj"` returns nothing. (2) Only one CMakeLists in the entire xcode/ tree. (3) Root CMakeLists has zero references to `xcode/`. (4) The "xcode-debug" / "xcode-release" presets refer to the **CMake Xcode generator**, not the source tree. (5) `xcode/KmiDi/KmiDi_CANON/body/plugin/PluginProcessor.cpp:18` includes `engine/KellyBrain.h` which only exists at `src/engine/KellyBrain.h`.
**Critical implication:** All findings in the xcode partition describe CODE THAT IS NOT SHIPPED. **However**, many of the same basenames exist in `src/ui/`, `src/plugin/`, `src/engine/`, `src/midi/`, `src/voice/` with **diverged content**. The same JUCE Component bugs almost certainly exist in `src/ui/*.cpp` (the actually-built copies). **Highest-leverage follow-up: re-audit `src/ui/*.cpp`** with Group E's checklist.

---

## SECTION 2 — High-Risk Design Problems

### H-1. `file(GLOB_RECURSE)` is the build's source of truth
**[`CMakeLists.txt:245-248`, `src_penta-core/CMakeLists.txt:50`, `KmiDi_PROJECT/source/cpp/src_penta-core/CMakeLists.txt:61`]**
Three CMakeLists use `file(GLOB_RECURSE ${dir}/*.cpp)`. CMake docs warn against this even with `CONFIGURE_DEPENDS`. New file added → silently compiled into the lib. Consequence: the entire ODR/duplicate-symbol class of bugs (C-1, C-5, C-23) only exists because globs picked up files nobody intended.

### H-2. `KELLY_CORE_SOURCES` is a 16-line `EXCLUDE REGEX` brittle list
**[`CMakeLists.txt:251-277`]**
16 hand-maintained `list(FILTER ... EXCLUDE REGEX)` lines remove files from the GLOB result that are also built in `daiw_core` or aren't supposed to compile into KellyCore. Any new file added to `libs/daiw/CMakeLists.txt` without a matching exclude → simultaneous compilation into both libs → ODR.

### H-3. `daiw_core` target name collision across 3 build trees
**[`libs/daiw/CMakeLists.txt`, `KmiDi_PROJECT/source/cpp/src/CMakeLists.txt:104`, `_archive/KmiDi_FINAL/engine/cpp_music_brain/CMakeLists.txt:104`]**
Three trees define `daiw_core/daiw_dsp/daiw_midi/daiw_harmony` with identical target names. The two orphan trees are not added by root, so no current collision. **However** `KmiDi_PROJECT/source/cpp/cpp_music_brain/CMakeLists.txt:60` pins **JUCE 7.0.9** vs root's JUCE 8 — instant ABI hell if anyone runs the orphan build.

### H-4. `penta_core` target name collision across 3 build trees
**[`src_penta-core/CMakeLists.txt`, `KmiDi_PROJECT/source/cpp/src_penta-core/CMakeLists.txt`, `penta_build/CMakeLists.txt`]**
Three CMakeLists files all build `penta_core` from different source trees. `penta_build/CMakeLists.txt` is in the source root and points at `KmiDi_PROJECT/source/cpp/src_penta-core/`.

### H-5. `src_penta-core/CMakeLists.txt` has double `cmake_minimum_required` + `project()`
**[`src_penta-core/CMakeLists.txt:1,8`]**
Two `cmake_minimum_required()` and two `project()` declarations in sequence (3.22 → 3.20, `penta_core` → `KmiDiPentaCore`). Second `project()` overwrites the first. Plus `set(CMAKE_CXX_STANDARD 17)` at line 4 vs root's required C++20 — when `penta_core` is linked into `KellyCore`, headers compile differently across TUs.

### H-6. `KellyFFI` "no double JUCE" rule defeated by KellyCore PUBLIC linkage
**[`CMakeLists.txt:484,294-308`]**
`target_link_libraries(KellyFFI PUBLIC KellyCore PRIVATE Qt6::Core ... juce::juce_*)`. The `PRIVATE` JUCE/Qt linkage is neutralized because `KellyCore` itself links JUCE/Qt PUBLIC. Anything depending on KellyFFI inherits all of JUCE/Qt at link time — defeating the entire isolation strategy. This is the *exact* failure mode the project history calls out as "PROVEN failure mode."

### H-7. `bindings/penta_core_native` brings JUCE into the Python process
**[`bindings/CMakeLists.txt:15-20`]**
`pybind11_add_module(penta_core_native ...)` then `target_link_libraries(... PRIVATE penta_core juce::juce_osc juce::juce_dsp)`. When Python imports the module, JUCE static initializers run inside the Python process. If the same process later loads `KellyFFI.dylib` via ctypes, two JUCE copies share one address space → "pointer being freed was not allocated" abort.

### H-8. 12 `-Wno-*` warnings disabled on KellyCore — masks real bugs
**[`CMakeLists.txt:348-364`]**
`-Wno-unused-lambda-capture`, `-Wno-sign-compare`, `-Wno-overloaded-virtual`, `-Wno-reorder`, `-Wno-switch`, plus 7 others. The dangerous ones:
- `-Wno-overloaded-virtual` masks bugs where a derived class shadows a base virtual.
- `-Wno-reorder` masks member-initializer-order bugs.
- `-Wno-sign-compare` masks signed/unsigned bugs.
- `-Wno-switch` masks missing case/enum coverage.

### H-9. `include/` contaminated with CPython internals on the public include path
**[`include/cellobject.h, asdl.h, ceval.h, abstract.h, bltinmodule.h, boolobject.h, ast.h` + `target_include_directories(KellyCore PUBLIC include)`]**
Top-level `include/` contains CPython internal headers shipped IN the source tree. `target_include_directories(KellyCore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)` means every consumer of KellyCore headers gets these on their include path.

### H-10. `CMAKE_OSX_DEPLOYMENT_TARGET` mismatch between root and presets
**[`CMakeLists.txt:73` (`10.12`) ↔ `CMakePresets.json:18` (`11.0`)]**

### H-11. Hardcoded Homebrew Qt paths in `CMakePresets.json`
**[`CMakePresets.json:22`]**

### H-12. `KmiDi_FINAL` does not exist at any preset path
**[`CMakePresets.json:21` ↔ filesystem]**
Both presets reference `${sourceDir}/KmiDi_FINAL` but it doesn't exist (only inside `.worktrees/integration-finalize/`). Documented Xcode preset is silently broken.

### H-13. `_archive/KmiDi_FINAL/CMakeLists.txt` is a fully functional parallel root build with same target names
**[`_archive/KmiDi_FINAL/CMakeLists.txt`]**
562-line `project(KmiDi)` defining `KellyCore`, `KellyApp`, `KellyPlugin`, `penta_core` — same target names as live root.

### H-14. `_archive/` and orphan trees: ~4,046 files of dead/orphan native code
- `_archive/`: 3,533 first-party native files
- `KmiDi_PROJECT/`: 80 first-party native files
- `xcode/KmiDi/KmiDi_CANON/`: 406 native files
- `KmiDi-puzzles/`: 7 plugin files
- 5 orphan top-level test files (~1,700 lines)
- `common/` 9 files

### H-15. KellyCore PUBLIC links 12 JUCE modules — header-only PIMPL impossible

### H-16. `RTState` is "atomic soup" — no cross-field consistency
**[`include/penta/common/RTState.h:33-75`]**
~30 separate `std::atomic` members. Cross-field consistency impossible: a reader can see new BPM but old emotion.

### H-17. `LockFreeQueue` reference in API surface vs actual mutex backing

### H-18. `LogicBridge` embeds Python interpreter and is callable from any thread
**[`include/daiw/logic_bridge.hpp:29-225`]**

### H-19. `MemoryManager::purgeDreamState()` invalidates all pmr-allocated containers; no enforcement

### H-20. `set_source_files_properties(... COMPILE_LANGUAGES OBJCXX)` — wrong CMake property name
**[`CMakeLists.txt:282-286`]**
`COMPILE_LANGUAGES` is **not a real CMake property** (correct property is `LANGUAGE`). The set is a no-op.

### H-21. `MidiBuilder` mutex held across full file build serializes concurrent operations
**[`src/midi/MidiBuilder.cpp:13,27,116,326,330,340`]**

### H-22. `kelly_ffi.cpp` stub getters return hardcoded values
**[`src/bridge/kelly_ffi.cpp:616-704`]**
`kelly_brain_get_emotion_state` returns hardcoded `{"valence":0.0,"arousal":0.5,...}`. Plus reads `wrapper->initialized` (non-atomic `bool`) without locking → data race.

### H-23. `kelly_ffi.cpp` hand-rolled JSON parser/serializer without escaping
**[`src/bridge/kelly_ffi.cpp:74-76,117-129,236-306`]**
Strings concatenated into JSON via `<<` without escaping `"`, `\`, control chars. The parser uses substring offsets that are wrong if JSON has whitespace around the colon.

### H-24. Mixed allocator across FFI boundary — `malloc` for strings, `new` for handles
**[`src/bridge/kelly_ffi.cpp:60-74,344,354-368`]**

### H-25. FFI functions are not `noexcept`
**[`src/bridge/kelly_ffi.cpp:312-927`]**
Every `extern "C"` function in the FFI surface lacks `noexcept`.

### H-26. `intent_ir_ffi.cpp` has empty `rust_eh_personality` stub
**[`src/bridge/intent_ir_ffi.cpp:25`]**
Only safe under `panic=abort`. No CI guard verifies this.

### H-27. ODR — Dead `engine/` tree shadows live `src/engine/` headers
**[`engine/{IntentPipeline,AdapterRegistry,CoreBridge,StateMachineConductor}.{h,cpp}` ↔ `src/engine/{same names}`]**

### H-28. `_archive/KmiDi_FINAL/engine/src/plugin/` has duplicate snake_case + PascalCase variants
**[`_archive/KmiDi_FINAL/engine/src/plugin/{PluginProcessor.cpp,plugin_processor.cpp,PluginEditor.cpp,plugin_editor.cpp,vst3/PluginProcessor.cpp}`]**

### H-29. `MidiExporter::addVocalNotes` off-by-one channel ambiguity
**[`src/midi/MidiExporter.cpp:443-457`]**

### H-30. `MidiExporter::expression curve` blanket-emits CC11 to all 16 channels even unused
**[`src/midi/MidiExporter.cpp:424-440`]**

### H-31. `cpp_music_brain` (orphan) uses JUCE 7.0.9 vs root JUCE 8
**[`KmiDi_PROJECT/source/cpp/cpp_music_brain/CMakeLists.txt:60`]**

### H-32. `src/ml/MLBridge` unbounded async-thread leak
**[`src/ml/MLBridge.cpp:696-707`]**

### H-33. `ONNXInference` raw pointers via `void*` PIMPL
**[`src/ml/ONNXInference.cpp:18-42,89-110,228`]**

### H-34. `tools/midi_ci_daemon/main.cpp` silently truncates non-ASCII chars to 7 bits
**[`tools/midi_ci_daemon/main.cpp:50`]**

### H-35. C++17 vs C++20 cross-TU layout drift
**[`src_penta-core/CMakeLists.txt:4`]**

### H-36. `engine.c::s_last_error` is process-global static, not thread-local
**[`engine/src/engine.c:19,79`]**

### H-37. `engine.c::kmidi_engine_process` is empty stub — RT harness measures empty path
**[`engine/src/engine.c:59-77`]**

### H-38. `GuardrailValidator` O(N²) algorithm via density-recompute-in-loop
**[`common/GuardrailValidator.h:167-170`]**
`while (!midi.chords.empty() && density(midi) > auth.maxRhythmicDensity)` — `density()` is O(N), called inside an O(N) loop. Algorithmic DoS.

### H-39. `xcode/.../body/plugin/PluginProcessor.cpp:1-62` — "two type systems same namespace" dead reconciliation cruft

### H-40. `final_kel/Body/onnx_runtime/JepaRunner.cpp:48-64` — `infer()` always throws

---

## SECTION 3 — RAII & Lifetime Violations

### R-1. `KellyBrain` pimpl destructor `= default` in header — see C-43
### R-2. `BiometricInput` destructor order — bridges deleted before streaming stop — see C-36
### R-3. `BiometricInput` `void*` storage of class instances with conflicting forward decl
**[`src/biometric/BiometricInput.h:150-154`]**
### R-4. `BiometricInput::streamingThread_` member declared but never started/joined
**[`src/biometric/BiometricInput.h:157`]**
### R-5. `MidiGenerator::generate()` returns pointer to `static thread_local` — see C-19
### R-6. `MidiIO::CallbackWrapper` holds back-pointer to owner; member-destruction order fragile
**[`src/midi/MidiIO.cpp:7-12,50`]**
### R-7. `final_kel/Body/onnx_runtime/JepaRunner.cpp:23-34` — `void*` member as heap owner
### R-8. `daiw::memory_pool` destructor walks pool and double-frees user objects
**[`include/daiw/memory_pool.hpp:36-43`]**
### R-9. `RTPoolPtr` move constructor lacks exception guard around placement new
**[`include/penta/common/RTMemoryPool.h:55-96`]**
### R-10. `MemoryManager::purgeDreamState` invalidates all pmr-allocated containers — see H-19
### R-11. JUCE Component dtor missing `setLookAndFeel(nullptr)` and `stopTimer()` — see C-38
### R-12. `ScoreEntryPanel` member declaration order causes UAF — see C-40
### R-13. `RTLogger::g_logger` non-thread-safe singleton + leak — see C-34
### R-14. Zero `JUCE_LEAK_DETECTOR` macros across `xcode/KmiDi/KmiDi_CANON` (and likely `src/`)

---

## SECTION 4 — Threading Risks

### T-1. Python C API without GIL — see C-6
### T-2. Detached thread captures `this` raw — see C-7
### T-3. RTState atomic-soup — see H-16
### T-4. Reader-only seqlock with no writer — see C-11
### T-5. `RTMemoryPool` ABA hazard — see C-16
### T-6. `daiw::memory_pool` ABA window in release CAS — see C-17
### T-7. `daiw::MPSCQueue` slot-reuse race in publish/consume cycle
**[`include/daiw/lock_free_queue.hpp:139-159,177`]**
### T-8. `LockFreeRingBuffer::availableToRead()` mixed memory order
**[`xcode/.../body/ml/LockFreeRingBuffer.h:91-94` + `src/ml/LockFreeRingBuffer.h:27-89`]**
### T-9. `MidiIO::callback_` set from main thread, read from audio thread, no sync — see R-6
### T-10. `MidiBuilder` mutex on RT-touched class held across full build — see H-21
### T-11. `BlockLatencyInstrument` race between RT recorder and stats reader
**[`include/penta/diagnostics/BlockLatencyInstrument.h:86-107,117-124`]**
### T-12. `PerformanceMonitor` declares `std::vector<std::atomic<uint64_t>>` (atomics not movable)
**[`include/penta/diagnostics/PerformanceMonitor.h:52`]**
### T-13. `BiometricInput::dataHistory_` mutated without lock from streaming thread
### T-14. `kelly_ffi.cpp` callback dispatched while holding wrapper mutex
**[`src/bridge/kelly_ffi.cpp:677-684,736,742`]**
### T-15. `kelly_brain_is_initialized` reads `wrapper->initialized` (non-atomic bool) without mutex
**[`src/bridge/kelly_ffi.cpp:395`]**
### T-16. `ParameterMorphEngine::getCurrentValue()` takes mutex
**[`src/engine/ParameterMorphEngine.cpp:14-86`]**
### T-17. `BridgeClient::voiceProcessor_` raw pointer cross-thread
### T-18. `WavetableSynth::currentWavetable_` shared_ptr swap during audio playback — see C-41
### T-19. `GlottalSource::process` function-local static across instances — see C-42
### T-20. `PitchTracker` `mutable` member buffers shared across concurrent callers
**[`src/prrot/PitchTracker.h:87-88`]**
### T-21. `SpectralAnalyzer` `mutable` member buffers same race surface
### T-22. `OSCHub::callbacks_` mutated without synchronization
**[`src_penta-core/osc/OSCHub.cpp:50-77`]**
### T-23. `GrooveEngine::analysis_.onsetPositions` written from RT, no documented reader sync
### T-24. `RTMemoryPool::getAvailableBlocks` walks free list while writers mutate
### T-25. `random_device` + `mt19937` constructed per call from RT-callable functions

---

## SECTION 5 — Real-Time Safety

### RT-1. `HarmonyEngine::suggestVoiceLeading(...) noexcept` returns `std::vector`
**[`include/penta/harmony/HarmonyEngine.h:54-56`]**

### RT-2. `VoiceLeading::findOptimalVoicing(...) noexcept` returns `std::vector`
**[`include/penta/harmony/VoiceLeading.h:34,67,73`]**

### RT-3. PRROT FFT path heap allocation in noexcept — see C-32

### RT-4. `PRROTEngine::processAudioSegment` 8+ allocations per call — see C-33

### RT-5. `RTMessageQueue::push(OSCMessage) noexcept` copies value type containing `std::string` + `std::vector<variant>`
**[`src_penta-core/osc/RTMessageQueue.cpp:17-28`]**

### RT-6. `RTNeuralProcessor::process` not noexcept, calls `juce::Logger::writeToLog` from RT path
**[`src/ml/RTNeuralProcessor.cpp:75-101`]**

### RT-7. `InputValidation` and `AudioValidator` `noexcept` build heap strings unconditionally

### RT-8. `AudioWorkerThread::WorkFn` is `std::function<void()>`
**[`include/penta/rt/AudioWorkerThread.h:33,111`]**

### RT-9. `AudioBuffer::getChannelData()` uses `assert()` for OOB protection
**[`include/penta/common/RTTypes.h:79-86`]**

### RT-10. `AudioBuffer::resize` allocates without RT marking

### RT-11. `engine.c::kmidi_engine_process` is empty stub — see H-37

### RT-12. `MasterEQProcessor::processBlock` Coefficients `Ptr` ref-count race
**[`src/plugin/MasterEQProcessor.cpp:131-139,158`]**

### RT-13. `PluginProcessor::processBlock` heavy work + ML inference + EQ + parameter atomics
**[`src/plugin/PluginProcessor.cpp:401-742`]**

### RT-14. `plugins/plugin/PluginProcessor::processBlock` blocking lock_guard
**[`plugins/plugin/PluginProcessor.cpp:483-526`]**

### RT-15. `MidiBuilder::buildMidiBuffer` allocates and locks on audio path
**[`src/midi/MidiBuilder.cpp:113-272`]**

### RT-16. `GrooveEngine::applyGroove` no floor clamp on shifted note positions
**[`src/midi/GrooveEngine.cpp:53-84`]**

### RT-17. `engine_rt.cpp:20` SPSC queue used without documented producer/consumer

### RT-18. `VoiceProcessor::queuePhonemes` racing audio — see C-37

### RT-19. `VoiceSynthesizer::synthesizeAudio` per-sample heap allocation in offline path
**[`src/voice/VoiceSynthesizer.cpp:297-303`]**

### RT-20. `VocoderEngine::processSample` per-sample formant filter coefficient recompute
**[`src/voice/VocoderEngine.cpp:226-232`]**

### RT-21. `PRROTEngine` float→size_t cast without NaN/Inf guard
**[`src/prrot/PRROTEngine.cpp:107-114`]**

### RT-22. `AudioWorkerThread::run()` busy yield loop
**[`include/penta/rt/AudioWorkerThread.h:68-72`]**

### RT-23. `F0Extractor::diffBuffer_::resize` in pitch-extraction path
**[`src/audio/F0Extractor.cpp:99-104`]**

### RT-24. `SpectralAnalyzer::computeSTFT` per-frame allocation
**[`src/audio/SpectralAnalyzer.cpp:91,121,198-207`]**

---

## SECTION 6 — Cross-Package / Monorepo Issues

### X-1. The 3-tree harmony/groove/osc ODR with diverged content (the headline bug) — see C-1
### X-2. `KellyFFI` allocator mismatch vs `bindings/penta_core_native` vs plugin builds — three JUCE entry points — see H-6, H-7, H-15
### X-3. `include/` contains CPython internals on the public include path — see H-9
### X-4. 3 daiw_core targets, 3 penta_core targets, 4 KmiDiCore variants — see H-3, H-4, H-13
### X-5. `src_penta-core/CMakeLists.txt` C++17 vs root C++20 cross-TU layout drift — see H-5, H-35
### X-6. `KmiDi_PROJECT/` 80-file orphan tree is naively buildable as standalone
### X-7. `engine/` (C lib) ↔ `src/engine/` (C++ lib) namespace + path collision — see H-27
### X-8. `BiometricInput.cpp` AND `.mm` both in glob → duplicate symbols — see C-5
### X-9. Missing CMake `EXCLUDE REGEX` for `harmony/`, `groove/`, `osc/` — see H-2
### X-10. `KmiDi_FINAL` doesn't exist at any preset path — see H-12
### X-11. JUCE 7.0.9 vs JUCE 8 in same monorepo — see H-31
### X-12. Two C ABI surfaces — `c/` standalone vs `src/bridge/kelly_ffi.h`
### X-13. Plugin compile defs applied twice (KellyPlugin and KellyPlugin_VST3)
**[`CMakeLists.txt:431-440`]**
### X-14. Stale build artifact `src/engines/VoiceLeading.cpp.o` for a deleted source file
### X-15. vst3 stub remnants — partially deleted (per memory)
### X-16. Worktree at `.worktrees/integration-finalize/` has full duplicated tree

---

## SECTION 7 — Performance Concerns

### P-1. `KELLY_CORE_SOURCES` is one giant monolithic translation set with 12 warning suppressions
### P-2. `GuardrailValidator::OutputPlanValidator::validate` O(N²) — see H-38
### P-3. `engine.c::kmidi_engine_process` empty stub — RT harness measures nothing — see H-37
### P-4. `ChordAnalyzerSIMD` `vBestScore` computed but never extracted
**[`src_penta-core/harmony/ChordAnalyzerSIMD.cpp:88-131`]**
### P-5. `ChordAnalyzerSIMD` per-template inner work is scalar despite SIMD framing
### P-6. `VocoderEngine` filter coefficients per sample — see RT-20
### P-7. `AudioEmotionRunner` 131 K scalar mults per inference, no SIMD
**[`src/ml/AudioEmotionRunner.cpp:218-222`]**
### P-8. `OnsetDetector::std::log2(config_.fftSize)` for integer log2
**[`src_penta-core/groove/OnsetDetector.cpp:36`]**
### P-9. `MidiSequence` `std::map<uint16_t, std::vector<MidiMessage>>` for active-note tracking
**[`src/midi/MidiSequence.cpp:17-45`]**
### P-10. `MidiExporter` rebuilds file by copying every event to insert lyric meta events
**[`src/midi/MidiExporter.cpp:68-102,140-152`]**
### P-11. `MidiExporter::expression curve` blanket 15-channel emission — see H-30
### P-12. `GrooveEngine` full `std::sort` on every call
**[`src/midi/GrooveEngine.cpp:105-109`]**
### P-13. `StateMachineConductor.h` 36 KB header-only `inline` logic
### P-14. `kelly_ffi.cpp` hand-rolled JSON serialization — see H-23
### P-15. `prrot_bindings::def_readonly` on `std::vector` returns by-copy on every property access
**[`bindings/prrot_bindings.cpp:120-126,116-119`]**
### P-16. `MLBridge::spawnAsyncTask` `std::function` capture allocations + thread creation per spawn — see H-32
### P-17. `OSCHub::matchPattern` recursion for each pattern — DoS via crafted addresses
**[`src_penta-core/osc/OSCHub.cpp:79-112`]**
### P-18. `MusicTheoryBrain::analyzeMIDI` ostringstream + per-chord/per-concept string concatenation
### P-19. `SpectralAnalyzer::calculateSpectralFlatness` underflows to zero
**[`src/audio/SpectralAnalyzer.cpp:324-345`]**
### P-20. `VADSystem::vector::erase(begin())` instead of deque
**[`src/engine/VADSystem.cpp:58-60`]**
### P-21. `F0Extractor::calculateDifferenceFunction` O(N×maxLag) with no SIMD
**[`src/audio/F0Extractor.cpp:124-137`]**
### P-22. `c/src/musical_intent.c` per-component `strdup` across copy paths

---

## SECTION 8 — Suggested Fixes

### Tier 0 — Critical (apply first; unblocks future fixes)

(See `whimsical-stargazing-anchor.md` plan file for the 24 in-place safety fixes scheduled for this session.)

The complete Tier-0 list includes 24 in-place fixes (no deletions), addressing the worst crash/UB primitives, the noexcept-lying RT functions, the GIL-less Python bridges, the BiometricInput dtor order bug, the KellyBrain pimpl, the MidiGenerator thread_local pointer escape, and the hardcoded developer paths. These are scheduled for application in this session as Phase 3.

### Tier 1 — High priority (deferred to future sessions)

Quarantine orphan trees, delete stale duplicates after divergence triage, replace `RTMemoryPool` with ABA-safe variant, fix `daiw::MPSCQueue::size_approx`, fix `daiw::audio_io.hpp` dead-on-arrival, use `juce::Component::SafePointer<>` for all async callbacks, fix Euclidean modulo for pitch class arithmetic, sanitize WAV chunk sizes, fix `RTLogger` Meyers singleton, fix `MasterEQProcessor` Coefficients race, replace `kelly_ffi.cpp` JSON parser with `nlohmann::json`, fix `MLBridge` thread leak, enforce Rust `panic=abort`, fix `MidiBuilder` mutex hold, fix `OSCHub` callbacks_ unsynchronized access, fix `BiometricInput` non-atomic shared state, fix `kelly_brain_get_*` stubs, fix `WavetableSynth::loadWavetable` shared_ptr swap.

### Tier 2 — Medium priority (hygiene)

Make `AudioBuffer::getChannelData()` runtime-checked, replace `daiw::memory_pool::release` with documented SPSC + thread-id assertion, replace `BlockLatencyInstrument::history_` with atomic array, replace `PerformanceMonitor::latencyHistory_` `std::vector<atomic>` with `std::array`, make `engine.c::s_last_error` `thread_local`, fix `MusicTheoryBrain::analyzeMIDI` string concatenation, add `JUCE_LEAK_DETECTOR`, fix `tools/midi_ci_daemon/main.cpp` UTF-8, delete `common/` 4-line stub files, remove hardcoded Homebrew Qt paths from `CMakePresets.json`, add Ninja preset, reconcile `CMAKE_OSX_DEPLOYMENT_TARGET`, fix `set_source_files_properties COMPILE_LANGUAGES`, replace `pythonbindings::def_readonly` on vectors with `def_property_readonly`.

---

## Top 10 urgent items

| # | Item | Section | Blast radius |
|---|------|---------|--------------|
| 1 | Diverged ODR — 4/5 harmony, 4/4 groove, 5/5 osc files in two libs both linked into Python module | C-1 | Silent semantic drift in shipped binaries; fix collapses ~10 issues |
| 2 | Two `kelly::Wound`/`EmotionNode`/`IntentResult`/`RuleBreakType` types in same namespace | C-21 | Persistent miscompiles; "worst single bug" per Group A |
| 3 | Python C API called without GIL across entire bridge layer + xcode CANON | C-6 | Crash, interpreter corruption, UB |
| 4 | `BiometricInput.cpp` AND `.mm` both compiled — duplicate symbols / ODR | C-5 | Link failure or runtime ODR |
| 5 | `KellyTests` references 4 nonexistent files — `BUILD_TESTS=ON` broken | C-4 | Zero C++ test coverage of shipped code |
| 6 | `KellyFFI` "no double JUCE" rule defeated by KellyCore PUBLIC linkage; Python bindings bring JUCE into Python process | H-6, H-7 | Allocator-mismatch crash (the documented project failure mode) |
| 7 | `HarmonyEngine::activeNotes_[note.pitch]` OOB on pitch≥128 + multiple noexcept-allocate functions in RT path | C-25, RT-1..7 | UB; possible RCE primitive in plugin context |
| 8 | `engine_rt.cpp` declares C ABI symbols never compiled; reader-only seqlock with no writer | C-10, C-11 | Phantom ABI; RT state always returns zero |
| 9 | `include/` contains CPython internals on PUBLIC include path | H-9 | Cross-language header collision |
| 10 | `src/plugin/PluginProcessor.cpp` ships hardcoded `/Users/seanburdges/...` paths | C-9 | Plugin broken on every machine except dev's; dev username leaked in shipped binary |

**Two dominant root causes account for ~70% of these:** (1) the harmony/groove/osc consolidation never finished, and (2) JUCE/Qt PUBLIC linkage on KellyCore defeats the entire FFI isolation strategy.

---

## UNKNOWNS

1. Whether `BiometricInput.cpp` or `BiometricInput.mm` actually wins at link time on the dev machine. Needs clean build verification.
2. Whether the Rust crate at `bridges/intent_ir_rust/` is built with `panic=abort`.
3. Whether `penta::ml::AudioEmotionRunner::pushSamples`/`updateParams` (called from `processBlock`) are RT-safe.
4. Whether `MasterEQProcessor`, `PluginLatencyManager`, `InferenceThreadManager`, `MultiModelProcessor`, `MLFeatureExtractor` allocate inside `processBlock`.
5. Whether `CIntentFrame` C struct layout matches Rust `CIntentFrame` exactly.
6. Whether the standalone `c/` library is actually consumed by anything outside its own `examples/` directory.
7. Whether the older `plugins/plugin/PluginProcessor.cpp` is still referenced in any build target.
8. **Whether `src/ui/EmotionWorkstation.cpp` and the other src/ui mirrors of the audited xcode files have the same JUCE Component lifetime bugs.** Highly likely — re-audit recommended.
9. Whether `tests/runtime_contract_tests.cpp` (1089 lines) and `tests/state_machine_conformance_tests.cpp` (221 lines) at the top level are worth resurrecting.
10. Exact runtime-call topology of `HarmonyEngine::suggestVoiceLeading` / `VoiceLeading::findOptimalVoicing`.
11. Whether `LogicBridge` (the embedded Python in `daiw/logic_bridge.hpp`) has any actual call sites or is only declared.
12. Whether `juce::Random` (used in `FormantSynthVoice::random_`) has RT-safety guarantees on the JUCE 8 version pinned in `external/JUCE`.
13. Whether `KellyBrainLegacy` from `Kelly.h` is built into `KellyCore`.
14. Whether the writer for `RTState` (the seqlock writer side) exists in `src-tauri/` (Rust) or somewhere outside the audited C++ tree.
15. Whether `MIDI_CHANNEL_MELODY` (in `common/MusicConstants.h`) is 0- or 1-indexed.
16. ARM build status — per CMakeLists `-mavx2` is conditionally added; on Apple Silicon `__AVX2__` should be undefined and **both** ChordAnalyzer scalar fallbacks compile → expected link error.

---

## Coverage map

**Audited at file:line precision:**
- **Group A** (RT/audio core, 184 files): KellyCore, daiw_core, audio engine, lock-free queues, ring buffers, RT state, F0 extractor, spectral analyzer, audio file reader, IntentIR, RTState, AudioWorkerThread, BlockLatencyInstrument, PerformanceMonitor, KellyBrain, IntentPipeline, IntentProcessor, several engine/* files
- **Group B** (DSP/ML/voice, 127 files): src_penta-core, harmony, groove, osc, ml, KellyML, prrot, voice — including byte-level diffs of harmony/ between src/ and src_penta-core/ confirming divergence
- **Group C** (FFI/plugins/ABI, 82 files): src/bridge/* (kelly_ffi, intent_ir_ffi, OSCBridge, all 8 Python bridges), src/plugin, src/biometric, plugins/, bindings/, c/ (full standalone C lib)
- **Group D** (KmiDi_PROJECT + legacy, 123 files): KmiDi_PROJECT/source/cpp/{src,src_penta-core,cpp_music_brain}, engine/, legacy/, final_kel/, src/midi, src/project, _archive/KmiDi_FINAL
- **Group E** (xcode CANON + UI, 509 files): xcode/KmiDi/KmiDi_CANON/body/{plugin, music_theory, audio, midi, engines, voice, biometric, ml, ui, harmony, bridge, common, brain/music_brain}. **Established via direct evidence that the xcode tree is orphan source — NOT a third build context, NOT shipped.**
- **Main thread** (build system + small partitions): root CMakeLists.txt, all 8 first-party CMakeLists files, CMakePresets.json, common/, KmiDi-puzzles/, tests/cpp/test_kelly_ffi.cpp, tools/midi_ci_daemon/main.cpp, RTTypes.h, HarmonyEngine.h, byte-level diffs of 5 harmony files between src/ and src_penta-core/

**Not audited at file:line precision (UNKNOWN status):**
- `src/ui/*.cpp` (57 files) — the actually-built UI code that mirrors the orphan xcode CANON UI files. **Highest-priority follow-up.**
- `src/ml/MultiModelProcessor`, `MelSpectrogram`, `MIDITokenizer`, `MLFeatureExtractor`, `NodeMLMapper`, `DDSPProcessor` — control-thread ML inference paths
- Most `src/voice/*.cpp` past `VoiceSynthesizer` and `VocoderEngine`
- Most `src/engines/*.cpp` (BassEngine, MelodyEngine, RhythmEngine)
- `src/KellyML/*` — not read
- `src/bridge/StateBridge.cpp`, `EngineIntelligenceBridge.cpp`, `PreferenceBridge.cpp`, `SuggestionBridge.cpp` — confirmed via grep to share the GIL anti-pattern; bodies not read line-by-line
- `bridges/intent_ir_rust/` — Rust crate, out of C++ scope
- `_archive/KmiDi_FINAL/engine/` past CMakeLists and one vst3 stub
- `external/JUCE/` — third party
- `src-tauri/` Rust frontend code — out of scope

---

*End of audit. See `/Users/seanburdges/.claude/plans/whimsical-stargazing-anchor.md` for the action plan and `docs/SALVAGE_CATALOG_2026Q2.md` for the per-file divergence disposition.*
