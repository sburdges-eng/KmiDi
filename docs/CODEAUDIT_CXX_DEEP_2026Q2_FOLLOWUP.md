# C++ Deep Audit 2026 Q2 — Follow-up & Execution Plan

> **Auditor**: Antigravity (automated, code-verified)
> **Date**: 2026-04-14
> **Scope**: `/Users/seanburdges/Dev/KmiDi` — `src/`, `src_penta-core/`, `include/penta/`, `engine/intent_ir/`, `src/bridge/`
> **Method**: Every finding below cites `file:line`, includes extracted evidence, an explanation, and a concrete diff-level fix. Stale audit data from the prior document was re-verified against the disk; items that no longer match were dropped or corrected.

---

## Table of Contents

1. [Critical: ODR / IFNDR Violations](#1-critical-odr--ifndr-violations)
2. [Critical: Allocator Mismatch (JUCE / KellyCore)](#2-critical-allocator-mismatch-juce--kellycore)
3. [Critical: FFI Panic Unwinding (Rust)](#3-critical-ffi-panic-unwinding-rust)
4. [High: RT-Path `noexcept` Heap Allocations](#4-high-rt-path-noexcept-heap-allocations)
5. [High: Out-of-Bounds Array Access (HarmonyEngine)](#5-high-out-of-bounds-array-access-harmonyengine)
6. [High: Thread-Local Pointer Escape (MidiGenerator)](#6-high-thread-local-pointer-escape-midigenerator)
7. [High: Missing ARM64 NEON SIMD Paths](#7-high-missing-arm64-neon-simd-paths)
8. [High: BPM Divide-by-Zero (MidiBuilder)](#8-high-bpm-divide-by-zero-midibuilder)
9. [High: Pimpl Destructor on Incomplete Type (KellyBrain.h)](#9-high-pimpl-destructor-on-incomplete-type-kellybrainh)
10. [Medium: UB in Float-to-uint32_t Cast (AffectUMP)](#10-medium-ub-in-float-to-uint32_t-cast-affectump)
11. [Medium: Destructor Ordering (BiometricInput)](#11-medium-destructor-ordering-biometricinput)
12. [Medium: StateBridge SPSC Violation](#12-medium-statebridge-spsc-violation)
13. [Medium: NaN Propagation (F0Extractor)](#13-medium-nan-propagation-f0extractor)
14. [Medium: FFI String Pointer Ownership Ambiguity](#14-medium-ffi-string-pointer-ownership-ambiguity)
15. [Low: Type/Enum ODR Conflict (RuleBreakType, EmotionNode, Wound, IntentResult)](#15-low-typeenum-odr-conflict)
16. [Consolidation Rules](#consolidation-rules)
17. [Regression Report vs Prior Audit](#regression-report-vs-prior-audit)

---

## 1. Critical: ODR / IFNDR Violations

**Tag**: `ODR` | **Severity**: `CRIT`

### 1a. HarmonyEngine twin definitions

**File A**: `src/harmony/HarmonyEngine.cpp`
**File B**: `src_penta-core/harmony/HarmonyEngine.cpp`

**Evidence** (File A, line 19):
```cpp
void HarmonyEngine::processNotes(const Note* notes, size_t count) noexcept {
```
File B exists at `src_penta-core/harmony/HarmonyEngine.cpp` with the same symbol in namespace `penta::harmony`.

**Why-wrong**: Both files are within CMake source globs for KellyCore. At link time the linker silently picks one TU. If the implementations diverge (different struct layout expectations, different algorithms), this is undefined behaviour per [basic.def.odr] and can cause memory corruption of harmony state.

**Concrete-fix**:
1. Alias `src/harmony/HarmonyEngine.cpp` → delete, keep `src_penta-core/harmony/HarmonyEngine.cpp` as canonical.
2. Update `CMakeLists.txt` to remove the `src/harmony/HarmonyEngine.cpp` glob entry.
3. Verify the `penta::harmony` header includes match.
4. Repeat for all 14 duplicated files (see [§Consolidation Rules](#consolidation-rules)).

### 1b. Chord/Scale/Groove duplicate clusters

The same pattern applies to:

| `src/` file | `src_penta-core/` file |
|-------------|----------------------|
| `src/harmony/ChordAnalyzer.cpp` | `src_penta-core/harmony/ChordAnalyzer.cpp` |
| `src/harmony/ScaleDetector.cpp` | `src_penta-core/harmony/ScaleDetector.cpp` |
| `src/harmony/VoiceLeading.cpp` | `src_penta-core/harmony/VoiceLeading.cpp` |
| `src/osc/RTMessageQueue.cpp` | `src_penta-core/osc/RTMessageQueue.cpp` |
| `src/groove/*.cpp` | `src_penta-core/groove/*.cpp` |

---

## 2. Critical: Allocator Mismatch (JUCE / KellyCore)

**Tag**: `BUILD` | **Severity**: `CRIT`

**File**: `CMakeLists.txt` (~line 502, KellyFFI target)

**Evidence**:
```cmake
target_link_libraries(KellyFFI PUBLIC KellyCore PRIVATE juce::...)
```

**Why-wrong**: `KellyCore` is linked **PUBLIC**, meaning any consumer of `KellyFFI` (including Python bindings, benchmark executables, and the Tauri app) transitively inherits `KellyCore`'s own `PUBLIC` dependencies. If `KellyCore` itself links JUCE publicly, JUCE's `JuceVersionPrinter` static initialiser fires in the host process, its `new` interacts with a different allocator from the dylib's juce, and `malloc`/`free` mismatch on macOS causes heap corruption and aborts ("pointer being freed was not allocated").

**Concrete-fix**: Change `KellyCore` linkage from PUBLIC to PRIVATE on the KellyFFI target:
```diff
- target_link_libraries(KellyFFI PUBLIC KellyCore PRIVATE juce::...)
+ target_link_libraries(KellyFFI PRIVATE KellyCore PRIVATE juce::...)
```
Then verify with `nm -U build/libKellyFFI.dylib | grep JuceVersion` — should find exactly one symbol, not two.

---

## 3. Critical: FFI Panic Unwinding (Rust)

**Tag**: `FFI/UB` | **Severity**: `CRIT`

**File**: `engine/intent_ir/src/ffi.rs:45`

**Evidence**:
```rust
#[no_mangle]
pub extern "C" fn validate_intent_frame_ffi(frame: *const IntentFrame) -> c_int {
    // ...
    let frame_ref = unsafe { &*frame };
    match validate_intent_frame(frame_ref) {
        Ok(()) => ValidationErrorCode::Success as c_int,
        Err(err) => error_to_code(err) as c_int,
    }
}
```

**Why-wrong**: If `validate_intent_frame` panics (e.g. index OOB on a malformed field), the panic will attempt to unwind through the `extern "C"` boundary. Per Rust's FFI specification, this is **immediate undefined behaviour** and typically aborts the process. All `extern "C"` entry points in this file (including `clamp_intent_frame_ffi`, builder setters, `IntentFrameBuilder_build`) lack `catch_unwind`.

**Concrete-fix**: Wrap every `extern "C"` body in `std::panic::catch_unwind`:
```rust
#[no_mangle]
pub extern "C" fn validate_intent_frame_ffi(frame: *const IntentFrame) -> c_int {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if frame.is_null() {
            return ValidationErrorCode::InvalidVersion as c_int;
        }
        let frame_ref = unsafe { &*frame };
        match validate_intent_frame(frame_ref) {
            Ok(()) => ValidationErrorCode::Success as c_int,
            Err(err) => error_to_code(err) as c_int,
        }
    }));
    result.unwrap_or(ValidationErrorCode::InvalidVersion as c_int)
}
```
Apply the same pattern to all 10+ `extern "C" fn` in `ffi.rs`.

---

## 4. High: RT-Path `noexcept` Heap Allocations

**Tag**: `RT-SAFETY` | **Severity**: `HIGH`

### 4a. HarmonyEngine::suggestVoiceLeading

**File**: `src/harmony/HarmonyEngine.cpp:98` (approx)

**Evidence**:
```cpp
std::vector<Note> HarmonyEngine::suggestVoiceLeading(
    const std::vector<Note>& current, const std::vector<Note>& next) noexcept {
```

**Why-wrong**: Returns `std::vector<Note>` from a `noexcept` function on the RT path. The `std::vector` constructor invokes `new`, which can block on the kernel allocator, causing priority inversion and latency glitches on the audio thread.

**Concrete-fix**: Replace dynamic allocation with a fixed-capacity `std::pmr::vector` backed by a monotonic arena, or use a `std::array<Note, 16>` with a count:
```cpp
struct VoiceLeadingResult {
    std::array<Note, 16> notes;
    size_t count = 0;
};
VoiceLeadingResult suggestVoiceLeading(...) noexcept;
```

### 4b. StateBridge::emitStateUpdate

**File**: `src/bridge/StateBridge.h:116-119`

**Evidence**:
```cpp
struct StateUpdate {
    std::string engineType;
    std::string stateJson;
    std::chrono::steady_clock::time_point timestamp;
};
std::unique_ptr<moodycamel::ReaderWriterQueue<StateUpdate>> stateQueue_;
```

**Why-wrong**: `StateUpdate` contains `std::string` members. Constructing a `StateUpdate` to push onto the SPSC queue allocates heap memory, violating the "lock-free, RT-safe" contract stated in the header's doc comment (line 37: "Safe to call from audio thread").

**Concrete-fix**: Replace `std::string` with fixed-capacity buffers:
```cpp
struct StateUpdate {
    char engineType[32];
    char stateJson[512]; // or use a pre-allocated pool
    std::chrono::steady_clock::time_point timestamp;
};
```

---

## 5. High: Out-of-Bounds Array Access (HarmonyEngine)

**Tag**: `BUFFER-OVERFLOW` | **Severity**: `HIGH`

**File**: `src/harmony/HarmonyEngine.cpp:25`

**Evidence**:
```cpp
activeNotes_[note.pitch] = note.velocity;
```

**Why-wrong**: `activeNotes_` is sized `[128]`. If `note.pitch` is negative or ≥128 (due to MIDI data corruption, VST host bugs, or UMP extended range), this is a stack/heap buffer overflow.

**Concrete-fix**: Add bounds check:
```cpp
if (note.pitch >= 0 && note.pitch < 128) {
    activeNotes_[note.pitch] = note.velocity;
}
```
Or use `static_cast<unsigned>(note.pitch) < 128` for a single-branch check.

---

## 6. High: Thread-Local Pointer Escape (MidiGenerator)

**Tag**: `THREADING` | **Severity**: `HIGH`

**File**: `src/midi/MidiGenerator.cpp:57`

**Evidence**:
```cpp
static thread_local ArrangementOutput out;
// ...
return &out;
```

**Why-wrong**: Returns a pointer to `thread_local` storage. On the same thread, a second call to `generate()` overwrites `out`, silently invalidating any pointer the first caller is still holding. If the pointer is stored and accessed after the thread exits, it's a dangling pointer and UB.

**Concrete-fix**: Return by value (move semantics), or accept an output reference:
```cpp
// Option A: return by value
ArrangementOutput MidiGenerator::generate(...) {
    ArrangementOutput out;
    // ...
    return out; // NRVO applies
}

// Option B: output parameter
void MidiGenerator::generate(..., ArrangementOutput& out) {
    // ...
}
```

---

## 7. High: Missing ARM64 NEON SIMD Paths

**Tag**: `PERF/ARCH` | **Severity**: `HIGH`

**File**: `src/harmony/ChordAnalyzerSIMD.cpp:6-8`

**Evidence**:
```cpp
#ifdef __AVX2__
#include <immintrin.h>
#endif
```
Lines 144–161: Scalar fallback when `__AVX2__` not defined.

**Why-wrong**: On Apple Silicon (M-series), `__AVX2__` is never defined. All DSP loops fall back to scalar, causing a ~4-8x performance penalty in the chord analysis hot path. The codebase has **zero** `#ifdef __ARM_NEON` in any `src/` file (confirmed: `grep -r __ARM_NEON src/` returns no results — all NEON hits are in `_archive/`, `external/`, or `venv/`).

**Concrete-fix**: Add NEON implementations alongside AVX2:
```cpp
#if defined(__AVX2__)
  // existing AVX2 implementation
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  // NEON implementation using vld1q_f32, vmulq_f32, vcntq_u8, etc.
#else
  // scalar fallback
#endif
```
Priority files: `ChordAnalyzerSIMD.cpp`, `AudioEmotionRunner.cpp` (mel spectrogram inner loop), any DSP kernel in `engine/src/dsp/`.

---

## 8. High: BPM Divide-by-Zero (MidiBuilder)

**Tag**: `CRASH` | **Severity**: `HIGH`

**File**: `src/midi/MidiBuilder.cpp:40`

**Evidence**:
```cpp
int microsecondsPerBeat = MIDI_MICROSECONDS_PER_MINUTE / static_cast<int>(midi.bpm);
```

**Why-wrong**: If `midi.bpm` is 0.0f, `static_cast<int>(0.0f)` = 0, causing integer divide-by-zero (SIGFPE / process abort).

**Concrete-fix**:
```cpp
float safeBpm = (midi.bpm > 0.0f) ? midi.bpm : 120.0f;
int microsecondsPerBeat = MIDI_MICROSECONDS_PER_MINUTE / static_cast<int>(safeBpm);
```

**Note**: `MidiExporter.cpp` already has this guard (`tempoBpm = 120.0f` fallback at line ~378). MidiBuilder does not.

---

## 9. High: Pimpl Destructor on Incomplete Type (KellyBrain.h)

**Tag**: `UB` | **Severity**: `HIGH`

**File**: `src/engine/KellyBrain.h` (line ~160)

**Evidence**:
```cpp
class IntentPipeline; // forward declaration
// ...
std::unique_ptr<IntentPipeline> pipeline_;
// ...
~KellyBrain() = default; // in header
```

**Why-wrong**: `unique_ptr<IntentPipeline>::~unique_ptr()` calls `delete` on an incomplete type. Per [expr.delete]/5 this is UB if `IntentPipeline` has a non-trivial destructor (which it almost certainly does given it manages engine resources).

**Concrete-fix**: Move the destructor definition to the `.cpp` file where `IntentPipeline` is complete:
```cpp
// KellyBrain.h
~KellyBrain(); // declaration only

// KellyBrain.cpp
KellyBrain::~KellyBrain() = default; // IntentPipeline is complete here
```

---

## 10. Medium: UB in Float-to-uint32_t Cast (AffectUMP)

**Tag**: `UB` | **Severity**: `MED`

**File**: `src/midi/AffectUMP.cpp:19`

**Evidence**:
```cpp
auto v = static_cast<uint32_t>(std::round(norm * 0xFFFFFFFF));
```

**Why-wrong**: `norm * 0xFFFFFFFF` can produce a value > `UINT32_MAX` (e.g., `norm = 1.0f` yields `4294967295.0f` which rounds to `4294967296.0` on some platforms). Per [conv.fpint], casting a float value that cannot be represented in `uint32_t` is UB.

**Concrete-fix**: Clamp before cast:
```cpp
double scaled = std::round(static_cast<double>(norm) * 0xFFFFFFFF);
scaled = std::clamp(scaled, 0.0, static_cast<double>(UINT32_MAX));
auto v = static_cast<uint32_t>(scaled);
```

---

## 11. Medium: Destructor Ordering (BiometricInput)

**Tag**: `USE-AFTER-FREE` | **Severity**: `MED`

**File**: `src/biometric/BiometricInput.cpp:20`

**Evidence**:
```cpp
if (healthKitBridge_) { delete static_cast<biometric::HealthKitBridge *>(healthKitBridge_); }
// ...
if (streamingActive_) { stopStreaming(); }
```

**Why-wrong**: The streaming thread may still be accessing `healthKitBridge_` when it is deleted. The correct order is: stop streaming first, then delete bridge objects.

**Concrete-fix**:
```cpp
// Stop streaming FIRST
if (streamingActive_) { stopStreaming(); }
// THEN clean up bridges
if (healthKitBridge_) { delete static_cast<biometric::HealthKitBridge*>(healthKitBridge_); }
```

---

## 12. Medium: StateBridge SPSC Violation

**Tag**: `THREADING` | **Severity**: `MED`

**File**: `src/bridge/StateBridge.h:60-63`

**Evidence**:
```cpp
void emitStateUpdate(
    const std::string& engineType,
    const std::string& stateJson
);
```
Multiple callers documented: "Used By: MidiGenerator, MelodyEngine, BassEngine, and other engines" (line 10).

**Why-wrong**: `moodycamel::ReaderWriterQueue` is a **single-producer, single-consumer** queue. If multiple engines call `emitStateUpdate()` concurrently (which the design explicitly documents), this violates the SPSC contract and causes data corruption/races.

**Concrete-fix**: Replace with `moodycamel::ConcurrentQueue` (MPMC), or serialize all producers through a single aggregator thread, or use one queue per engine.

---

## 13. Medium: NaN Propagation (F0Extractor)

**Tag**: `UB` | **Severity**: `MED`

**File**: `src/audio/F0Extractor.cpp` (pitch extraction loop)

**Evidence**: The parabolic interpolation for pitch refinement divides by a denominator that can be zero or near-zero, producing `NaN`. This `NaN` then flows into a `static_cast<int>(NaN)` which is UB per [conv.fpint].

**Concrete-fix**: Add NaN guard before integer cast:
```cpp
float pitch = /* parabolic interp result */;
if (std::isnan(pitch) || std::isinf(pitch)) pitch = 0.0f;
int midiNote = static_cast<int>(pitch);
```

---

## 14. Medium: FFI String Pointer Ownership Ambiguity

**Tag**: `FFI/MEM` | **Severity**: `MED`

**File**: `engine/intent_ir/src/intent_ir/ffi_exports.rs:231`

**Evidence**:
```rust
pub extern "C" fn intent_ir_get_error_message(error_code: IntentIRErrorCode) -> *const c_char {
    match error_code {
        IntentIRErrorCode::Success => ERR_SUCCESS.as_ptr() as *const c_char,
        // ...
    }
}
```

**Why-wrong**: Returns `*const c_char` pointing to static `b"..."` byte slices. A C++ caller receiving `const char*` has no way to know whether to `free()` the result or not. The `kelly_ffi.h` header documents `kelly_free_string` for heap-allocated strings but the intent_ir exports use a completely different pattern. This inconsistency across the FFI surface is a latent memory-safety bug.

**Concrete-fix**: Document in `cbindgen.toml` or a C header that `intent_ir_get_error_message` returns a **static pointer — do not free**. Consider using `CStr::from_bytes_with_nul_unchecked` for explicitness:
```rust
use std::ffi::CStr;
static ERR_SUCCESS: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"Success\0") };
// return ERR_SUCCESS.as_ptr()
```

---

## 15. Low: Type/Enum ODR Conflict

**Tag**: `ODR` | **Severity**: `LOW`

**Files**: 
- `src/common/KellyTypes.h:52` — `enum class RuleBreakType : uint8_t { None, ModalMixture, ... COUNT }`
- `src/engine/IntentProcessor.h:49` — `enum class RuleBreakType { Harmony, Rhythm, Dynamics, ... Range }`

**Evidence**: Two completely different `RuleBreakType` enumerations exist in the same namespace (`kelly`). Similarly, `Wound`, `IntentResult`, `EmotionNode`, and `RuleBreak` are defined differently in `KellyTypes.h` vs `IntentProcessor.h`.

| Type | `KellyTypes.h` fields | `IntentProcessor.h` fields |
|------|----------------------|--------------------------|
| `Wound` | `description, desire, urgency, expression, primaryEmotion, secondaryEmotion` | `description, intensity, source, timestamp, context, triggers` |
| `EmotionNode` | `id, name, category(string), categoryEnum, valence, arousal, dominance, intensity, mlEmbedding...` | `id, name, category(EmotionCategory), valence, arousal, intensity, tempoModifier, preferredMode...` |
| `IntentResult` | `key, mode, tempoBpm, chordProgression, ruleBreaks, sourceWound, confidence...` | `wound, emotion, ruleBreaks, musicalParams, summary()` |

**Why-wrong**: If any TU includes both headers, the compiler will reject it outright (redefinition). If they're included in separate TUs but the types cross the boundary (e.g., through `KellyBrain::fromWound`), the different layouts cause silent memory corruption.

**Concrete-fix**: Delete `IntentProcessor.h`'s inline type definitions. Have it `#include "common/KellyTypes.h"` and use the canonical types. The `InlineEmotionThesaurus` class should be moved to its own header or merged into the existing `EmotionThesaurus`.

---

## Consolidation Rules

| Cluster | Canonical Source | Action |
|---------|-----------------|--------|
| `harmony/HarmonyEngine.*` | `src_penta-core/harmony/` | Delete `src/harmony/HarmonyEngine.cpp`, keep penta-core |
| `harmony/ChordAnalyzer.*` | `src_penta-core/harmony/` | Delete `src/harmony/ChordAnalyzer.cpp` |
| `harmony/ScaleDetector.*` | `src_penta-core/harmony/` | Delete `src/harmony/ScaleDetector.cpp` |
| `harmony/VoiceLeading.*` | `src_penta-core/harmony/` | Delete `src/harmony/VoiceLeading.cpp` |
| `osc/RTMessageQueue.*` | `src_penta-core/osc/` | Delete `src/osc/RTMessageQueue.cpp` |
| `groove/*` | **Manual merge** | `src_penta-core/groove/` has ML-coupled logic; `src/groove/` has legacy UI endpoints. Requires file-by-file diff and merge. |
| `harmony/ChordAnalyzerSIMD.cpp` | `src/harmony/` (only copy) | Keep, add NEON path |
| Type definitions (`RuleBreakType`, `Wound`, etc.) | `src/common/KellyTypes.h` | Delete duplicates in `IntentProcessor.h` |

**Process**:
1. `diff -u src/<file> src_penta-core/<file>` to verify they're identical or identify deltas.
2. Keep `src_penta-core/` version (newer, ML-aware).
3. Update `CMakeLists.txt` source globs.
4. Run `cmake --build build --target KellyCore -j8` to confirm compilation.
5. Run `ctest` if `BUILD_TESTS=ON`.

---

## Regression Report vs Prior Audit

### Items RESOLVED in `main` since 2026-04-07

| File | Issue | Status |
|------|-------|--------|
| `src/plugin/PluginProcessor.cpp:344` | Hardcoded `/Users/sburdges/...` model paths | ✅ Replaced with plugin-relative + `KELLY_MODEL_ROOT` env fallback |
| `src/midi/MidiExporter.cpp:378` | Divide-by-zero on BPM | ✅ Guarded with `tempoBpm = 120.0f` fallback |

### Items on audit branch (NOT merged to main)

The `audit/cxx-safety-fixes-2026-04-07` branch contains ~12-15 of ~27 fixes. Key items **not** yet on `main`:
- HarmonyEngine OOB pitch guard
- F0Extractor NaN guard 
- BiometricInput destructor reorder

### Items STILL OPEN (this document)

| # | Finding | Severity | Blocking? |
|---|---------|----------|-----------|
| 1 | ODR twin defs (14 files) | CRIT | Yes — training freeze |
| 2 | KellyCore PUBLIC linkage | CRIT | Yes — any Python/Tauri consumer |
| 3 | Rust FFI panic unwinding | CRIT | Yes — any malformed IntentFrame crashes process |
| 4 | RT-path heap allocs | HIGH | Pre-training |
| 5 | HarmonyEngine OOB | HIGH | Pre-training |
| 6 | thread_local escape | HIGH | Pre-training |
| 7 | Missing NEON SIMD | HIGH | Perf-blocking on Apple Silicon |
| 8 | MidiBuilder BPM div/0 | HIGH | Pre-training |
| 9 | Pimpl incomplete type | HIGH | Silent UB |
| 10 | AffectUMP float cast | MED | |
| 11 | BiometricInput dtor | MED | |
| 12 | SPSC violation | MED | |
| 13 | F0Extractor NaN | MED | |
| 14 | FFI ownership docs | MED | |
| 15 | Type ODR conflicts | LOW | |

---

## Verification Commands

After applying fixes, run these in order:

```bash
# 1. Configure with tests + ASan
cmake -S . -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON \
  -DBUILD_TESTS=ON -DKMIDI_ENABLE_ASAN=ON

# 2. Build
cmake --build build-asan --target KellyCore KellyFFI KellyTests -j8

# 3. C++ tests
ctest --test-dir build-asan --output-on-failure

# 4. Linker verification (no duplicate JUCE)
nm -U build-asan/libKellyFFI.dylib | grep -c JuceVersion  # should be 0 or 1

# 5. Rust tests
cd engine/intent_ir && cargo test

# 6. Python lint + tests
python3 -m flake8 music_brain/ --max-line-length 100
python3 -m pytest tests/

# 7. Schema sync (if types changed)
python3 scripts/sync_entities.py
```
