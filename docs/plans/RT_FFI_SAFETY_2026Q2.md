# RT / FFI Safety Integration — 2026 Q2

> **Branch**: `audit/rt-ffi-safety-2026q2` (worktree: `~/.config/superpowers/worktrees/KmiDi/audit-rt-ffi-safety-2026q2/`)
> **Base**: `main @ a2e1b357`
> **Scope**: consolidate prior audit findings (`docs/CODEAUDIT_CXX_DEEP_2026Q2_FOLLOWUP.md`) with new 2026-04-21 live findings. Single PR, one commit per task.
> **Execution model**: subagent-driven development (`superpowers:subagent-driven-development`). Implementer + spec review + code quality review per task.

---

## Task ordering (prerequisite → leaf)

```
T0  baseline unblock              (pre-existing test build failure)
T1  ODR + build consolidation     (line numbers stable for later tasks)
T2  FFI boundary hardening        (header + ABI; independent of T3..T7)
T3  RT state & SPSC correctness   (seqlock + cache padding + MPMC fix)
T4  Python GIL safety             (independent; crash-class)
T5  RT-path correctness bugs      (6 discrete fixes in hot code)
T6  RT-path performance           (cleanup; no behavior change)
T7  NEON SIMD coverage            (large surface; last, isolated)
```

Each task commits independently. No task depends on the output of the next. T1 must land first so line numbers referenced in T5/T7 remain stable.

---

## Global acceptance gate (must pass after every task commit)

```bash
cd $WORKTREE

# Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON
cmake --build build --target KellyCore KellyFFI -j8

# Rust FFI
cargo test --manifest-path engine/intent_ir/Cargo.toml

# If task touches headers consumed by tests, also:
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON \
  -DBUILD_TESTS=ON -DKMIDI_ENABLE_ASAN=ON
cmake --build build-asan --target KellyCore KellyFFI KellyTests -j8
ctest --test-dir build-asan --output-on-failure
```

Additional per-task verification is listed in the task.

---

## T0 — Unblock pre-existing baseline

### Problem
`engine/intent_ir/tests/validator_tests.rs:5-7` uses `use crate::...::*` paths. Rust integration tests live in a separate crate and must reference the library by name. Build fails before any audit work runs, preventing clean CI signal.

### Evidence
```rust
// engine/intent_ir/tests/validator_tests.rs:1-7
//! Unit tests for Intent IR validator
#![cfg(test)]
use crate::types::*;
use crate::validator::*;
use crate::builder::*;
```
Compilation error:
```
error[E0432]: unresolved import `crate::validator`
  --> tests/validator_tests.rs:7:12
```

### Required change
Replace `use crate::` with `use intent_ir::` throughout `engine/intent_ir/tests/*.rs`. Do not change any test bodies. If `builder` or `validator` modules are not re-exported from `lib.rs`, add `pub mod validator;` / `pub mod builder;` at the crate root (whichever makes existing doctests / public API work). **Verify `src/lib.rs` first before assuming the fix.**

### Acceptance
```bash
cargo test --manifest-path engine/intent_ir/Cargo.toml
# Expect: all tests pass, 0 errors, 0 failures.
```

### Files
- `engine/intent_ir/tests/validator_tests.rs`
- Any other test file under `engine/intent_ir/tests/` with the same pattern
- Possibly `engine/intent_ir/src/lib.rs` (module re-exports)

### Non-scope
Do NOT fix any other audit finding in this commit. This is baseline cleanup only.

---

## T1 — ODR + build consolidation

### Problem
Two parallel source trees (`src/` and `src_penta-core/`) contain duplicate implementations of harmony, chord, scale, voice-leading, OSC, and groove modules. Both are globbed into `KellyCore` at build time. The linker silently picks one translation unit per symbol; divergent implementations corrupt engine state with no diagnostic. Additionally, `KellyCore` is linked `PUBLIC` on `KellyFFI`, propagating JUCE to every FFI consumer and causing allocator mismatch crashes on macOS.

Related: two `RuleBreakType` / `Wound` / `EmotionNode` / `IntentResult` type definitions coexist in the `kelly` namespace across `KellyTypes.h` and `IntentProcessor.h` with different layouts.

### Evidence
Live at HEAD:
```bash
diff -u src/harmony/HarmonyEngine.cpp src_penta-core/harmony/HarmonyEngine.cpp
# Both exist, both globbed by CMakeLists.txt.
```
```cmake
# CMakeLists.txt (KellyFFI target, ~line 502)
target_link_libraries(KellyFFI PUBLIC KellyCore PRIVATE juce::...)
```
```cpp
// src/common/KellyTypes.h:52   enum class RuleBreakType : uint8_t { None, ModalMixture, ... COUNT }
// src/engine/IntentProcessor.h:49   enum class RuleBreakType { Harmony, Rhythm, ... Range }
```

### Required changes

1. **File consolidation.** For each pair in the table below, `diff` the two files, keep the `src_penta-core/` version as canonical (per prior-audit Consolidation Rules), delete the `src/` version, update `CMakeLists.txt` source globs so only one copy is compiled.

   | Delete (src/) | Keep (src_penta-core/) |
   |---|---|
   | `src/harmony/HarmonyEngine.cpp` | `src_penta-core/harmony/HarmonyEngine.cpp` |
   | `src/harmony/ChordAnalyzer.cpp` | `src_penta-core/harmony/ChordAnalyzer.cpp` |
   | `src/harmony/ScaleDetector.cpp` | `src_penta-core/harmony/ScaleDetector.cpp` |
   | `src/harmony/VoiceLeading.cpp` | `src_penta-core/harmony/VoiceLeading.cpp` |
   | `src/osc/RTMessageQueue.cpp` | `src_penta-core/osc/RTMessageQueue.cpp` |
   | `src/groove/*.cpp` | `src_penta-core/groove/*.cpp` |

   **Keep** `src/harmony/ChordAnalyzerSIMD.cpp` (only one copy; T7 will extend it). **Do not delete** headers that the deleted `.cpp` files pair with unless the header is also duplicated — verify first with `diff`.

   If any `diff` shows behavioral divergence that looks intentional (new algorithm, bug fix), STOP and report `DONE_WITH_CONCERNS` listing the divergent file and the observed delta. The controller will decide how to merge.

2. **Linkage fix.** In `CMakeLists.txt`, change the `KellyFFI` target's `KellyCore` linkage from `PUBLIC` to `PRIVATE`:
   ```diff
   - target_link_libraries(KellyFFI PUBLIC KellyCore PRIVATE juce::...)
   + target_link_libraries(KellyFFI PRIVATE KellyCore PRIVATE juce::...)
   ```

3. **Type consolidation.** Delete the duplicate enum/struct definitions (`RuleBreakType`, `Wound`, `EmotionNode`, `IntentResult`, `RuleBreak`) from `src/engine/IntentProcessor.h`. Have it `#include "common/KellyTypes.h"` and use the canonical types. If `IntentProcessor.h` has other logic that depends on its private type layout, report `DONE_WITH_CONCERNS` and leave that part for a later task.

### Acceptance
```bash
# 1. No duplicate TU for consolidated symbols.
nm -U build/libKellyCore.a | grep ' T _ZN5kelly7harmony13HarmonyEngine' | wc -l
# Expect: exactly 1 per symbol (not 2).

# 2. KellyFFI does not re-export JUCE symbols as PUBLIC.
nm -U build/libKellyFFI.dylib | grep JuceVersion | wc -l
# Expect: ≤ 1 (was previously 2 when PUBLIC caused double-link).

# 3. Clean build.
cmake --build build --target KellyCore KellyFFI -j8
# Expect: zero "duplicate symbol" warnings.

# 4. Tests still green (no behavior regression).
ctest --test-dir build-asan --output-on-failure
```

### Files
- 14 deletions under `src/harmony/`, `src/osc/`, `src/groove/`
- `CMakeLists.txt` (source globs + KellyFFI link directive)
- `src/engine/IntentProcessor.h` (delete duplicate types, add include)

### Non-scope
- Do NOT modify the contents of `src_penta-core/` files — they are canonical.
- Do NOT add NEON to `ChordAnalyzerSIMD.cpp` — that's T7.

---

## T2 — FFI boundary hardening

### Problem
Three independent FFI-surface bugs:

1. `IntentFrame` is declared `#[repr(C, packed)]`. Packed references are UB in Rust (will hard-error on `derive(Debug)` as the toolchain tightens), and the C/C++ mirror struct must match the packing exactly or field reads corrupt silently.
2. Every `#[no_mangle] extern "C"` function in `engine/intent_ir/src/ffi.rs` lacks `std::panic::catch_unwind`. `panic = "abort"` in `Cargo.toml` converts UB to `SIGABRT` but still terminates the host DAW and loses unsaved user work.
3. `src/engine/KellyBrain.h` holds `std::unique_ptr<IntentPipeline> pipeline_` with `~KellyBrain() = default` in the header. `IntentPipeline` is only forward-declared there. Calling `delete` on an incomplete type with a non-trivial destructor is UB per `[expr.delete]/5`.

FFI string ownership (`intent_ir_get_error_message` → static byte slices) is not documented alongside `kelly_free_string` heap-owned return strings, inviting double-free or missed-free.

### Evidence
```rust
// engine/intent_ir/src/types.rs:77-86
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct IntentFrame { ... }
```
```rust
// engine/intent_ir/src/ffi.rs:44-55 (no catch_unwind)
#[no_mangle]
pub extern "C" fn validate_intent_frame_ffi(frame: *const IntentFrame) -> c_int {
    if frame.is_null() { return ValidationErrorCode::InvalidVersion as c_int; }
    let frame_ref = unsafe { &*frame };
    match validate_intent_frame(frame_ref) { ... }
}
```
```cpp
// src/engine/KellyBrain.h (~160)
class IntentPipeline;
std::unique_ptr<IntentPipeline> pipeline_;
~KellyBrain() = default;    // UB: deletes incomplete type
```

### Required changes

1. **Remove `packed`**. In `engine/intent_ir/src/types.rs:77`, change `#[repr(C, packed)]` to `#[repr(C)]`. Add crate-root static assertions:
   ```rust
   // Add to engine/intent_ir/src/lib.rs
   const _: () = {
       assert!(core::mem::size_of::<types::IntentFrame>() == N);  // compute from layout
       assert!(core::mem::align_of::<types::IntentFrame>() == 8);
   };
   ```
   Compute `N` by running `cargo test -- --nocapture` on a debug print of `size_of` or by inspection of the field list (`types.rs:79-86`). If cbindgen-emitted C headers exist in `engine/intent_ir/` or under `bindings/`, regenerate them and verify the C declaration has no `__attribute__((packed))`. If the regenerated header has drift, note it as `DONE_WITH_CONCERNS` — header regeneration may be out of scope.

2. **Wrap every `extern "C"` with `catch_unwind`**. Every `#[no_mangle]` function in `engine/intent_ir/src/ffi.rs` (there are 11) must catch panics and return an error-code sentinel (or silently succeed for void functions). Use `AssertUnwindSafe` for closures capturing `&mut T`. Example:
   ```rust
   #[no_mangle]
   pub extern "C" fn validate_intent_frame_ffi(frame: *const IntentFrame) -> c_int {
       std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
           if frame.is_null() {
               return ValidationErrorCode::InvalidVersion as c_int;
           }
           let f = unsafe { &*frame };
           match validate_intent_frame(f) {
               Ok(()) => ValidationErrorCode::Success as c_int,
               Err(e) => error_to_code(e) as c_int,
           }
       })).unwrap_or(ValidationErrorCode::InvalidVersion as c_int)
   }
   ```
   For `void`-returning setters (e.g., `IntentFrameBuilder_set_emotion`), do:
   ```rust
   let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| { /* body */ }));
   ```
   Because `lib.rs` declares `#![no_std]`, confirm whether `std::panic::catch_unwind` is available in the build. If not, either drop `#![no_std]` (it's `crate-type = ["staticlib"]` and already carries a libc allocator) or switch to `core::panic::catch_unwind` behind a cfg. Test with `cargo test`.

3. **Fix Pimpl dtor**. In `src/engine/KellyBrain.h`, change `~KellyBrain() = default;` to `~KellyBrain();` (declaration only). In `src/engine/KellyBrain.cpp`, add `KellyBrain::~KellyBrain() = default;` at the bottom — below where `IntentPipeline` is included and complete. Apply the same pattern to copy/move special members if they're `= default` in the header.

4. **Document FFI ownership**. Add a header comment block to `src/bridge/kelly_ffi.h` near `kelly_free_string`:
   ```c
   /*
    * FFI string ownership contract:
    *  - Functions returning `char*` (e.g. kelly_brain_from_text): heap-allocated,
    *    caller MUST pass to kelly_free_string.
    *  - Functions returning `const char*` (e.g. kelly_get_last_error,
    *    intent_ir_get_error_message): static storage, caller MUST NOT free.
    *  - When in doubt, check the return type's const-qualification.
    */
   ```
   If `intent_ir_get_error_message` is declared with non-const return elsewhere, tighten the declaration to `const char*`.

### Acceptance
```bash
# Packed removal compiles + no UAR lint.
cargo build --manifest-path engine/intent_ir/Cargo.toml
cargo clippy --manifest-path engine/intent_ir/Cargo.toml -- -D warnings

# catch_unwind coverage: every extern "C" in ffi.rs is now wrapped.
grep -c '#\[no_mangle\]' engine/intent_ir/src/ffi.rs
grep -c 'catch_unwind' engine/intent_ir/src/ffi.rs
# Expect: the two counts match (or catch_unwind count ≥ no_mangle count).

# Pimpl fix: incomplete-type delete warning gone.
cmake --build build --target KellyCore -j8 2>&1 | grep -i "delete.*incomplete" | wc -l
# Expect: 0
```

### Files
- `engine/intent_ir/src/types.rs`
- `engine/intent_ir/src/lib.rs` (size asserts, maybe drop `no_std`)
- `engine/intent_ir/src/ffi.rs` (catch_unwind on 11 functions)
- `src/engine/KellyBrain.h` + `src/engine/KellyBrain.cpp`
- `src/bridge/kelly_ffi.h` (ownership docs)

### Non-scope
- Do NOT change the builder allocation strategy here (that's a later task, not in this PR).
- Do NOT regenerate cbindgen output unless build actually requires it.

---

## T3 — RT state & SPSC correctness

### Problem
Three issues that compound into observable state tearing and cache-line thrashing:

1. `RTState` is read by `kelly_brain_get_rt_state` as ~27 independent relaxed loads with a single acquire on `sequence` at the end. Writer (`AudioEmotionRunner::updateParams`) stores 7 atomics with no sequence bump. Reader can observe half-new / half-old state — precisely the "state tearing" failure mode.
2. All atomics in `RTState` are packed contiguously with no alignment directive. Apple Silicon cache line = 128 bytes; every writer store invalidates the reader's cache line and vice versa.
3. `LockFreeRingBuffer::writePos_` and `readPos_` are adjacent `std::atomic<size_t>` → producer/consumer ping-pong on every push/pop.

The StateBridge queue type (`moodycamel::ReaderWriterQueue`, SPSC) is documented as being fed by MelodyEngine, BassEngine, MidiGenerator — multi-producer. SPSC contract violation corrupts cursors.

### Evidence
```cpp
// include/penta/common/RTState.h:33-75
struct RTState {
    std::atomic<double>   bpm{120.0};       // no alignas
    std::atomic<uint64_t> samplePosition{0};
    // ... 38 more atomics, no padding ...
    std::atomic<uint64_t> sequence{0};
};
```
```cpp
// src/bridge/kelly_ffi.cpp:866-909
out_state->bpm = s.bpm.load(std::memory_order_relaxed);
// ... 27 relaxed loads of different atomics ...
out_state->sequence = s.sequence.load(std::memory_order_acquire); // too late
```
```cpp
// src/ml/AudioEmotionRunner.cpp:381-389  (writer)
state.valence.store(..., std::memory_order_relaxed);
state.arousal.store(..., std::memory_order_relaxed);
// ... no sequence bump anywhere ...
```
```cpp
// src/ml/LockFreeRingBuffer.h:129-131
std::array<T, Capacity> buffer_;
std::atomic<size_t> writePos_;
std::atomic<size_t> readPos_;  // same cache line as writePos_
```
```cpp
// src/bridge/StateBridge.h:10, 60-63, 121
// "Used By: MidiGenerator, MelodyEngine, BassEngine, and other engines"
std::unique_ptr<moodycamel::ReaderWriterQueue<StateUpdate>> stateQueue_;
```

### Required changes

1. **Seqlock on RTState**. Introduce a canonical publish/snapshot protocol. Writer increments `sequence` to odd before stores, bumps to next even after. Reader retries on odd or mismatched seq.

   In `include/penta/common/RTState.h`, add helpers:
   ```cpp
   struct RTState {
       // ... existing fields, but add alignment (see step 2) ...

       // Writer-side: call before and after publishing a batch of stores.
       uint64_t begin_publish() noexcept {
           uint64_t s = sequence.fetch_add(1, std::memory_order_acq_rel);
           return s;  // now odd
       }
       void end_publish() noexcept {
           sequence.fetch_add(1, std::memory_order_release);  // back to even
       }
   };

   // Reader-side helper (not a member of RTState to avoid pulling reader state in):
   template <typename F>
   bool rt_state_snapshot(const RTState& s, F&& copy_all) noexcept {
       for (int attempt = 0; attempt < 8; ++attempt) {
           uint64_t seq1 = s.sequence.load(std::memory_order_acquire);
           if (seq1 & 1ull) continue;
           copy_all();  // lambda does all loads with memory_order_relaxed
           std::atomic_thread_fence(std::memory_order_acquire);
           uint64_t seq2 = s.sequence.load(std::memory_order_relaxed);
           if (seq1 == seq2) return true;
       }
       return false;  // caller should use last-known or return EAGAIN
   }
   ```

   In `src/ml/AudioEmotionRunner.cpp:381-389`, wrap the store block:
   ```cpp
   state.begin_publish();
   state.valence.store(impl_->slewValence.process(), std::memory_order_relaxed);
   // ... existing stores ...
   state.trackParams[2].store(impl_->slewDriveAmount.process(), std::memory_order_relaxed);
   state.end_publish();
   ```

   In `src/bridge/kelly_ffi.cpp:866-909`, rewrite `kelly_brain_get_rt_state` to use `rt_state_snapshot`. Return `KELLY_ERROR_AGAIN` (add to `KellyErrorCode` if not present; otherwise reuse a suitable existing code) if the snapshot fails after 8 retries.

   Every writer that stores to `RTState` must use the begin/end pair. Grep for `\.store(` on `RTState` members in the codebase and update every call site.

2. **Cache-line padding**. In `include/penta/common/RTState.h`:
   ```cpp
   #if defined(__cpp_lib_hardware_interference_size)
     inline constexpr std::size_t kCacheLine = std::hardware_destructive_interference_size;
   #elif defined(__aarch64__) && defined(__APPLE__)
     inline constexpr std::size_t kCacheLine = 128;  // M-series
   #else
     inline constexpr std::size_t kCacheLine = 64;
   #endif
   ```
   Group fields by writer and align each group:
   ```cpp
   struct RTState {
       alignas(kCacheLine) std::atomic<uint64_t> sequence{0};

       // Group 1: Audio thread writes
       alignas(kCacheLine) std::atomic<double>   bpm{120.0};
       std::atomic<uint64_t> samplePosition{0};
       std::atomic<uint64_t> barStart{0};
       std::atomic<uint32_t> bar{0};
       std::atomic<uint32_t> beat{0};
       std::atomic<uint32_t> numerator{4};
       std::atomic<uint32_t> denominator{4};
       std::atomic<bool>     playing{false};

       // Group 2: ML worker writes
       alignas(kCacheLine) std::atomic<float> valence{0.0f};
       std::atomic<float> arousal{0.5f};
       std::atomic<float> dominance{0.5f};
       std::atomic<int16_t> discreteEmotionId{-1};
       std::atomic<float> emotionIntensity{0.0f};
       std::atomic<float> emotionConfidence{0.0f};

       // Group 3: Bridge/UI writes (infrequent)
       alignas(kCacheLine) std::atomic<float> tempoBias{0.0f};
       std::atomic<float> rhythmicDensity{0.5f};
       std::atomic<float> grooveStrength{0.5f};
       std::atomic<float> harmonicTension{0.5f};
       std::atomic<float> harmonicMotion{0.5f};
       std::atomic<float> melodicActivity{0.5f};
       std::atomic<float> textureDensity{0.5f};
       std::atomic<float> dynamicRange{0.5f};

       // Group 4: Track params
       alignas(kCacheLine) std::array<std::atomic<float>, kMaxTrackParams> trackParams{};

       // Trailing pad so whatever follows in the aggregate doesn't share.
       alignas(kCacheLine) std::byte _tail_pad[1]{};

       // Existing ctor / delete copy/move remain.
   };
   ```
   Preserve the existing `static_assert(std::atomic<T>::is_always_lock_free, ...)` block.

3. **Pad LockFreeRingBuffer cursors**. In `src/ml/LockFreeRingBuffer.h`:
   ```cpp
   private:
       std::array<T, Capacity> buffer_;
       alignas(64) std::atomic<size_t> writePos_{0};  // 64 is fine here; buffer is POD
       alignas(64) std::atomic<size_t> readPos_{0};
   ```
   Use 64 (or `std::hardware_destructive_interference_size` if you want the same flexibility). The optional reader-side cached cursor optimization (cachedReadPos_ / cachedWritePos_) is NOT in scope for this task — out-of-scope perf tuning.

4. **StateBridge multi-producer**. In `src/bridge/StateBridge.h`, replace the SPSC queue with MPMC:
   ```cpp
   #include <concurrentqueue.h>   // moodycamel MPMC
   std::unique_ptr<moodycamel::ConcurrentQueue<StateUpdate>> stateQueue_;
   ```
   Update `StateBridge.cpp` to use `try_enqueue` / `try_dequeue` (same API surface). **Note:** this task does NOT replace `std::string` inside `StateUpdate` — that's prior-audit §4b, remains for a later pass. Mark as `DONE_WITH_CONCERNS` in the report so the follow-up is tracked.

### Acceptance
```bash
# Seqlock unit test: dispatch a writer thread bumping state and a reader thread
# snapshotting. Over 1M iterations, reader must never see torn state.
# Author a new test at tests/cpp/test_rt_state_seqlock.cpp (use existing test
# harness pattern from tests/cpp/test_kelly_ffi.cpp).

# TSan must see zero data races on RTState after this change.
cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON \
  -DBUILD_TESTS=ON -DKMIDI_ENABLE_TSAN=ON
cmake --build build-tsan --target KellyTests -j8
ctest --test-dir build-tsan --output-on-failure
# Expect: 0 TSan reports on test_rt_state_seqlock.

# Padding: offset check
# In a dev assertion or compile-time static_assert:
static_assert(offsetof(RTState, valence) - offsetof(RTState, bpm) >= 128);

# StateBridge: simulate 3 producer threads, 1 consumer. No lost / corrupted updates.
```

### Files
- `include/penta/common/RTState.h` (padding + seqlock helpers)
- `src/ml/AudioEmotionRunner.cpp` (wrap updateParams writes)
- `src/bridge/kelly_ffi.cpp` (seqlock-retry reader)
- `src/ml/LockFreeRingBuffer.h` (cursor padding)
- `src/bridge/StateBridge.h` + `StateBridge.cpp` (MPMC)
- `tests/cpp/test_rt_state_seqlock.cpp` (new)

### Non-scope
- Do NOT replace `std::string` in `StateUpdate` — separate task.
- Do NOT add SPSC batch push APIs to LockFreeRingBuffer — that's T6.
- Do NOT touch PluginProcessor's `emotionRTState_` beyond what's needed to compile.

---

## T4 — Python GIL safety

### Problem
`PyObject_CallObject` / `PyTuple_New` / `PyUnicode_FromString` are invoked from C++ in `PythonBridgeBase::callPythonFunction` and `StateBridge` without acquiring the GIL. If the calling thread is not the one that imported Python, this is immediate UB (documented by CPython).

In particular, `StateBridge::processStateQueue` is declared to be called asynchronously by a worker thread (`class StateWorkerThread` at `StateBridge.h:126`). That worker must hold the GIL before touching any PyObject.

### Evidence
```cpp
// src/bridge/PythonBridgeBase.cpp:128-187  (no GIL)
PyObject* pyArgs = PyTuple_New(...);
PyObject* result = PyObject_CallObject(func, pyArgs);
```
```cpp
// src/bridge/StateBridge.cpp:156-215, 222-255 (no GIL, runs on worker thread)
PyObject* result = PyObject_CallObject(func, nullptr);
```

### Required changes

1. Add a RAII helper at `src/bridge/PythonBridgeBase.h` (private section or anon namespace inside `.cpp`):
   ```cpp
   #ifdef PYTHON_AVAILABLE
   struct PyGILGuard {
       PyGILState_STATE state;
       PyGILGuard()  : state(PyGILState_Ensure()) {}
       ~PyGILGuard() { PyGILState_Release(state); }
       PyGILGuard(const PyGILGuard&) = delete;
       PyGILGuard& operator=(const PyGILGuard&) = delete;
   };
   #endif
   ```

2. Wrap every Python C API access in these files with `PyGILGuard`. Audit all call sites:
   - `src/bridge/PythonBridgeBase.cpp`: `importModule`, `getFunction`, both `callPythonFunction` overloads
   - `src/bridge/StateBridge.cpp`: `getCurrentState`, `getEngineState`, `processStateQueue`, `shutdown` (if it decref's)
   - Any other files identified by `grep -l "PyObject\|PyGILState\|PyUnicode" src/bridge/`

   Every `Py_DECREF`, `Py_XDECREF` must also be inside the guard scope (decref touches the object graph and needs the GIL).

3. Initialize the Python thread state system at interpreter startup:
   ```cpp
   // PythonBridgeBase::initializePython (after Py_Initialize):
   if (!PyGILState_Check()) {
       PyEval_InitThreads();  // deprecated in 3.9+, noop in 3.13, still safe
   }
   // then release the main-thread GIL so other threads can acquire:
   mainThreadState_ = PyEval_SaveThread();
   ```
   On shutdown, restore:
   ```cpp
   PyEval_RestoreThread(mainThreadState_);
   Py_Finalize();
   ```
   If `mainThreadState_` is not already a member, add it. Wrap in `#ifdef PYTHON_AVAILABLE`.

### Acceptance
```bash
# Build with Python enabled (check existing CMake flag name; likely -DENABLE_PYTHON=ON
# or similar — grep CMakeLists.txt).
cmake --build build --target KellyCore -j8

# Integration test: construct StateBridge from a non-main thread, call
# emitStateUpdate + processStateQueue repeatedly. Must not SIGSEGV, must not
# assert in CPython debug builds.
# Add tests/cpp/test_python_gil.cpp if one doesn't exist.

# TSan must not flag GIL-related races.
ctest --test-dir build-tsan -R python_gil --output-on-failure
```

### Files
- `src/bridge/PythonBridgeBase.h` / `.cpp`
- `src/bridge/StateBridge.cpp`
- Any other bridge file identified by grep
- `tests/cpp/test_python_gil.cpp` (new, if test suite permits)

### Non-scope
- Do NOT refactor the Python module discovery path.
- Do NOT add new Python functions.

---

## T5 — RT-path correctness bugs

### Problem
Six discrete bugs in the hot audio path or its close neighbors. Each is small; bundling them avoids six separate commits and review cycles for two-line fixes. All are verified against live code at HEAD.

### Sub-tasks

**T5.1 — `pthread_set_qos_class_self_np` per callback.**

*Location*: `src/plugin/PluginProcessor.cpp:421-424`.
Set QoS once, not every block:
```cpp
// PluginProcessor.h: add member
std::once_flag qosSetOnce_;

// processBlock:
#if JUCE_MAC
std::call_once(qosSetOnce_, []{
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
});
#endif
```
Better: move the call into `audioWorkgroupContextChanged` (already a hook at line 403) and delete from processBlock entirely. Try the clean move first; fall back to `call_once` if the workgroup hook is not reliably invoked pre-render.

**T5.2 — HarmonyEngine OOB pitch.**

*Location*: `src_penta-core/harmony/HarmonyEngine.cpp:25` (post-T1 canonical path).
```cpp
if (static_cast<unsigned>(note.pitch) < 128) {
    activeNotes_[note.pitch] = note.velocity;
}
```

**T5.3 — `thread_local` pointer escape in MidiGenerator.**

*Location*: `src/midi/MidiGenerator.cpp:57` (`return &out;`).
Change the function signature to return by value and let NRVO / move handle it. Callers must update:
```cpp
ArrangementOutput MidiGenerator::generate(...) {
    ArrangementOutput out;
    // ...
    return out;
}
```
Grep `MidiGenerator::generate` callers and update to consume by value. If any call site stores the pointer long-term, convert to local ownership.

**T5.4 — MidiBuilder BPM divide-by-zero.**

*Location*: `src/midi/MidiBuilder.cpp:40`.
```cpp
const float safeBpm = (midi.bpm > 0.0f) ? midi.bpm : 120.0f;
int microsecondsPerBeat = MIDI_MICROSECONDS_PER_MINUTE / static_cast<int>(safeBpm);
```
Mirror the guard MidiExporter already has.

**T5.5 — AffectUMP float→uint32_t UB.**

*Location*: `src/midi/AffectUMP.cpp:19`.
```cpp
double scaled = std::round(static_cast<double>(norm) * static_cast<double>(0xFFFFFFFFu));
scaled = std::clamp(scaled, 0.0, static_cast<double>(UINT32_MAX));
auto v = static_cast<uint32_t>(scaled);
```

**T5.6 — BiometricInput destructor ordering.**

*Location*: `src/biometric/BiometricInput.cpp:20`. Swap the two if-blocks so `stopStreaming()` runs before `delete healthKitBridge_`.

**T5.7 — F0Extractor NaN guard.**

*Location*: `src/audio/F0Extractor.cpp`, wherever the parabolic interpolation divides. Add:
```cpp
if (std::isnan(pitch) || std::isinf(pitch)) pitch = 0.0f;
int midiNote = static_cast<int>(pitch);
```

### Acceptance
```bash
# Build + all existing tests pass.
cmake --build build-asan --target KellyCore KellyFFI KellyTests -j8
ctest --test-dir build-asan --output-on-failure

# Targeted: smoke test MidiBuilder with bpm=0, HarmonyEngine with pitch=200,
# AffectUMP with norm=1.0f. Add one test per sub-task in tests/cpp/ or extend
# existing ones. Use the plugin test harness if present.
```

### Files
- `src/plugin/PluginProcessor.cpp` + `PluginProcessor.h`
- `src_penta-core/harmony/HarmonyEngine.cpp`
- `src/midi/MidiGenerator.cpp` + callers
- `src/midi/MidiBuilder.cpp`
- `src/midi/AffectUMP.cpp`
- `src/biometric/BiometricInput.cpp`
- `src/audio/F0Extractor.cpp`

### Non-scope
- Do NOT tune any performance behavior — that's T6.
- Do NOT replace thread_local with pmr arenas or other architectural changes; return-by-value is the spec.

---

## T6 — RT-path performance

### Problem
Ghost work in the audio callback and ML worker. None are correctness bugs; all waste CPU on the hot path. Bundled because they share a verification surface (profiler traces, not unit tests).

### Sub-tasks

**T6.1 — Per-sample modulo in lookahead ring.**

*Location*: `src/plugin/PluginProcessor.cpp:438-442, 521-525`.

In `prepareToPlay`, size `lookaheadBuffer_` to `juce::nextPowerOfTwo(desiredSamples)`. Store `lookaheadMask_ = size - 1` as a member. In `processBlock`, replace the per-sample loop with two memcpys:
```cpp
const size_t wIdx = lookaheadWritePos_ & lookaheadMask_;
const size_t first = std::min(static_cast<size_t>(numSamples),
                              static_cast<size_t>(lookaheadBuffer_.getNumSamples() - wIdx));
std::memcpy(dst + wIdx, src, first * sizeof(float));
if (static_cast<size_t>(numSamples) > first) {
    std::memcpy(dst, src + first, (numSamples - first) * sizeof(float));
}
lookaheadWritePos_ = (lookaheadWritePos_ + numSamples) & lookaheadMask_;
```
Mirror for the read-side loop.

**T6.2 — Bulk SPSC enqueue in pushSamples.**

*Location*: `src/ml/AudioEmotionRunner.cpp:332-339`.

`impl_->sampleRing` is currently `moodycamel::ReaderWriterQueue<float>` (likely; confirm by reading the impl). Either:
- switch to `LockFreeRingBuffer<float, N>` and call its existing `push(samples, count)` bulk path, or
- keep moodycamel but use `try_enqueue_bulk` if the queue type supports it.

Whichever is chosen, the implementation must:
1. memcpy `count` samples in at most two contiguous writes
2. increment `droppedSamples` once with the overflow delta
3. preserve the `noexcept` declaration

**T6.3 — APVTS pointer cache.**

*Location*: `src/plugin/PluginProcessor.cpp:486, 494, 496, 506, 509, 545, 553` (and any other `apvts_.getRawParameterValue` in processBlock).

Cache pointers once in `prepareToPlay`:
```cpp
// PluginProcessor.h:
std::atomic<float>* paramMlInfluence_ = nullptr;
std::atomic<float>* paramValence_     = nullptr;
std::atomic<float>* paramArousal_     = nullptr;
std::atomic<float>* paramEqBypass_    = nullptr;
std::atomic<float>* paramBypass_      = nullptr;

// prepareToPlay:
paramMlInfluence_ = apvts_.getRawParameterValue(PARAM_ML_INFLUENCE);
// ... etc.

// processBlock — replace every call.
const float blend = paramMlInfluence_->load(std::memory_order_relaxed);
```
Keep a null check + diagnostic log in prepareToPlay (not processBlock) in case a parameter ID is misspelled.

**T6.4 — FTZ/DAZ platform helper.**

Factor a shared helper at `include/penta/common/Platform.h`:
```cpp
inline void set_denormals_off() noexcept {
#if defined(__SSE__)
    _mm_setcsr(_mm_getcsr() | 0x8040u);
#elif defined(__aarch64__)
    std::uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ull << 24);
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
}
```
Replace the inline block at `src/ml/AudioEmotionRunner.cpp:178-187` with one call. Call the same helper at the top of any non-JUCE render thread (search `src/` for `processBlock`, audio worker threads).

**T6.5 — Drop-oldest result queue policy.**

*Location*: `src/ml/InferenceThreadManager.h:131-134` and `LockFreeRingBuffer` if it's used.

Add an overwrite-on-full variant to `LockFreeRingBuffer`:
```cpp
bool pushOverwrite(const T* data, size_t count) {
    if (count > Capacity) return false;
    while (availableToWrite() < count) {
        T discard;
        pop(&discard, 1);  // drop oldest
    }
    return push(data, count);
}
```
`inferenceLoop` then calls `resultBuffer_.pushOverwrite(&result, 1)`. Audio thread still calls regular `pop`. Old samples in the request queue keep drop-newest (samples aren't a state, they're a stream).

**T6.6 — Semaphore wake instead of spin-sleep.**

*Location*: `src/ml/InferenceThreadManager.h:137`.

Add a counting semaphore, `release()` on every `submitRequest`, `try_acquire_for(10ms)` in `inferenceLoop`. Remove the `sleep_for(100us)` line. Must compile on C++20 / clang / gcc.

**T6.7 — droppedSamples telemetry consumer.**

Add a readout accessible from the RT telemetry path (wherever `lastInferenceMs` is surfaced). At minimum: expose `dropRate()` in droppedSamples/sec (EMA or rolling window) and log-once-per-10s-if-nonzero via JUCE logger gated on `JUCE_DEBUG`. Do NOT log from the audio thread — log from a UI-thread timer callback that samples the atomic.

### Acceptance
```bash
# Build.
cmake --build build --target KellyCore KellyFFI KellyPluginStandalone -j8

# Optional benchmark regression if bench exists:
ctest --test-dir build -R rt_harness --output-on-failure
# Or run the RT harness manually and compare before/after CPU % in Activity Monitor.

# No new data races:
ctest --test-dir build-tsan --output-on-failure
```

### Files
- `src/plugin/PluginProcessor.cpp` / `.h`
- `src/ml/AudioEmotionRunner.cpp` / `.h`
- `src/ml/InferenceThreadManager.h`
- `src/ml/LockFreeRingBuffer.h` (pushOverwrite)
- `include/penta/common/Platform.h` (set_denormals_off)

### Non-scope
- NEON SIMD — T7.
- Any new inference model behavior — out of scope entirely.

---

## T7 — NEON SIMD coverage

### Problem
Every SIMD-accelerated DSP kernel gates on `__AVX2__` with a scalar fallback. On Apple Silicon (primary target) AVX2 is never defined and every hot DSP loop runs scalar. The prior audit flagged `ChordAnalyzerSIMD.cpp` and mel-spectrogram inner loops specifically.

### Evidence
```cpp
// src/harmony/ChordAnalyzerSIMD.cpp:6-8
#ifdef __AVX2__
#include <immintrin.h>
#endif
// Lines 144-161: scalar fallback.
```
```
$ grep -r __ARM_NEON src/  # returns no results
```

### Required changes

1. **Add a shared SIMD abstraction** at `include/penta/common/SIMDKernels.h` (the file already exists — extend it). Provide NEON implementations of whatever primitives ChordAnalyzerSIMD and MelSpectrogram need:
   - `simd::dot_product_f32(const float* a, const float* b, size_t n)`
   - `simd::multiply_add_f32(float* dst, const float* a, const float* b, size_t n)`
   - `simd::max_element_f32(const float* a, size_t n)`
   - (Add others as dictated by the two callers.)

   Each primitive:
   ```cpp
   inline float dot_product_f32(const float* a, const float* b, size_t n) noexcept {
   #if defined(__AVX2__)
       // existing AVX2 path (lift from ChordAnalyzerSIMD.cpp)
   #elif defined(__ARM_NEON) || defined(__ARM_NEON__)
       float32x4_t acc = vdupq_n_f32(0.0f);
       size_t i = 0;
       for (; i + 4 <= n; i += 4) {
           float32x4_t va = vld1q_f32(a + i);
           float32x4_t vb = vld1q_f32(b + i);
           acc = vfmaq_f32(acc, va, vb);
       }
       float s = vaddvq_f32(acc);
       for (; i < n; ++i) s += a[i] * b[i];
       return s;
   #else
       // scalar
       float s = 0.0f;
       for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
       return s;
   #endif
   }
   ```

2. **Refactor ChordAnalyzerSIMD.cpp** to call the new primitives. Delete the scalar fallback (it now lives inside the primitive). Delete the inline AVX2 code once it's promoted into the header.

3. **Refactor MelSpectrogram's mel filter inner loop** (`src/ml/MelSpectrogram.cpp`; find the hot loop by grep for `mel` and `for`). Use the same primitives.

4. **Document architecture coverage** in a short comment at the top of `SIMDKernels.h`:
   ```cpp
   // SIMD primitives for audio DSP kernels.
   // Dispatch at compile time:
   //   - AVX2   (x86_64 Intel/AMD)
   //   - NEON   (AArch64, including Apple Silicon)
   //   - scalar (fallback, correctness only)
   ```

### Acceptance
```bash
# Release build on arm64 must pick the NEON path.
cmake -S . -B build-arm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_KELLY_CORE=ON
cmake --build build-arm --target KellyCore -j8

# Verify NEON intrinsics actually emitted:
otool -tv build-arm/libKellyCore.a | grep -E 'fmla|vld1' | head
# Expect: non-empty output from the ChordAnalyzerSIMD and MelSpectrogram TUs.

# Unit test parity: each primitive has a test that compares NEON/scalar for
# random input. Allow an epsilon of 1e-5 for FMA-vs-split reassociation.
ctest --test-dir build-asan -R simd --output-on-failure
```

### Files
- `include/penta/common/SIMDKernels.h`
- `src/harmony/ChordAnalyzerSIMD.cpp`
- `src/ml/MelSpectrogram.cpp`
- `tests/cpp/test_simd_kernels.cpp` (new)

### Non-scope
- Do NOT port every scalar loop in the codebase. Only the two files above.
- Do NOT introduce runtime dispatch; compile-time arch selection is sufficient.

---

## Final code review (after all 7 tasks merge)

Dispatch `superpowers:requesting-code-review` (or the equivalent reviewer subagent) against the entire branch:

```
BASE_SHA: main @ a2e1b357
HEAD_SHA: audit/rt-ffi-safety-2026q2 @ HEAD
WHAT_WAS_IMPLEMENTED: RT/FFI safety integration — 7 tasks, see task summaries above.
```

Then use `superpowers:finishing-a-development-branch` to decide integration path (single PR vs. merge to main).
