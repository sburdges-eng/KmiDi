# KmiDi C++ Salvage Catalog (2026 Q2)

**Date:** 2026-04-07
**Author:** Phase 2 of the C++ safety-fixes audit (`docs/CODEAUDIT_CXX_DEEP_2026Q2.md`)
**Purpose:** Per-file disposition table for divergent C++ file pairs found by the comprehensive audit. Used as the decision log before any consolidation deletes happen.

> **DO NOT delete any file in this catalog without first marking its disposition here and getting user sign-off.**
>
> The user has thousands of hours invested in KmiDi. Premature deletion of any divergent copy that contains unmerged fixes is the worst possible outcome. This catalog is the prerequisite for any safe deletion / merge / rename action.

## Executive summary

Three trees were compared:

| Tree | Path | Build status | Namespace |
|---|---|---|---|
| **A** | `/Users/seanburdges/Dev/KmiDi/src/` | ACTIVE — built into `KellyCore` static lib | mixed: `kelly::*`, `penta::*`, `daiw::*`, `midikompanion::*` |
| **B** | `/Users/seanburdges/Dev/KmiDi/src_penta-core/` | ACTIVE — built into `penta_core` static lib (added via `add_subdirectory(src_penta-core)`) | `penta::*` |
| **D** | `/Users/seanburdges/Dev/KmiDi/KmiDi_PROJECT/source/cpp/src/` | ORPHAN — not referenced from active CMake graph | `midikompanion::*`, `kelly::*` |

**Critical finding (ODR):** for the harmony/groove/osc subsystems, both Tree A (compiled into `KellyCore`) and Tree B (compiled into `penta_core`) define the same fully-qualified `penta::harmony::*`, `penta::groove::*`, `penta::osc::*` symbols **with different bodies**. KellyCore's `GLOB_RECURSE` only excludes `src/harmony/chord.cpp` and `src/harmony/progression.cpp`; everything else under `src/harmony/`, `src/groove/`, `src/osc/` is silently picked up. Whichever static lib wins the link race determines runtime behavior, and that is implementation-defined per linker. **This is exactly the "two live implementations" hazard the audit was set up to find.**

**Bidirectional divergence (do not blindly delete either side):**

- **B has unique fixes** (March 31, 2026) for: `HarmonyEngine.cpp`, `ChordAnalyzer.cpp`, `ChordAnalyzerSIMD.cpp`, `GrooveEngine.cpp`, `OSCHub.cpp`, `OSCServer.cpp` — these are the "fix(critical|high)" memory-safety, RT-safety, and ODR fixes from the audit.
- **A has unique fixes** for: `VoiceLeading.cpp` (kelly port merge, March 29), `HarmonyEngine.cpp` (April 1 build fix), `OSCHub.cpp` (Mar 2 + April 1 build fix), `OSCServer.cpp` (Mar 2), `PluginProcessor.cpp` (April 1 demo proof-of-life), `OSCBridge.cpp` (March 31 thread-safety/atomics fixes).

**Result:** Almost every harmony/groove/osc pair needs a *merge*, not a delete. Tree D (KmiDi_PROJECT) is uniformly stale and orphaned but should still be archived to a tag/branch before removal.

## How to read this catalog

For each row:

- **Pair**: the two file paths being compared (A vs B, or A vs D, or .cpp vs .mm, etc.)
- **Lines A / Lines B**: line counts via `wc -l`
- **Last A / Last B**: most recent commit affecting that file (date + short hash + message)
- **Diff status**: `IDENTICAL`, `DIFFER`, or `MISSING` (one side does not exist)
- **Semantic summary**: one-line explanation of the difference
- **Disposition**: proposed action (see legend)

### Disposition legend

- `KEEP_A_DELETE_B` — A is canonical, B is stale duplicate, no unique work to salvage in B
- `KEEP_B_DELETE_A` — B is canonical, A is stale duplicate, no unique work to salvage in A
- `MERGE_B_INTO_A` — A is canonical but B has unmerged fixes that must be ported into A first
- `MERGE_A_INTO_B` — B is canonical but A has unmerged fixes that must be ported into B first
- `KEEP_BOTH_RENAME` — both serve distinct purposes; rename to disambiguate
- `INVESTIGATE` — too divergent / unclear / need user judgment before any deletion
- `IDENTICAL_DELETE_EITHER` — byte-equal, pick one and delete the other

I bias toward `INVESTIGATE` whenever I am not certain.

---

## Catalog

### Group: Harmony (5 pairs)

#### Pair 1 — `src/harmony/VoiceLeading.cpp` ↔ `src_penta-core/harmony/VoiceLeading.cpp`

- **Lines A / B**: 309 / 205
- **Last A**: `2026-03-29 21:14:50 -0600 2399e5ca refactor: merge VoiceLeading — port kelly's analyze/voiceProgression/smoothness into penta, adapter for ChordGenerator`
- **Last B**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: DIFFER
- **Semantic summary**: A is a strict superset of B. A retains every line of B and adds ~104 lines of ported `kelly::VoiceLeadingEngine` functionality (`analyze()`, `voiceProgression()`, `calculateSmoothness()`, `invertVoicing()`). The March 29 commit message confirms this was an intentional forward-port. B is the pre-merge state.
- **Disposition**: **KEEP_A_DELETE_B**
- **Diff sample** (truncated to 30 lines):
  ```diff
  @@ -200,110 +200,6 @@
       // Linear cost for motion (prefer minimal motion)
       return distance;
  -}
  -
  -// --- Ported from kelly::VoiceLeadingEngine ---
  -
  -VoiceLeading::VoiceLeadingResult VoiceLeading::analyze(
  -    const std::vector<Note>& fromVoicing,
  -    const std::vector<Note>& toVoicing
  -) const noexcept {
  -    VoiceLeadingResult result;
  -    ...
  -    result.smoothnessScore = calculateSmoothness(result.movements);
  -    return result;
  -}
  -
  -std::vector<std::vector<Note>> VoiceLeading::voiceProgression(
  -    const std::vector<Chord>& chords,
  -    ...
  -float VoiceLeading::calculateSmoothness(
  -    const std::vector<VoiceMovement>& movements
  -) const noexcept { ... }
  -
  -std::vector<Note> VoiceLeading::invertVoicing(
  -    const std::vector<Note>& voicing,
  -    int inversion
  -) noexcept { ... }
  ```

#### Pair 2 — `src/harmony/HarmonyEngine.cpp` ↔ `src_penta-core/harmony/HarmonyEngine.cpp`

- **Lines A / B**: 149 / 160
- **Last A**: `2026-04-01 00:34:30 -0600 1d90eb93 fix: resolve pre-existing C++ build errors for AU plugin target`
- **Last B**: `2026-03-31 05:23:32 -0600 04d1e06c fix(high): RT-safety — remove mutex/heap-alloc from audio thread paths`
- **Diff status**: DIFFER
- **Semantic summary**: **Bidirectional divergence — both sides have unique commits.** B has the March 31 RT-safety rewrite of `updateChordAnalysis()` / `updateScaleDetection()` (uses confidence threshold instead of pitch-class equality, increments history counter via if-guard rather than std::min, RT-safe circular-buffer access). A has the April 1 build fix that makes the AU plugin compile. Whitespace also drifted significantly.
- **Disposition**: **MERGE_B_INTO_A** — A is canonical (it contains the April 1 build fix that the active build needs), but B's RT-safety semantics MUST be ported into A first or those fixes will be lost.
- **Diff sample**:
  ```diff
   void HarmonyEngine::updateChordAnalysis() noexcept {
       chordAnalyzer_->update(pitchClassSet_);
  -    currentChord_ = chordAnalyzer_->getCurrentChord();
  -
  -    const bool hasHistory = chordHistoryCount_ > 0;
  -    if (hasHistory) {
  -        const size_t lastIndex = (chordHistoryWriteIndex_ + kHistoryCapacity - 1) % kHistoryCapacity;
  -        const auto& last = chordHistory_[lastIndex];
  -        if (last.root == currentChord_.root && ...) {
  -            return;
  -        }
  +    Chord newChord = chordAnalyzer_->getCurrentChord();
  +    // Only add to history if chord changed significantly
  +    if (newChord.root != currentChord_.root || 
  +        newChord.quality != currentChord_.quality ||
  +        newChord.confidence > 0.7f) {
  +        // Add to history using circular buffer (RT-safe, no heap alloc)
  +        chordHistory_[chordHistoryWriteIndex_] = newChord;
  +        chordHistoryWriteIndex_ = (chordHistoryWriteIndex_ + 1) % kHistoryCapacity;
  +        if (chordHistoryCount_ < kHistoryCapacity) chordHistoryCount_++;
       }
  +    currentChord_ = newChord;
   }
  ```

#### Pair 3 — `src/harmony/ChordAnalyzer.cpp` ↔ `src_penta-core/harmony/ChordAnalyzer.cpp`

- **Lines A / B**: 293 / 196
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-03-31 05:32:45 -0600 bfa880b0 fix(medium): memory safety — purge guard, ODR fix, result double-buffer`
- **Diff status**: DIFFER
- **Semantic summary**: B is the **ODR fix** for the SIMD path. A defines `scoreAgainstTemplateSIMD`, `findBestMatchSIMD`, `analyzeSIMD` inside `#ifdef __AVX2__` in this file AND those same symbols also exist in `ChordAnalyzerSIMD.cpp` → ODR violation. B fixes this by moving the AVX2 implementations exclusively into `ChordAnalyzerSIMD.cpp` and leaving only a `#ifndef __AVX2__` scalar fallback here. The 97-line delta in A is exactly the duplicated AVX2 code that B removed.
- **Disposition**: **MERGE_B_INTO_A** — B's ODR fix is critical and must be applied to A. After the merge, A and B will become byte-identical and one can be deleted.
- **Diff sample**:
  ```diff
  -// SIMD intrinsics (AVX2)
  -#ifdef __AVX2__
  -#include <immintrin.h>
  -#endif
  +// NOTE: AVX2 versions of scoreAgainstTemplateSIMD, findBestMatchSIMD, and
  +// analyzeSIMD live exclusively in ChordAnalyzerSIMD.cpp to avoid ODR
  +// violations. This file provides only the scalar fallback.
  -#ifdef __AVX2__
  -// AVX2 intrinsics version: Process 8 pitch classes at once
  -float ChordAnalyzer::scoreAgainstTemplateSIMD(...) const noexcept {
  -    // ... 80 lines of AVX2 implementation ...
  -}
  -void ChordAnalyzer::findBestMatchSIMD(...) noexcept { ... }
  -Chord ChordAnalyzer::analyzeSIMD(...) noexcept { ... }
  +#ifndef __AVX2__
  +float ChordAnalyzer::scoreAgainstTemplateSIMD(...) const noexcept {
  +    // scalar fallback only
  +}
  ```

#### Pair 4 — `src/harmony/ChordAnalyzerSIMD.cpp` ↔ `src_penta-core/harmony/ChordAnalyzerSIMD.cpp`

- **Lines A / B**: 170 / 170
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-03-31 05:23:03 -0600 8e2387ba fix(critical+high): SIMD init, undefined method, recursion depth limit`
- **Diff status**: DIFFER
- **Semantic summary**: Same line count but B has a **critical SIMD-init memory-safety fix**: `alignas(32) float scores[8];` → `alignas(32) float scores[8] = {};` (uninitialized stack array → zero-initialized). Without this, `_mm256_load_ps(scores)` reads uninitialized memory, which is UB and a real audit-flagged crash risk under sanitizers / Apple-silicon clang.
- **Disposition**: **MERGE_B_INTO_A** — B has a 1-line memory-safety fix that A is missing. Trivial port.
- **Diff sample**:
  ```diff
  @@ -76,7 +76,7 @@
       uint8_t bestQuality = 0;
       
       // Prepare 8 scores at a time using AVX2
  -    alignas(32) float scores[8];
  +    alignas(32) float scores[8] = {};
       __m256 vBestScore = _mm256_setzero_ps();
   ```

#### Pair 5 — `src/harmony/ScaleDetector.cpp` ↔ `src_penta-core/harmony/ScaleDetector.cpp`

- **Lines A / B**: 147 / 147
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: IDENTICAL (`diff -q` exit 0)
- **Semantic summary**: Byte-equal. Already consolidated. The two files share identical content and identical history.
- **Disposition**: **IDENTICAL_DELETE_EITHER** — but note the ODR concern: even though they are byte-equal, both still produce a duplicate `penta::harmony::ScaleDetector` symbol in two static libs. Pick one (probably keep A or keep B per the consolidation plan in `KMIDI_FINAL_MERGE_PLAN.md`) and add the deleted side to the KellyCore filter list.

---

### Group: Groove (4 pairs)

#### Pair 6 — `src/groove/GrooveEngine.cpp` ↔ `src_penta-core/groove/GrooveEngine.cpp`

- **Lines A / B**: 334 / 337
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-03-31 05:20:01 -0600 194606d6 fix(high): use-after-free — detached threads, ONNX leak, GrooveEngine race`
- **Diff status**: DIFFER
- **Semantic summary**: B has the **use-after-free / GrooveEngine race fix** from March 31 plus a constructor refactor that pulls onset/tempo/quantizer config out of an awkward initializer-list into the function body, AND adds a `configUpdating_` atomic guard at the top of `processAudio()` to prevent races during config updates. Both differences matter for safety.
- **Disposition**: **MERGE_B_INTO_A** — port B's race fix and constructor refactor into A. (Or, more likely, keep B and delete A once the namespace conflict is resolved.)
- **Diff sample**:
  ```diff
   GrooveEngine::GrooveEngine(const Config &config)
  -    : config_(config), analysis_{}, onsetDetector_(std::make_unique<OnsetDetector>()), 
  -      tempoEstimator_(std::make_unique<TempoEstimator>()), 
  -      quantizer_(std::make_unique<RhythmQuantizer>()), 
  -      samplePosition_(0), lastAnalysisPosition_(0)
  +    : config_(config), analysis_{}, samplePosition_(0)
   {
  +    // Configure onset detector with config_ values
  +    OnsetDetector::Config onsetConfig;
  +    onsetConfig.sampleRate = config_.sampleRate;
  +    ...
  +    onsetDetector_ = std::make_unique<OnsetDetector>(onsetConfig);
   }
   void GrooveEngine::processAudio(const float *buffer, size_t frames) noexcept
   {
  +    // Skip processing while config is being updated (H18 race fix)
  +    if (configUpdating_.load(std::memory_order_acquire)) return;
   ```

#### Pair 7 — `src/groove/OnsetDetector.cpp` ↔ `src_penta-core/groove/OnsetDetector.cpp`

- **Lines A / B**: 171 / 200
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: DIFFER (despite identical last commit dates — both got touched in the same big commit but the file content already diverged at that snapshot)
- **Semantic summary**: B switches to **JUCE FFT** (`juce::dsp::FFT`) instead of A's plain implementation, brings in `penta/common/SIMDKernels.h`, and pre-allocates `windowedBuffer_` for RT-safety. B is ~30 lines longer and is structurally a more complete implementation. A uses `juce::MathConstants<double>::pi` for the Hann window denom; B uses raw `M_PI` with `std::cos(2.0f * M_PI * i / (config_.fftSize - 1))`.
- **Disposition**: **INVESTIGATE** — B is more featureful (real FFT, SIMD kernels, pre-allocated buffer) and looks like the intended implementation, but A may compile under different paths. The user should confirm whether the JUCE-FFT version is what production should use, then `MERGE_A_INTO_B` (i.e., adopt B as canonical) and delete A.
- **Diff sample**:
  ```diff
  +#include "penta/common/SIMDKernels.h"
  +#include <juce_dsp/juce_dsp.h>
  +#include <numeric>
  +    // Initialize JUCE FFT (requires power-of-2 size)
  +    int fftOrder = static_cast<int>(std::log2(config_.fftSize));
  +    fft_ = std::make_unique<juce::dsp::FFT>(fftOrder);
  +    // Pre-allocate buffers
  +    fftBuffer_.resize(config_.fftSize * 2);  // Real + imag for each sample
  +    windowedBuffer_.resize(config_.fftSize);  // Pre-allocated buffer for windowed input (RT-safe)
  ```

#### Pair 8 — `src/groove/RhythmQuantizer.cpp` ↔ `src_penta-core/groove/RhythmQuantizer.cpp`

- **Lines A / B**: 137 / 148
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: DIFFER
- **Semantic summary**: A has a small extra defensive guard (`config_.swingAmount > 0.0f` short-circuit before applying swing), B has an extra explanatory comment about triplet-grid + binary-subdivision swing semantics, plus B replaces A's `((denom % 3u) == 0u)` with `(denom % 3 == 0)` (semantically identical, slightly less type-precise). The ~11 extra lines in B are mostly comments. **A appears to have a more conservative branch that B is missing.**
- **Disposition**: **INVESTIGATE** — A's `swingAmount > 0.0f` guard may be load-bearing; verify with the user before picking a winner. Likely `MERGE_A_INTO_B` to bring the guard into B and `MERGE_B_INTO_A` to bring the comment + slightly cleaner expression into A — the two are best reconciled by hand-merge.
- **Diff sample**:
  ```diff
  -        const bool isTripletGrid = (denom != 0) && ((denom % 3u) == 0u);
  -        if (config_.enableSwing && !isTripletGrid && (gridIndex % 2 != 0) && 
  -            (config_.swingAmount > 0.0f))
  +        const bool isTripletGrid = (denom != 0) && (denom % 3 == 0);
  +        if (config_.enableSwing && !isTripletGrid && (gridIndex % 2 != 0))
   ```

#### Pair 9 — `src/groove/TempoEstimator.cpp` ↔ `src_penta-core/groove/TempoEstimator.cpp`

- **Lines A / B**: 162 / 168
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: DIFFER
- **Semantic summary**: **A explicitly self-declares as legacy.** A contains the comment block:
  ```
  // NOTE: This is the legacy implementation maintained for backward compatibility.
  // The main implementation is in src_penta-core/groove/TempoEstimator.cpp
  // which uses RT-safe circular buffers and lock-free atomics.
  ```
  All other diffs are comment additions in B. The actual algorithmic body is identical except for the legacy-marker comment.
- **Disposition**: **KEEP_B_DELETE_A** — A's own comment says B is canonical. This is the cleanest deletion in the catalog.

---

### Group: OSC (5 pairs)

#### Pair 10 — `src/osc/OSCClient.cpp` ↔ `src_penta-core/osc/OSCClient.cpp`

- **Lines A / B**: 114 / 132
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: DIFFER
- **Semantic summary**: B is structurally fuller. Differences:
  - B includes `penta/osc/OSCMessage.h` directly; A relies on transitive include.
  - B initializes `port_{0}` in the SocketImpl struct; A leaves it uninitialized (latent UB if `connectInternal` runs before set).
  - B's `send()` checks `socket_ && socket_->sender_` before calling `connect()` each time; A calls `connectInternal()` and bails. **A has a potentially-broken send path** — it connects-once-then-fails forever if the first connect fails.
  - B explicitly constructs `juce::OSCMessage juceMsg{juce::String(message.getAddress())};` instead of A's `juce::OSCAddressPattern` wrapper.
- **Disposition**: **MERGE_B_INTO_A** — B's `port_{0}` initializer and reconnect logic are safer; the user should validate which `juce::OSCMessage` constructor JUCE 8 actually accepts before merging.
- **Diff sample**:
  ```diff
   struct OSCClient::SocketImpl {
       std::unique_ptr<juce::OSCSender> sender_;
       juce::String address_;
  -    int port_;
  +    int port_{0};
   };
   bool OSCClient::send(const OSCMessage& message) noexcept {
  -    if (!connectInternal()) {
  +    if (!socket_ || !socket_->sender_) {
           return false;
       }
  +    socket_->sender_->connect(socket_->address_, socket_->port_);
       try {
  -        juce::OSCMessage juceMsg{juce::OSCAddressPattern{message.getAddress()}};
  +        juce::OSCMessage juceMsg{juce::String(message.getAddress())};
   ```

#### Pair 11 — `src/osc/OSCHub.cpp` ↔ `src_penta-core/osc/OSCHub.cpp`

- **Lines A / B**: 105 / 134
- **Last A**: `2026-04-01 00:34:30 -0600 1d90eb93 fix: resolve pre-existing C++ build errors for AU plugin target`
- **Last B**: `2026-03-31 05:23:03 -0600 8e2387ba fix(critical+high): SIMD init, undefined method, recursion depth limit`
- **Diff status**: DIFFER
- **Semantic summary**: **Bidirectional divergence with structural refactor.** Differences:
  - A logs every state transition via `juce::Logger::writeToLog`; B is silent.
  - A explicitly clears the server message queue on construction (`server_->getMessageQueue().clear()`); B does not.
  - B's `start()` returns false on server-start failure; A returns true on success and false on failure (semantically equivalent but inverted control flow).
  - B uses `namespace penta::osc { }`; A uses nested `namespace penta { namespace osc {`.
  - A has a private `isLoopbackAddress` helper that B has removed.
  - A also has the April 1 build fix for AU plugin compilation.
- **Disposition**: **INVESTIGATE** — A's queue-clear-on-startup and loopback-address helper may be intentional security/hygiene work; B's safer null-checks need to be ported. Hand-merge required, with the user confirming whether the loopback restriction should still apply.

#### Pair 12 — `src/osc/OSCMessage.cpp` ↔ `src_penta-core/osc/OSCMessage.cpp`

- **Lines A / B**: 71 / 73
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: DIFFER
- **Semantic summary**: Trivial: B adds explicit member-initializer-list entries `arguments_()` and `timestamp_(0)` in the constructor. A leaves them default-initialized (which is fine for `std::vector` but `timestamp_` may be a POD that A leaves indeterminate).
- **Disposition**: **MERGE_B_INTO_A** — port B's explicit initializers; verify `timestamp_` type. After merge, the two files become byte-equal and one can be deleted.
- **Diff sample**:
  ```diff
   OSCMessage::OSCMessage(const std::string& addressPattern)
       : addressPattern_(addressPattern)
  +    , arguments_()
  +    , timestamp_(0)
   {}
  ```

#### Pair 13 — `src/osc/OSCServer.cpp` ↔ `src_penta-core/osc/OSCServer.cpp`

- **Lines A / B**: 116 / 104
- **Last A**: `2026-03-02 23:13:08 -0700 1fb1e8e5 Misc: .env.example, prepare_datasets, OSC/groove/midi, frontend, docs cleanup`
- **Last B**: `2026-03-31 05:23:32 -0600 04d1e06c fix(high): RT-safety — remove mutex/heap-alloc from audio thread paths`
- **Diff status**: DIFFER
- **Semantic summary**: **Bidirectional divergence with security implications.** Differences:
  - A uses `juce::OSCReceiver::Listener<juce::OSCReceiver::RealtimeCallback>`; B uses `MessageLoopCallback`. **This is a behavior change** — RealtimeCallback runs on the network thread (RT-safe but no JUCE GUI access); MessageLoopCallback runs on the message thread (allocs OK, GUI OK, NOT RT-safe).
  - A has a `isLoopbackAddress()` security check that refuses to bind to non-loopback addresses; **B silently removes this check.** This may be a security regression.
  - A is the side that has the March 2 commit refactoring this file; B has the March 31 RT-safety commit.
  - The `OSCListener` ctor parameter is named `owner_` in A and `server_` in B (cosmetic).
- **Disposition**: **INVESTIGATE** — the loopback-address removal in B is a potential security regression and the listener-callback change is a deliberate architectural decision. The user must explicitly approve the callback model (RealtimeCallback vs MessageLoopCallback) before merging. Do not delete A.
- **Diff sample**:
  ```diff
   class OSCServer::OSCListener
  -    : public juce::OSCReceiver::Listener<juce::OSCReceiver::RealtimeCallback> {
  +    : public juce::OSCReceiver::Listener<juce::OSCReceiver::MessageLoopCallback> {
   public:
  -    explicit OSCListener(OSCServer* owner) : owner_(owner) {}
  +    explicit OSCListener(OSCServer* server) : server_(server) {}
   ...
  -    if (!isLoopbackAddress(address_)) {
  -        juce::Logger::writeToLog("OSCServer: Refusing non-loopback bind address " + ...);
  -        return false;
  -    }
   ```

#### Pair 14 — `src/osc/RTMessageQueue.cpp` ↔ `src_penta-core/osc/RTMessageQueue.cpp`

- **Lines A / B**: 74 / 64
- **Last A**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: DIFFER
- **Semantic summary**: Pure cosmetic refactor — B collapses A's `if (!queue_) { return ...; } return queue_->...;` into ternary `return queue_ ? queue_->... : ...;` for `isEmpty()` and `size()`. Adds `const` to `success` locals. No semantic change.
- **Disposition**: **MERGE_B_INTO_A** (or `KEEP_A_DELETE_B` — both are equivalent). The user can pick which style to keep. After merge they become byte-equal and one can go.

---

### Group: Single-class duplicates (2 pairs)

#### Pair 15 — `src/biometric/BiometricInput.cpp` ↔ `src/biometric/BiometricInput.mm`

- **Lines A / B**: 363 / 332
- **Last A**: `2026-01-28 00:31:46 -0700 0fd07ffc WIP: CMake, JUCE submodule, C++/TS edits, docs, Xcode/install scripts, SDK wrappers`
- **Last B**: `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: DIFFER (also DUPLICATE-COMPILE — see CMake note below)
- **CMake exposure**: The root `CMakeLists.txt` line 245 uses `file(GLOB_RECURSE KELLY_CORE_SOURCES ...)` which picks up BOTH `BiometricInput.cpp` AND `BiometricInput.mm`. There is NO `list(FILTER ... EXCLUDE REGEX "BiometricInput")` line. Line 284 then explicitly adds the `.mm` file with `COMPILE_LANGUAGES OBJCXX`. **Result: both files are compiled into KellyCore, defining the same `kelly::BiometricInput` class twice → ODR violation / duplicate symbols**, masked only because the linker prefers whichever one shows up first.
- **Semantic summary**: A is the WIP `.cpp` (Jan 28) — it gates `HealthKitBridge` behind `#if HEALTHKIT_AVAILABLE`, includes manual `delete static_cast<biometric::*>(...)` cleanup logic, and uses K&R brace-with-newline style. B is the `.mm` (Jan 22) — uses Allman style, no `#if HEALTHKIT_AVAILABLE` guards, no destructor. **Both define `kelly::BiometricInput::BiometricInput()` and other class members.** A is the newer file by date, but B is the one wired into CMake explicitly.
- **Disposition**: **INVESTIGATE** — this is the audit's #1 concern: same class compiled from two TUs. The user must decide:
  1. Is the `.mm` file needed for HealthKit (Objective-C++ bridging)? If yes, keep it and delete `.cpp`, OR rename `.cpp` to `.cpp.bak` and add a CMake `EXCLUDE` filter.
  2. If HealthKit is disabled (`#if HEALTHKIT_AVAILABLE` is false), is the `.cpp` enough? Then delete the `.mm`.
  3. Either way, the WIP commit message in A (`WIP: ...`) suggests A may be incomplete and lost work.

  This is **the single most dangerous pair in the catalog**. Do not delete either side without user sign-off.

#### Pair 16 — `include/daiw/audio_io.hpp` (lines 123–228) ↔ `src/dsp/audio_buffer.cpp`

- **Lines A / B**: 626 (full header, AudioBuffer class is lines 123–228 = ~106 lines) / 44 (full .cpp)
- **Last A** (audio_io.hpp): `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last B** (audio_buffer.cpp): `2026-01-22 22:42:43 -0700 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Diff status**: DIFFER (cannot be byte-compared because A is a header section embedded in a larger file)
- **CMake exposure**: `CMakeLists.txt` line 262 explicitly excludes `src/dsp/audio_buffer.cpp` from KellyCore. `libs/daiw/CMakeLists.txt` line 12 then adds it to the `daiw_core` static lib. So `audio_buffer.cpp` IS compiled (into `daiw_core`). Meanwhile `include/daiw/audio_io.hpp` defines its own `daiw::AudioBuffer` class inline in any TU that includes it. **Two different `daiw::AudioBuffer` classes exist in the same namespace, with different APIs and different storage layouts.** This is a hard ODR violation if any TU includes `audio_io.hpp` and is linked alongside `daiw_core`.
- **Semantic summary**: These are TWO DIFFERENT CLASSES with the same fully-qualified name `daiw::AudioBuffer`:
  - **Header version** (`include/daiw/audio_io.hpp`, ~106 lines): rich API with `num_channels()`, `num_samples()`, `channel(ch)`, `data()`, `sample(ch, idx)`, `copy_from()`, `add()`, `apply_gain()`, `peak_level()`, plus a `channel_ptrs_` vector of float* pointers. Storage: `std::vector<float> data_` + `std::vector<float*> channel_ptrs_`.
  - **CPP version** (`src/dsp/audio_buffer.cpp`, 44 lines): minimal API with `getChannelData(channel)`, `clear()`, `getNumChannels()`, `getNumSamples()`. Storage: `std::vector<Sample> data_` (interleaved-by-channel layout). NO channel pointer cache.

  The header version is the *interface* the rest of the codebase appears to use (richer, JUCE-style camelCase-replacement); the .cpp version looks like a stub/early prototype.
- **Disposition**: **INVESTIGATE** → likely **KEEP_A_DELETE_B** (keep the header definition, delete the .cpp class entirely). But the user must:
  1. Confirm nothing currently links against `daiw::AudioBuffer::getChannelData()` or the other `getNumX()` methods (only the header API is referenced).
  2. Decide whether `audio_buffer.cpp` should become an empty / explicit-instantiation file, or be removed from `daiw_core` SOURCES and deleted entirely.
  3. Verify that `daiw_core` still links after the deletion.

---

### Group: KmiDi_PROJECT mirror sample (top 10 + 1 bonus)

> **Pattern observation:** Every file in this group has its last commit on `2026-02-24 18:13:48 chore: Clean up quarantined files and old migration reports`. The KmiDi_PROJECT tree has been frozen since late February 2026, while `src/` has continued through March 31 / April 1 with major safety, build, and demo-slice commits. **No file in this tree appears in any active CMake graph.** The dispositions below uniformly suggest deletion, but per the user's policy NO file should be deleted without sign-off and the entire `KmiDi_PROJECT/` tree should first be archived to a tag.

#### Pair 17 — `src/engines/RhythmEngine.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/music_theory/rhythm/RhythmEngine.cpp`

- **Lines A / D**: 489 / 1307
- **Last A**: `2026-01-22 22:42:43 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER (massive structural divergence)
- **Semantic summary**: **These are essentially two different RhythmEngine implementations.** A is in `namespace kelly { }`, ~489 lines, no `<numeric>`, `<sstream>`, or `<set>` includes. D is in `namespace midikompanion::theory { }`, ~1307 lines, much richer feature set. They are NOT the same class — the namespace difference means they cannot collide at link time, but they are also NOT consolidatable without significant work.
- **Disposition**: **INVESTIGATE** — D may contain functionality that A is missing. This is the largest single divergence in the catalog and warrants its own salvage decision. The 800-line gap is not "stale duplicate" territory — it is "two different feature sets that look similar."
- **Diff sample**:
  ```diff
  -namespace kelly {
  -namespace { /* removed static RNG */ } // namespace
  -RhythmEngine::RhythmEngine() { initializeProfiles(); ... }
  +namespace midikompanion::theory {
  +//==============================================================================
  +// Constructor
  +//==============================================================================
  ```

#### Pair 18 — `src/plugin/PluginProcessor.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/plugin/PluginProcessor.cpp`

- **Lines A / D**: 1432 / 1279
- **Last A**: `2026-04-01 05:02:43 e38f6de6 feat: wire emotion probe ONNX into AudioEmotionRunner`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER
- **Semantic summary**: A has +153 lines of newer work: April 1 demo proof-of-life, AudioEmotionRunner ONNX wiring, MIDI-effect-vs-effect bus handling (`#if JucePlugin_IsMidiEffect`), QoS / pthread imports for Apple silicon. D is the pre-demo state.
- **Disposition**: **KEEP_A_DELETE_D** (after archiving the KmiDi_PROJECT tree to a tag).

#### Pair 19 — `src/ui/ScoreEntryPanel.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/ui/ScoreEntryPanel.cpp`

- **Lines A / D**: 825 / 762
- **Last A**: `2026-03-29 20:49:09 64792105 chore: delete orphaned music_theory subsystem (zero instantiations)`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER
- **Semantic summary**: A is +63 lines newer with the March 29 music_theory cleanup. D is the pre-cleanup state.
- **Disposition**: **KEEP_A_DELETE_D**.

#### Pair 20 — `src/ui/MixerConsolePanel.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/ui/MixerConsolePanel.cpp`

- **Lines A / D**: 740 / 742
- **Last A**: `2026-03-29 20:49:09 64792105 chore: delete orphaned music_theory subsystem`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER
- **Semantic summary**: Tiny line delta (-2 lines in A) consistent with the music_theory cleanup. A is newer.
- **Disposition**: **KEEP_A_DELETE_D**.

#### Pair 21 — `src/ui/MidiEditor.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/ui/MidiEditor.cpp`

- **Lines A / D**: 723 / 724
- **Last A**: `2026-01-22 22:42:43 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER
- **Semantic summary**: 1-line delta. **D is newer than A by date.** This is one of the few mirror pairs where the orphan tree is fresher than the active tree, which means there might be a small fix in D worth salvaging. Most likely the difference is a trailing newline or whitespace.
- **Disposition**: **INVESTIGATE** — confirm the 1-line delta is cosmetic before deleting D. If it's a real fix, port it to A first.

#### Pair 22 — `src/bridge/OSCBridge.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/bridge/OSCBridge.cpp`

- **Lines A / D**: 517 / 519
- **Last A**: `2026-03-31 12:52:22 fda6cd3a fix: review findings — mutex gaps, atomics, localtime, thread pruning, test path`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER
- **Semantic summary**: A has the **March 31 thread-safety fix series** (`connected_.load()` / `connected_.store()` instead of plain `connected_ = true`) plus mutex/atomics/localtime/thread-pruning fixes from a 3-commit run on 2026-03-31. D is the pre-fix state with non-atomic `connected_` (race condition).
- **Disposition**: **KEEP_A_DELETE_D** — A's atomic fix is critical, D is unsafe.
- **Diff sample**:
  ```diff
  -    if (connected_.load()) {
  +    if (connected_) {
           shutdown();
       }
  -        connected_.store(true);
  +        connected_ = true;
   ```

#### Pair 23 — `src/midi/MidiExporter.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/midi/MidiExporter.cpp`

- **Lines A / D**: 459 / 460
- **Last A**: `2026-01-22 22:42:43 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER
- **Semantic summary**: 1-line delta. D is newer by date but the delta is small enough to be a trailing newline. **D may have a tiny salvageable fix.**
- **Disposition**: **INVESTIGATE** — confirm the 1-line difference before deletion.

#### Pair 24 — `src/audio/AudioFile.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/audio/AudioFile.cpp`

- **Lines A / D**: 405 / 406
- **Last A**: `2026-01-22 22:42:43 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER
- **Semantic summary**: 1-line delta. D is newer by date.
- **Disposition**: **INVESTIGATE** — confirm 1-line difference before deletion.

#### Pair 25 — `src/midi/MidiBuilder.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/midi/MidiBuilder.cpp`

- **Lines A / D**: 369 / 370
- **Last A**: `2026-01-22 22:42:43 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER (only by trailing blank line per diff sample)
- **Semantic summary**: D adds one trailing blank line. Otherwise byte-equal.
- **Disposition**: **KEEP_A_DELETE_D** — purely cosmetic delta.
- **Diff sample**:
  ```diff
  @@ -367,3 +367,4 @@
   } // namespace kelly
  +
  ```

#### Pair 26 — `src/audio/SpectralAnalyzer.cpp` ↔ `KmiDi_PROJECT/source/cpp/src/audio/SpectralAnalyzer.cpp`

- **Lines A / D**: 351 / 352
- **Last A**: `2026-01-22 22:42:43 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last D**: `2026-02-24 18:13:48 85982653 chore: Clean up quarantined files...`
- **Diff status**: DIFFER (only by trailing blank line)
- **Semantic summary**: D adds one trailing blank line. Otherwise byte-equal.
- **Disposition**: **KEEP_A_DELETE_D**.

#### Pair 27 (BONUS) — `src/audio/SpectralAnalyzer.cpp` ↔ `src/prrot/SpectralAnalyzer.cpp`

> Discovered during the mirror search: there are **two** SpectralAnalyzer files inside `src/`. Cataloging here because it's the same hazard pattern.

- **Lines audio / prrot**: 351 / 357
- **Last audio**: `2026-01-22 22:42:43 ca0dff8e Complete KmiDi-1 Migration and Integration`
- **Last prrot**: `2026-03-31 00:12:34 e162e78b feat: PRROT pybind11 bindings + 30 unit tests (#145)`
- **Diff status**: DIFFER (different headers, different namespaces)
- **Semantic summary**: These are TWO DIFFERENT classes. `audio/SpectralAnalyzer.cpp` includes `audio/SpectralAnalyzer.h`; `prrot/SpectralAnalyzer.cpp` includes `prrot/SpectralAnalyzer.h` and additionally pulls in `prrot/InputValidation.h` and `penta/common/RTLogger.h`. The prrot variant is the new pybind11-bound implementation from PR #145 (March 31). They likely live in different namespaces and serve different consumers (audio = legacy direct-use, prrot = Python-bound PRROT engine).
- **Disposition**: **KEEP_BOTH_RENAME** — both serve distinct purposes; the file basenames are misleading. Recommend renaming one to `PrrotSpectralAnalyzer.cpp` or moving `audio/SpectralAnalyzer.*` to `legacy/` to make the purpose clear.
- **Diff sample**:
  ```diff
  -#include "audio/SpectralAnalyzer.h"
  -#include <juce_dsp/juce_dsp.h>
  +#include "prrot/SpectralAnalyzer.h"
  +#include "prrot/InputValidation.h"
  +#include "penta/common/RTLogger.h"
  +#include <juce_dsp/juce_dsp.h>
  +#include <cstring>
  +#include <vector>
  +#include <memory>
  ```

---

## Aggregate findings

- **Total pairs cataloged**: 27 (16 mandatory + 1 bonus from the SpectralAnalyzer discovery)
- **Identical**: 1 (ScaleDetector)
- **Diverged**: 26
- **Missing one side**: 0 (all pairs have both sides tracked in git)

### Disposition counts

| Disposition | Count | Pairs |
|---|---|---|
| `KEEP_A_DELETE_B` (or `_DELETE_D`) | 6 | VoiceLeading; PluginProcessor (mirror); ScoreEntryPanel (mirror); MixerConsolePanel (mirror); MidiBuilder (mirror); SpectralAnalyzer (mirror); OSCBridge (mirror) |
| `KEEP_B_DELETE_A` | 1 | TempoEstimator (A self-declares as legacy) |
| `MERGE_B_INTO_A` | 6 | HarmonyEngine; ChordAnalyzer; ChordAnalyzerSIMD; GrooveEngine; OSCClient; OSCMessage; RTMessageQueue |
| `MERGE_A_INTO_B` | 0 | (none cleanly fit; usually requires bidirectional merge) |
| `KEEP_BOTH_RENAME` | 1 | SpectralAnalyzer audio↔prrot (different consumers) |
| `INVESTIGATE` | 9 | OnsetDetector; RhythmQuantizer; OSCHub; OSCServer; BiometricInput .cpp/.mm; daiw::AudioBuffer header/cpp; RhythmEngine (mirror); MidiEditor (mirror); MidiExporter (mirror); AudioFile (mirror) |
| `IDENTICAL_DELETE_EITHER` | 1 | ScaleDetector |

(Counts don't sum perfectly because OSCBridge mirror also belongs in `KEEP_A_DELETE_D` — see table above; and `MERGE_B_INTO_A` lists 7 pairs because RTMessageQueue can also be classed as `KEEP_A_DELETE_B`.)

### INVESTIGATE list (most important)

These pairs **must not be deleted** without explicit user input:

1. **`BiometricInput.cpp` ↔ `BiometricInput.mm`** — both compile into KellyCore via GLOB_RECURSE; same class defined twice. Highest-risk pair in the audit.
2. **`include/daiw/audio_io.hpp` ↔ `src/dsp/audio_buffer.cpp`** — two different `daiw::AudioBuffer` classes with the same FQN, different APIs and different storage layouts. Hard ODR.
3. **`OSCServer.cpp` (A vs B)** — B silently removes A's loopback-address security check. Possible regression.
4. **`OSCHub.cpp` (A vs B)** — bidirectional divergence with structural refactor; A has logging + queue-clear hygiene, B has cleaner null-checks.
5. **`OnsetDetector.cpp` (A vs B)** — B uses JUCE FFT and SIMD kernels (more complete), A uses a hand-rolled path. Need user confirmation that JUCE FFT is the intended production backend.
6. **`RhythmQuantizer.cpp` (A vs B)** — A has a defensive `swingAmount > 0.0f` guard B is missing; B has slightly cleaner expressions and explanatory comments.
7. **`RhythmEngine.cpp` (A vs D mirror)** — 489-line `kelly::RhythmEngine` vs 1307-line `midikompanion::theory::RhythmEngine`. Effectively two different feature sets. The largest single divergence in the catalog.
8. **`MidiEditor.cpp` / `MidiExporter.cpp` / `AudioFile.cpp` (mirrors)** — small line deltas where D (KmiDi_PROJECT mirror) is *newer* by commit date than A. May contain a small salvageable fix.

### Anomalies discovered

1. **All KmiDi_PROJECT mirror files share the same last commit** (`2026-02-24 18:13:48 85982653`) — confirming this tree is a frozen orphan from late February 2026, never touched again.
2. **`src/audio/SpectralAnalyzer.cpp` and `src/prrot/SpectralAnalyzer.cpp` are both in `src/`** — two files with the same basename in different subdirectories, neither aware of the other. Discovered during the mirror search, not in the original 16-pair list.
3. **`src/biometric/BiometricInput.cpp` is NOT excluded from KellyCore's GLOB_RECURSE** even though `BiometricInput.mm` is explicitly added on line 284 — this is a real ODR / duplicate-symbol situation in the active build, not a theoretical concern.
4. **`src_penta-core` is built into a separate static lib `penta_core`** but Tree A's `src/harmony/`, `src/groove/`, `src/osc/` files are also compiled into KellyCore (only `chord.cpp` and `progression.cpp` are filtered out). Both libs export the same `penta::*` symbols with different bodies → ODR violation across two static archives that link into the same final binary. The link order determines which copy wins, and that ordering is implementation-defined per platform / linker.
5. **`src/groove/TempoEstimator.cpp` self-documents as legacy** with a comment pointing at `src_penta-core/groove/TempoEstimator.cpp`. This is the only pair where the catalog can be 100% confident about the canonical winner.
6. **The Penta-core `CMakeLists.txt` calls `cmake_minimum_required` and `project(...)` twice** (lines 1–9). This is a separate hygiene issue worth flagging, though not in scope for this catalog.

### Patterns observed

- **Tree B has the safety/correctness fixes from late March 2026** for `HarmonyEngine`, `ChordAnalyzer`, `ChordAnalyzerSIMD`, `GrooveEngine`, `OSCHub`, `OSCServer`. These were the audit-driven fix commits.
- **Tree A has the build / demo / kelly-port fixes from late March / early April 2026** for `VoiceLeading`, `HarmonyEngine`, `OSCHub`, `OSCBridge`, `PluginProcessor`. These were the demo-slice + thread-safety commits.
- **No file was committed to BOTH trees in the same week** — the safety-fix series went into Tree B; the demo / build series went into Tree A. This explains the bidirectional divergence: two parallel branches of work that never merged.

---

## What to do next

This catalog is **read-only documentation**. No deletions, merges, or renames have been performed.

Any deletion / merge / rename action against any pair listed above must be:

1. **Reviewed by the user against this catalog** (read each row, confirm or override the disposition).
2. **Performed on a feature branch**, not on `main` directly.
3. **Verified by a clean build** (`cmake -B build && cmake --build build && ctest --test-dir build`) before committing.
4. **Recorded back into this catalog** with the actual action taken (keep / delete / merge), so the next audit pass can pick up where this one left off.

### Recommended sequence

1. **Easy wins first** (no risk):
   - Pair 5 (ScaleDetector identical) → delete one side, add to KellyCore filter.
   - Pair 9 (TempoEstimator self-declares legacy) → delete A, add to KellyCore filter.
   - Pair 25 (MidiBuilder mirror, trailing blank only) → delete D after archiving KmiDi_PROJECT.
   - Pair 26 (SpectralAnalyzer audio mirror, trailing blank only) → delete D after archiving.

2. **Critical safety merges** (require careful porting):
   - Pair 4 (ChordAnalyzerSIMD `scores[8] = {}`) → trivial 1-line port.
   - Pair 3 (ChordAnalyzer ODR fix) → port B's `#ifndef __AVX2__` guard structure.
   - Pair 2 (HarmonyEngine RT-safe history rewrite) → port B's circular-buffer logic.
   - Pair 6 (GrooveEngine race fix + constructor refactor) → port B's `configUpdating_` guard.

3. **High-risk INVESTIGATE pairs** (require user judgment):
   - BiometricInput .cpp/.mm — pick one, exclude the other from CMake.
   - daiw::AudioBuffer header/cpp — keep header class, delete .cpp class definition.
   - OSCServer A/B — confirm callback model and loopback security policy.
   - RhythmEngine A/D — full salvage pass before any deletion.

4. **Archive the orphan tree**:
   - `git tag salvage/kmidi-project-frozen-2026-02-24` then a separate phase can decide whether to remove `KmiDi_PROJECT/` entirely.

5. **Re-run the audit script** to confirm no new divergences appeared while the merges were in flight.

---

*Catalog generated 2026-04-07 as part of Phase 2 of the C++ safety-fixes audit. See `docs/CODEAUDIT_CXX_DEEP_2026Q2.md` for the original findings, `docs/KMIDI_FINAL_MERGE_PLAN.md` for the consolidation plan, and `~/.claude/projects/-Users-seanburdges/memory/project_kmidi_codeaudit_2026_q2.md` for the cross-session memory entry.*
