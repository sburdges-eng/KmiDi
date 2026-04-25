# RT / FFI follow-ups — 2026 Q2

> **Source**: explicitly declared "Known follow-ups (out of scope)" in PR
> [#149 (`audit/rt-ffi-safety-2026q2`)](https://github.com/sburdges-eng/KmiDi/pull/149).
>
> This document scopes the two named follow-ups so each can be landed
> as its own PR (one logical change per commit), independent of the
> 13-PR audit stack that #149 anchors.

---

## Follow-up A — Full `kelly::Wound` consolidation

### Status after PR #150

PR #150 (`follow-up/wound-odr-consolidation`) deletes two of the three
divergent `kelly::Wound` definitions:

| Site | PR #150 action |
|---|---|
| `common/Types.h` (3-field) | **Removed**, comment block left in place. |
| `src/engine/IntentProcessor.h` (6-field) | **Removed**, replaced by explicit `#include "../common/KellyTypes.h"`. |
| `src/common/KellyTypes.h` (8-field) | **Canonical**, extended with `timestamp / context / triggers` from the IntentProcessor.h layout. |

The Bugbot HIGH finding on #150 (also Codex P1) flagged that this still leaves
four legacy consumers broken — addressed by [#163](https://github.com/sburdges-eng/KmiDi/pull/163).
Once #163 lands, the **header-level** ODR is gone.

### What remains

What PR #150 explicitly defers ("the 'intensity is alias for urgency'
semantic preserved as two separate fields. Aliasing happens at the
consumer layer; unifying them is a behaviour change and out of scope for
this cleanup."):

The canonical `kelly::Wound` carries both `urgency` *and* `intensity` as
separate fields. Real consumers diverge on which one they write:

| Consumer | Writes | Reads (downstream) |
|---|---|---|
| `src/plugin/PluginProcessor.cpp` | `wound.intensity` | `IntentProcessor::processWound` reads `.intensity` ✓ |
| `src/engine/KellyBrain.cpp` | both `urgency` *and* `intensity` (mirrored) | various |
| `engine/StateMachineConductor.h` (post #163) | both, mirrored | `IntentPipeline::process(const Wound&)` |
| `tests/reference/test_emotion_engine_reference.cpp` (post #163) | both, mirrored | reference test |

Every consumer that reads `intensity` works. Anyone who reads `urgency`
without first checking `intensity` may see the documented default
(`0.5f`) instead of caller intent. **Today this manifests as no bug**
because no live caller reads `urgency` exclusively; the failure mode is
latent and ships the day someone writes new code that does.

### Concrete tasks

1. **Audit which consumers mutate / read each field.**
   - Run `rg -nC2 'wound\.(intensity|urgency)' src/ tests/ engine/ plugins/`.
   - For each writer that sets only one field, document the producer side
     in the same commit as the read-side change.

2. **Pick one canonical name in `KellyTypes.h::Wound`.**
   - Recommend keeping `intensity` (matches `KellyTypes.h::EmotionState::intensity`
     and the plugin/UI side of the codebase). Rename `urgency` → keep it as a
     non-canonical accessor, OR remove the field outright after consumers migrate.

3. **Sweep ~20 consumers.** Files known to reference `Wound`:
   ```
   src/bridge/kelly_ffi.{h,cpp}        src/engine/AdaptiveGenerator.{h,cpp}
   src/bridge/IntentBridge.cpp         src/engine/IntentPipeline.{h,cpp}
   src/bridge/kelly_bridge.cpp         src/engine/IntentProcessor.h
   src/common/IntentIRAdapter.cpp      src/engine/Kelly.h
   src/common/TypeAdapter.h            src/engine/KellyBrain.{h,cpp}
   src/engine/EmotionThesaurus.h       src/engine/MidiKompanionBrain.h
   src/engine/WoundProcessor.{h,cpp}   src/engine/test_kelly.cpp
   src/ml/MLBridge.{h,cpp}             src/plugin/PluginEditor.{h,cpp}
   src/plugin/PluginProcessor.{h,cpp}  src/plugin/PluginState.{h,cpp}
   src/ui/EmotionWorkstation.{h,cpp}   src/voice/LyricGenerator.{h,cpp}
   src/voice/VoiceSynthesizer.{h,cpp}  tests/cpp/test_kelly_ffi.cpp
   tests/runtime_contract_tests.cpp    plugins/plugin/PluginProcessor.{h,cpp}
   ```
   Per-file work: `urgency` write → use accessor, or migrate to `intensity`.

4. **Drop `src/common/TypeAdapter.h::convertToUnifiedWound /
   convertToLegacyWound`** if no caller remains after the sweep.

### Acceptance

- `rg 'struct\s+Wound' --type=cpp --type=h` shows exactly **one** definition (`src/common/KellyTypes.h:209`).
- `rg 'wound\.urgency'` returns zero hits, OR every hit also writes `wound.intensity` to the same value.
- `cmake --build build --target KellyCore KellyFFI -j8` clean.
- `cargo test --manifest-path engine/intent_ir/Cargo.toml` clean.

### Non-scope

- Renaming `Wound` itself (e.g. to `MotivationalWound` or `IntentWound`).
- Adding fields beyond what's in PR #150's canonical layout.

---

## Follow-up B — RT-safe arena-backed `StateBridge::emitStateUpdate`

### Status after PR #149

`src/bridge/StateBridge.{h,cpp}` was switched to `moodycamel::ConcurrentQueue`
(MPMC) so multiple engine writers can enqueue concurrently. The PR header
documents that `emitStateUpdate` is **not** RT-safe:

> StateBridge `emitStateUpdate` is documented as NOT audio-thread safe
> (std::string copy + ConcurrentQueue first-touch allocates). A fixed-size
> arena state channel for RT emission is a separate task.

`StateBridge::emitStateUpdate` is currently called from `KellyBrain::generateMidi`
(non-RT, UI/pipeline context). Wiring it from `processBlock` or any RT
worker would deadlock or allocate.

### What needs to happen

Two things stand between this and "RT-safe":

1. **No `std::string` copies on the producer side.** The current API:
   ```cpp
   void emitStateUpdate(std::string source, std::string json);
   ```
   constructs and copies two `std::string`s. RT path needs a fixed-capacity
   buffer with byte-count + small-buffer-optimization equivalent.

2. **No first-touch allocation in `ConcurrentQueue`.** `moodycamel::ConcurrentQueue`
   pre-allocates blocks per-producer-token, but only on first `enqueue`.
   RT side needs to acquire a producer token at init time (non-RT) and reuse
   it for every enqueue.

### Proposed API shape

```cpp
struct RTStateMessage {
    std::array<char, 32>  source;     // null-padded ASCII tag
    std::array<char, 224> payload;    // JSON, fixed cap, no escape needed
    uint16_t              source_len;
    uint16_t              payload_len;
    uint64_t              monotonic_seq;
};
static_assert(sizeof(RTStateMessage) == 256, "fits in two cache lines");

class StateBridge {
public:
    // Non-RT: acquire a token at startup (one per writer thread).
    using ProducerToken = moodycamel::ProducerToken;
    [[nodiscard]] ProducerToken acquireRTProducerToken();

    // RT-safe: noexcept, allocation-free, no locks.
    bool emitStateUpdateRT(ProducerToken& tok,
                           std::string_view source,
                           std::string_view payload) noexcept;

    // Existing emitStateUpdate(std::string, std::string) stays for
    // non-RT callers (UI thread, generation pipeline).
};
```

### Concrete tasks

1. **Add `RTStateMessage` POD** to `src/bridge/StateBridge.h`.
2. **Add `MPMCQueue<RTStateMessage>` member** sized at compile-time
   (e.g. `kRTStateQueueDepth = 1024`).
3. **Add `acquireRTProducerToken()`** that pre-warms the queue's per-producer
   block by calling `try_enqueue` then `try_dequeue` on a sentinel.
4. **Add `emitStateUpdateRT`** with the truncate-on-overflow contract
   (return `false` if `source.size() > 32` or `payload.size() > 224`).
5. **Worker thread reader loop**: drain `RTStateMessage` queue, convert to
   `std::string` only at the Python boundary.
6. **Test**: `tests/cpp/test_state_bridge_rt.cpp` — N producer threads
   pumping `emitStateUpdateRT`, single consumer thread, assert no
   message-loss / no allocation (use a `MallocHookSentinel` style guard
   if available).
7. **Wire one RT call site** as proof-of-life (likely
   `PluginProcessor::processBlock` or the `RTLogger::publish` path).

### Acceptance

- `pmap`-style allocation counter on the producer side shows zero
  allocations across 1M `emitStateUpdateRT` calls.
- Existing `KellyBrain::generateMidi` call site keeps the
  `std::string` API unchanged (no behaviour regression).
- `cmake --build build --target KellyCore KellyFFI KellyTests -j8` clean.
- `ctest --test-dir build-asan -R state_bridge_rt` clean.

### Non-scope

- Wiring `emitStateUpdateRT` to every potential RT producer in the codebase.
  This follow-up establishes the API + one validated caller; broader rollout
  comes after PR #149 + #163 ship.
- Backpressure / queue-overflow recovery beyond truncate-on-overflow.

---

## Tracking

| Follow-up | Branch (proposed) | PR (when opened) | Depends on |
|---|---|---|---|
| A — Wound full sweep | `audit/wound-consumer-sweep-2026q2` | TBD | #150, #163 |
| B — StateBridge RT arena | `audit/statebridge-rt-arena-2026q2` | TBD | #149, #151 |

Both can be developed independently after the audit stack lands;
neither blocks the other.
