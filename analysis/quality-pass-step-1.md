# Quality Pass: Step 1 — Fix 5 ODR Hazards + Delete Dead Stubs
**Date:** 2026-03-29
**Commit:** 6ff9d7a6

## Immediate Verification

| Check | Result |
|-------|--------|
| vst3/ stubs deleted | **PASS** |
| Lowercase plugin stubs deleted | **PASS** |
| Repo-root HarmonyEngine.cpp deleted | **PASS** |
| engine/MidiGenerator.h deleted | **PASS** — only midi/MidiGenerator.h remains |
| daiw MemoryPool renamed | **PASS** — now `MutexMemoryPool` in memory.hpp |
| daiw RingBuffer renamed | **PASS** — now `EventRingBuffer` in memory.hpp (line 97) |
| EmotionThesaurus class renamed | **PASS** — now `InlineEmotionThesaurus` |
| KellyBrain class renamed | **PASS** — now `KellyBrainLegacy` |

## BUGS FOUND

### P0: Constructor name mismatches (will fail to compile)

1. **IntentProcessor.h:127** — Constructor still `EmotionThesaurus()`, should be `InlineEmotionThesaurus()`
   - Class renamed at line 125 but constructor at line 127 not updated

2. **Kelly.h:32** — Constructor still `KellyBrain(int tempo, unsigned int seed)`, should be `KellyBrainLegacy(...)`
   - Class renamed at line 30 but constructor at line 32 not updated

**These WILL cause compile errors.** The implementation session likely hasn't tried to build yet.

### P1: kelly_bridge.cpp:207 TODO not cleaned up

The TODO says "Fix MidiGenerator.h structure conflicts and re-enable KellyBrain" — the conflict IS now resolved (engine/MidiGenerator.h deleted). The TODO should be removed or updated to reflect remaining work (re-enable KellyBrain export).

## Commit Stats

11 files changed, 34 insertions(+), 1,381 deletions(-)

## ODR Fix Summary

| ODR Hazard | Fix Applied | Status |
|-----------|------------|--------|
| daiw::MemoryPool | Renamed to `MutexMemoryPool` | **FIXED** |
| daiw::RingBuffer (memory.hpp vs ring_buffer.hpp) | Renamed to `EventRingBuffer` | **FIXED** |
| kelly::EmotionThesaurus | Renamed to `InlineEmotionThesaurus` | **BUG — constructor mismatch** |
| kelly::KellyBrain | Renamed to `KellyBrainLegacy` | **BUG — constructor mismatch** |
| kelly::MidiGenerator | Deleted engine/ version | **FIXED** |

## Status: PASS — 2 constructor bugs fixed by quality pass agent

Fixes applied:
- IntentProcessor.h:127 — `EmotionThesaurus()` → `InlineEmotionThesaurus()`
- Kelly.h:32 — `KellyBrain(...)` → `KellyBrainLegacy(...)`
