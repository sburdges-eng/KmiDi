# Src / Penta-Core ODR Collision Survey (2026 Q2)

**Date:** 2026-04-21
**Scope:** Compare `src/` and `src_penta-core/` trees for genuine same-namespace same-class `.cpp` collisions that link into the same binary (`KellyCore` publicly links `penta_core`).
**Method:** basename join across both trees → namespace/class verification per hit → git log + line count + `diff -q` per pair.

## Correction to the prior audit memory

The 2026-04-07 C++ deep audit cited "4/5 harmony, 4/4 groove, 5/5 osc files actively diverged in body." That claim is **outdated**. Current directory listings:

| Directory | `src/` | `src_penta-core/` |
|-----------|--------|-------------------|
| `harmony/` | `chord.cpp`, `progression.cpp` (daiw_core, excluded from KellyCore), `VoiceLeading.cpp` | `ChordAnalyzer.cpp`, `ChordAnalyzerSIMD.cpp`, `HarmonyEngine.cpp`, `ScaleDetector.cpp` |
| `groove/` | empty (only `ACTIVE_DEVELOPMENT.md`) | `GrooveEngine.cpp`, `OnsetDetector.cpp`, `RhythmQuantizer.cpp`, `TempoEstimator.cpp` |
| `osc/` | `OSCClient.cpp`, `OSCHub.cpp`, `OSCServer.cpp` | `OSCMessage.cpp`, `RTMessageQueue.cpp` |

**Zero basename collisions in harmony/groove/osc.** These trees are disjoint today (consolidation already landed in prior branches).

## Actual collisions (cross-tree and intra-tree)

A broader basename join across ALL subdirectories found **5 genuinely colliding `.cpp` basenames** that link into the same library graph:

```
comm -12 <(find src -name '*.cpp' -printf '%f\n' | sort -u) \
         <(find src_penta-core -name '*.cpp' -printf '%f\n' | sort -u)
→ AudioAnalyzer.cpp
→ DiagnosticsEngine.cpp
→ GrooveEngine.cpp
→ PerformanceMonitor.cpp
→ RTLogger.cpp
```

Not all are genuine ODR — several are same-basename / different-namespace. Per-collision analysis below.

---

## ODR collisions (must resolve)

### 1. `RTLogger.cpp` — real ODR, **divergent content**

| Copy | Path | Lines | Namespace | Last change |
|------|------|-------|-----------|-------------|
| A | `src/common/RTLogger.cpp` | 109 | `penta::` | 9bd8a722 thread-safety fix |
| B | `src_penta-core/common/RTLogger.cpp` | 101 | `penta::` | 9bd8a722 thread-safety fix |

Both define methods of `penta::RTLogger` (declared in `include/penta/common/RTLogger.h`). `src/` copy has extra pthread/QoS platform code that `src_penta-core/` lacks. Linker either errors or silently picks one; observed behavior depends on link order.

**Recommended canonical:** `src_penta-core/common/RTLogger.cpp` (lives in the library that owns the header via `penta::` namespace). Port the pthread/QoS platform code from `src/` into it if still needed. Then delete `src/common/RTLogger.cpp`.

**Blast radius:** Anything that includes `penta/common/RTLogger.h`. No API change — same class, same methods.

---

### 2. `DiagnosticsEngine.cpp` — real ODR, **divergent content**

| Copy | Path | Lines | Namespace | Last change |
|------|------|-------|-----------|-------------|
| A | `src/diagnostics/DiagnosticsEngine.cpp` | 196 | `penta::diagnostics` | ca0dff8e KmiDi-1 migration |
| B | `src_penta-core/diagnostics/DiagnosticsEngine.cpp` | 201 | `penta::diagnostics` | ca0dff8e KmiDi-1 migration |

Same namespace, same class. Both were last touched in the same squash-merge commit but have drifted 5 lines apart since. `diff` required to confirm which has the bug fixes.

**Recommended canonical:** `src_penta-core/diagnostics/DiagnosticsEngine.cpp` (penta_core owns the `penta::diagnostics` namespace). Diff the two to find which version has fixes that need porting before deletion.

---

### 3. `PerformanceMonitor.cpp` — real ODR, **penta_core version is newer**

| Copy | Path | Lines | Namespace | Last change |
|------|------|-------|-----------|-------------|
| A | `src/diagnostics/PerformanceMonitor.cpp` | 129 | `penta::diagnostics` | f347fa9c AU plugin proof-of-life |
| B | `src_penta-core/diagnostics/PerformanceMonitor.cpp` | **152** | `penta::diagnostics` | **9bd8a722 thread-safety — atomic stats, singleton race, data races** |

Same namespace, same class. Penta_core version is 23 lines larger and received the thread-safety hardening commit (9bd8a722). `src/` version did not.

**Recommended canonical:** `src_penta-core/diagnostics/PerformanceMonitor.cpp`. The src/ copy is an older snapshot without the thread-safety fix. Delete `src/diagnostics/PerformanceMonitor.cpp`.

---

### 4. `AudioAnalyzer.cpp` — real ODR but **currently byte-identical**

| Copy | Path | Lines | Namespace | Status |
|------|------|-------|-----------|--------|
| A | `src/diagnostics/AudioAnalyzer.cpp` | 174 | `penta::diagnostics` | identical to B |
| B | `src_penta-core/diagnostics/AudioAnalyzer.cpp` | 174 | `penta::diagnostics` | identical to A |
| C | `src/audio/AudioAnalyzer.cpp` | 97 | `midikompanion::audio` | unrelated — different class |

A and B are bit-identical. Still a link-time duplicate-symbol hazard (some linkers dedupe, others error or pick arbitrarily). C is a different class in a different namespace, no ODR conflict.

**Recommended canonical:** `src_penta-core/diagnostics/AudioAnalyzer.cpp`. Delete `src/diagnostics/AudioAnalyzer.cpp`. Leave `src/audio/AudioAnalyzer.cpp` (unrelated class) alone.

---

## Not ODR (same basename, different class)

### 5. `GrooveEngine.cpp` — the memory's "groove divergence" was mostly a mirage

Three copies, but they split two ways:

| Copy | Path | Lines | Namespace / Class | Notes |
|------|------|-------|-------------------|-------|
| A | `src/midi/GrooveEngine.cpp` | **557** | `kelly::GrooveEngine` | Most recent: 1fb1e8e5 (OSC/groove/midi cleanup) |
| B | `src/engines/GrooveEngine.cpp` | 251 | `kelly::GrooveEngine` | ca0dff8e KmiDi-1 migration |
| C | `src_penta-core/groove/GrooveEngine.cpp` | 337 | `penta::groove::GrooveEngine` | d102d492 audit crit-fixes |

- **A and B are an intra-`src/` ODR** — both in `namespace kelly` with the same class name. This is a real collision and NOT involving `src_penta-core` at all.
- **C is a different class** (`penta::groove::GrooveEngine`) — no ODR conflict with A or B.

**Recommendation:**

- Keep **A** (`src/midi/GrooveEngine.cpp`) — largest, most recent. Delete **B** (`src/engines/GrooveEngine.cpp`). Needs your confirmation that B has no features missing from A; diff the two before deleting.
- Keep **C** (`src_penta-core/groove/GrooveEngine.cpp`) — it's a distinct class in a distinct namespace; both can coexist.

---

## Summary table

| # | Collision | Severity | Canonical pick | Delete |
|---|-----------|----------|----------------|--------|
| 1 | `RTLogger.cpp` | **ODR, divergent** | `src_penta-core/common/` | `src/common/RTLogger.cpp` (after porting pthread/QoS code if needed) |
| 2 | `DiagnosticsEngine.cpp` | **ODR, divergent** | `src_penta-core/diagnostics/` | `src/diagnostics/DiagnosticsEngine.cpp` (after diff-review) |
| 3 | `PerformanceMonitor.cpp` | **ODR, divergent** | `src_penta-core/diagnostics/` (newer, thread-safety fix) | `src/diagnostics/PerformanceMonitor.cpp` |
| 4 | `AudioAnalyzer.cpp` | **ODR, identical** | `src_penta-core/diagnostics/` | `src/diagnostics/AudioAnalyzer.cpp` |
| 5 | `GrooveEngine.cpp` (intra-src) | **ODR within src/** | `src/midi/` (largest, newest) | `src/engines/GrooveEngine.cpp` (after diff-review) |

The `src_penta-core/groove/GrooveEngine.cpp` case (5C) is NOT an ODR issue — different namespace.

---

## Why "Tree A vs Tree B" is the wrong framing for this repo

The audit memory called for a whole-tree canonical choice. That framing doesn't match current state:

- `src_penta-core/` is a self-contained library with `project(penta_core)` and `add_subdirectory(src_penta-core)` in root CMake.
- `src/` adds KellyCore-specific code on top.
- KellyCore **publicly links** penta_core — they are intentionally compiled together.

So the fix is per-file, not per-tree:

1. Any `.cpp` with `namespace penta::` or `penta::<sub>::` lives in `src_penta-core/`.
2. Any `.cpp` with `namespace kelly::` lives in `src/`.
3. Anything that's been copied into both trees is a leftover from earlier migration — `src/` is the obsolete copy.

Then for intra-src duplicates like `GrooveEngine.cpp` (A vs B), the larger / more-recent version wins.

---

## Recommended execution order (for a follow-up PR)

Small, reversible commits:

1. **Commit 1** — delete `src/diagnostics/AudioAnalyzer.cpp` (identical to penta_core copy, zero-risk).
2. **Commit 2** — port pthread/QoS platform code from `src/common/RTLogger.cpp` into `src_penta-core/common/RTLogger.cpp` (or conditionally compile if platform-gated), then delete `src/common/RTLogger.cpp`.
3. **Commit 3** — diff+port any unique fixes from `src/diagnostics/DiagnosticsEngine.cpp` into `src_penta-core/diagnostics/DiagnosticsEngine.cpp`, then delete `src/` copy.
4. **Commit 4** — delete `src/diagnostics/PerformanceMonitor.cpp` (older, missing thread-safety fix — nothing to port).
5. **Commit 5** — diff `src/midi/GrooveEngine.cpp` vs `src/engines/GrooveEngine.cpp`; port any `src/engines/` features into `src/midi/` if needed, then delete `src/engines/GrooveEngine.cpp`.
6. **Commit 6** — verify no new basename collisions introduced by subsequent branches: add a CI check `python3 scripts/check_basename_collisions.py src/ src_penta-core/`.

Each deletion must be verified by:
- `cargo test --manifest-path engine/intent_ir/Cargo.toml` → unchanged
- `cmake -S . -B build -DBUILD_KELLY_CORE=ON && cmake --build build --target KellyCore -j8` → still builds (requires JUCE in the build environment)

---

## Blast radius and risk

**Zero API change.** All five collisions involve the same class/namespace; deleting a duplicate `.cpp` leaves the canonical copy in place. Consumers of `penta::RTLogger`, `penta::diagnostics::*`, and `kelly::GrooveEngine` see the same headers and the same method signatures.

**Risk: missing fixes ported.** For RTLogger (1) and DiagnosticsEngine (2), the `src/` copy has content the `src_penta-core/` copy lacks. Forgetting to port those lines before deletion produces a silent regression.

---

## Files that are NOT in scope for this survey

- `_archive/KmiDi_FINAL/**` — archived, not compiled.
- `KmiDi_PROJECT/source/cpp/src/harmony/**` — legacy project tree with its own CMake flags; CLAUDE.md: "Do not mix with root CMake options."
- `.worktrees/*` — per-branch worktrees.

These trees contain additional copies (e.g. `_archive/KmiDi_FINAL/engine/src/harmony/HarmonyEngine.cpp`, `KmiDi_PROJECT/source/cpp/src/harmony/VoiceLeading.cpp`) but they don't link into the canonical KellyCore/penta_core build.

---

## Next step

Decide execution order above. The safe starting point is Commit 1 (AudioAnalyzer, identical content, zero-risk delete). After that, sequence the remaining four in any order with per-commit verification.
