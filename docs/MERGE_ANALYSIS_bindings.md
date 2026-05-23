# Bindings Stack — Merge Analysis

**Scope:** pybind11 C++ bindings in `bindings/` (target `penta_core_native`) and any
CMake/build files affecting that target.
**Date:** 2026-05-23
**Mode:** Analysis only — no source modified, no commits/merges.

## Summary (TL;DR)

**None of the 11 feature branches touch the bindings stack.** Every branch is
pure-Python work confined to `music_brain/**` and `tests/unit/**`. There are
**zero** changes to `bindings/`, `CMakeLists.txt`, `bindings/CMakeLists.txt`,
`cmake/`, `include/`, `engine/`, or `src_penta-core/` on any branch — confirmed
with both two-dot (`origin/main..`) and three-dot (`origin/main...`) diffs.

The `penta_core_native` target and the C++ engine it wraps are **identical**
across `origin/main` and all 11 branch tips, so merging any or all of them
introduces **no bindings risk and no C++ rebuild requirement.**

### The bindings target (for reference)

`bindings/CMakeLists.txt` builds one pybind11 module, `penta_core_native`, from:
`bindings.cpp`, `harmony_bindings.cpp`, `groove_bindings.cpp`,
`diagnostics_bindings.cpp`, `osc_bindings.cpp`, `ml_bindings.cpp`,
`prrot_bindings.cpp`. It links `penta_core`, `KellyCore` (when present), and
`Python3::Python`, and adds `${CMAKE_SOURCE_DIR}/src` to its include path.
None of these files or their transitive C++ dependencies are modified by any
branch under analysis.

---

## Branches Touching This Stack

**None.**

No branch modifies any file in `bindings/`, nor any CMake/build file affecting
`penta_core_native`, nor any header/source the bindings wrap
(`include/`, `engine/`, `src_penta-core/`, `src/`).

---

## Branches NOT Touching This Stack

All 11 branches. Each is Python-only. File footprint per branch (from
`git diff --name-only origin/main...origin/feat/<branch>`):

| # | Branch | Commits | Files changed | Touches bindings/C++? |
|---|--------|--------:|---------------|:---------------------:|
| 1 | `feat/constrained-decoding` | 3 | `music_brain/audio/chunking.py`, `music_brain/decoding/{__init__,constrained}.py`, `music_brain/latent/{__init__,normalization}.py`, +3 tests | No |
| 2 | `feat/context-window` | 1 | `music_brain/generation/{__init__,context_window}.py`, +1 test | No |
| 3 | `feat/dual-representation` | 1 | `music_brain/audio/dual_clip.py`, +1 test | No |
| 4 | `feat/generation-scope` | 2 | `music_brain/generation/{__init__,latency_budget,scope}.py`, +2 tests | No |
| 5 | `feat/linear-projection` | 1 | `music_brain/latent/{__init__,projection}.py`, +1 test | No |
| 6 | `feat/multimodal-fusion` | 1 | `music_brain/latent/{__init__,fusion}.py`, +1 test | No |
| 7 | `feat/stem-bus` | 1 | `music_brain/audio/stems.py`, +1 test | No |
| 8 | `feat/symbolic-realization` | 1 | `music_brain/symbolic/{__init__,realize}.py`, +1 test | No |
| 9 | `feat/transition-model` | 1 | `music_brain/prediction/{__init__,transition}.py`, +1 test | No |
| 10 | `feat/ttg-energy-gating` | 4 | `music_brain/api_schemas/ttg_adapter.py`, `music_brain/pipeline/intent_pipeline.py`, +3 tests | No |
| 11 | `feat/world-state` | 1 | `music_brain/session/world_state.py`, +1 test | No |

> **Note on diffstat shape.** A two-dot `git diff --stat origin/main..origin/feat/<branch>`
> reports ~6,100 *deletions* for every branch. These are **not** deletions made
> by the branches — they are main-side additions the branches predate (see
> "Notes for Consolidation"). The two-dot bindings-path diff is empty, and the
> three-dot diff (branch-side changes only) confirms each branch adds only the
> Python files listed above.

---

## Conflict Matrix

### Bindings files × branches (all empty — no conflicts)

| Binding file | Branches modifying it | Conflict risk |
|--------------|-----------------------|:-------------:|
| `bindings/bindings.cpp` | — | none |
| `bindings/diagnostics_bindings.cpp` | — | none |
| `bindings/groove_bindings.cpp` | — | none |
| `bindings/harmony_bindings.cpp` | — | none |
| `bindings/ml_bindings.cpp` | — | none |
| `bindings/osc_bindings.cpp` | — | none |
| `bindings/prrot_bindings.cpp` | — | none |
| `bindings/CMakeLists.txt` | — | none |

### Bindings-level conflicts: NONE

No two branches modify the same binding file, because **no branch modifies any
binding file.** There is no overlap on `bindings/*.cpp`, `bindings/CMakeLists.txt`,
or any C++ header/source the bindings wrap. Merging the full set in any order
produces no conflict within the bindings stack.

### Adjacent (non-bindings) Python conflicts — flagged for the Python stack owner, out of scope here

These do not affect `penta_core_native`, but they are real merge conflicts the
Python-stack consolidation must resolve. Recording them so they are not lost:

- **`music_brain/latent/__init__.py`** — modified by 3 branches:
  `feat/constrained-decoding`, `feat/linear-projection`, `feat/multimodal-fusion`.
- **`music_brain/generation/__init__.py`** — modified by 2 branches:
  `feat/context-window`, `feat/generation-scope`.

These are package `__init__` export aggregations; expect textual conflicts on the
export lists when merging the second and subsequent branch of each group. They
are unrelated to bindings.

---

## Recommended Merge Order

From the **bindings perspective specifically**, ordering is unconstrained — all
11 branches carry zero bindings impact and cannot conflict with the bindings
stack regardless of sequence.

1. **Independent branches (no bindings changes) — merge first, no bindings risk:**
   **All 11 branches** fall in this tier. None require a C++/bindings rebuild and
   none can break `penta_core_native`.
2. **Branches with isolated bindings changes — merge next:** *None.*
3. **Branches with overlapping bindings changes — merge last:** *None.*

> The only ordering constraint anywhere in this set lives in the **Python** stack
> (the shared `__init__.py` files above), not in bindings. Defer sequencing of
> `feat/constrained-decoding` / `feat/linear-projection` / `feat/multimodal-fusion`
> (latent group) and `feat/context-window` / `feat/generation-scope` (generation
> group) to the Python merge analysis.

---

## Test Gaps & CI Recommendations

- **Current state:** No `build/build.ninja` exists — CMake has not been configured
  in this checkout. The `penta_core_native` target has therefore never been
  configured or built here.
- **What this means for these merges:** Because no branch alters `bindings/`,
  CMake, or any wrapped C++ source, the merged tree's bindings build is
  byte-identical to `main`'s. **No new bindings build/test work is created by
  merging these branches.** Validation effort belongs to the **Python** layer
  (`pytest tests/unit/`), where all the new code and tests live.
- **What would need to happen to validate bindings at all (independent of these
  merges):**
  1. Ensure JUCE present at `external/JUCE/` (required by `penta_core`/`KellyCore`).
  2. Configure: `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON` (plus whatever option enables the `bindings/` subdirectory / `penta_core` target in the top-level `CMakeLists.txt`).
  3. Build: `cmake --build build --target penta_core_native -j8`.
  4. Smoke test: `python -c "import penta_core_native"` against `build/python/`.
- **Recommended CI steps for the bindings target (general, not gating these merges):**
  - Add a job that configures CMake and builds `penta_core_native`, then runs the
    import smoke test, so future branches that *do* touch `bindings/` or wrapped
    C++ are caught. None of the 11 branches here would exercise it.

---

## Notes for Consolidation

- **Common stale base.** All 11 branches share the identical merge-base
  `676581c5` ("Merge pull request #184 …", 2026-05-22). `main` has since advanced
  to `abf90773` — PR **#196** "feat(music_brain/latent): add latent control core
  across 5 waves". `main` made **no** changes to `bindings/` or any C++ path since
  the merge-base, which is why every branch's bindings diff is clean on both sides.
  The large uniform "deletions" in two-dot diffstats are #196's additions, not
  branch deletions. Rebasing the branches onto current `main` before merge is
  advisable but is a Python concern, not a bindings one.

- **Latent overlap risk (Python, flagged here for visibility).** Three branches
  add code under `music_brain/latent/`
  (`normalization.py`, `projection.py`, `fusion.py`) while `main`'s #196 just
  landed a "latent control core." Reviewer should check these new primitives for
  duplication/redundancy against what #196 already provides. This is a functional
  overlap, not a bindings or file-level git conflict.

- **Binding coverage gap (intentional, not a defect for these merges).** Every new
  module — `latent`, `generation`, `decoding`, `audio`, `symbolic`, `prediction`,
  `session` — is **pure Python** with pure-Python unit tests. None call into
  `penta_core_native`, and none require a binding. If any of these primitives are
  later intended to delegate to the C++ engine (e.g. groove/harmony/diagnostics
  already exposed via `*_bindings.cpp`), that wiring does **not** exist on any
  branch today. Adding it would be a **separate future bindings task**, not part
  of this consolidation.

- **No consolidation opportunity within the bindings stack.** Since no branch
  changes bindings, there is nothing to batch, dedupe, or sequence at the C++/FFI
  boundary for this set of branches.
