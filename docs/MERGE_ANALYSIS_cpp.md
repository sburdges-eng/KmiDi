# Merge Analysis — C++ Stack

**Stack owner:** CPP agent
**Date:** 2026-05-23
**Scope:** Native C++ engine and build system only — `engine/` (C++), `include/`, `src_penta-core/`,
`libs/daiw/`, `cmake/`, all `CMakeLists.txt` / `*.cmake`, every `*.cpp/*.cc/*.cxx/*.h/*.hpp/*.hh/*.mm`,
`BUILD.md`, plus the two contract surfaces the C++ side consumes: `shared_schemas/` (intent source of
truth) and `engine/intent_ir/` (Rust C-ABI staticlib linked into KellyFFI).
**Method:** read-only inspection of `origin/main..<branch>` and `676581c5..<branch>`. No code modified.

---

## TL;DR

**None of the 11 branches touch the C++ stack. Zero C++ merge risk, zero CPP merge-ordering
constraints, and the native build/test gates do not need to run for any of these merges.**

Every branch is pure Python (`music_brain/**/*.py` + `tests/**/*.py`). A path-restricted diff of all
11 branches against `origin/main`, filtered to every C++/CMake/header/FFI/schema path, returned
**empty for all 11**. The C ABI surface (`kelly_*`, `IntentFrameBuilder_*`, `validate_intent_frame_ffi`)
and the intent contract (`shared_schemas/CompleteSongIntentRequest.json` →
`engine/intent_ir/src/generated/intent.rs`) are untouched.

---

## ⚠️ Read this before reviewing any of these branches with `git diff main..<branch>`

All 11 branches forked from merge-base **`676581c5`**. `origin/main` is **exactly 1 commit ahead** of
that base — commit **`abf90773` "feat(music_brain/latent): add latent control core across 5 waves
(#196)"**. The branches predate #196.

Because of this, a **two-dot diff** (`git diff origin/main..<branch>`) reports the entire #196 latent
core (~6,100 lines across `music_brain/latent/*.py`, their tests, and two `docs/` files) as
**deletions**. **These deletions are a diff artifact, not real changes.** A correct 3-way merge
(branch → main) keeps the #196 files: main added them, the branch never touched them. Reviewing these
PRs with a naive two-dot diff will falsely suggest they revert #196 and delete generated/contract code.

**For the CPP stack specifically:** even under the misleading two-dot view, the deletions are confined
to `music_brain/` and `docs/` (Python + docs). No C++, CMake, header, FFI, or `shared_schemas/` path
appears as deleted or modified in any view. The native tree is inert with respect to these branches.

Always review these branches against the merge-base (`git diff 676581c5..<branch>`) or after rebasing
onto `origin/main`.

---

## Per-branch C++ footprint (true footprint vs. merge-base `676581c5`)

| Branch | Commits ahead | C++ / CMake / FFI / schema files touched | Actual files (all Python/tests) |
|--------|:---:|:---:|---|
| `feat/constrained-decoding` | +3 | **none** | `music_brain/audio/chunking.py`, `music_brain/decoding/{__init__,constrained}.py`, `music_brain/latent/{__init__,normalization}.py`, 3 tests |
| `feat/context-window` | +1 | **none** | `music_brain/generation/{__init__,context_window}.py`, 1 test |
| `feat/dual-representation` | +1 | **none** | `music_brain/audio/dual_clip.py`, 1 test |
| `feat/generation-scope` | +2 | **none** | `music_brain/generation/{__init__,latency_budget,scope}.py`, 2 tests |
| `feat/linear-projection` | +1 | **none** | `music_brain/latent/{__init__,projection}.py`, 1 test |
| `feat/multimodal-fusion` | +1 | **none** | `music_brain/latent/{__init__,fusion}.py`, 1 test |
| `feat/stem-bus` | +1 | **none** | `music_brain/audio/stems.py`, 1 test |
| `feat/symbolic-realization` | +1 | **none** | `music_brain/symbolic/{__init__,realize}.py`, 1 test |
| `feat/transition-model` | +1 | **none** | `music_brain/prediction/{__init__,transition}.py`, 1 test |
| `feat/ttg-energy-gating` | +4 | **none** | `music_brain/api_schemas/ttg_adapter.py`, `music_brain/pipeline/intent_pipeline.py`, 3 tests |
| `feat/world-state` | +1 | **none** | `music_brain/session/world_state.py`, 1 test |

Verification command (run per branch; empty output = no C++ footprint):

```bash
git diff --stat origin/main origin/feat/<branch> -- \
  '*.cpp' '*.cc' '*.cxx' '*.h' '*.hpp' '*.hh' '*.mm' '*.cmake' 'CMakeLists.txt' \
  'cmake/' 'BUILD.md' 'libs/daiw/' 'src_penta-core/' 'include/' 'engine/' 'shared_schemas/'
# → empty for all 11 branches
```

---

## Conflicts between branches (C++ files)

**None.** No two branches modify any shared C++/CMake/header/FFI/schema file, because no branch
modifies any such file at all. There is no C++ conflict surface to order around.

---

## Recommended merge order (C++ perspective)

The C++ stack imposes **no ordering constraint**. These branches can merge in any order with respect
to the native engine; sequencing is driven entirely by the Python/bindings stacks (e.g. the shared
`music_brain/latent/__init__.py` export edits in `constrained-decoding`, `linear-projection`, and
`multimodal-fusion` — a Python-stack concern, out of CPP scope; flagged here only as a cross-stack
pointer, not a CPP finding).

---

## Current Build Status

**Branch checked:** `feature/agent-swarm` (current working branch). Because all 11 feature branches
are pure-Python (see footprint table), the native build state of `feature/agent-swarm` is
representative of every post-merge tree from the C++ point of view.

**Build directory used:** `build/`. The directories named in the task brief (`build-check/`,
`build-asan/`, `build-demo/`) **do not exist** in this workspace. The only configured CMake tree
present is `build/`; `build_fileio/` holds a stray `CMakeLists.txt` only and is not a usable build
dir. Per task constraints, **no `cmake` configure was run** — only the existing tree was used.

**`build/` configuration** (read from `build/CMakeCache.txt`): Ninja generator,
`CMAKE_BUILD_TYPE=Release`, `BUILD_KELLY_CORE=ON`, `BUILD_KELLY_FFI=ON`, `BUILD_TESTS=OFF`,
`KMIDI_ENABLE_ASAN=OFF`, `KMIDI_ENABLE_TSAN=OFF`.

**Result:** ✅ **PASS (artifact-level).** `build/libKellyCore.a` is present (23,328,912 bytes,
mtime 2026-05-23) alongside a populated `build.ninja` and `compile_commands.json` — i.e. KellyCore
was configured and successfully archived in this tree today.

**Caveat — live recompile not re-run this session.** `cmake --build build --target KellyCore` (and a
`ninja -C build KellyCore` dry-run) are blocked by this environment's bash permission gate and were
not approved, so a fresh compile-from-source was not executed. The PASS above is evidence from a
build product dated today, **not** a re-run compile. Given the C++ delta from all 11 branches is
**zero**, no recompile is required for any of these merges regardless.

**Note for the final ctest gate:** the present `build/` was configured with `BUILD_TESTS=OFF`, so the
consolidated `ctest` smoke recommended below would require a reconfigure with `-DBUILD_TESTS=ON`
(no such test-enabled build tree currently exists in this workspace).

---

## Contract / FFI integrity check (the one thing CPP cares about)

The C++ side consumes the intent contract through two generated/owned surfaces:

1. `shared_schemas/CompleteSongIntentRequest.json` (source of truth) →
   `engine/intent_ir/src/generated/intent.rs` (Rust, compiled into the KellyFFI dylib).
2. The KellyFFI C ABI: `kelly_*` (C++) + `IntentFrameBuilder_*` / `validate_intent_frame_ffi`
   (embedded Rust `intent_ir` staticlib).

**Neither is touched by any branch.** `shared_schemas/` and `engine/` returned empty in the
path-restricted diff for all 11. The C ABI surface is stable; no `sync_entities.py` re-sync is required
as a result of these merges, and no contract drift reaches the native consumers.

Note on a potential false alarm: the Python file `music_brain/intent_ir/emitter.py` appears in the
two-dot diffs of every branch. That is **#196 artifact**, not branch work — it does not appear in any
branch's merge-base footprint (`676581c5..<branch>`). It is also a Python module (IR emission), not the
Rust `engine/intent_ir/` C-ABI library; even if it had changed, it would be a Python/bindings concern,
not CPP.

---

## Test gaps (C++)

No C++ tests are added, removed, or invalidated by any branch. The native test suite
(`ctest --test-dir build`), the RT harness (`BUILD_RT_HARNESS`), and the sanitizer builds
(`KMIDI_ENABLE_ASAN` / `KMIDI_ENABLE_TSAN`) are all unaffected. There is no C++ coverage gap to
flag — there is no C++ delta to cover.

---

## CI recommendations (C++)

- **Skip the native C++ build matrix for all 11 merges.** Running `cmake … --target KellyCore/KellyFFI`,
  `ctest`, or the ASan/TSan jobs adds no signal: the merges produce no C++ delta and cannot change
  native build or runtime behavior. This is a pure-Python consolidation from the engine's point of view.
- **One guard worth keeping (cheap):** add a CI assertion that the path-restricted C++ diff is empty for
  each branch before merge, so an unexpected native edit (e.g. a stray regenerated `intent.rs`) is caught:

  ```bash
  test -z "$(git diff --name-only origin/main HEAD -- \
    '*.cpp' '*.cc' '*.cxx' '*.h' '*.hpp' '*.hh' '*.mm' '*.cmake' 'CMakeLists.txt' \
    'cmake/' 'libs/daiw/' 'src_penta-core/' 'include/' 'engine/' 'shared_schemas/')"
  ```

- **At final integration:** after all Python branches land, a single C++ smoke build
  (`KellyCore` + `KellyFFI`) + `ctest` confirms the consolidated tree still links — a one-time
  sanity gate, not a per-branch requirement.
- **Do not rely on two-dot PR diffs** for review gating on these branches (see warning above); base
  CI diffs on the merge-base or post-rebase tree.

---

## Summary for the unified plan

> **C++ stack: no-op.** All 11 branches are pure-Python and add nothing to `engine/`, `include/`,
> `src_penta-core/`, `libs/daiw/`, `cmake/`, headers, the KellyFFI C ABI, or `shared_schemas/`.
> No CPP conflicts, no CPP ordering constraints, no CPP test gaps. Native CI gates can be skipped
> per-branch; keep one path-restricted "no C++ delta" guard and one final consolidated link+`ctest`
> smoke build. Reviewers must use merge-base / post-rebase diffs — the two-dot diff falsely shows the
> #196 latent core as deleted, but never shows any C++/schema path as affected.
