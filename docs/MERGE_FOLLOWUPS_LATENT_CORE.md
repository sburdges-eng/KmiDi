# Merge follow-ups — Latent Control Core (PR #196)

This doc enumerates the pre-existing CI failures that PR #196
(`feat/latent-control-core`) leaves untouched. They predate this PR
(reproducible on `main` after the latest CI-fix sequence in
`082c6b0d` / `0f1e09be` / `02927f9b` / `676581c5`) and would have
blocked merge if treated as PR-blockers. Each item below has a
diagnosed root cause, a proposed fix scope, and an owner/PR sketch.

> The user explicitly approved merging PR #196 on the condition that
> this plan **exists** prior to merge; its execution can be deferred.
> All gates *attributable to PR #196* are green:
>
> - 218 latent unit + integration tests pass
> - `flake8 --max-line-length 100` clean on every new + modified file
> - `black --check` clean on every new + modified file
> - 6 inline review findings from Codex + Cursor applied; 6 stale
>   duplicates verified as already-resolved against HEAD

---

## 1. `httpx` import error in `tests/unit/test_api_audit_fixes.py`

**Symptom (excerpt from `Python Tests` job):**

```
RuntimeError: The starlette.testclient module requires the httpx package to be installed.
    $ pip install httpx
```

**Root cause:** `httpx` is listed at `pyproject.toml:42` *inside the
`dev` optional-dependency extra*. The failing CI jobs install the
package via `pip install -e .` (or equivalent) **without** the `[dev]`
extra, so `httpx` is missing at test time. Eleven tests in
`tests/unit/test_api_audit_fixes.py` use FastAPI's `TestClient`, which
imports `starlette.testclient`, which requires `httpx`.

**Proposed fix (one-line, separate PR):**

Either of:

- Move `httpx>=0.24` from `[project.optional-dependencies].dev` to
  `[project.optional-dependencies].test` (and define a `test` extra
  in `pyproject.toml`), then install `pip install -e ".[test]"` in CI.
- Or move `httpx>=0.24` to `[project.dependencies]` if the API audit
  tests are considered part of the always-on test surface.

**Suggested PR scope:** ~3 lines in `pyproject.toml`, a 1-line change
in each affected CI workflow if a new extra is introduced. Title:
`chore(ci): install httpx for FastAPI TestClient audit tests`.

---

## 2. `test_midi_mutation.py` collection error on Python 3.9

**Symptom (excerpt from `Sprint 1 – Core Testing & Quality` job):**

```
ERROR collecting tests/unit/test_midi_mutation.py
TypeError: unsupported operand type(s) for |: 'type' and 'type'
```

**Root cause:** the file uses PEP 604 union syntax (`X | Y`) at
runtime in a function signature or type alias. PEP 604 runtime support
landed in Python 3.10. Sprint 1 runs on Python 3.9 → collection fails
before any test runs.

**Proposed fix (one-line, separate PR):**

Add `from __future__ import annotations` at the top of
`tests/unit/test_midi_mutation.py` so the annotations are evaluated
lazily as strings on 3.9. *Or* drop Python 3.9 from the matrix if
3.10 is now the floor for the rest of the project.

**Suggested PR scope:** 1 line. Title:
`fix(tests): defer annotations in test_midi_mutation for Py3.9`.

---

## 3. Repo-wide `black` non-compliance (480 files)

**Symptom (excerpt from `Python Lint & Format` job):**

```
480 files would be reformatted, 111 files would be left unchanged.
```

**Root cause:** the repo has had `black` enforcement turned on in CI
for some time, but most of the existing tree was authored before
adopting `black` (or with different settings). The 480-file diff is
purely formatting and predates this PR. Every file *added or modified
by PR #196* is already `black`-compliant under `--max-line-length 100`.

**Proposed fix (mechanical, separate PR):**

Run `black .` repo-wide on a chore PR with zero behavior changes.
Coordinate the PR landing window with active branches because it
touches 480 files and will conflict with anything in flight. Mark the
SHA in `.git-blame-ignore-revs` so `git blame` history stays useful.

**Suggested PR scope:** mechanical reformat only. Title:
`chore(format): apply repo-wide black baseline`.

---

## 4. C++ / JUCE build failures

**Failing jobs:** `JUCE Plugin Validation (macOS)`,
`C++ Tests (ubuntu-latest, default | gcc-11 | clang-14)`,
`C++ Tests (macos-latest, default)`, `C++ Tests (windows-latest, default)`,
`Code Coverage`, `Memory Testing (Valgrind)`,
`Performance Benchmarks`, `Performance Regression Tests`,
`Plugin Integration Tests`, `Runtime Contract C++ Tests`,
`RT-Safety Validation`, `Valgrind Memory Testing`, `cpp-ctest`,
`test-cpp`, `Build Headless Core and Package App (*)`,
`C++ Build`.

**Symptom (representative — `Code Coverage` failure):**

```
CMake Error at CMakeLists.txt:112 (find_package):
-- Configuring incomplete, errors occurred!
```

JUCE Plugin Validation succeeds at CMake configure but exits 1
during the build step (full log is mostly compiler ABI detection;
the actual build error is below the head sample).

**Root cause (provisional):** the repo recently bumped JUCE to
`8.0.13` (commit `396661d2 fix(ci): bump JUCE submodule to 8.0.13 and
add Linux X11/freetype deps`) and a follow-up
(`0f1e09be fix(ci): make install_juce_linux_deps skip non-Linux hosts
correctly`) plus downstream CMake unblocking (`082c6b0d`). Several
build targets still depend on configuration that hasn't caught up
with the new JUCE — see `CMakeLists.txt:112` `find_package(...)`.
Diagnosis-only at this point; the actual `find_package` argument
needs inspection.

**Proposed fix (separate PR per layer):**

1. `chore(ci): pull full JUCE plugin build log` — wrap the
   `ninja iDAW_Core` step in `--verbose` so the real error surfaces
   above the line-count truncation.
2. `fix(cmake): pin find_package for JUCE 8.0.13` — once the
   `find_package` failure at `CMakeLists.txt:112` is known, either
   bump the version constraint or vendor the missing module.
3. `fix(ci): re-enable C++ test matrix` after (2) lands.

**Suggested PR scope:** investigation first, then per-issue fixes.
Not blocking PR #196 because *no C++ source touched by this PR*; the
latent-core changes are pure Python.

---

## 5. GitNexus index re-analysis

**Symptom:** every commit on this branch emits

```
GitNexus index is stale (last indexed: 441d542)
```

and `gitnexus_impact` could not resolve `LatentFrame`,
`MultimodalFusion`, or `WorldModel.rollout` because the new
`music_brain/latent/` symbols aren't indexed yet.

**Proposed fix (single command, separate task):**

```bash
npx gitnexus analyze
```

after PR #196 merges. Then re-run impact analyses on the newly
indexed symbols (`LatentFrame`, `CompanionSession`, `LatentMemory`,
`KVCache`, …) and update `docs/AI_DEV_MEMORY.md` if any high-risk
upstream call sites surface.

**Suggested PR scope:** none (workspace operation). Owner: whoever
maintains the GitNexus index.

---

## Tracking

| # | Item | Severity | Suggested PR title |
|---|------|----------|--------------------|
| 1 | httpx test-dep missing | Low (env) | `chore(ci): install httpx for FastAPI TestClient audit tests` |
| 2 | `test_midi_mutation.py` PEP 604 on 3.9 | Low (test scope) | `fix(tests): defer annotations in test_midi_mutation for Py3.9` |
| 3 | Repo-wide black non-compliance | Low (cosmetic) | `chore(format): apply repo-wide black baseline` |
| 4 | C++/JUCE build matrix | Medium (build) | three PRs, see §4 |
| 5 | GitNexus reindex | Low (maintenance) | n/a, workspace task |

Items 1, 2, 3, and 5 each fit in a 1-day window. Item 4 likely takes
a focused engineer one or two days to investigate + fix.
