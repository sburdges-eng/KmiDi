# Phase 1 — Audit confirmation

Re-checked the blocker list against the repo. No downgrades; a few sharpenings.

## Confirmed facts

- **BUILD.md:** Line 16 `KmiDi_recovery_20260218-0329`; lines 26–39 `build_out`, `KellyPlugin`, `KellyApp`, `KellyTests`. Root CMake uses `build`; AGENTS/FULL_STACK_BUILD document `KellyFFI`, `KellyPlugin_VST3`. KellyCore, KellyPlugin, KellyApp, KellyTests exist in CMake but BUILD.md must use canonical dir `build` and V1-relevant target names.
- **build_v1.sh:** Line 27 `pip install -r "${ROOT_DIR}/requirements.txt"`. No top-level `requirements.txt` (only in subdirs). Confirmed blocker.
- **README / DEVELOPMENT.md:** "Full v1 build" points to build_v1.sh only; FULL_STACK_BUILD.md is "React/Tauri/C++ integration." Two pipelines not labeled as (A) penta_core + PyInstaller + Tauri vs (B) KellyFFI + Tauri.
- **package.json:** No `test:all`, `test:cpp`, `test:rust`, `test:integration`, `build:all-release`, `build:plugins`, `lint`, `format`. Confirmed.
- **DEVELOPMENT.md line 80:** "C++ file watcher and rebuilder" — dev:all only runs React, Tauri, Python API. Confirmed.
- **FULL_STACK_BUILD.md:** Hardcoded `/Users/seanburdges/Dev/KmiDi` at lines 32, 55, 114. Confirmed.
- **QUICK_START.md:** Title "Quick Start" but content is Kelly Companion Python usage only. Confirmed.
- **QUICK_REFERENCE.md:** References `scripts/verify_imports.py`, `scripts/test_python_integration.py`, `scripts/verify_build.py`, `scripts/setup_build.sh` — all exist. References `START_HERE.md`, `NEXT_DEVELOPMENT_PHASE.md`, `BUILD_STATUS.md`, `PHASE_1_PROGRESS.md`, `WORKSPACE_SETUP.md` — START_HERE, NEXT_DEVELOPMENT_PHASE, WORKSPACE_SETUP exist under `docs/`; BUILD_STATUS.md and PHASE_1_PROGRESS.md do not exist. Line 139 hardcoded path `/Users/seanburdges/KmiDi-1`. Sharpen: fix or annotate missing BUILD_STATUS.md, PHASE_1_PROGRESS.md and path.
- **BUILD.md CMake:** Says >= 3.22; bootstrap/AGENTS use 3.27+. Confirmed minor.
- **dev-setup.sh line 36:** Echo says "npm run tauri dev"; package has "npm run dev:tauri". Confirmed minor.
- **ARCHITECTURE.md / API.md:** Both exist at `docs/ARCHITECTURE.md`, `docs/API.md`. DEVELOPMENT.md "refer to" is in same doc dir so relative refs are valid. No change needed for 11.
- **scripts/build-all.sh:** Exists. DEVELOPMENT.md line 117 reference is valid.
- **Schema:** Not run yet; deferred to Phase 4/5.

## Final prioritized execution list (this run)

| Phase | Priority | Item | Action |
|-------|----------|------|--------|
| 2 | Critical | BUILD.md | Fix repo name to KmiDi, build dir to `build`, align targets to KellyFFI / KellyPlugin_VST3 and CMake 3.27+ |
| 2 | Critical | build_v1.sh | Replace `pip install -r requirements.txt` with `pip install -e .` (or equivalent) so packaging step runs |
| 2 | Critical | Two V1 pipelines | In README and DEVELOPMENT.md, label (A) build_v1.sh = penta_core + PyInstaller + Tauri; (B) build-full-stack.sh / FULL_STACK_BUILD = KellyFFI + Tauri |
| 3 | Major | DEVELOPMENT.md npm scripts | Replace nonexistent test/build/lint/format with actual commands or remove claims |
| 3 | Major | DEVELOPMENT.md dev:all | Remove "C++ file watcher"; list only React, Tauri, Python API |
| 3 | Major | FULL_STACK_BUILD.md | Replace hardcoded paths with "repo root" / portable wording |
| 3 | Major | QUICK_START.md | Add repo quick start at top or relabel as Kelly Companion usage |
| 3 | Major | QUICK_REFERENCE.md | Fix doc paths to docs/ where needed; annotate or remove BUILD_STATUS.md, PHASE_1_PROGRESS.md; fix hardcoded path |
| 4 | Minor | BUILD.md | CMake >= 3.27 (if not done in critical fix) |
| 4 | Minor | dev-setup.sh | Echo "npm run dev:tauri" |
| 4 | Minor | Schema | Run sync + pytest test_api_schema; record result |
| 5 | — | Validation | Run schema sync, pytest, frontend typecheck/build as applicable |
| 6 | — | Freeze report | Emit status, fixes, validations, blockers, non-goals, review steps |
