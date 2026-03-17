# Phase 1: Freeze Audit

**Goal:** Identify contradictions, stale paths, missing commands, mismatched build targets, schema drift risks, and references that break the canonical V1 story.

**Artifacts inspected:** README.md, AGENTS.md, BUILD.md, QUICK_START.md, docs/DEVELOPMENT.md, docs/FULL_STACK_BUILD.md, docs/QUICK_REFERENCE.md, package.json, scripts/build_v1.sh, scripts/build-full-stack.sh, scripts/dev-setup.sh, scripts/bootstrap.sh, scripts/sync_entities.py, shared_schemas/CompleteSongIntentRequest.json, tests/unit/test_api_schema.py, CMakeLists.txt.

---

## Critical blockers

1. **BUILD.md contradicts canonical V1**
   - Refers to repo name `KmiDi_recovery_20260218-0329` (stale).
   - Uses build dir `build_out`; AGENTS.md and scripts use `build`.
   - Documents targets `KellyPlugin`, `KellyApp`, `KellyTests`; root project uses `KellyPlugin_VST3`, `KellyFFI`, and has no `KellyApp` in the same way. CMake defaults force `BUILD_DESKTOP`/`BUILD_PLUGINS` OFF unless Qt/JUCE UI enabled.
   - Risk: New contributors follow BUILD.md and get wrong commands/paths.

2. **build_v1.sh depends on missing requirements.txt**
   - Script runs `pip install -r "${ROOT_DIR}/requirements.txt"` before PyInstaller. No top-level `requirements.txt` exists (deps are in pyproject.toml).
   - Result: Documented “full V1 build” fails at packaging step unless fixed or documented.

3. **Two distinct “V1” pipelines not clearly distinguished**
   - **Pipeline A (`build_v1.sh`):** sync_entities → C++ penta_core/penta_core_native (no KellyFFI) → PyInstaller music_brain API → Tauri build. No KellyFFI/KellyCore in this path.
   - **Pipeline B (FULL_STACK_BUILD.md + build-full-stack.sh):** KellyFFI → optional KellyPlugin_VST3 → Tauri cargo check. This is the React → Tauri → KellyFFI → KellyCore path.
   - README/DEVELOPMENT.md call `build_v1.sh` the “full v1 build” but do not state that the native desktop integration path (KellyFFI) is the other pipeline. Readers can assume one “V1” and get the wrong workflow.

---

## Major blockers

4. **DEVELOPMENT.md references npm scripts that do not exist**
   - `npm run test:all`, `npm run test:cpp`, `npm run test:rust`, `npm run test:integration`, `npm run build:all-release`, `npm run build:plugins`, `npm run lint`, `npm run format` are documented but not in package.json.
   - Contributes to unreproducible “documented” test/build path.

5. **DEVELOPMENT.md “Start All Services” implies C++ file watcher**
   - Text says dev:all starts “C++ file watcher and rebuilder.” package.json dev:all only runs React, Tauri, and Python API; no C++ watcher.
   - Misleading for expectations of out-of-the-box full-stack dev.

6. **FULL_STACK_BUILD.md uses hardcoded absolute path**
   - Contains `/Users/seanburdges/Dev/KmiDi` in examples and “Use the root workspace path.” Should use “repo root” or `$(pwd)`-style wording for portability and copy-paste.

7. **QUICK_START.md is mis-scoped**
   - Titled “Quick Start” but content is Kelly Companion Python usage (emotion thesaurus, engines, harmony). Not repo quick start or build/dev. Confusing for “quick start” from repo root.

8. **docs/QUICK_REFERENCE.md references scripts and docs that may not match canonical path**
   - Points to `scripts/verify_imports.py`, `scripts/verify_build.py`, `scripts/setup_build.sh`, `START_HERE.md`, `NEXT_DEVELOPMENT_PHASE.md`, `BUILD_STATUS.md`, etc. Some of these may be legacy; not cross-checked for consistency with AGENTS.md/dev-setup/build_v1.

---

## Minor blockers / consistency

9. **BUILD.md CMake version**
   - BUILD.md says CMake `>= 3.22`; AGENTS.md and bootstrap.sh use 3.27+. Should align to 3.27+ for freeze.

10. **dev-setup.sh final echo**
    - Says “npm run tauri dev” as alternative; package.json exposes “npm run dev:tauri”. Functionally equivalent (both run tauri dev) but naming inconsistency.

11. **AGENTS.md Reference docs table**
    - Lists ARCHITECTURE.md, API.md in DEVELOPMENT.md “refer to” section; existence not verified (could be missing or moved).

12. **Schema/API contract**
    - sync_entities.py and test_api_schema.py use `music_brain.engine_api.schema.CompleteSongIntentRequest`; shared_schemas/CompleteSongIntentRequest.json exists. Sync flow is documented; no drift detected in inspected files. Remaining risk: JSON file not regenerated from Pydantic (sync is code → TS/Rust; schema source is Python model).

---

## Summary table

| Priority   | Item | Location | Fix direction |
|-----------|------|----------|----------------|
| Critical  | BUILD.md wrong repo name, build dir, targets | BUILD.md | Align to KmiDi repo, `build`, KellyFFI/KellyPlugin_VST3; or clearly mark as “Kelly-only” reference. |
| Critical  | build_v1.sh requires requirements.txt | scripts/build_v1.sh | Add requirements.txt from pyproject deps or use pip install -e . and document; or document as optional path. |
| Critical  | Two V1 pipelines not distinguished | README, DEVELOPMENT.md, AGENTS | Document both: (1) build_v1.sh = penta_core + PyInstaller + Tauri; (2) build-full-stack.sh = KellyFFI + Tauri. |
| Major     | DEVELOPMENT.md npm scripts missing | docs/DEVELOPMENT.md | Remove or replace with real commands (e.g. pytest tests/, cargo test, npm run build). |
| Major     | dev:all does not start C++ watcher | docs/DEVELOPMENT.md | Correct copy: list only React, Tauri, Python API. |
| Major     | Hardcoded path in FULL_STACK_BUILD.md | docs/FULL_STACK_BUILD.md | Replace with “repo root” or relative wording. |
| Major     | QUICK_START.md is Python-only usage | QUICK_START.md | Rename or add repo quick start at top and point to dev-setup + AGENTS. |
| Minor     | BUILD.md CMake 3.22 vs 3.27 | BUILD.md | Unify to 3.27+. |
| Minor     | dev-setup echo “tauri dev” vs “dev:tauri” | scripts/dev-setup.sh | Use “npm run dev:tauri” in echo. |

---

## What was not changed in Phase 1

- No edits to code or config; audit only.
- Legacy surfaces (KmiDi_FINAL, legacy/ui) and external path (swif:xcode/KmiDi/KmiDi_CANON) were not modified.
- Schema sync and test_api_schema were not run; reserved for Phase 3.
