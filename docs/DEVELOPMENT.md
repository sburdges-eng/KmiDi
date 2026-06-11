# KmiDi Development Guide

Status: current operational guide aligned to the 2026 architecture authority set
Last updated: 2026-06-08

Start here when you need to build, run, test, or debug the current canonical repo.
For architecture decisions, do not treat this file as the source of truth. Use:
- `docs/ARCHITECTURE.md`
- `docs/REPO_MODULE_MAP.md`
- `docs/INTENT_IR_AUTHORITY.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`
- `docs/PERSISTENCE_AND_MIGRATION.md`

## 1. What is canonical right now

KmiDi is a plugin-first, engine-separable music creation system.
The currently canonical development surfaces in this repo are:
- React frontend in `src/`
- Python Music Brain API in `music_brain/`
- Rust Intent IR contract layer in `engine/intent_ir/`
- C++ native engine / KellyFFI / plugin code in the root CMake project

Important clarification:
- `package.json` does not currently define `npm run dev:tauri`.
- Treat older references to that command as historical/legacy drift unless they are deliberately revalidated and restored.
- For day-to-day development, the reliable combined dev entrypoint is `npm run dev:all` for React + Music Brain API.

## 2. Prerequisites

Minimum toolchain expected by the current repo:
- Node 20+
- npm
- Python 3.10+
- Rust stable
- CMake 3.27+
- Ninja

Platform notes:
- macOS: Xcode Command Line Tools
- Linux: GCC 9+ or Clang 10+
- Native/plugin work additionally expects JUCE at `external/JUCE/`

## 3. One-command setup

From repo root:

```bash
./scripts/dev-setup.sh
```

What it does today:
1. runs `scripts/bootstrap.sh`
2. runs `npm install`
3. runs `python3 -m pip install -e .`
4. best-effort installs `pydantic` and `uvicorn`

Notes:
- `scripts/dev-setup.sh` now points to the current supported dev entrypoints only.
- If you only need frontend + API work, you do not need the full native/plugin toolchain.

## 4. Daily run commands

### Frontend only

```bash
npm run dev
```

Equivalent aliases currently defined:
- `npm run dev`
- `npm run dev:react`
- `npm run dev:ui`

Dev server:
- URL: `http://localhost:1420`
- Host bind: `0.0.0.0`
- HMR port: `1421`

### Python API only

```bash
npm run dev:python
```

Equivalent direct command:

```bash
python3 -m uvicorn music_brain.api:app --reload --port 8000
```

API endpoints:
- base: `http://localhost:8000`
- docs: `http://localhost:8000/docs`

### Frontend + API together

```bash
npm run dev:all
```

This is the primary combined development command in the current repo.
It starts:
- Vite React dev server
- FastAPI Music Brain service

### Legacy/diagnostic Python boot helper

`run_brain.py` still exists and can be useful for import checks or legacy service bring-up:

```bash
python run_brain.py check
python run_brain.py gui
python run_brain.py penta
python run_brain.py orchestrator
```

Use it as an operational helper, not as architecture authority.
Its `gui` mode currently starts only the FastAPI service on `127.0.0.1:8000`.

## 5. Build workflows

## Frontend build

```bash
npx tsc --noEmit
npm run build
```

`npm run build` runs:
- TypeScript compile check via `tsc`
- Vite production build

## Python install / lint / test

Install:

```bash
python3 -m pip install -e .
```

Lint:

```bash
python3 -m flake8 music_brain/ --max-line-length 100
```

Tests:

```bash
python3 -m pytest tests/
```

Important:
- Do not pass `--timeout`; `pytest-timeout` is not a project dependency.

## Rust Intent IR

Crate location:
- `engine/intent_ir/`

Common commands:

```bash
cd engine/intent_ir && cargo test
cd engine/intent_ir && cargo clippy
```

Role of this crate:
- validates and represents canonical Intent IR
- exports Rust-side C ABI pieces consumed through KellyFFI

## C++ / KellyFFI / plugin builds

Use the root CMake project.
Example configure:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_KELLY_CORE=ON \
  -DBUILD_KELLY_FFI=ON \
  -DKMIDI_BUILD_JUCE_UI=ON \
  -DBUILD_PLUGINS=ON
```

Common builds:

```bash
cmake --build build --target KellyFFI -j8
cmake --build build --target KellyCore -j8
cmake --build build --target KellyPlugin_VST3 -j8
```

Important constraints:
- plugin/native work requires JUCE in `external/JUCE`
- KellyFFI intentionally links JUCE privately
- do not introduce duplicate JUCE linkage into executables that consume KellyFFI

## 6. CMake presets and editor-integrated native workflows

`CMakePresets.json` currently defines:
- `ninja-debug`
- `ninja-asan`
- `ninja-tsan`
- `xcode-debug`
- `xcode-release`

Current build presets target the RT harness:
- `ninja-debug-rt-harness`
- `ninja-asan-rt-harness`
- `ninja-tsan-rt-harness`

Useful commands:

```bash
cmake --preset ninja-debug
cmake --build --preset ninja-debug-rt-harness
cmake --preset ninja-asan
cmake --build --preset ninja-asan-rt-harness
```

## 7. Schema sync workflow

Source of truth for engine-facing intent schema:
- `shared_schemas/CompleteSongIntentRequest.json`

After changing schema or generated contract surfaces, run:

```bash
python3 scripts/sync_entities.py
```

This updates generated artifacts including:
- `src/types/Intent.ts`
- `engine/intent_ir/src/generated/intent.rs`
- Python validation mirrors

Minimum follow-up checks:

```bash
python3 -m pytest tests/unit/test_api_schema.py
cd engine/intent_ir && cargo test
```

Rules:
- generated artifacts are not hand-edited
- engine-facing and persisted intent must converge through validated Intent IR

## 8. Recommended task-oriented workflows

### Frontend/UI work
1. `npm run dev`
2. make changes in `src/`
3. run `npx tsc --noEmit`
4. run `npm run build`

### API/backend work
1. `npm run dev:python`
2. edit `music_brain/`
3. run `python3 -m flake8 music_brain/ --max-line-length 100`
4. run `python3 -m pytest tests/`

### Intent contract work
1. edit `shared_schemas/` and/or Intent IR validation logic
2. run `python3 scripts/sync_entities.py`
3. run Python schema tests
4. run `cd engine/intent_ir && cargo test`
5. if semantics changed, ensure migration docs and compatibility tests are updated

### Native engine / FFI / plugin work
Before changing these surfaces, read and follow:
- `AGENTS.md` native safety section
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`

Then:
1. configure/build affected CMake targets
2. run native tests if enabled
3. run sanitizer builds for risky changes
4. re-check schema and Python tests if bridge or contract layers changed

## 9. Debugging notes

### Frontend
- Vite serves on `1420`
- browser access is supported because host bind is `0.0.0.0`
- Tauri API imports are stubbed by `vite.config.ts` when `TAURI_PLATFORM` is not set, so the web build remains usable without a Tauri shell

### Python API
- docs available at `/docs`
- if `uvicorn` is missing, reinstall via `python3 -m pip install -e .`

### Native / RT work
- prefer preset-based debug, ASan, and TSan builds
- use RT harness targets for low-level runtime investigation
- no exceptions, locks, or allocations on RT paths

## 10. Verification matrix before merge

Choose the subset that matches the surfaces you touched.

### Frontend-only
```bash
npx tsc --noEmit
npm run build
```

### Python-only
```bash
python3 -m flake8 music_brain/ --max-line-length 100
python3 -m pytest tests/
```

### Intent/schema work
```bash
python3 scripts/sync_entities.py
python3 -m pytest tests/unit/test_api_schema.py
cd engine/intent_ir && cargo test
```

### Native/FFI/plugin work
```bash
cmake --build build --target <affected-target>
ctest --test-dir build --output-on-failure
cd engine/intent_ir && cargo test
python3 -m pytest tests/
```

Add ASan/TSan runs when changing ownership, FFI, lifecycle, or RT-sensitive code.

## 11. Known operational drift

These facts are intentionally documented so future work does not re-derive them:
- `README.md` still describes a Tauri-centered canonical UI surface and an external canonical Rust layer; that does not match the current architecture authority set in `docs/ARCHITECTURE.md`.
- `run_brain.py` remains useful, but it is not the primary product boot surface described by the architecture handoff.

If you update those surfaces later, keep this guide aligned with the actual runnable commands rather than historical intent.

## 12. Related docs

- `AGENTS.md`
- `BUILD.md`
- `docs/FULL_STACK_BUILD.md`
- `docs/ENVIRONMENT.md`
- `docs/BOOT.md`
- `docs/WORKSPACE_SETUP.md`
- `docs/SWARM_MATRIX.md`
