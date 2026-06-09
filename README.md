# KmiDi Canonical Repository

Status: current repo entrypoint aligned to the 2026 architecture authority set
Last updated: 2026-06-08

KmiDi is a plugin-first, engine-separable AI-assisted music creation system.
The current canonical architecture authority begins with:
- `docs/ARCHITECTURE.md`
- `docs/REPO_MODULE_MAP.md`
- `docs/INTENT_IR_AUTHORITY.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`
- `docs/PERSISTENCE_AND_MIGRATION.md`

If older docs conflict with those, follow the authority set above.

## What is canonical in this repo

Primary active surfaces:
- `src/` — React frontend
- `music_brain/` — Python FastAPI backend and orchestration
- `engine/intent_ir/` — Rust Intent IR contract layer embedded into KellyFFI
- root CMake project (`engine/`, `src/`, `include/`, `src_penta-core/`) — C++ native engine, KellyFFI, plugin/runtime code
- `shared_schemas/` — canonical schema source for engine-facing/persisted intent contracts

Important clarification:
- `package.json` does not currently define `npm run dev:tauri`.
- Tauri-coupled code paths still exist in places as compatibility/latent dual-mode surfaces, but they are not the current operational center described by the architecture handoff.
- The most reliable combined development boot path today is `npm run dev:all`.

## Product and architecture center

- Product launch priority: plugin/runtime first
- Internal architecture requirement: preserve a future standalone-native-engine path
- Canonical intent truth: validated Intent IR
- Live runtime truth: native engine
- Project/session/persistence truth: plugin/runtime project layer
- AI is additive; plugin load, playback, editing, and saved project loading must survive AI/backend failure

## Quick start

### Setup

```bash
./scripts/dev-setup.sh
```

This currently runs:
- bootstrap helper
- `npm install`
- `python3 -m pip install -e .`

### Run the active dev stack

```bash
npm run dev:all
```

This starts:
- React/Vite frontend on `http://localhost:1420`
- Music Brain API on `http://localhost:8000`

Run separately if needed:

```bash
npm run dev
npm run dev:python
```

### Build checks

```bash
npx tsc --noEmit
npm run build
python3 -m pytest tests/
```

## Native / plugin build path

For C++ / KellyFFI / plugin work, use the root CMake project.
Example:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_KELLY_CORE=ON \
  -DBUILD_KELLY_FFI=ON \
  -DKMIDI_BUILD_JUCE_UI=ON \
  -DBUILD_PLUGINS=ON

cmake --build build --target KellyFFI -j8
```

Use these docs for the detailed native story:
- `BUILD.md`
- `docs/FULL_STACK_BUILD.md`
- `docs/DEVELOPMENT.md`
- `AGENTS.md`

## Repository map

Top-level high-signal paths:
- `src/` — React app
- `music_brain/` — FastAPI app, engine API, orchestration, ML-adjacent Python modules
- `engine/intent_ir/` — Rust validator/builder/FFI half of Intent IR
- `shared_schemas/` — source schema for generated TS/Rust/Python contract artifacts
- `scripts/` — setup, sync, build, env, acquisition helpers
- `tests/` — Python tests
- `docs/` — operational and architecture docs
- `external/JUCE/` — JUCE dependency for native/plugin builds

Supporting/legacy-adjacent surfaces still present in-tree:
- `libs/daiw/`
- `include/penta/`, `include/prrot/`
- `src_penta-core/`
- `legacy/ui/`
- older review/audit docs that may preserve historical Tauri-era assumptions

## Intent contract rule

When changing engine-facing or persisted intent:

```bash
python3 scripts/sync_entities.py
python3 -m pytest tests/unit/test_api_schema.py
cd engine/intent_ir && cargo test
```

Generated artifacts are not hand-edited.

## Operational docs

Use these for current runnable truth:
- `docs/DEVELOPMENT.md`
- `docs/ENVIRONMENT.md`
- `docs/BOOT.md`
- `docs/WORKSPACE_SETUP.md`

Use these for architecture truth:
- `docs/ARCHITECTURE.md`
- `docs/REPO_MODULE_MAP.md`
- `docs/INTENT_IR_AUTHORITY.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`
- `docs/PERSISTENCE_AND_MIGRATION.md`

## Known drift still in-tree

Historical and legacy surfaces still preserved in-tree may mention superseded Tauri-era assumptions, including:
- `npm run dev:tauri`
- Tauri as the canonical desktop shell
- an external canonical Rust layer outside this repo

Treat those references as archaeology, not runnable truth, unless they are explicitly revalidated against the authority set and the actual checked-in root scripts.
