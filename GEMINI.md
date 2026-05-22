# GEMINI.md

This file provides guidance to Gemini CLI when working with code in this repository.

## Project overview

KmiDi / iDAW is an **AI-powered music creation platform** (monorepo). Four layers:

| Layer | Tech | Location |
|-------|------|----------|
| Frontend | React 19 + Vite + TypeScript + Tailwind | `src/` |
| Desktop shell | Tauri 2 + Rust | `engine/intent_ir/` |
| Native engine | C++20 (KellyCore, KellyFFI, JUCE 8) | `engine/`, `src/`, `include/`, `src_penta-core/` |
| Backend API | Python FastAPI (`music_brain`) | `music_brain/` |

Data flow: **React** → `invoke()` → **Tauri/Rust** → FFI → **KellyFFI (C ABI)** → **KellyCore (C++)**.
API flow: **React** → HTTP → **Music Brain API** (port 8000, `/generate`, `/docs`).

Architecture principle: Side A (C++ real-time, lock-free, no allocs) ↔ ring buffer ↔ Side B (Python AI + UI). Emotional intent feeds production rules. Human imperfection (timing/pitch drift) is a feature.

## Repository layout

```
KmiDi/
├── apps/kmidi/             # CLI app stub (pyproject.toml placeholder)
├── libs/
│   ├── ai_core/            # ML core library stub (pyproject.toml)
│   ├── daiw/               # C++ RT-safe primitives (daiw_core static lib, JUCE)
│   └── jepa/               # Workspace entry → music_brain/jepa/ (README + pyproject.toml)
├── src/                    # React app (components, hooks, types)
├── engine/intent_ir/       # Rust intent crate (commands, bridge, build.rs)
├── music_brain/            # Python FastAPI app + engine API
│   ├── jepa/               #   JEPA models (Audio-JEPA, Chord-JEPA, Stem-JEPA, trainer)
│   ├── penta_core/         #   Penta-core ML (emotion runners, diagnostics, PID Flow, etc.)
│   └── ...                 #   50+ submodules (see music_brain/ listing)
├── python/mcp/             # MCP servers (daiw_mcp, penta_swarm, mcp_todo, mcp_workstation)
├── .claude/                # Multi-agent orchestration (agents, commands, skills, verify)
├── shared_schemas/         # Single source of truth for intent (JSON → sync to TS/Rust)
├── scripts/                # sync_entities.py, dev-setup, build, env, acquire/
├── training/scripts/       # JEPA training entrypoints (train_jepa.py)
├── tests/                  # Python tests (pytest)
├── engine/, include/, src/ # C++ (engine, bridge, plugin, DSP)
├── include/prrot, include/penta  # PRROT/penta headers
├── src_penta-core/         # Penta-core C++ (harmony, groove, diagnostics)
├── external/JUCE/          # JUCE 8 (required for C++/plugins/FFI)
├── cmake/                  # CMake helpers
├── config/                 # Training/config YAML, source_manifest.yaml
├── docs/                   # DEVELOPMENT.md, ENVIRONMENT.md, FULL_STACK_BUILD.md, etc.
├── BUILD.md                # C++ / CMake / Tauri build reference
├── pyproject.toml          # Python deps (music_brain, fastapi, uvicorn, pydantic)
└── package.json            # npm scripts (dev, build, tauri)
```

## Common commands

### Setup

```bash
./scripts/dev-setup.sh          # One-command: JUCE submodule, npm install, pip install -e .
```

### Running services

```bash
npm run dev:all                  # React (localhost:1420) + Music Brain API (localhost:8000)
npm run dev                      # React only (Vite, localhost:1420)
npm run dev:python               # Music Brain API only (uvicorn, localhost:8000)
npm run dev:tauri                # Tauri desktop app (run dev:python separately for API)
```

### Building

```bash
npm run build                    # Frontend: tsc + vite build
pip install -e .                 # Python: install music_brain as editable package

# C++ (requires JUCE in external/JUCE/)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON
cmake --build build --target KellyFFI -j8
```

### Testing

```bash
python3 -m pytest tests/                            # All Python tests
python3 -m pytest tests/unit/test_prrot_bindings.py # Single test file
python3 -m pytest tests/ -k "test_name"             # Single test by name
python3 -m pytest tests/ -m unit                    # By marker (unit, integration, slow, cpp)
ctest --test-dir build --output-on-failure           # C++ tests (BUILD_TESTS=ON)
cd engine/intent_ir && cargo test                           # Rust/Tauri tests
```

### Linting

```bash
python3 -m flake8 music_brain/ --max-line-length 100   # Python lint (CI-enforced)
npx tsc --noEmit                                        # TypeScript type-check
```

### Schema sync

```bash
python3 scripts/sync_entities.py    # Sync shared_schemas/ → TS types + Rust types + Python validation
```

## Key build options (CMake)

| Option | Default | Notes |
|--------|---------|-------|
| `BUILD_KELLY_CORE` | ON | Core C++ library |
| `BUILD_KELLY_FFI` | ON | Shared lib for Tauri/Rust FFI |
| `BUILD_PLUGINS` | ON | VST3/CLAP (requires `KMIDI_BUILD_JUCE_UI=ON`) |
| `KMIDI_BUILD_JUCE_UI` | OFF | Must enable for plugin builds |
| `BUILD_TESTS` | OFF | C++ test suite |
| `BUILD_RT_HARNESS` | ON | Headless RT callback harness |
| `ENABLE_RTNEURAL` / `ENABLE_ONNX_RUNTIME` | OFF | ML inference backends |
| `KMIDI_OFFLINE_BUILD` | OFF | Disable FetchContent for offline builds |

## Architecture notes

- **KellyFFI** links JUCE and Qt **PRIVATE**. Executables that link only KellyFFI must not also link JUCE (avoids allocator mismatch / static-init crashes).
- **Intent schema source of truth:** `shared_schemas/CompleteSongIntentRequest.json`. Changes must be synced via `sync_entities.py` to `src/types/Intent.ts`, `engine/intent_ir/src/generated/intent.rs`, and Python validation.
- **`/generate` API** uses `GenerateRequest`/`EmotionalIntent` (defined in `music_brain/api.py`), **not** `CompleteSongIntentRequest`. The `instruments` field takes dicts `{"instrument": "piano"}`, not plain strings. `structure` names must match `^(intro|verse|chorus|bridge|outro|build|drop)$`.
- **App shell entrypoint:** `AppConsole` (in `main.tsx`). `App.tsx` is legacy/alternate and not imported.
- **Two build contexts:** Root CMake uses `BUILD_PLUGINS`, `KMIDI_BUILD_JUCE_UI`, `BUILD_KELLY_FFI`. Legacy DAIW in `KmiDi_FINAL/engine/cpp_music_brain` uses `DAIW_BUILD_VST3` / `DAIW_BUILD_AU` — do not mix.

## C++ conventions

- RT-safe audio callbacks: `noexcept`, no heap allocations on audio thread, prefer `std::pmr` arenas.
- AVX2 SIMD with scalar fallback (see `include/penta/` and DSP code in `libs/daiw/`).
- C++20 required. JUCE 8 must be present at `external/JUCE/`.

## Python conventions

- Line length: 100 (flake8/black enforced).
- Do not pass `--timeout` to pytest (pytest-timeout is not installed).
- `asyncio_mode = "auto"` in pytest config.

## Data governance

- **Datasets** live in `~/Datasets` only (env var: `KELLY_AUDIO_DATA_ROOT`). Never commit audio, MIDI, or large binary files.
- **Model weights/checkpoints** live in `~/Models/checkpoints/` (env var: `KELLY_MODEL_ROOT`). Never commit `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`.
- Hardcoded `/Users/<name>/...` or `/Volumes/...` paths in source are prohibited.
- Every training run requires a `run_manifest.yaml` before launch.

## Reference docs

| Doc | Content |
|-----|---------|
| `AGENTS.md` | Full agent context; **§ Native safety, FFI ownership, and verification map** for FFI/JUCE/RT before native edits |
| `BUILD.md` | C++ / CMake / Tauri build reference |
| `docs/DEVELOPMENT.md` | Dev guide, workflows, debugging |
| `docs/ENVIRONMENT.md` | Env vars, file layout, validation |
| `docs/FULL_STACK_BUILD.md` | React ↔ Tauri ↔ KellyFFI ↔ KellyCore integration |
| `docs/DATASETS_LAYOUT.md` | Dataset volume layout and acquisition |
| `docs/NATIVE_SAFETY_AND_FFI.md` | FFI, JUCE/ODR, RT safety, verification (mirror of `AGENTS.md`) |

## Shared agentic infrastructure

Shared agents, hooks, and skills live in `workspace-scaffold/`:

- `workspace-scaffold/agents/` — innovator, strategic-implementer, security-reviewer
- `workspace-scaffold/hooks/` — credential guard, scope enforcement, lint-on-write
- `workspace-scaffold/skills/` — dataset-packaging, training-ops-cockpit, training-pipeline-orchestrator
