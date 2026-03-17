# AGENTS.md

Canonical agent and developer context for KmiDi / iDAW: project structure, build, services, and gotchas.

---

## Project overview

KmiDi / iDAW is an **AI-powered music creation platform** (monorepo). Stack:

| Layer | Tech | Purpose |
|-------|------|---------|
| Frontend | React (Vite) + TypeScript | UI, intent builder, emotion wheel |
| Desktop shell | Tauri 2 + Rust | Native host, FFI to C++ |
| Native engine | C++20 (KellyCore, KellyFFI) | AI/emotion engine, audio, plugins |
| Backend API | Python (FastAPI, music_brain) | Generate endpoint, engine API |

**Required for development (e.g. Cursor Cloud):** React + Music Brain API.  
**Optional (need native toolchains):** Tauri desktop app, C++ KellyCore/KellyFFI, VST3/CLAP plugins, Streamlit mixer, Android.

---

## Repository layout

```
KmiDi/
├── src/                    # React app (components, hooks, types)
├── src-tauri/              # Tauri app (Rust commands, bridge, build.rs)
├── music_brain/            # Python FastAPI app and engine API
├── shared_schemas/         # Single source of truth for intent (JSON → sync to TS/Rust)
├── scripts/                # sync_entities.py, dev-setup, build, env, acquire/ (source_manifest)
├── tests/                  # Python tests (pytest)
├── engine/, include/, src/  # C++ (engine, bridge, plugin)
├── external/JUCE/          # JUCE 8 (required for C++/plugins/FFI)
├── cmake/                  # CMake helpers
├── config/                 # Training/config YAML, source_manifest.yaml (external sources)
├── docs/                   # DEVELOPMENT.md, ENVIRONMENT.md, FULL_STACK_BUILD.md, DATASETS_LAYOUT.md
├── BUILD.md                # C++ / CMake / Tauri build reference
├── pyproject.toml          # Python deps (music_brain, fastapi, uvicorn, pydantic)
└── package.json            # npm scripts (dev, build, tauri)
```

Data flow: **React** → `invoke()` / events → **Tauri** → **Rust FFI** → **KellyFFI (C ABI)** → **KellyCore (C++)**.  
API flow: **React** → HTTP → **Music Brain API** (`/generate`, `/docs`).

---

## Prerequisites

- **CMake** 3.27+, **Ninja**, **Node** 20+, **Python** 3.11+ (3.9+ in pyproject), **Rust** (stable)
- **macOS:** Xcode Command Line Tools; **Linux:** GCC 9+ / Clang 10+, ALSA/JACK, X11
- **C++/Tauri/plugins:** JUCE in `external/JUCE`, Qt6 (if building desktop/plugins)

One-command setup from repo root:

```bash
./scripts/dev-setup.sh
```

Runs: bootstrap (JUCE submodule, version checks), `npm install`, `pip install -e .` (music_brain + tests).  
In cloud VMs, JUCE submodule step can be skipped if C++ build is not needed.

---

## Running services

| Context | Command | Notes |
|---------|---------|--------|
| **Any (React + API)** | `npm run dev:all` | React (Vite) + Music Brain API only; no port collision |
| **Desktop (Tauri)** | `npm run dev:tauri` | Run **separately**; Tauri starts its own Vite. For API, run `npm run dev:python` in another terminal |

| Service | Command | URL |
|---------|---------|-----|
| React (Vite) | `npm run dev` (or `npm run dev:react`) | http://localhost:1420 |
| Music Brain API | `npm run dev:python` or `python3 -m uvicorn music_brain.api:app --reload --port 8000 --host 0.0.0.0` | http://localhost:8000, docs at /docs |
| Tauri desktop | `npm run dev:tauri` | Starts Vite then opens http://localhost:1420 in the native window |

- Python API needs `fastapi`, `uvicorn`, `pydantic` (and `pip install -e .`). Ensure `$HOME/.local/bin` is on `PATH` if uvicorn is installed with `--user`.
- Vite is bound to `0.0.0.0` (see `vite.config.ts`) so the frontend is reachable outside the host.

---

## Build (full context)

### Frontend

- **Dev:** `npm run dev` (Vite, port 1420).
- **Type-check:** `npx tsc --noEmit`.
- **Production:** `npm run build` (runs `tsc && vite build`).
- **TypeScript:** `strict` is off in tsconfig; enable incrementally and fix errors before enabling.

`package-lock.json` is committed; use `npm ci` in CI for reproducible installs, or `npm install` for local dev.

### Python

- **Install:** `pip install -e .` (from repo root; installs `music_brain` and deps).
- **Run API:** `python3 -m uvicorn music_brain.api:app --reload --port 8000`.
- **Lint:** `python3 -m flake8 music_brain/ --max-line-length 100` (CI-enforced; respect repo flake8 config).
- **Tests:** `python3 -m pytest tests/` (config in `pytest.ini`). Do **not** pass `--timeout` (pytest-timeout not installed).

### C++ / CMake (root project)

- **Configure (example):**  
  `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON -DKMIDI_BUILD_JUCE_UI=ON -DBUILD_PLUGINS=ON`
- **Key options:**

  | Option | Default | Effect |
  |--------|---------|--------|
  | `BUILD_KELLY_CORE` | ON | Kelly core library, app, plugins |
  | `BUILD_KELLY_FFI` | ON | KellyFFI shared lib for Tauri/Rust |
  | `BUILD_PLUGINS` | ON | VST3/CLAP (requires `KMIDI_BUILD_JUCE_UI=ON`) |
  | `BUILD_DESKTOP` | OFF* | Desktop GUI (set ON if `KMIDI_BUILD_QT_UI=ON`) |
  | `BUILD_TESTS` | OFF | Native/unit tests |
  | `KMIDI_BUILD_JUCE_UI` | OFF | Legacy JUCE UI; enables plugin build path |
  | `KMIDI_BUILD_QT_UI` | OFF | Legacy Qt6 UI |
  | `USE_KMI_DI_FINAL` | OFF | Use KmiDi_FINAL components (DSP, etc.) |
  | `BUILD_RT_HARNESS` | ON | Headless RT callback harness |
  | `ENABLE_TRACY` | OFF | Tracy profiling |
  | `ENABLE_RTNEURAL` / `ENABLE_ONNX_RUNTIME` | OFF | ML inference |
  | `KMIDI_OFFLINE_BUILD` | OFF | No FetchContent; vendor deps in external/ |

  \*With default options, `BUILD_DESKTOP` and `BUILD_PLUGINS` are forced OFF unless Qt/JUCE UI are enabled.

- **Key targets:**
  - `KellyCore` — core C++ library
  - `KellyFFI` — shared library for Tauri (output e.g. `build/libKellyFFI.dylib`; CMake copies to `src-tauri/resources/`)
  - `KellyPlugin_VST3` — VST3 plugin (e.g. `build/KellyPlugin_artefacts/Release/VST3/...`)
  - `KellyFFIBenchmark` — FFI benchmark (requires KellyFFI; do not link JUCE in the benchmark exe — KellyFFI links JUCE PRIVATE)
  - `KellyTests` — C++ tests (when `BUILD_TESTS=ON` and Catch2 present)

- **Build:** `cmake --build build --target KellyFFI -j8` (or `KellyCore`, `KellyPlugin_VST3`, etc.).
- **JUCE:** Must be present at `external/JUCE/CMakeLists.txt` (full clone with extras/Build/CMake). JUCE 8–aligned.
- **KellyFFI:** Links JUCE and Qt **PRIVATE** so executables that link only KellyFFI do not get a second JUCE (avoids allocator mismatch / "pointer being freed was not allocated" in static init).

Full-stack build and Tauri link paths: `docs/FULL_STACK_BUILD.md`. Optional helper: `./scripts/build-full-stack.sh` (e.g. `--debug`, `--no-plugins`, `--build-dir`, `--no-tauri`).

### Tauri

- **Dev:** `npm run dev:tauri` (after KellyFFI is built if using native backend).
- **Build:** `npm run tauri build` (or `npm ci && npm run tauri build`).
- **Rust tests:** `cd src-tauri && cargo test`.
- `src-tauri/build.rs` looks for `libKellyFFI` in `../build`, `../build/debug`, `../build/release`; for custom build dirs, adjust build.rs or copy the dylib.

### Schema sync (UI–engine contract)

- **Source of truth:** `shared_schemas/CompleteSongIntentRequest.json`.
- **Sync:** `python3 scripts/sync_entities.py` — updates `src/types/Intent.ts`, `src-tauri/src/generated/intent.rs`, and Python validation.
- **CI:** Verifies no drift (e.g. `tests/unit/test_api_schema.py`).

---

## Lint / test / build (summary)

| Area | Lint / check | Test | Build |
|------|--------------|------|--------|
| Frontend | `npx tsc --noEmit` | (Vitest if configured) | `npm run build` |
| Python | `python3 -m flake8 music_brain/ --max-line-length 100` | `python3 -m pytest tests/` | `pip install -e .` |
| C++ | — | `ctest --test-dir build --output-on-failure` (if BUILD_TESTS=ON) | `cmake --build build --target <target>` |
| Rust/Tauri | `cargo clippy` (optional) | `cargo test` in src-tauri | `npm run tauri build` |

---

## Environment and config

- **Env files:** `.env`, `.env.development`, `.env.production`; feature-specific under `env/`; user overrides in `.env.local` (git-ignored). Load: `source scripts/load-env.sh` (or `scripts/load-env.sh tauri ml`).
- **Validation:** `./scripts/validate-env.sh`.
- **Important vars (see `docs/ENVIRONMENT.md`):** `KELLY_MODELS_PATH`, `KMIDI_API_URL` (default `http://127.0.0.1:8000`), `TAURI_DEV_HOST`, `RUST_LOG`, `VITE_*` for frontend.
- **Build options** are CMake flags, not env vars; set via `-D` on the `cmake` command.

---

## API: `/generate` and intent

- The **HTTP** `/generate` endpoint uses **`GenerateRequest`** and **`EmotionalIntent`** (defined inline in `music_brain/api.py`), **not** `CompleteSongIntentRequest`. At minimum you must provide:
  - `intent.emotional_intent` (required string — mood/emotion description)
  - `intent.core_desire` (string)
  - `intent.technical.genre`, `intent.technical.key` (string with space, e.g. `"C major"`), `intent.technical.bpm`, `intent.technical.structure`, `intent.technical.instruments`
  - **Structure:** list of dicts with `name` matching `^(intro|verse|chorus|bridge|outro|build|drop)$` and `bars` (int)
  - **Instruments:** list of **dicts** with an `instrument` key (not plain strings)
- Note: `CompleteSongIntentRequest` in `music_brain/engine_api/schema.py` is a separate strict schema used at the engine boundary, not the `/generate` API payload.

Minimal working example:

```json
{
  "intent": {
    "emotional_intent": "intimate and powerful pop ballad",
    "core_desire": "emotional pop ballad",
    "technical": {
      "genre": "pop",
      "key": "C major",
      "bpm": 120,
      "structure": [
        { "name": "intro", "bars": 4 },
        { "name": "verse", "bars": 8 },
        { "name": "chorus", "bars": 8 }
      ],
      "instruments": [
        { "instrument": "piano" },
        { "instrument": "bass" },
        { "instrument": "drums" }
      ]
    }
  }
}
```

---

## Gotchas

- **Lockfile:** `package-lock.json` is committed; use `npm ci` in CI and `npm install` for local dev.
- **App shell:** Default entrypoint is `AppConsole` (main.tsx). `App.tsx` is legacy/alternate and not imported.
- **Cloud vs local:** `npm run dev:all` runs React + API only (safe everywhere). For desktop, run `npm run dev:tauri` separately.
- **JUCE:** Large submodule; in cloud-only workflows bootstrap can skip `git submodule update` for JUCE if C++ is not built.
- **Two build contexts:** Root CMake uses `BUILD_PLUGINS`, `KMIDI_BUILD_JUCE_UI`, `BUILD_KELLY_FFI`. Legacy DAIW in `KmiDi_FINAL/engine/cpp_music_brain` uses `DAIW_BUILD_VST3` / `DAIW_BUILD_AU` — do not mix.
- **KellyFFI and JUCE:** KellyFFI is built with JUCE linked **PRIVATE** and `JUCE_DISABLE_JUCE_VERSION_PRINTING=1` to avoid two copies of JUCE and static-init crashes when the benchmark (or any exe that only links the dylib) runs.
- **pytest:** Do not pass `--timeout`; pytest-timeout is not a project dependency.

---

## Reference docs

| Doc | Content |
|-----|--------|
| `docs/DEVELOPMENT.md` | Full dev guide, workflows, debugging, C++/Rust/React structure |
| `docs/ENVIRONMENT.md` | Env vars, file layout, loading, validation |
| `docs/FULL_STACK_BUILD.md` | React ↔ Tauri ↔ KellyFFI ↔ KellyCore, build order, integration tests |
| `docs/DATASETS_LAYOUT.md` | Canonical dataset volume layout (by_source, by_domain), KMIDI_DATASETS_PATH, acquisition paths |
| `docs/SOURCE_INTEGRATION_PLAN.md` | Source integration and download plan; external briefings in `docs/research/sources/` |
| `docs/AU_PLUGIN_ARCHITECTURE.md` | Audio Unit (AU) plugin architecture: macOS, iOS AUv3, build contexts |
| `docs/SAGEMAKER_SETUP.md` | SageMaker AI training (JEPA): IAM, S3, ECR, image build, launch jobs |
| `BUILD.md` | C++ / CMake / Tauri build instructions and prerequisites |
