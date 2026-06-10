# AGENTS.md

Canonical agent and developer context for KmiDi / iDAW: project structure, build, services, and gotchas.

## Dev-root boundary

This file is project-local. `~/Dev` is only the workspace container; once a session is inside this repo, use this file plus `CLAUDE.md` and project docs as the active rules.

- Scope searches, build commands, tests, schema syncs, and model/runtime checks to this repository unless a sibling repo is explicitly named.
- Load project-specific code intelligence, schema checks, audio/runtime AI tooling, and CI checks from this repo's docs only.
- If work touches another folder inside `~/Dev`, switch to that folder's own `AGENTS.md` before editing there.
- Do not refresh or rely on a `~/Dev`-wide index as the source of truth for this project.

**Before changing C++, KellyFFI, Rust intent_ir FFI, or real-time audio paths:** read and follow **§ [Native safety, FFI ownership, and verification map](#native-safety-ffi-ownership-and-verification-map)** below (file paths, ownership rules, commands). Do not rely on memory of this doc from prior sessions.

---

## Project overview

KmiDi / iDAW is an **AI-powered music creation platform** (monorepo). Stack:

| Layer | Tech | Purpose |
|-------|------|---------|
| Frontend | React (Vite) + TypeScript | UI, intent builder, emotion wheel |
| Intent IR | Rust staticlib (C ABI) `engine/intent_ir/` | `IntentFrame` validator/builder/types; linked **into** KellyFFI dylib |
| Native engine | C++20 (KellyCore, KellyFFI dylib) | AI/emotion engine, audio, plugins (VST3/CLAP/AU) |
| Backend API | Python (FastAPI, music_brain) | Generate endpoint, engine API |

KellyFFI dylib exposes one combined C ABI: `kelly_*` (from C++) plus `IntentFrameBuilder_*` / `validate_intent_frame_ffi` (from the embedded Rust `intent_ir` staticlib). Consumers are Python bindings, C++ tests/benchmarks, and plugin hosts. **There is no Tauri desktop shell in the canonical tree.**

**Required for development (e.g. Cursor Cloud):** React + Music Brain API.  
**Optional (need native toolchains):** C++ KellyCore/KellyFFI, VST3/CLAP/AU plugins, Streamlit mixer, Android.

---

## Repository layout

```
KmiDi/
├── apps/kmidi/             # CLI app stub (pyproject.toml placeholder)
├── libs/
│   ├── ai_core/            # ML core library stub (pyproject.toml)
│   ├── daiw/               # C++ RT-safe primitives (daiw_core static lib, JUCE)
│   └── jepa/               # Workspace entry → music_brain/jepa/ (README + pyproject.toml)
├── src/                    # React app (components, hooks, types)
├── engine/intent_ir/       # Rust staticlib: IntentFrame C ABI (src/ffi.rs, validator, builder, types, cbindgen.toml) linked into KellyFFI
├── music_brain/            # Python FastAPI app + engine API
│   ├── jepa/               #   JEPA models (Audio-JEPA, Chord-JEPA, Stem-JEPA, trainer)
│   ├── penta_core/         #   Penta-core ML (emotion runners, diagnostics, PID Flow, etc.)
│   └── ...                 #   50+ submodules (see music_brain/ listing)
├── python/mcp/             # MCP servers (daiw_mcp, penta_swarm, mcp_todo, mcp_workstation)
├── .claude/                # Multi-agent orchestration (agents, commands, skills, verify)
├── shared_schemas/         # Single source of truth for intent (JSON → sync to TS/Rust)
├── scripts/                # sync_entities.py, dev-setup, build, env, acquire/ (source_manifest)
├── training/scripts/       # JEPA training entrypoints (train_jepa.py)
├── tests/                  # Python tests (pytest)
├── engine/, include/, src/ # C++ (engine, bridge, plugin); engine/src/dsp, engine/intent_ir (merged)
├── include/penta           # penta headers (merged from KmiDi_FINAL)
├── src_penta-core/         # Penta-core C++ (harmony, groove, diagnostics, etc.)
├── external/JUCE/          # JUCE 8 (required for C++/plugins/FFI)
├── cmake/                  # CMake helpers
├── config/                 # Training/config YAML, source_manifest.yaml (external sources)
├── docs/                   # DEVELOPMENT.md, ENVIRONMENT.md, FULL_STACK_BUILD.md, DATASETS_LAYOUT.md
├── BUILD.md                # C++ / CMake build reference
├── pyproject.toml          # Python deps (music_brain, fastapi, uvicorn, pydantic)
└── package.json            # npm scripts (dev, build, preview)
```

API flow: **React** → HTTP → **Music Brain API** (`/generate`, `/docs`).  
Native flow: a process loads the **KellyFFI** dylib and calls its C ABI (`kelly_*` for engine state/generation; `IntentFrameBuilder_*` / `validate_intent_frame_ffi` for intent IR). Inside the dylib, the C++ side links **KellyCore**; the embedded Rust **intent_ir** staticlib provides the intent ABI half. There is no Tauri shell in the canonical tree.

---

## Prerequisites

- **CMake** 3.27+, **Ninja**, **Node** 20+, **Python** 3.11+ (3.9+ in pyproject), **Rust** (stable)
- **macOS:** Xcode Command Line Tools; **Linux:** GCC 9+ / Clang 10+, ALSA/JACK, X11
- **C++/plugins:** JUCE in `external/JUCE`, Qt6 (if building desktop UI or plugins)

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

| Service | Command | URL |
|---------|---------|-----|
| React (Vite) | `npm run dev` (or `npm run dev:react`) | http://localhost:1420 |
| Music Brain API | `npm run dev:python` or `python3 -m uvicorn music_brain.api:app --reload --port 8000 --host 0.0.0.0` | http://localhost:8000, docs at /docs |

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
  | `BUILD_KELLY_FFI` | ON | KellyFFI dylib (C ABI; embeds Rust `intent_ir` staticlib) |
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
  | `KMIDI_ENABLE_ASAN` | OFF | AddressSanitizer + UBSan (Debug builds; mutually exclusive with TSan) |
  | `KMIDI_ENABLE_TSAN` | OFF | ThreadSanitizer (Debug builds; mutually exclusive with ASan) |

  \*With default options, `BUILD_DESKTOP` and `BUILD_PLUGINS` are forced OFF unless Qt/JUCE UI are enabled.

- **Key targets:**
  - `KellyCore` — core C++ library
  - `KellyFFI` — shared library exposing the combined C ABI (output e.g. `build/libKellyFFI.dylib`); embeds the Rust `intent_ir` staticlib
  - `KellyPlugin_VST3` — VST3 plugin (e.g. `build/KellyPlugin_artefacts/Release/VST3/...`)
  - `KellyFFIBenchmark` — FFI benchmark (requires KellyFFI; do not link JUCE in the benchmark exe — KellyFFI links JUCE PRIVATE)
  - `KellyTests` — C++ tests (when `BUILD_TESTS=ON` and Catch2 present)

- **Build:** `cmake --build build --target KellyFFI -j8` (or `KellyCore`, `KellyPlugin_VST3`, etc.).
- **JUCE:** Must be present at `external/JUCE/CMakeLists.txt` (full clone with extras/Build/CMake). JUCE 8–aligned.
- **KellyFFI:** Links JUCE and Qt **PRIVATE** so executables that link only KellyFFI do not get a second JUCE (avoids allocator mismatch / "pointer being freed was not allocated" in static init).

Full-stack build paths: `docs/FULL_STACK_BUILD.md`. Optional helper: `./scripts/build-full-stack.sh` (e.g. `--debug`, `--no-plugins`, `--build-dir`).

### Rust intent_ir

- **Crate:** `engine/intent_ir/` (staticlib + rlib; `panic = "abort"` in release for FFI safety).
- **Build:** linked into KellyFFI by the root CMake when `BUILD_KELLY_FFI=ON`.
- **Tests:** `cd engine/intent_ir && cargo test`.
- **Header generation:** `cbindgen.toml` produces the C header for the Rust-side ABI consumed by C++ and external callers.

### Schema sync (UI–engine contract)

- **Source of truth:** `shared_schemas/CompleteSongIntentRequest.json`.
- **Sync:** `python3 scripts/sync_entities.py` — updates `src/types/Intent.ts`, `engine/intent_ir/src/generated/intent.rs`, and Python validation.
- **CI:** Verifies no drift (e.g. `tests/unit/test_api_schema.py`).

---

## Lint / test / build (summary)

| Area | Lint / check | Test | Build |
|------|--------------|------|--------|
| Frontend | `npx tsc --noEmit` | (Vitest if configured) | `npm run build` |
| Python | `python3 -m flake8 music_brain/ --max-line-length 100` | `python3 -m pytest tests/` | `pip install -e .` |
| C++ | — | `ctest --test-dir build --output-on-failure` (if BUILD_TESTS=ON) | `cmake --build build --target <target>` |
| Rust (intent_ir) | `cargo clippy` | `cd engine/intent_ir && cargo test` | — |

---

## Environment and config

- **Env files:** `.env`, `.env.development`, `.env.production`; feature-specific under `env/`; user overrides in `.env.local` (git-ignored). Load: `source scripts/load-env.sh` (or `scripts/load-env.sh ml`).
- **Validation:** `./scripts/validate-env.sh`.
- **Important vars (see `docs/ENVIRONMENT.md`):** `KELLY_MODELS_PATH`, `KMIDI_API_URL` (default `http://127.0.0.1:8000`), `RUST_LOG`, `VITE_*` for frontend.
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
- **Cloud vs local:** `npm run dev:all` runs React + API only (safe everywhere). Native plugin/engine work requires a local C++ toolchain (CMake + JUCE submodule).
- **JUCE:** Large submodule; in cloud-only workflows bootstrap can skip `git submodule update` for JUCE if C++ is not built.
- **Two build contexts:** Root CMake uses `BUILD_PLUGINS`, `KMIDI_BUILD_JUCE_UI`, `BUILD_KELLY_FFI`. Legacy DAIW in `KmiDi_FINAL/engine/cpp_music_brain` uses `DAIW_BUILD_VST3` / `DAIW_BUILD_AU` — do not mix.
- **KellyFFI and JUCE:** KellyFFI is built with JUCE linked **PRIVATE** and `JUCE_DISABLE_JUCE_VERSION_PRINTING=1` to avoid two copies of JUCE and static-init crashes when the benchmark (or any exe that only links the dylib) runs.
- **pytest:** Do not pass `--timeout`; pytest-timeout is not a project dependency.

---

## On-device and alignment tools

- **MIDI-CI daemon:** Optional build `-DBUILD_MIDI_CI_DAEMON=ON` produces `tools/midi_ci_daemon/midi_ci_daemon`; reads JSON microformat from stdin and sends MIDI-CI Property Exchange SysEx via libremidi. See `tools/midi_ci_daemon/README.md`.
- **Core ML LLM (stateful KV-cache):** `scripts/export_llm_coreml.py` wraps ExecuTorch export with Core ML state + b4w quantize; `tools/coreml_llm_runner` Swift app runs state-threaded greedy decode. Sub-15 ms/token target on M4. See `docs/FULL_STACK_BUILD.md` and `tools/coreml_llm_runner/README.md`.
- **PID Flow diagnostic:** `penta_core.ml.diagnostics.pid_flow` computes layer-wise Redundant / Text Unique / Audio Unique / Synergy to detect modality collapse. Run `scripts/run_pid_flow.py --dummy` to validate (use `PYTHONPATH=music_brain` from repo root).
- **Canonicalization (vector DB hot-swap):** `penta_core.ml.canonicalize_embeddings.fit_orthogonal_map(anchor_old, anchor_new)` plus `apply_map` for backward-compatible retrieval after encoder upgrades. CLI: `scripts/canonicalize_embeddings.py --old-embeddings ... --new-embeddings ... --output map.npz`.
- **APSC Multi-Stem Wrapper:** `music_brain/penta_core/ml/apsc_wrapper.py` mitigates position bias via prompt permutation and majority vote.
- **StructXLIP Symbolic Preprocessor:** `music_brain/penta_core/ml/structxlip/` extracts audio edge maps (onset, flux) and alignment losses for structure-aware training.

---

## Native safety, FFI ownership, and verification map

Single place for **memory/FFI/JUCE/RT** concerns. Use this when touching native code, adding FFI entry points, or debugging crashes around the dylib.

Human-oriented copy (same content, readable in docs navigation): [`docs/NATIVE_SAFETY_AND_FFI.md`](docs/NATIVE_SAFETY_AND_FFI.md). **Edit `AGENTS.md` first**, then align that file in the same PR.

### FFI buffer ownership (who allocates / frees)

The KellyFFI dylib exposes **one combined C ABI** with two halves:

- **`kelly_*` half** — implemented in C++ (`src/bridge/kelly_ffi.cpp`), declared in `src/bridge/kelly_ffi.h`. Owns `kelly_free_string` for caller-frees returns.
- **`IntentFrameBuilder_*` / `validate_intent_frame_ffi` half** — implemented in Rust (`engine/intent_ir/src/ffi.rs`) and linked **into** KellyFFI as a staticlib. Header generated via cbindgen.

Both halves must follow the same ownership discipline. There is **no separate "Rust consumer of KellyFFI"** in the canonical tree — Rust is inside the dylib, not above it.

| What | Where | Agent action |
|------|--------|----------------|
| C++ ABI contract (which `char*` to free) | `src/bridge/kelly_ffi.h` — `kelly_free_string`, comments on static vs heap | Every new `kelly_*` return type must be documented (caller frees vs static, e.g. `kelly_get_error_message`). |
| C++ ABI implementation | `src/bridge/kelly_ffi.cpp` | Pair every heap allocation returned across the boundary with `kelly_free_string` (or document static storage). |
| Rust ABI implementation | `engine/intent_ir/src/ffi.rs` | All `extern "C"` symbols use `Box::into_raw` / `Box::from_raw` for handles, wrap bodies in `catch_unwind` (release builds use `panic = "abort"` as backstop), and document who owns out-pointers. |
| Rust ABI header generation | `engine/intent_ir/cbindgen.toml` | When adding/changing an `extern "C"` symbol in `ffi.rs`, regenerate the C header so C++ callers see the same prototype. |
| External callers of either half | C++ tests/benchmarks, Python bindings (`bindings/`), plugin code | Match the header: heap → callee documents the free fn; static/thread-local → **never** free. |
| Regression tests | `tests/cpp/test_kelly_ffi.cpp` (C++ side); `engine/intent_ir` `cargo test` (Rust side) | Extend when adding FFI; run with `BUILD_TESTS=ON` and C++ tests enabled. |

### Duplicate JUCE / ODR / allocator mismatch

| What | Where | Agent action |
|------|--------|----------------|
| CMake policy (PRIVATE JUCE on KellyFFI) | `CMakeLists.txt` (~KellyFFI target: `target_link_libraries(KellyFFI ... PRIVATE juce::...)`, `JUCE_DISABLE_JUCE_VERSION_PRINTING`) | New executables/libs must **not** also link JUCE if they only consume `KellyFFI` the wrong way — see comments in CMake next to `KellyFFIBenchmark`. |
| Benchmark pattern | `CMakeLists.txt` — `KellyFFIBenchmark` links **only** `KellyFFI` | Do not add `juce::` targets to small harness exes that already link KellyFFI. |
| Manual verification | Linker / `nm` / `otool -L` (macOS) | If suspicious double-free or JUCE init crashes: confirm a single JUCE inside `libKellyFFI`, not a second copy in the host. |

### RT allocations, locks, audio thread

| What | Where | Agent action |
|------|--------|----------------|
| Lock-free RT snapshot type | `include/penta/common/RTState.h` | Atomics only; `static_assert` lock-freedom. No new heap use in RTState hot paths. |
| Plugin callback | `src/plugin/PluginProcessor.cpp`, `PluginProcessor.h` | Review any change inside `processBlock` for heap alloc, blocking locks, or unbounded work. |
| Headless harness | `rt_harness/` (enabled by `BUILD_RT_HARNESS` in root `CMakeLists.txt`) | Use for callback regression when changing RT behavior. |
| Sanitizers | Debug build: `KMIDI_ENABLE_ASAN=ON` (see `CLAUDE.md`) | Run `ctest` (and affected targets) after native changes that might introduce UB or lifetime bugs. |

### Version and contract drift (not the same as HTTP `intent_schema_version`)

| What | Where | Agent action |
|------|--------|----------------|
| KellyFFI ABI / dylib | `CMakeLists.txt` — `KellyFFI` `VERSION` / `SOVERSION`; `engine/intent_ir/Cargo.toml`; cbindgen-generated header | Breaking the C ABI (either `kelly_*` half or Rust `IntentFrameBuilder_*` / `validate_intent_frame_ffi` half) requires a version bump and a coordinated update of all C++/Python consumers. |
| TS / Rust / Python intent shapes | `shared_schemas/`, `scripts/sync_entities.py`, `src/types/Intent.ts`, `engine/intent_ir/src/generated/` | After schema edits, run sync and commit generated files; run Python schema tests. |
| HTTP API only | `music_brain/api_schemas/` | REST contract evolution; does not fix C++ memory by itself. |

### Commands to run (when native/FFI touched)

- Configure and build affected targets, e.g. `KellyFFI`, `KellyCore`, plugins: see `BUILD.md` and root `CMakeLists.txt`.
- C++ tests: `BUILD_TESTS=ON`, then `ctest --test-dir build --output-on-failure` (when enabled).
- Sanitizer: `KMIDI_ENABLE_ASAN=ON` Debug build + `ctest` per `CLAUDE.md`.
- Rust: `cd engine/intent_ir && cargo test`.
- Python (if API/schemas touched): `flake8 music_brain/`, `pytest tests/`.

---

## Integration gate (merge checklist)

Every PR or feature branch touching native code must satisfy all of the following before merge:

- [ ] **Clean build** of every affected target (`KellyCore`, `KellyFFI`, plugins, tests) with no warnings promoted to errors.
- [ ] **Sanitizer clean:** Debug build with `KMIDI_ENABLE_ASAN=ON` passes all tests with zero ASan/UBSan findings. Document any waiver with a tracking ticket.
- [ ] **No new heap allocations or locks on RT paths.** Audio callbacks must remain `noexcept`, allocation-free, and lock-free. Review any code that runs inside `processBlock` or the RT callback harness.
- [ ] **No duplicate JUCE linkage / ODR violations.** KellyFFI links JUCE PRIVATE. Any new executable or shared library must not also link JUCE directly — verify with `nm` or linker diagnostics if in doubt.
- [ ] **FFI ownership.** Any new `extern "C"` pointer contract on the C++ half is documented in `src/bridge/kelly_ffi.h`; on the Rust half it lives in `engine/intent_ir/src/ffi.rs` with the cbindgen header regenerated. Free vs static must be explicit. See **§ Native safety, FFI ownership, and verification map** above.
- [ ] **Single canonical tree.** New code goes into the repo root, not `KmiDi_FINAL/`, `KmiDi_PROJECT/`, or worktree-only paths. If importing from KmiDi_FINAL, copy into root and delete the worktree copy in the same PR.
- [ ] **Schema sync.** If `shared_schemas/` changed, `scripts/sync_entities.py` was run and generated files are committed.
- [ ] **Python lint + tests pass.** `flake8 music_brain/` and `pytest tests/` green.

---

## Reference docs

| Doc | Content |
|-----|--------|
| `docs/prd/KMIDI_PRD.md` | Product requirements: bounded contexts, hybrid modular monolith, TTG, RT hazards, APIs, MVP |
| `docs/DEVELOPMENT.md` | Full dev guide, workflows, debugging, C++/Rust/React structure |
| `docs/ENVIRONMENT.md` | Env vars, file layout, loading, validation |
| `docs/FULL_STACK_BUILD.md` | React ↔ Music Brain API ↔ KellyFFI ↔ KellyCore, build order, integration tests |
| `docs/DATASETS_LAYOUT.md` | Canonical dataset volume layout (by_source, by_domain), KMIDI_DATASETS_PATH, acquisition paths |
| `docs/SOURCE_INTEGRATION_PLAN.md` | Source integration and download plan; external briefings in `docs/research/sources/` |
| `docs/AU_PLUGIN_ARCHITECTURE.md` | Audio Unit (AU) plugin architecture: macOS, iOS AUv3, build contexts |
| `docs/SAGEMAKER_SETUP.md` | SageMaker AI training (JEPA): IAM, S3, ECR, image build, launch jobs |
| `docs/LATENT_ARCHITECTURE.md` | Six high-leverage tools (stateful KV-cache, MIDI-CI, canonicalization, APSC, StructXLIP, PID Flow) |
| `BUILD.md` | C++ / CMake build instructions and prerequisites |
| `AGENTS.md` (this file): Native safety, FFI ownership, and verification map | FFI frees, duplicate JUCE, RT paths, contract drift, commands |
| `docs/NATIVE_SAFETY_AND_FFI.md` | Human-readable mirror of the map above; keep in sync with this section |
| `docs/SWARM_MATRIX.md` | Hermes Tmux Matrix operator guide: `init_matrix.sh` + `kmidi_swarm.py`, six-stack parallel execution, dry-run, troubleshooting |
