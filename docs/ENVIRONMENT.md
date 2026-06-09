# KmiDi Environment Configuration

Status: current environment loading and variable reference aligned to repo scripts
Last updated: 2026-06-08

This file documents the environment behavior that is actually implemented by:
- `scripts/load-env.sh`
- `scripts/validate-env.sh`
- `.env*` files at repo root
- optional feature env files under `env/`

For architecture boundaries, do not use this file as authority. Use `docs/ARCHITECTURE.md` and its companion authority docs.

## 1. What environment loading actually does

The canonical loader is:

```bash
source scripts/load-env.sh
```

Optional feature selection:

```bash
source scripts/load-env.sh tauri ml training mcp
```

If you pass no feature list, the loader defaults to:
- `tauri`
- `ml`
- `training`
- `mcp`

## 2. Load order

`scripts/load-env.sh` loads files in this order, with later files overriding earlier ones:

1. `.env`
2. `.env.production` if `NODE_ENV=production` or `KMIDI_ENV=production`
3. otherwise `.env.development` when environment mode resolves to development
4. feature files under `env/` based on selected features:
   - `env/.env.tauri`
   - `env/.env.ml`
   - `env/.env.training`
   - `env/.env.mcp`
5. `.env.local`

Operational notes:
- `.env.local` is the highest-priority override layer.
- Feature files are optional; the loader skips missing files.
- The loader parses simple `KEY=VALUE` lines and ignores blank lines and comment lines.

## 3. Environment mode selection

The loader chooses mode with:
- `NODE_ENV` first
- then `KMIDI_ENV`
- default: `development`

That means these are equivalent examples:

```bash
NODE_ENV=production source scripts/load-env.sh
KMIDI_ENV=production source scripts/load-env.sh
```

## 4. Fast validation

Validate current environment resolution with:

```bash
./scripts/validate-env.sh
```

The validator currently:
- sources `scripts/load-env.sh`
- checks required and optional variables
- checks whether configured paths exist
- flags placeholder-looking API keys

Current hard requirement enforced by the validator:
- `KELLY_MODELS_PATH` must be set

Important nuance:
- the validator requires `KELLY_MODELS_PATH` to be set
- it does not currently require that path to exist; missing paths are warnings, not hard failures

## 5. Files you should care about

Repo-root files:
- `.env.example` — committed template
- `.env` — local base config
- `.env.development` — committed dev defaults
- `.env.production` — committed production-style defaults/template
- `.env.local` — git-ignored local override layer

Feature files referenced by the loader:
- `env/.env.tauri`
- `env/.env.ml`
- `env/.env.training`
- `env/.env.mcp`

Caveat:
- this repo currently documents feature env files more heavily than it commits them. Missing feature files are acceptable because the loader treats them as optional.

## 6. Important variables in the current repo

## Core paths

### `KELLY_MODELS_PATH`
- required by the validator
- default in templates/dev config: `./models`
- used as the native/C++ model path reference

### `PYTHON_MODEL_PATH`
- Python-side model path
- defaults to `./models` in templates/dev config

### `TRAINING_DATA_PATH`
- training data location
- default in template: `./data/training`

### `CHECKPOINT_PATH`
- training checkpoint location

### `LOG_PATH`
- log directory

### `KMIDI_DATA_ROOT`, `KMIDI_DATASETS_PATH`, `KMIDI_CACHE_ROOT`
- optional external-drive/data-root style variables from `.env.example`
- useful when datasets or caches live off-repo

## Service URLs and ports

### `KMIDI_API_URL`
- default dev value: `http://127.0.0.1:8000`
- should point at the Music Brain API for integrated frontend/backend work

### `MUSIC_BRAIN_API_URL`
- optional override in templates
- template comment says consumers should prefer this first, then fall back to `KMIDI_API_URL`

### `TAURI_DEV_HOST`
- present in templates/dev config
- historical Tauri-related variable still used as part of env layering

### `TAURI_PLATFORM`
- used by the Vite config to decide whether `@tauri-apps/api/*` imports should be stubbed
- when unset, the React web build uses stubs so development can proceed without a Tauri shell

### `ML_INFERENCE_URL`
- optional ML inference service base URL

### `MCP_SERVER_PORT`
- optional MCP service port

## Frontend build-time variables

These are inlined by Vite at build time rather than read dynamically at runtime:
- `VITE_KMIDI_USE_API`
- `VITE_API_BASE`
- any other `VITE_*` variables you introduce

Rule:
- only variables prefixed with `VITE_` are exposed to frontend code via `import.meta.env`

## Feature flags and logging

Common flags in templates:
- `KMIDI_USE_API`
- `ENABLE_ML_INFERENCE`
- `ENABLE_MCP_SERVERS`

Common logging vars:
- `RUST_LOG`
- `RUST_BACKTRACE`
- `CXX_LOG_LEVEL`
- `PYTHON_LOG_LEVEL`

## Training-related vars

Current templates/validator mention:
- `LOCAL_RANK`
- `WORLD_SIZE`
- `CUDA_VISIBLE_DEVICES`
- `TRAINING_BATCH_SIZE`

## 7. Recommended local setup patterns

### Minimal frontend + API development

Usually enough:

```bash
cp .env.example .env
printf '\nKELLY_MODELS_PATH=./models\n' >> .env.local
source scripts/load-env.sh
npm run dev:all
```

If `./models` does not exist yet, the validator will warn but not fail.

### Keep secrets out of committed files

Put secrets in `.env.local`, for example:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GITHUB_TOKEN=...
```

## 8. Access patterns by layer

### Shell scripts

```bash
source scripts/load-env.sh
```

### Python

The repo does not enforce a single dotenv loader pattern inside every Python entrypoint.
If a Python script needs env file loading outside the shell wrapper, load explicitly in that script or launch it from a shell where `scripts/load-env.sh` was sourced.

### C++

Use normal process environment access such as `std::getenv(...)` where required.
Do not invent a second configuration authority for build options.

### Frontend

Read only `VITE_*` variables through `import.meta.env`.
Do not assume arbitrary shell variables are available in browser code.

## 9. Build options are not environment variables

This is one of the most important distinctions in the repo.
CMake build configuration belongs on the `cmake` command line or in CMake presets, not in `.env` files.

Examples:

```bash
cmake -S . -B build -G Ninja -DBUILD_KELLY_FFI=ON -DBUILD_PLUGINS=ON
cmake --preset ninja-debug
```

Do not treat these as env-controlled runtime settings:
- `BUILD_KELLY_CORE`
- `BUILD_KELLY_FFI`
- `BUILD_PLUGINS`
- `KMIDI_BUILD_JUCE_UI`
- `KMIDI_ENABLE_ASAN`
- `KMIDI_ENABLE_TSAN`

## 10. Common problems

### Variables seem missing
Use:

```bash
source scripts/load-env.sh
./scripts/validate-env.sh
```

Common causes:
- you executed the loader instead of sourcing it
- `.env.local` overrides a value you forgot about
- you expected a missing optional feature env file to exist

### Frontend cannot reach backend
Check:
- `npm run dev:python` is running
- `KMIDI_API_URL` or `VITE_API_BASE` points to the expected host/port
- API docs respond at `http://127.0.0.1:8000/docs`

### Tauri-specific behavior is inconsistent
Current repo state is mixed:
- environment templates still include Tauri-era variables
- Vite still knows how to stub Tauri imports
- but the canonical product handoff is no longer “Tauri is the product center”

Treat Tauri-related env as compatibility/configuration residue unless an actively supported shell restores those paths.

### Validator fails immediately
The most likely reason is:
- `KELLY_MODELS_PATH` is unset

Set it in `.env` or `.env.local`, then re-run validation.

## 11. Durable facts worth preserving

These facts should stop future re-derivation:
- `.env.local` is the highest-priority override layer.
- `scripts/load-env.sh` defaults to loading all four feature buckets: `tauri ml training mcp`.
- `TAURI_PLATFORM` affects Vite stubbing behavior.
- build flags belong to CMake, not `.env`.
- the validator enforces presence of `KELLY_MODELS_PATH`, but path existence is only a warning today.

## 12. Related docs

- `docs/DEVELOPMENT.md`
- `docs/BOOT.md`
- `docs/FULL_STACK_BUILD.md`
- `docs/DATASETS_LAYOUT.md`
- `AGENTS.md`
