# KmiDi Boot Surfaces

Status: current operational boot reference
Last updated: 2026-06-08

Purpose:
- document how to start the currently useful runtime surfaces in this repo
- distinguish the primary combined dev boot path from legacy or diagnostic helpers
- avoid re-deriving which commands are real versus historical

For architecture authority, use `docs/ARCHITECTURE.md` and companion docs. This file is about operational bring-up only.

## 1. The primary development boot path

For current day-to-day work, the main supported combined startup is:

```bash
npm run dev:all
```

What it starts:
- Vite React frontend on `http://localhost:1420`
- Music Brain FastAPI service on `http://localhost:8000`

This is the most reliable “boot the active stack” command in the current repo because it matches actual `package.json` scripts.

## 2. Individual boot surfaces

### React frontend

```bash
npm run dev
```

Aliases currently defined:
- `npm run dev`
- `npm run dev:react`
- `npm run dev:ui`

Expected URL:
- `http://localhost:1420`

### Music Brain API

```bash
npm run dev:python
```

Equivalent direct command:

```bash
python3 -m uvicorn music_brain.api:app --reload --port 8000
```

Expected URLs:
- API root: `http://localhost:8000`
- docs: `http://localhost:8000/docs`

## 3. `run_brain.py`: what it is now

`run_brain.py` still exists as a deterministic Python-side helper and diagnostic entrypoint.
It supports four modes:

```bash
python run_brain.py check
python run_brain.py penta
python run_brain.py orchestrator
python run_brain.py gui
```

Current behavior by mode:
- `check` — import-checks required Python modules and exits
- `penta` — starts `music_brain.penta_core` server logic
- `orchestrator` — starts the Python AI orchestrator loop
- `gui` — starts the FastAPI service on `127.0.0.1:8000`

Important limits:
- despite the historical name, `gui` does not launch a supported desktop shell from current `package.json`
- use `run_brain.py` as a helper for Python service bring-up, import validation, or older operational flows
- do not treat it as the canonical product boot path

## 4. Recommended boot order by task

### Frontend/API feature work
1. optional: `source scripts/load-env.sh`
2. `npm run dev:all`

### API-only debugging
1. optional: `source scripts/load-env.sh`
2. `npm run dev:python`

### Python module sanity check before a longer run
1. optional: `source scripts/load-env.sh`
2. `python run_brain.py check`

### Penta/orchestrator investigation
1. ensure Python deps and any needed API keys are available
2. optional: `source scripts/load-env.sh`
3. run `python run_brain.py penta` or `python run_brain.py orchestrator`

### Native/plugin work
This file does not define a single one-command native boot path.
For native bring-up, use:
- `docs/DEVELOPMENT.md`
- `BUILD.md`
- `docs/FULL_STACK_BUILD.md`
- root CMake commands/presets

## 5. Pre-flight checks

Before booting longer-lived services, the fastest checks are:

### Frontend/backend stack

```bash
npx tsc --noEmit
python3 -m pytest tests/unit/test_api_schema.py
```

### Python import sanity

```bash
python run_brain.py check
```

### Environment sanity

```bash
./scripts/validate-env.sh
```

## 6. Expected dependencies and failure modes

### `npm run dev:all` fails
Check:
- `npm install` has been run
- Python dependencies were installed with `python3 -m pip install -e .`
- port `1420` or `8000` is not already occupied

### `npm run dev:python` fails
Check:
- `uvicorn` and `fastapi` are installed
- `music_brain` imports succeed
- your active Python is the one where `pip install -e .` ran

### `run_brain.py check` fails
Its current required imports include:
- `music_brain`
- `music_brain.penta_core`
- `music_brain.orchestrator`
- `music_brain.session`
- `music_brain.emotion`
- `music_brain.kelly_companion`

A failure here means the Python environment is not bootable enough for that flow.

## 7. Loop/restart mode in `run_brain.py`

`run_brain.py` also supports automatic restart for non-zero exits:

```bash
python run_brain.py gui --loop --delay 5 --max-restarts 10
```

Use this only for operational experimentation or tmux-driven workflows.
It is not the main developer boot path for everyday frontend/API work.

## 8. Known drift and interpretation rules

These facts are deliberate because they affect bring-up decisions:
- The repo still contains Tauri-era wording in some places.
- `package.json` does not currently define `npm run dev:tauri`.
- The active combined boot path you can rely on is `npm run dev:all`.

When docs disagree, prefer:
1. actual scripts in `package.json`
2. actual behavior in `run_brain.py`
3. current architecture authority docs
4. older narrative docs last

## 9. Related docs

- `docs/DEVELOPMENT.md`
- `docs/ENVIRONMENT.md`
- `BUILD.md`
- `docs/FULL_STACK_BUILD.md`
- `AGENTS.md`
