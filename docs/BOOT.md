# Brain Boot Sequence

> **Governance: BOOT LAW** – The project must maintain a deterministic boot path.
> A runnable system is more valuable than an advanced but fragile one.
> Ref: `.cursor/rules/engineering-governance.mdc`, `TODO.md`.

---

## Overview

All Brain services are started through the single entry point **`run_brain.py`** at the repository root.  
Four modes are supported:

| Mode | Purpose |
|---|---|
| `check` | Pre-flight import check – no server started. Safe for CI / pre-push. |
| `penta` | Start the **penta_core ML server** (`music_brain.penta_core`). |
| `orchestrator` | Start the **AI orchestrator** (`music_brain.orchestrator`). |
| `gui` | Launch the **Music Brain FastAPI** on port 8000 (Tauri/React desktop shell optional). |

---

## Quick Start

```bash
# 1. Verify the environment is bootable (no side-effects, zero servers started):
python run_brain.py check

# 2. Start the ML layer:
python run_brain.py penta

# 3. Start the AI orchestrator:
python run_brain.py orchestrator

# 4. Launch the full GUI (Music Brain API on port 8000):
python run_brain.py gui
```

---

## Dependency Order

The Brain components must be initialised in the following order to avoid
import races and missing-model errors:

```
1. music_brain              – core package (sessions, intent schema, emotion mapping)
   └─ music_brain.penta_core  – C++/Python ML bridge (penta mode)
   └─ music_brain.orchestrator – AI pipeline coordinator (orchestrator mode)
   └─ music_brain.session      – intent / rule schema
   └─ music_brain.emotion      – production + emotion mapping
   └─ music_brain.groove_kmidi – humanization / groove engine

2. mcp_workstation          – multi-AI orchestration (optional, python/mcp/)
   └─ Requires music_brain to be importable first.

3. kmidi_gui (Tauri + React)
   └─ Communicates with Music Brain API at http://127.0.0.1:8000.
   └─ Start the API (gui mode) before launching the desktop shell.
```

---

## `run_brain.py check` – Pre-flight Import Verification

Running `python run_brain.py check` imports each required module in order and
prints a pass/fail result **without starting any server**.  
Use this as a fast, side-effect-free gate before any boot step.

Example output on a healthy system:

```
Brain pre-flight check
========================================
  ok  music_brain core package (music_brain)
  ok  penta_core ML layer (music_brain.penta_core)
  ok  AI orchestrator (music_brain.orchestrator)
  ok  session / intent schema (music_brain.session)
  ok  emotion mapping (music_brain.emotion)
  ok  Kelly Companion (music_brain.kelly_companion)
========================================
All modules OK – brain is bootable.
```

Exit code is **0** on success, **1** if any module fails to import.

---

## CI Integration

`run_brain.py check` is included in the **CI Preflight Gate**
(`.github/workflows/ci-preflight.yml`) as the **Brain Boot Check** job.
It runs on every PR and push to `main` / `Kelly-Master`, providing an early
warning if a dependency is missing or broken.

To add the same check as a local pre-push hook:

```bash
# .git/hooks/pre-push  (create or append)
#!/bin/sh
python run_brain.py check || { echo "Brain pre-flight failed. Aborting push."; exit 1; }
```

---

## Module Reference

### `music_brain.penta_core` (penta mode)

- Python package at `music_brain/penta_core/`.
- Contains the `server.py` FastMCP entry point (`mcp.run()`).
- Requires: `mcp` (or `fastmcp`) and API keys set in `.env`
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`).

### `music_brain.orchestrator` (orchestrator mode)

- Python package at `music_brain/orchestrator/`.
- `AIOrchestrator` is the async context manager / pipeline runner.
- Depends on: `music_brain.orchestrator.{pipeline,interfaces,logging_utils}`.

### Music Brain FastAPI (gui mode)

- Entry point: `music_brain.api:app` (FastAPI application).
- Default port: **8000**. Docs at `http://127.0.0.1:8000/docs`.
- Tauri/React shell communicates with this API; start the API first.
- Env var `UVICORN_PORT` overrides the default port if needed.

### `mcp_workstation` (optional)

- Located at `python/mcp/mcp_workstation/`.
- Part of the MCP servers collection at `python/mcp/` (daiw_mcp, mcp_penta_swarm, mcp_todo, mcp_workstation).
- Start with: `python -m mcp_workstation status` (from `python/mcp/`).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check` reports `FAIL music_brain` | `pip install -e .` from repo root. |
| `check` reports `FAIL music_brain.penta_core` | Ensure `music_brain/penta_core/__init__.py` exists. |
| `gui` fails with "uvicorn not found" | `pip install uvicorn fastapi`. |
| Tauri shell shows blank / cannot connect | Verify Music Brain API is running (`curl http://127.0.0.1:8000/emotions`). |
| Orchestrator exits immediately | Missing optional deps; check `pip install -e ".[dev]"`. |
