# Discovery: run_brain.py — xcode vs workspace diff

**Date:** 2026-03-10

## Summary

- **Workspace** `run_brain.py` is a refactor for the monorepo: it uses `music_brain.*` packages, no `KmiDi_CANON` or `sys.path` hacks, and `gui` mode starts the Music Brain FastAPI server (uvicorn). **xcode** version assumes `KmiDi_CANON/brain` layout and has extra options for long-running restarts.

## Differences

| Aspect | xcode (kmidi-companion-dev) | Workspace (Dev/KmiDi) |
|--------|------------------------------|------------------------|
| **Layout** | Adds `ROOT`, `BRAIN = ROOT / "KmiDi_CANON" / "brain"` to `sys.path` | No path changes; relies on `music_brain` from repo |
| **Check** | Simple module check | `_REQUIRED_MODULES` list; importlib; clearer pass/fail output |
| **Loop** | `--loop`, `--delay`, `--max-restarts` for restart-on-exit (tmux-friendly) | Added (parity with xcode) |
| **Orchestrator** | Imports `KmiDi_CANON.brain.mcp_workstation.orchestrator`; passes `extra_argv` | Imports `music_brain.orchestrator`; has KeyboardInterrupt handling |
| **GUI** | Stub: "kmidi_gui runs via Tauri app. Use: cargo tauri dev" | Starts `uvicorn music_brain.api:app --reload --port 8000 --host 127.0.0.1` |

## What was not dropped

Workspace did not drop logic; it replaced xcode’s KmiDi_CANON-based entry with the monorepo’s `music_brain` API and orchestrator. The only behavioral “loss” is the **--loop / --delay / --max-restarts** option set. If you want restarts for long-running Brain in tmux, re-add a small loop in `main()` (or a wrapper script) that reruns the chosen mode with a delay on non-zero exit.

## Suggested follow-up

- ~~Optional: add `--loop`, `--delay`, `--max-restarts` to workspace `run_brain.py` for parity with xcode when running under tmux.~~ **Done:** workspace `run_brain.py` now has `--loop`, `--delay`, `--max-restarts` (ignored for `check` mode).
