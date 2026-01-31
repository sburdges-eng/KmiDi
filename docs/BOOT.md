# KmiDi Brain — Deterministic Boot Path

Single entry point: **`run_brain.py`** at repo root. A runnable system is more valuable than an advanced but fragile one.

## Modes

| Mode           | Command                    | Purpose |
|----------------|----------------------------|--------|
| **penta**      | `python run_brain.py penta` | Load penta_core ML (inference); default mode |
| **orchestrator** | `python run_brain.py orchestrator` | Run mcp_workstation Orchestrator (requires music_brain) |
| **gui**        | `python run_brain.py gui`   | Stub; GUI via Tauri: `cargo tauri dev` |
| **check**      | `python run_brain.py check` | Report module availability (no run) |

## Dependency order

1. **penta_core** — ML inference; lives under `KmiDi_CANON/brain/penta_core/ml/` (e.g. `inference.py`).
2. **mcp_workstation** — Orchestrator; depends on `music_brain.session` / `music_brain.tier1`.
3. **kmidi_gui** — Control surface; typically run via Tauri, not this script.

## Pre-flight check

Before relying on orchestrator or full stack, run:

```bash
python run_brain.py check
```

Reports: `penta_core/ml`, `mcp_workstation`, `kmidi_gui`, `music_brain/session` — OK or MISSING.

If **music_brain** is missing: restore from sburdges-eng/KmiDi forensic or rebuild. Not present on all online branches.

## Boot sequence (recommended)

1. From repo root: `python run_brain.py check`
2. If penta only: `python run_brain.py penta`
3. If full orchestrator: ensure music_brain restored, then `python run_brain.py orchestrator`
4. For GUI: use Tauri from `tauri-app/`: `cargo tauri dev`

## Reliable recursive run (restart on failure)

For long-running modes (e.g. orchestrator), run in a loop so the Brain restarts on crash:

```bash
python run_brain.py orchestrator --loop
```

Options:

- `--loop` — restart after non-zero exit (use with tmux).
- `--delay SEC` — seconds to wait before restart (default: 2).
- `--max-restarts N` — stop after N restarts (0 = infinite, default: 0).

Example in tmux:

```bash
tmux new -s kmidi
micromamba activate kmidi
python run_brain.py orchestrator --loop
# Detach: Ctrl+B then D
```

## CI / automation

Consider adding `run_brain.py check` to CI or a pre-push hook to catch missing modules before they break remote runs.
