# KmiDi Brain — Deterministic Boot Path

Single entry point: **`run_brain.py`** at repo root. A runnable system is more valuable than an advanced but fragile one.

**Kelly brain:** The Brain started and checked here is the Kelly brain (intent → MIDI, orchestrator, music_brain, penta_core). Code and UI use "Kelly" (e.g. KellyBrain, kelly_brain_*); repo name is KmiDi MIDI Companion.

**Roadmap:** See [docs/PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for current status and next steps.

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

**Check list (spine order):** `penta_core/ml`, `music_brain/session`, `music_brain/session/intent_processor`, `music_brain/tier1`, `mcp_workstation`, `kmidi_gui` — OK or MISSING. See `docs/CONTRACTS.md` for contract owners.

If **music_brain** or **tier1** is missing: restore from sburdges-eng/KmiDi forensic or rebuild. Not present on all online branches. **Recovery rule:** If there is no recoverable code path for a module, see `docs/CONTRACTS.md` §10 and `.cursor/rules/recovery-code-path.mdc`.

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

## Orchestrator phase order

Per `docs/CONTRACTS.md`: LLM intent → from_flat or use forensic intent → MIDI pipeline → optional image → optional audio. Error handling: phase failure sets result dict to `status: "error"` or `"failed"` with `details`; workflow still returns the intent.

## Stub vs failed vs completed

- **stubbed** — Capability not loaded (e.g. model missing); placeholder result; orchestrator continues.
- **failed** — Capability ran but returned error.
- **completed** — Capability ran and produced output.
- **error** — Exception or unrecoverable failure.

Orchestrator never blocks on stub; it records status and continues.

## Optional model paths (image / audio)

`run_brain.py check` runs without any of these. Set only when using image or audio generation.

| Env | Purpose | Default / note |
|-----|---------|----------------|
| `KMI_DI_IMAGE_MODEL_PATH` or `STABLE_DIFFUSION_MODEL_PATH` | Image pipeline (diffusers hub id or path) | `runwayml/stable-diffusion-v1-5`; stub if diffusers/torch not installed. |
| `KMI_DI_AUDIO_MODEL_ID` or `AUDIOCRAFT_MODEL_ID` | Audio model (MusicGen hub id) | `musicgen-small`; stub if audiocraft not installed. |

Check mode does not load models; orchestrator preloads image/audio engines and continues with stubbed result if load fails.

## JEPA coexist mode (optional)

When JEPA models are available, CapabilityRouter in `KmiDi_CANON/brain/ml/` can run current vs JEPA backends.

| Env | Purpose | Default |
|-----|---------|---------|
| `KMI_DI_MODEL_MODE` | `current` \| `shadow` \| `jepa` | `current` |
| `KMI_DI_SHADOW_LOG_DIR` | Shadow log directory for side-by-side metrics | `logs/shadow` |
| `KMI_DI_SHADOW_TIMEOUT_MS` | Timeout (ms) for shadow JEPA inference | `200` |

- **current** — Use existing models only.
- **shadow** — Run JEPA in parallel, log comparison, return current output; JEPA diagnostics under `debug` only.
- **jepa** — Use JEPA output when registered.

Per DATA_AND_TRAINING, for production logs use `~/Models/logs/shadow` via `KMI_DI_SHADOW_LOG_DIR`.

## Brain HTTP API (port 8000)

The body (useMusicBrain.ts) talks to `http://127.0.0.1:8000`. **Use uvicorn** (canonical ASGI server for FastAPI). From repo root:

```bash
PYTHONPATH=KmiDi_CANON/brain uvicorn KmiDi_CANON.brain.api_server:app --host 127.0.0.1 --port 8000
```

Optional: `--reload` for dev auto-reload. Alternative: `python3 -m KmiDi_CANON.brain.api_server` (same PYTHONPATH; that module runs uvicorn internally).

Endpoints: `/emotions`, `/generate`, `/interrogate`, `/config/humanizer`, `POST /spectocloud/render`, `/lyrics`, `/health`. Spectocloud backend: `music_brain/visualization/spectocloud.py` (CONTRACTS §5b).

## Optional imports (music_brain)

Many music_brain modules use optional imports with `*_AVAILABLE` flags for graceful degradation:

- **realtime/events.py** — Falls back to minimal NoteEvent stub when Logic/comprehensive_engine unavailable.
- **vocal/parrot.py** — `LIBROSA_AVAILABLE`: librosa/soundfile for voice learning; degrades to numpy-only if missing.
- **agents/crewai_music_agents.py** — `CREWAI_AVAILABLE`: CrewAI framework; agents unavailable if missing.
- **agents/ableton_bridge.py, unified_hub.py, daiw_mcp_server.py** — Optional DAW/MCP integrations.
- **audio/** — librosa, essentia optional; theory_analyzer, framework_integrations degrade gracefully.
- **voice/** — torch, transformers, pyaudio optional; neural_voice, auto_tune, synthesizer degrade.

Check imports pass silently; boot check and tests don't require all optional deps. For full functionality, install: `librosa soundfile torch transformers pyaudio essentia crewai`.

## CI / automation

Consider adding `run_brain.py check` to CI or a pre-push hook to catch missing modules before they break remote runs. **CI:** `.github/workflows/ci.yml` runs brain check and stub-creep on push/PR. **Pre-push:** `cp scripts/pre-push-hook.sh .git/hooks/pre-push && chmod +x .git/hooks/pre-push`. Run brain tests: from repo root, `python -m pytest tests/ -v` (requires `tests/conftest.py` path setup). For Phase 4 stub-creep: `python scripts/check_stub_creep.py --allow-docs`.
