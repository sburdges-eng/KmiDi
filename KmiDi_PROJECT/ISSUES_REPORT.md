# Issues Report

## Scope
- Focused on runtime-critical sources in `KmiDi_PROJECT/source/python/mcp_workstation`, `music_brain`, and `KmiDi_PROJECT/source/frontend/src-tauri`.
- External/third-party code (for example `KmiDi_PROJECT/external`) and large training/output datasets were not deeply reviewed.

## Findings

### Blockers
1) Missing module import prevents the orchestrator from starting.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py:13` imports `music_brain.tier1.midi_pipeline_wrapper`, but `music_brain/tier1/` does not exist in the repo.
- Impact: `python -m mcp_workstation` and any orchestration flow will fail at import time.

2) `get_workstation()` is incompatible with the current `Orchestrator` signature.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py:18-35` requires `llm_model_path`.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py:227-229` exposes `get_workstation()` with no required args.
- `KmiDi_PROJECT/source/python/mcp_workstation/cli.py:166` and `KmiDi_PROJECT/source/python/mcp_workstation/server.py:365` call `get_workstation()` with no arguments.
- Impact: CLI/server paths will raise `TypeError` before doing any work.

### High
3) CLI/server call a proposal/task API that does not exist on `Orchestrator`.
- `KmiDi_PROJECT/source/python/mcp_workstation/cli.py:168-258` calls methods like `get_status`, `submit_proposal`, `get_phase_progress`.
- `KmiDi_PROJECT/source/python/mcp_workstation/server.py:369-458` calls the same proposal/task API surface.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py` only defines `execute_workflow` and lock helpers.
- Impact: even if the constructor mismatch is fixed, these calls will raise `AttributeError` at runtime.

4) Image/audio generation paths are effectively stubbed in normal flows.
- `KmiDi_PROJECT/source/python/mcp_workstation/llm_reasoning_engine.py:70-73` constructs `ImageGenerationEngine`/`AudioGenerationEngine` but never loads models.
- `KmiDi_PROJECT/source/python/mcp_workstation/image_generation_engine.py:82-96` returns a stub unless `_load_pipeline()` has been called.
- `KmiDi_PROJECT/source/python/mcp_workstation/audio_generation_engine.py:62-75` returns a stub unless `_load_model()` has been called.
- `KmiDi_PROJECT/source/python/mcp_workstation/orchestrator.py:30-35` never calls `_load_pipeline()` or `_load_model()`.
- Impact: image/audio generation will always return placeholder data unless a caller manually loads models.

### Medium
5) Audio generation is permanently disabled by a hardcoded flag.
- `KmiDi_PROJECT/source/python/mcp_workstation/audio_generation_engine.py:7-12` comments out the audiocraft imports and sets `AUDIOCRAFT_AVAILABLE = False` unconditionally.
- Impact: `AudioGenerationEngine` never transitions out of stub mode even if audiocraft is installed.

6) `music_brain` is used as a package but has no `__init__.py` at its root.
- `KmiDi_PROJECT/source/python/kmidi_gui/core/preset.py:12-13` imports `music_brain.session.intent_schema`.
- `music_brain/` lacks an `__init__.py` file, making it a namespace package and potentially breaking tooling/packaging assumptions.
- Impact: imports may fail depending on the runtime packaging or how PYTHONPATH is configured.

### Low
7) Tauri HTTP bridge has no timeouts for local API requests.
- `KmiDi_PROJECT/source/frontend/src-tauri/src/bridge/musicbrain.rs:7-75` uses `reqwest::Client::new()` and `.send().await?` without a timeout.
- Impact: UI commands can hang indefinitely if the local service is down or unresponsive.

