# KmiDi Canonical Repository

`KmiDi/` is the active canonical workspace for the KmiDi product line.

## Canonical UI Surface (V1)

The only supported V1 desktop shell is **Tauri + React**:
- `engine/intent_ir/` — Rust intent crate (FFI bridge, types, validator)
- React frontend (web layer)

## Supporting Layers

- `src/ui/` — JUCE/C++ audio visualization and controls (not standalone UI)
- `plugin/` — JUCE audio/MIDI plugin implementation
- `KmiDi_FINAL/engine/` — Engine components
- `src_penta-core/`, `python/`, `training/` — Model and toolchain layers
- `music_brain/` — Python intent pipeline
- `build/`, `tools/`, `scripts/` — Build and maintenance utilities

## Legacy UI Surfaces (Deprecated)

Per [ADR 001](docs/adr/001-one-ui-path.md), legacy UI surfaces have been moved to `legacy/ui/`:
- `legacy/ui/appkit_shell/` — Native macOS AppKit shell
- `legacy/ui/qt_gui/` — Qt6 UI surface

See `legacy/ui/README.md` for details on deprecated surfaces.

## Canonical Rust Layer

The canonical Rust layer is maintained outside this repository at:

- `~/Dev/swif:xcode/KmiDi/KmiDi_CANON/`

That path includes:

- `body/` — core runtime and engine code
- `brain/` — orchestration and model-facing logic
- `training/` — model training and assertions
- `ui/` — UI bindings for Rust-facing surfaces (external, not in V1 build)

## V1 build and dev

- **Dev setup:** `./scripts/dev-setup.sh` then `npm run dev:all` (see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)).

Two V1 build paths (use the one that matches your goal):

- **V1 pipeline A — penta_core + PyInstaller + Tauri:** `./scripts/build_v1.sh`. Builds: sync entities → C++ penta_core / Python bindings → PyInstaller-packaged Music Brain API → Tauri app. No KellyFFI.
- **V1 pipeline B — KellyFFI + Tauri (native desktop integration):** See [docs/FULL_STACK_BUILD.md](docs/FULL_STACK_BUILD.md) and `./scripts/build-full-stack.sh`. Builds KellyFFI shared lib (and optional KellyPlugin_VST3) for React → Tauri → KellyFFI → KellyCore. Use this path for plugin build verification and DAW/automation validation.

## Operational Notes

- Preserve the separation between UI systems and avoid editing duplicate snapshots.
- If you need to add or adjust a workflow, prefer updating this repo and matching canonical references in `swif:xcode/KmiDi/KmiDi_CANON/` where applicable.
- Keep this repository as the workspace source of truth for active UI work.
