# KmiDi Canonical Repository

`KmiDi/` is the active canonical workspace for the KmiDi product line.

## Canonical UI Surface (V1)

The only supported V1 desktop shell is **Tauri + React**:
- `src-tauri/` — Rust Tauri host bindings
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
- **Full v1 build:** `./scripts/build_v1.sh` (entities → C++/Python → Tauri).
- **React/Tauri/C++ integration:** See [docs/FULL_STACK_BUILD.md](docs/FULL_STACK_BUILD.md) for KellyFFI linkage, plugin build verification, and DAW/automation validation procedures.

## Operational Notes

- Preserve the separation between UI systems and avoid editing duplicate snapshots.
- If you need to add or adjust a workflow, prefer updating this repo and matching canonical references in `swif:xcode/KmiDi/KmiDi_CANON/` where applicable.
- Keep this repository as the workspace source of truth for active UI work.
