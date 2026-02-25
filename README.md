# KmiDi Canonical Repository

`KmiDi/` is the active canonical workspace for the KmiDi product line.

## Canonical Surfaces

- `KmiDi_FINAL/` — Canonical shipped UI surfaces.
  - `engine/` and `apps/` contain React/Tauri and AppKit entry points.
- `src/ui/` — JUCE/C++ implementation.
- `src-tauri/` — Rust Tauri host bindings.
- `src_penta-core/`, `python/`, `training/` — Supporting model and toolchain layers.
- `build/`, `tools/`, `scripts/` — Build and maintenance utilities.

## Core Canonical UI Entry Checks

- Confirm these before major edits:
  - `KmiDi_FINAL/engine/src/components/EmotionWheel.tsx`
  - `src/ui/EmotionWheel.cpp`
  - `KmiDi_FINAL/apps/macOS/AppKitShell/Sources/KmiDiApp/AppDelegate.swift`

## Canonical Rust Layer

The canonical Rust layer is maintained outside this repository at:

- `~/Dev/swif:xcode/KmiDi/KmiDi_CANON/`

That path includes:
- `body/` — core runtime and engine code
- `brain/` — orchestration and model-facing logic
- `training/` — model training and assertions
- `ui/` — UI bindings for Rust-facing surfaces

## Operational Notes

- Preserve the separation between UI systems and avoid editing duplicate snapshots.
- If you need to add or adjust a workflow, prefer updating this repo and matching canonical references in `swif:xcode/KmiDi/KmiDi_CANON/` where applicable.
- Keep this repository as the workspace source of truth for active UI work.
