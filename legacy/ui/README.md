# Legacy UI Surfaces

**Status:** Deprecated (per [ADR 001](../../docs/adr/001-one-ui-path.md))

This directory contains legacy UI surfaces that are out of the V1 build matrix.
They are preserved for reference and potential future use, but are not part of
the supported development path.

## Contents

### `appkit_shell/`
Native macOS AppKit shell. Originally at `KmiDi_FINAL/apps/macOS/AppKitShell/`.

### `qt_gui/`
Legacy Qt6 UI surface. Originally at `src/gui/`.

## Primary UI Surface (V1)

The only supported V1 desktop shell is **Tauri + React**, located at:
- `engine/intent_ir/` - Rust Tauri host bindings
- React frontend (web layer)

## JUCE Audio/MIDI Rendering

JUCE is restricted to audio/MIDI rendering and DSP support (not a standalone UI).
Located at:
- `plugin/` - JUCE plugin implementation
- `src/ui/` - JUCE components for audio visualization/controls

## Migration Notes

If you need functionality from these legacy surfaces:
1. Port the specific feature to the Tauri/React shell
2. Follow the UI boundary rules in `docs/UI_BOUNDARY_RULES.md`
3. Ensure API/schema hardening at the UI-to-engine boundary

## Re-enabling Legacy UI Builds (Not Recommended)

Legacy UI surfaces can be built by enabling CMake options:
```bash
cmake -DKMIDI_BUILD_QT_UI=ON -DBUILD_NATIVE_MACOS_APP=ON ..
```

These options are disabled by default for V1 stabilization.
