# Source Directory Index

**Status:** ACTIVE DEVELOPMENT
**Last Updated:** 2026-03-03

## Directory Structure

```
src/
├── core/              [ACTIVE] Core library and Intent IR system
│   └── intent_ir/    Intent IR framework
├── plugin/           [ACTIVE] VST3/CLAP audio plugin source files
├── ui/               [ACTIVE] JUCE UI components (audio/DSP controls, not standalone UI)
├── bridge/           [ACTIVE] FFI bridge for language bindings
└── stubs/            [ACTIVE] Stub implementations for testing

legacy/ui/            [DEPRECATED] Legacy UI surfaces (per ADR 001)
├── qt_gui/           Legacy Qt6 desktop GUI (was src/gui/)
└── appkit_shell/     Legacy macOS AppKit shell
```

## Primary UI Surface (V1)

The only supported V1 desktop shell is **Tauri + React** at `src-tauri/`.
The Qt GUI has been moved to `legacy/ui/qt_gui/` per ADR 001.

## Plugin Files (`src/plugin/`)

### Core Plugin Files
- `PluginProcessor.cpp/h` - Main plugin audio processor
- `PluginEditor.cpp/h` - Plugin UI editor component
- `PluginState.cpp/h` - Plugin state management

### Purpose
These files implement the VST3/CLAP audio plugin interface. The plugin processes audio and provides a GUI for user interaction.

## JUCE UI Components (`src/ui/`)

- JUCE/C++ components for audio visualization and DSP controls
- Not a standalone desktop UI - restricted to audio/MIDI rendering support

### Purpose
JUCE UI components that can be embedded in plugins or shells. Per ADR 001,
JUCE is restricted to audio/MIDI rendering and DSP support with no v1 standalone JUCE UI commitment.

## Bridge Files (`src/bridge/`)

- `kelly_ffi.cpp/h` - Foreign Function Interface for language bindings

### Purpose
FFI bridge allowing other languages (Python, Rust, etc.) to interact with the Kelly core library.

## Core Library (`src/core/`)

- `intent_ir/` - Intent Intermediate Representation system
  - `IntentFrame.cpp/h` - Core intent frame structure
  - `EngineContract.h` - Engine interface contract
  - `IntentFrameAdapter.h` - Frame adapter utilities
  - `Assertions.h` - Assertion macros

## Quick Reference

### For V1 Desktop UI Development
- Work with Tauri/React at `src-tauri/` and React frontend
- See docs/DEVELOPMENT.md for setup

### For Plugin Development
- Edit files in `src/plugin/`
- Build with `BUILD_PLUGINS=ON` in CMake

### For Legacy Qt GUI (Deprecated)
- Files now at `legacy/ui/qt_gui/`
- Build with `BUILD_DESKTOP=ON` in CMake (disabled by default)

### For FFI/Bindings
- Edit files in `src/bridge/`
- Header `kelly_ffi.h` is the public interface

## Migration Notes

All files in this directory were migrated from `KmiDi_FINAL/engine/src/` on 2026-01-21.
Qt GUI moved to `legacy/ui/qt_gui/` on 2026-03-03 per ADR 001.
See `PROJECT_SOURCE_MANIFEST.md` for complete migration details.
