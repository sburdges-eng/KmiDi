# Source Directory Index

**Status:** HISTORICAL INDEX WITH ACTIVE FILE REFERENCES
**Last Updated:** 2026-06-08

Historical note
- This index preserves older source-layout and shell assumptions and is not architecture authority.
- When it conflicts with the current repo architecture, follow `docs/ARCHITECTURE.md` and `docs/REPO_MODULE_MAP.md`.

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

## Historical note on primary UI wording

This section reflects an older Tauri-centered shell description.
Current repo authority is plugin-first / engine-separable and is documented in `docs/ARCHITECTURE.md` and `docs/REPO_MODULE_MAP.md`.
Do not use this file alone to infer the canonical product shell.

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
- Work with Tauri/React at `engine/intent_ir/` and React frontend
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
