# Source Directory Index

**Status:** ACTIVE DEVELOPMENT
**Last Updated:** 2026-01-21

## Directory Structure

```
src/
├── core/              [ACTIVE] Core library and Intent IR system
│   └── intent_ir/    Intent IR framework
├── plugin/           [ACTIVE] VST3/CLAP audio plugin source files
├── gui/              [ACTIVE] Desktop GUI application files
├── bridge/           [ACTIVE] FFI bridge for language bindings
└── stubs/            [ACTIVE] Stub implementations for testing
```

## Plugin Files (`src/plugin/`)

### Core Plugin Files
- `PluginProcessor.cpp/h` - Main plugin audio processor
- `PluginEditor.cpp/h` - Plugin UI editor component
- `PluginState.cpp/h` - Plugin state management

### Purpose
These files implement the VST3/CLAP audio plugin interface. The plugin processes audio and provides a GUI for user interaction.

## GUI Application Files (`src/gui/`)

- `main.cpp` - Application entry point
- `main_window.cpp/h` - Main application window

### Purpose
Desktop application files for the standalone Kelly application.

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

### For Plugin Development
- Edit files in `src/plugin/`
- Build with `BUILD_PLUGINS=ON` in CMake

### For GUI Development
- Edit files in `src/gui/`
- Build with `BUILD_DESKTOP=ON` in CMake

### For FFI/Bindings
- Edit files in `src/bridge/`
- Header `kelly_ffi.h` is the public interface

## Migration Notes

All files in this directory were migrated from `KmiDi_FINAL/engine/src/` on 2026-01-21.
See `PROJECT_SOURCE_MANIFEST.md` for complete migration details.
