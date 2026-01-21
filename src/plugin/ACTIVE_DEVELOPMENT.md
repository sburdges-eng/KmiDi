# 🟡 ACTIVE DEVELOPMENT - Plugin Source

**Status:** 🟡 ACTIVE
**Directory:** `src/plugin/`
**Color Code:** Yellow/Gold (🟡)

## Purpose

VST3 and CLAP audio plugin source files. These files are required by CMakeLists.txt for building plugins.

## Active Files

- `PluginProcessor.cpp/h` - Main plugin audio processor
- `PluginEditor.cpp/h` - Plugin UI editor
- `PluginState.cpp/h` - Plugin state management

## Build Configuration

- CMake target: `KellyPlugin`
- Formats: VST3, CLAP
- Categories: Fx, Synth
- Required by: `BUILD_PLUGINS=ON`

## Migration

Migrated from `KmiDi_FINAL/engine/src/plugin/` on 2026-01-21.
