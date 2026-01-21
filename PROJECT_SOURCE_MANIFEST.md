# Project Source File Manifest

**Generated:** 2026-01-21
**Total Source Files:** 407
**Status:** Pre-Migration Inventory

## Overview

This manifest documents all source files in `KmiDi_FINAL/engine/src/` that need to be migrated to the expected `src/` directory structure as defined in `CMakeLists.txt`.

## Critical Build Files (Required by CMakeLists.txt)

### Plugin Files
- **Source:** `KmiDi_FINAL/engine/src/plugin/PluginProcessor.cpp`
  - **Target:** `src/plugin/PluginProcessor.cpp`
  - **Status:** ✅ Exists
  - **Size:** 49,347 bytes
  - **Header:** `KmiDi_FINAL/engine/src/plugin/PluginProcessor.h` → `src/plugin/PluginProcessor.h`

- **Source:** `KmiDi_FINAL/engine/src/plugin/PluginEditor.cpp`
  - **Target:** `src/plugin/PluginEditor.cpp`
  - **Status:** ✅ Exists
  - **Size:** 39,041 bytes
  - **Header:** `KmiDi_FINAL/engine/src/plugin/PluginEditor.h` → `src/plugin/PluginEditor.h`

- **Source:** `KmiDi_FINAL/engine/src/plugin/PluginState.cpp`
  - **Target:** `src/plugin/PluginState.cpp`
  - **Status:** ✅ Exists
  - **Size:** 26,521 bytes
  - **Header:** `KmiDi_FINAL/engine/src/plugin/PluginState.h` → `src/plugin/PluginState.h`

### GUI Application Files
- **Source:** `KmiDi_FINAL/engine/src/gui/main.cpp`
  - **Target:** `src/gui/main.cpp`
  - **Status:** ✅ Exists
  - **Size:** 288 bytes

- **Source:** `KmiDi_FINAL/engine/src/gui/main_window.cpp`
  - **Target:** `src/gui/main_window.cpp`
  - **Status:** ✅ Exists
  - **Size:** 1,551 bytes
  - **Header:** `KmiDi_FINAL/engine/src/gui/main_window.h` → `src/gui/main_window.h`

### FFI Bridge Files
- **Source:** `KmiDi_FINAL/engine/src/bridge/kelly_ffi.cpp`
  - **Target:** `src/bridge/kelly_ffi.cpp`
  - **Status:** ✅ Exists
  - **Size:** 27,331 bytes

- **Source:** `KmiDi_FINAL/engine/src/bridge/kelly_ffi.h`
  - **Target:** `src/bridge/kelly_ffi.h`
  - **Status:** ✅ Exists
  - **Size:** 9,524 bytes

## Migration Status

- [x] Inventory complete
- [ ] Files moved to `src/`
- [ ] Include paths verified
- [ ] Build system tested
- [ ] Documentation updated

## Notes

- Total source files discovered: 407
- All critical files for CMakeLists.txt exist
- No path dependencies found that would break migration

## Migration Status

- [x] Inventory complete
- [x] Files moved to `src/`
- [x] Include paths verified (no hardcoded paths found)
- [ ] Build system tested
- [x] Documentation updated

## Migration Completed: 2026-01-21

All critical files have been successfully migrated to their expected locations:
- ✅ Plugin files: `src/plugin/`
- ✅ GUI files: `src/gui/`
- ✅ Bridge files: `src/bridge/`

All files verified with checksums to ensure integrity.
