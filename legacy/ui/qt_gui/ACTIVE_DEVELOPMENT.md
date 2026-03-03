# ⚠️ DEPRECATED - Legacy Qt GUI Application

**Status:** ⚠️ DEPRECATED (per ADR 001)
**Directory:** `legacy/ui/qt_gui/`
**Previous Location:** `src/gui/`

## Purpose

Legacy Qt6 desktop GUI application source files for the standalone Kelly application.
**Deprecated in favor of Tauri/React shell (`src-tauri/`).**

## Files

- `main.cpp` - Application entry point
- `main_window.cpp/h` - Main application window

## Build Configuration (Disabled by Default)

- CMake target: `KellyApp`
- Required by: `BUILD_DESKTOP=ON` (disabled by default per ADR 001)
- Enable with: `cmake -DKMIDI_BUILD_QT_UI=ON -DBUILD_DESKTOP=ON ..`

## Migration History

- Originally at `KmiDi_FINAL/engine/src/gui/`
- Migrated to `src/gui/` on 2026-01-21
- Moved to `legacy/ui/qt_gui/` on 2026-03-03 per ADR 001

## Primary V1 UI

The supported V1 desktop shell is **Tauri + React** at `src-tauri/`.
See `docs/adr/001-one-ui-path.md` for details.
