# Xcode Import Playbook

Historical note
- This playbook contains Tauri-era framing for some native targets.
- It remains useful for Xcode project generation, but it is not architecture authority.
- When target descriptions conflict with the current repo architecture, follow `docs/ARCHITECTURE.md`, `docs/REPO_MODULE_MAP.md`, and `AGENTS.md`.

Date: 2026-02-21  
Purpose: Generate an `.xcodeproj` for KmiDi with KmiDi_FINAL integration so it opens cleanly in Xcode.

## Prerequisites
- Xcode 15.1+ (with Command Line Tools)  
- CMake 3.27+ (`brew install cmake`)  
- Qt 6 (`brew install qt`) — CMake presets default to `/opt/homebrew/opt/qt/lib/cmake` (Intel: `/usr/local/opt/qt/lib/cmake`)  
- JUCE submodule initialized at `external/JUCE`

Initialize JUCE after clone:
```bash
cd /Users/seanburdges/Dev/KmiDi
git submodule sync --recursive
git submodule update --init --recursive external/JUCE
```

## One-liner (recommended)
```bash
cd /Users/seanburdges/Dev/KmiDi
./scripts/configure_xcode.sh xcode-debug   # or xcode-release
```
This applies JUCE patches, runs the preset, generates `build/xcode-*/Kelly.xcodeproj`,
and opens it in Xcode.

## Manual commands (if you prefer)
```bash
cd /Users/seanburdges/Dev/KmiDi
cmake --preset xcode-debug   # generator = Xcode, arm64, KmiDi_FINAL + native macOS app on
open build/xcode-debug/Kelly.xcodeproj
```

## What the presets do
- Generator: `Xcode`
- Architecture: `arm64`
- Deployment target: `11.0`
- Enables KmiDi_FINAL: `USE_KMI_DI_FINAL=ON`
- Enables native macOS app target: `BUILD_NATIVE_MACOS_APP=ON`
- Points KMI_DI_FINAL_ROOT at the in-repo `KmiDi_FINAL`
- Sets a Qt search path (`CMAKE_PREFIX_PATH`) for Homebrew installs

## Common issues
- **Qt not found**: set `Qt6_DIR` manually  
  ```bash
  cmake --preset xcode-debug -DQt6_DIR=/opt/homebrew/opt/qt/lib/cmake/Qt6
  ```
- **Missing Command Line Tools**: run `xcode-select --install`.
- **Universal build needed**: reconfigure with  
  ```bash
  cmake --preset xcode-debug -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
  ```

## Key Xcode targets/schemes
- `KellyCore` (static library)
- `KellyApp` (desktop Qt app, optional)
- `KellyPlugin` (VST3/CLAP)
- `KellyFFI` (shared lib for Tauri/bridge)
- `native_macos_app` (custom build step that runs `apps/macOS/build_macos_app.sh`)
- `dsp_core_test` (if KmiDi_FINAL DSP is present)

## Clean/reconfigure
```bash
rm -rf build/xcode-debug build/xcode-release
cmake --preset xcode-debug
```

## JUCE patch pipeline
- JUCE patches live under `third_party/patches/juce/*.patch`.
- `scripts/configure_xcode.sh` and `scripts/generate_xcode_project.sh` call
  `scripts/juce/apply_patches.sh` before CMake configure.
- The patch script is idempotent:
  - if patch is already applied, it no-ops;
  - if patch cannot apply cleanly, it fails with a drift error.

Manual patch apply:
```bash
cd /Users/seanburdges/Dev/KmiDi
./scripts/juce/apply_patches.sh
```

## Clean-state checks
```bash
cd /Users/seanburdges/Dev/KmiDi
git status --short
git submodule status
git ls-files -s | rg "KmiDi_PROJECT/external/JUCE" || true
```

Expected: no legacy `KmiDi_PROJECT/external/JUCE` gitlink and successful submodule status.

You can now open `Kelly.xcodeproj` and build/debug inside Xcode without extra wiring.
