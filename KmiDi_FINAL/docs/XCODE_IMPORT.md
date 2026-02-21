# Xcode Import Playbook (KmiDi_FINAL)

Date: 2026-02-21  
Purpose: Generate and open an `.xcodeproj` for `KmiDi_FINAL`.

## Prerequisites
- Xcode 15.1+ with Command Line Tools
- CMake 3.27+ (`brew install cmake`)
- Qt 6 (`brew install qt`)
- Rust toolchain available (used by `engine/intent_ir`)
- JUCE available in one of:
  - `build/external/JUCE`
  - `_build/external/JUCE`
  - `external/JUCE`
  - `../external/JUCE`

## Fast path
```bash
cd /Users/seanburdges/Dev/KmiDi/KmiDi_FINAL
./scripts/configure_xcode.sh xcode-debug
```

## Manual path
```bash
cd /Users/seanburdges/Dev/KmiDi/KmiDi_FINAL
cmake --preset xcode-debug
open _build/xcode-debug/KmiDi.xcodeproj
```

## Preset behavior
- Uses CMake `Xcode` generator
- Builds for `arm64`
- Sets deployment target to `11.0`
- Disables optional tests by default in preset
- Enables `BUILD_NATIVE_MACOS_APP=ON`
- Adds Homebrew Qt paths through `CMAKE_PREFIX_PATH`

## Useful targets
- `KellyCore`
- `KellyApp`
- `KellyPlugin`
- `native_macos_app`
- `intent_ir_unit_test` (if tests are enabled)
- `intent_ir_integration_test` (if tests are enabled)

## Common fixes
- Qt not found:
  ```bash
  cmake --preset xcode-debug -DQt6_DIR=/opt/homebrew/opt/qt/lib/cmake/Qt6
  ```
- Need universal binaries:
  ```bash
  cmake --preset xcode-debug -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"
  ```
- Need tests:
  ```bash
  cmake --preset xcode-debug -DBUILD_TESTS=ON
  ```
