# Kelly - Build Instructions

This document is the canonical build reference for the Kelly project.

## Prerequisites

- CMake `>= 3.22`
- Ninja build system
- Python `>= 3.10`
- Node.js `>= 20`
- Xcode Command Line Tools (macOS)

## Quick Start

```bash
git clone <your-kelly-repo-url>
cd KmiDi_recovery_20260218-0329
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configure

```bash
cmake -S . -B build_out -G Ninja \
  -DBUILD_PLUGINS=ON \
  -DBUILD_DESKTOP=ON \
  -DBUILD_TESTS=ON
```

## Build Targets

```bash
cmake --build build_out --target KellyCore
cmake --build build_out --target KellyPlugin
cmake --build build_out --target KellyApp
cmake --build build_out --target KellyTests
```

## Run Tests

```bash
ctest --test-dir build_out --output-on-failure
pytest tests -q
```

## Frontend / Tauri (if enabled)

```bash
npm install
npm run build
```

For desktop development:

```bash
npm run tauri dev
```

## Common Build Flags

- `BUILD_PLUGINS=ON|OFF` - JUCE plugin targets.
- `BUILD_DESKTOP=ON|OFF` - Desktop host app targets.
- `BUILD_TESTS=ON|OFF` - Native/unit test targets.

## JUCE 8 Note

This project is aligned to JUCE 8. If you encounter SDK issues, confirm the JUCE subtree in `external/JUCE` resolves to a JUCE 8 tag or a JUCE 8-compatible commit.

## Troubleshooting

- **CMake configure fails:** remove `build_out` and reconfigure.
- **Missing Python modules:** re-run `pip install -e .`.
- **Node/Tauri errors:** remove `node_modules` and reinstall.
- **Plugin host issues:** rebuild `KellyPlugin` and verify output binary paths.
