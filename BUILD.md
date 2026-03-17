# KmiDi / Kelly — Build Instructions

This document is the canonical build reference for the KmiDi monorepo (Kelly C++ engine, KellyFFI, Tauri shell).

## Prerequisites

- CMake `>= 3.27`
- Ninja build system
- Python `>= 3.10`
- Node.js `>= 20`
- Xcode Command Line Tools (macOS)

## Quick Start

```bash
git clone <your-kmidi-repo-url>
cd KmiDi
pip install -e .
./scripts/dev-setup.sh
```

## Configure (canonical build dir: `build`)

For the V1 native path (KellyFFI + optional plugin), from repo root:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DKMIDI_BUILD_JUCE_UI=ON \
  -DBUILD_PLUGINS=ON \
  -DBUILD_KELLY_FFI=ON
```

To also enable desktop app or native tests: `-DBUILD_DESKTOP=ON` / `-DBUILD_TESTS=ON` (require Qt/JUCE UI).

## Build Targets (V1-relevant)

```bash
cmake --build build --target KellyCore
cmake --build build --target KellyFFI
cmake --build build --target KellyPlugin_VST3
```

Optional: `KellyApp` (desktop host), `KellyTests` (when `BUILD_TESTS=ON`).

## Run Tests

```bash
ctest --test-dir build --output-on-failure
pytest tests -q
```

## Frontend / Tauri (if enabled)

```bash
npm install
npm run build
```

For desktop development:

```bash
npm run dev:tauri
```

## Common Build Flags

- `BUILD_KELLY_FFI=ON` - Shared library for Tauri (V1 native path).
- `BUILD_PLUGINS=ON` / `KMIDI_BUILD_JUCE_UI=ON` - JUCE plugin targets (e.g. KellyPlugin_VST3).
- `BUILD_DESKTOP=ON|OFF` - Desktop host app targets.
- `BUILD_TESTS=ON|OFF` - Native/unit test targets.

## JUCE 8 Note

This project is aligned to JUCE 8. If you encounter SDK issues, confirm the JUCE subtree in `external/JUCE` resolves to a JUCE 8 tag or a JUCE 8-compatible commit.

## Troubleshooting

- **CMake configure fails:** remove `build` and reconfigure.
- **Missing Python modules:** re-run `pip install -e .`.
- **Node/Tauri errors:** remove `node_modules` and reinstall.
- **Plugin host issues:** rebuild `KellyPlugin_VST3` and verify artifact under `build/KellyPlugin_artefacts/`.
