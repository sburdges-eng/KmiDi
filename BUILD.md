# KmiDi / Kelly — Build Instructions

Status: current build reference aligned to checked-in scripts and architecture authority
Last updated: 2026-06-08

This document is the operational build reference for the current repo.
For architecture authority, use `docs/ARCHITECTURE.md` and companion authority docs.

## 1. Toolchain prerequisites

Expected tools:
- CMake 3.27+
- Ninja
- Python 3.10+
- Node 20+
- Rust stable
- Xcode Command Line Tools on macOS

Native/plugin builds additionally expect:
- JUCE at `external/JUCE/`

## 2. Quick setup

```bash
./scripts/dev-setup.sh
```

That currently runs:
- bootstrap helper
- `npm install`
- `python3 -m pip install -e .`

## 3. Frontend and API build/run surfaces

Frontend build:

```bash
npx tsc --noEmit
npm run build
```

API run:

```bash
npm run dev:python
```

Combined dev run:

```bash
npm run dev:all
```

Important clarification:
- `package.json` does not currently define `npm run dev:tauri`.
- Treat older references to that command as historical/legacy drift, not current runnable truth.

## 4. Root CMake native build

For KellyFFI and plugin/runtime work, build from repo root.
Example configure:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_KELLY_CORE=ON \
  -DBUILD_KELLY_FFI=ON \
  -DKMIDI_BUILD_JUCE_UI=ON \
  -DBUILD_PLUGINS=ON
```

Example targets:

```bash
cmake --build build --target KellyCore -j8
cmake --build build --target KellyFFI -j8
cmake --build build --target KellyPlugin_VST3 -j8
```

Other useful options include:
- `BUILD_TESTS=ON`
- `BUILD_DESKTOP=ON` when desktop host targets are intentionally enabled
- `KMIDI_ENABLE_ASAN=ON`
- `KMIDI_ENABLE_TSAN=ON`

## 5. Preset-based native workflows

The repo also provides `CMakePresets.json`.
Current configure presets:
- `xcode-debug`
- `xcode-release`
- `ninja-debug`
- `ninja-asan`
- `ninja-tsan`

Current build presets:
- `xcode-debug`
- `xcode-release`
- `ninja-debug-rt-harness`
- `ninja-asan-rt-harness`
- `ninja-tsan-rt-harness`

Examples:

```bash
cmake --preset ninja-debug
cmake --build --preset ninja-debug-rt-harness

cmake --preset ninja-asan
cmake --build --preset ninja-asan-rt-harness
```

## 6. Intent/schema sync build rule

If you change engine-facing or persisted intent schema surfaces:

```bash
python3 scripts/sync_entities.py
python3 -m pytest tests/unit/test_api_schema.py
cd engine/intent_ir && cargo test
```

Generated files are not hand-edited.

## 7. Test commands

Python:

```bash
python3 -m pytest tests/
```

Rust Intent IR:

```bash
cd engine/intent_ir && cargo test
```

Native tests when enabled:

```bash
ctest --test-dir build --output-on-failure
```

Lint/check examples:

```bash
python3 -m flake8 music_brain/ --max-line-length 100
npx tsc --noEmit
```

## 8. Native safety reminders

Before changing C++/FFI/RT-sensitive code, read:
- `AGENTS.md`
- `docs/NATIVE_RUNTIME_OWNERSHIP.md`
- `docs/JUCE_RT_RULES.md`
- `docs/FFI_OWNERSHIP_AND_ABI.md`

Key constraints:
- no exceptions across FFI
- no exceptions, locks, or allocations on RT paths
- KellyFFI must not create duplicate JUCE linkage situations
- exported ABI changes require human review

## 9. Troubleshooting

### CMake configure fails
- remove or inspect the build directory and reconfigure
- verify JUCE exists at `external/JUCE/`
- verify required toolchains are installed

### Python module/import issues
- rerun `python3 -m pip install -e .`
- confirm the active Python interpreter is the one you installed into

### Frontend build issues
- rerun `npm install`
- check Node version

### Plugin build issues
- verify `KMIDI_BUILD_JUCE_UI=ON` and `BUILD_PLUGINS=ON`
- rebuild `KellyPlugin_VST3`
- inspect artifacts under `build/KellyPlugin_artefacts/`

## 10. Related docs

- `docs/DEVELOPMENT.md`
- `docs/FULL_STACK_BUILD.md`
- `docs/BOOT.md`
- `docs/ENVIRONMENT.md`
- `AGENTS.md`
