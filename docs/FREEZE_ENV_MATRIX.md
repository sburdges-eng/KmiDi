# Freeze environment validation matrix (Phase 1)

| Dependency | Present | Affects |
|------------|---------|---------|
| Python 3 | Yes (3.14.2) | dev-setup, build_v1.sh, sync_entities, pytest |
| pip / install -e . | Yes | dev-setup, build_v1.sh |
| PyInstaller | Yes (6.17.0) | build_v1.sh step 2 |
| Node/npm | Yes (v24, npm 11.6) | dev-setup, dev:all, build_v1.sh step 3 |
| Rust/cargo | Yes (1.93.1) | build_v1.sh Tauri, build-full-stack, cargo test |
| CMake | Yes (4.2.1, >3.27) | build_v1.sh, build-full-stack, Pipeline B |
| Ninja | Yes | CMake generator |
| JUCE | Yes (external/JUCE) | Pipeline B, KellyFFI, KellyPlugin_VST3 |
| Qt6 | Yes (pkg-config) | Pipeline B when BUILD_DESKTOP/plugins (optional for KellyFFI-only) |
| pybind11 | Yes (Python env) | build_v1.sh penta_core_native; CMake needs pybind11_DIR in clean config |
| C++ compiler | Yes (Apple Clang 17) | All native builds |

Note: A clean CMake config did not find pybind11 until pybind11_DIR is set (e.g. from bootstrap preset or `-Dpybind11_DIR=...`). build_v1.sh passes it via PYBIND_DIR from Python.
