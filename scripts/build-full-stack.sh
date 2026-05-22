#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"
BUILD_TYPE="Release"
BUILD_PLUGINS="ON"
RUN_TAURI_CHECK="ON"
JOBS="${JOBS:-8}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --debug                Use Debug build type
  --no-plugins           Skip KellyPlugin_VST3 build
  --build-dir <path>     Override build directory
  --no-tauri             Skip Tauri cargo build validation
  -h, --help             Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --no-plugins)
      BUILD_PLUGINS="OFF"
      shift
      ;;
    --build-dir)
      BUILD_DIR="${2:?missing path for --build-dir}"
      shift 2
      ;;
    --no-tauri)
      RUN_TAURI_CHECK="OFF"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "[1/4] Configuring CMake in $BUILD_DIR"
PYBIND_DIR="$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null)" || true
CMAKE_EXTRA=()
[[ -n "$PYBIND_DIR" ]] && CMAKE_EXTRA+=(-Dpybind11_DIR="$PYBIND_DIR")
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DBUILD_KELLY_CORE=ON \
  -DKMIDI_BUILD_JUCE_UI=ON \
  -DBUILD_PLUGINS="$BUILD_PLUGINS" \
  -DBUILD_KELLY_FFI=ON \
  "${CMAKE_EXTRA[@]}"

echo "[2/4] Building KellyFFI"
cmake --build "$BUILD_DIR" --target KellyFFI -j"$JOBS"

if [[ "$BUILD_PLUGINS" == "ON" ]]; then
  echo "[3/4] Building KellyPlugin_VST3"
  cmake --build "$BUILD_DIR" --target KellyPlugin_VST3 -j"$JOBS"
else
  echo "[3/4] Skipping plugin build (--no-plugins)"
fi

if [[ "$RUN_TAURI_CHECK" == "ON" ]]; then
  echo "[4/4] Validating Tauri Rust build"
  cargo build --manifest-path "$ROOT_DIR/engine/intent_ir/Cargo.toml"
else
  echo "[4/4] Skipping Tauri cargo validation (--no-tauri)"
fi

echo
echo "Build complete."
echo "  Root:      $ROOT_DIR"
echo "  Build dir: $BUILD_DIR"
echo "  Type:      $BUILD_TYPE"
