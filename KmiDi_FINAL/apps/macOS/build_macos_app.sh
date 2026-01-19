#!/usr/bin/env bash
set -euo pipefail

# Build a distributable macOS .app bundle for the C++ desktop target (Qt6).
#
# Prereqs:
# - Xcode command line tools
# - CMake (and Ninja recommended)
# - Qt6 (with `macdeployqt` available for packaging)
#
# Usage:
#   ./build_macos_app.sh [--config Release|Debug] [--clean]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="Release"
CLEAN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --clean)
      CLEAN=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

BUILD_DIR="${PROJECT_ROOT}/_build/macos"
DIST_DIR="${PROJECT_ROOT}/dist/macos"

if [[ "${CLEAN}" == "true" ]]; then
  rm -rf "${BUILD_DIR}" "${DIST_DIR}"
fi

mkdir -p "${BUILD_DIR}" "${DIST_DIR}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "Error: cmake not found" >&2
  exit 1
fi

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
fi

cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" "${GENERATOR_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE="${CONFIG}" \
  -DBUILD_DESKTOP=ON \
  -DBUILD_PLUGINS=OFF \
  -DBUILD_TESTS=OFF

cmake --build "${BUILD_DIR}" --config "${CONFIG}" --target KellyApp

APP_PATH=""
if [[ -d "${BUILD_DIR}/KmiDi.app" ]]; then
  APP_PATH="${BUILD_DIR}/KmiDi.app"
else
  APP_PATH="$(find "${BUILD_DIR}" -maxdepth 4 -type d \\( -name "KmiDi.app" -o -name "KellyApp.app" \\) | head -n 1 || true)"
fi

if [[ -z "${APP_PATH}" || ! -d "${APP_PATH}" ]]; then
  echo "Error: could not locate built .app bundle under ${BUILD_DIR}" >&2
  exit 1
fi

OUT_APP="${DIST_DIR}/KmiDi.app"
rm -rf "${OUT_APP}"
ditto "${APP_PATH}" "${OUT_APP}"

if command -v macdeployqt >/dev/null 2>&1; then
  MACDEPLOYQT_VERBOSE="${MACDEPLOYQT_VERBOSE:-0}"
  QT_LIBPATH=""
  if command -v qmake6 >/dev/null 2>&1; then
    QT_LIBPATH="$(qmake6 -query QT_INSTALL_LIBS || true)"
  elif command -v qtpaths6 >/dev/null 2>&1; then
    QT_LIBPATH="$(qtpaths6 --query QT_INSTALL_LIBS || true)"
  elif command -v qmake >/dev/null 2>&1; then
    QT_LIBPATH="$(qmake -query QT_INSTALL_LIBS || true)"
  elif command -v qtpaths >/dev/null 2>&1; then
    QT_LIBPATH="$(qtpaths --query QT_INSTALL_LIBS || true)"
  fi

  if [[ -n "${QT_LIBPATH}" ]]; then
    macdeployqt "${OUT_APP}" "-verbose=${MACDEPLOYQT_VERBOSE}" -always-overwrite "-libpath=${QT_LIBPATH}" >/dev/null
  else
    macdeployqt "${OUT_APP}" "-verbose=${MACDEPLOYQT_VERBOSE}" -always-overwrite >/dev/null
  fi
else
  echo "Note: macdeployqt not found; ${OUT_APP} may require Qt frameworks on the target machine." >&2
fi

echo "Built: ${OUT_APP}"
