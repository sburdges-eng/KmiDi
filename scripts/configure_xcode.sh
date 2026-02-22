#!/usr/bin/env bash
# Configure the CMake project using the Xcode generator and open it in Xcode.
# Usage: ./scripts/configure_xcode.sh [xcode-debug|xcode-release]

set -euo pipefail

PRESET="${1:-xcode-debug}"
PROJECT_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake not found. Install via Homebrew: brew install cmake"
    exit 1
fi

if ! xcodebuild -version >/dev/null 2>&1; then
    echo "Xcode command line tools not found. Run: xcode-select --install"
    exit 1
fi

case "${PRESET}" in
    xcode-debug)   BINARY_DIR="${PROJECT_ROOT}/build/xcode-debug" ;;
    xcode-release) BINARY_DIR="${PROJECT_ROOT}/build/xcode-release" ;;
    *) echo "Unknown preset: ${PRESET}. Use xcode-debug or xcode-release."; exit 1 ;;
esac

echo "Applying local JUCE patches..."
"${PROJECT_ROOT}/scripts/juce/apply_patches.sh"

echo "Configuring preset '${PRESET}'..."
cmake --preset "${PRESET}"

PROJECT_FILE="${BINARY_DIR}/Kelly.xcodeproj"
if [ -d "${PROJECT_FILE}" ]; then
    echo "Opening Xcode project: ${PROJECT_FILE}"
    open "${PROJECT_FILE}"
else
    echo "Xcode project was not generated at ${PROJECT_FILE}."
    echo "Check the CMake output above for errors (common issues: Qt6 not found)."
    exit 1
fi
