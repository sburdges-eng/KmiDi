#!/usr/bin/env bash
# KmiDi Workspace Bootstrap Script
# Initializes submodules, checks dependencies, and prepares the environment.
set -euo pipefail

echo "=> Bootstrapping KmiDi Workspace..."

# 1) Initialize all submodules
echo "-> Syncing submodules..."
git submodule update --init --recursive
if [ ! -f "external/JUCE/CMakeLists.txt" ]; then
  echo "FATAL: JUCE submodule failed to initialize. Please check your git configuration."
  exit 1
fi

# 2) Tool version checks
echo "-> Checking build tools..."
if ! command -v cmake >/dev/null 2>&1; then
  echo "FATAL: cmake is not installed. Please install CMake >= 3.27.0."
  exit 1
fi

CMAKE_VER="$(cmake --version | awk 'NR==1{print $3}')"
REQUIRED_CMAKE="3.27.0"
if [ "$(printf "%s\n" "$REQUIRED_CMAKE" "$CMAKE_VER" | sort -V | head -n1)" != "$REQUIRED_CMAKE" ]; then
  echo "FATAL: CMake >= $REQUIRED_CMAKE required. Found ${CMAKE_VER}."
  exit 1
fi

if command -v node >/dev/null 2>&1; then
  NODE_VER="$(node -v | sed 's/^v//')"
  REQUIRED_NODE="20.0.0"
  if [ "$(printf "%s\n" "$REQUIRED_NODE" "$NODE_VER" | sort -V | head -n1)" != "$REQUIRED_NODE" ]; then
    echo "WARNING: Node v$REQUIRED_NODE+ recommended for Tauri. Found v${NODE_VER}."
  fi
fi

# 3) Handle pybind11
echo "-> Checking for pybind11..."
if ! python3 -c "import pybind11" 2>/dev/null; then
  echo "-> pybind11 not found. Attempting to install..."
  python3 -m pip install pybind11 --quiet || {
    echo "WARNING: Failed to install pybind11 automatically. You may need to run: pip install pybind11"
  }
fi

PYBIND_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null || true)"
if [ -z "${PYBIND_DIR}" ]; then
  echo "WARNING: pybind11 still not found. CMake may fail for Python bindings."
else
  echo "   pybind11 found at: ${PYBIND_DIR}"
fi

echo "=> Bootstrap complete. Environment ready."
echo "   To configure C++ build:"
echo "   cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=${PYBIND_DIR}"
