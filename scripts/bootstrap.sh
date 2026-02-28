#!/usr/bin/env bash
set -euo pipefail

echo "=> Bootstrapping KmiDi v1 Workspace..."

# 1) Fix the JUCE submodule blocker
echo "-> Syncing external/JUCE..."
git submodule update --init --recursive external/JUCE
if [ ! -f "external/JUCE/CMakeLists.txt" ]; then
  echo "FATAL: JUCE submodule failed to initialize."
  exit 1
fi

# 2) Enforce tool versions (source of truth over stale docs)
if ! command -v cmake >/dev/null 2>&1; then
  echo "FATAL: cmake is not installed."
  exit 1
fi

CMAKE_VER="$(cmake --version | awk 'NR==1{print $3}')"
if [ "$(printf '%s\n' "3.27.0" "$CMAKE_VER" | sort -V | head -n1)" != "3.27.0" ]; then
  echo "FATAL: CMake >= 3.27.0 required. Found ${CMAKE_VER}."
  exit 1
fi

# Node check for Tauri
if command -v node >/dev/null 2>&1; then
  NODE_VER="$(node -v | sed 's/^v//')"
  if [ "$(printf '%s\n' "20.0.0" "$NODE_VER" | sort -V | head -n1)" != "20.0.0" ]; then
    echo "WARNING: Node v20+ recommended for Tauri. Found v${NODE_VER}."
  fi
fi

# 3) Fix pybind11 discovery
echo "-> Locating pybind11..."
PYBIND_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null || true)"
if [ -z "${PYBIND_DIR}" ]; then
  echo "WARNING: pybind11 not found in current Python environment. CMake may fail for bindings."
fi

echo "Bootstrap complete. Environment locked and ready."
echo "Run: cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=${PYBIND_DIR}"
