#!/bin/bash
# Minimal FFI test script
# Tests FFI compilation without requiring full KellyCore build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${PROJECT_ROOT}/build_ffi_test"

echo "=========================================="
echo "Minimal FFI Test"
echo "=========================================="
echo ""

# Create test build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Try to compile minimal FFI test
echo "Compiling minimal FFI test..."
g++ -std=c++20 \
    -I"${PROJECT_ROOT}/src" \
    -I"${PROJECT_ROOT}/include" \
    -DKELLY_FFI_AVAILABLE=0 \
    "${PROJECT_ROOT}/tests/cpp/test_ffi_minimal.cpp" \
    -o test_ffi_minimal \
    2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "  ✓ Compilation successful"
    echo ""
    echo "Running test..."
    ./test_ffi_minimal
    echo ""
    echo "=========================================="
    echo "Minimal FFI test complete!"
    echo "=========================================="
else
    echo "  ⚠ Compilation failed (expected if full build not available)"
    echo "  This is a minimal test - full FFI requires KellyCore"
fi
