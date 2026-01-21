#!/bin/bash
# Comprehensive build environment setup script
# Fixes common build issues and prepares the environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

echo "=========================================="
echo "KmiDi Build Environment Setup"
echo "=========================================="
echo ""

# 1. Setup Tauri icons
echo "Step 1: Setting up Tauri icons..."
"${SCRIPT_DIR}/setup-icons.sh"
echo ""

# 2. Check JUCE configuration
echo "Step 2: Checking JUCE configuration..."
JUCE_DIR="${PROJECT_ROOT}/external/JUCE"
JUCE_MODULE_SUPPORT="${JUCE_DIR}/extras/Build/CMake/JUCEModuleSupport.cmake"

if [ ! -f "${JUCE_MODULE_SUPPORT}" ]; then
    echo "  ⚠ JUCE module support file not found at: ${JUCE_MODULE_SUPPORT}"
    echo ""
    echo "  Options to fix:"
    echo "  1. Use KmiDi_FINAL JUCE (if available):"
    echo "     cmake -DUSE_KMI_DI_FINAL=ON -DKMI_DI_FINAL_ROOT=../KmiDi-1/KmiDi_FINAL"
    echo ""
    echo "  2. Update JUCE to a version with CMake support"
    echo "  3. Manually add JUCEModuleSupport.cmake"
    echo ""
    
    # Check if KmiDi_FINAL has JUCE
    KMI_DI_FINAL_JUCE="../KmiDi-1/KmiDi_FINAL/build/external/JUCE/extras/Build/CMake/JUCEModuleSupport.cmake"
    if [ -f "${KMI_DI_FINAL_JUCE}" ]; then
        echo "  ✓ Found JUCE in KmiDi_FINAL, you can use:"
        echo "    cmake -DUSE_KMI_DI_FINAL=ON .."
    fi
else
    echo "  ✓ JUCE module support found"
fi
echo ""

# 3. Check Rust/Cargo
echo "Step 3: Checking Rust environment..."
if command -v cargo &> /dev/null; then
    echo "  ✓ Cargo found: $(cargo --version)"
    
    # Check for Tauri CLI
    if cargo tauri --version &> /dev/null; then
        echo "  ✓ Tauri CLI found"
    else
        echo "  ⚠ Tauri CLI not found, installing..."
        cargo install tauri-cli --locked || echo "  ⚠ Failed to install Tauri CLI"
    fi
else
    echo "  ✗ Cargo not found. Install Rust: https://rustup.rs/"
fi
echo ""

# 4. Check Node.js/npm
echo "Step 4: Checking Node.js environment..."
if command -v npm &> /dev/null; then
    echo "  ✓ npm found: $(npm --version)"
    
    # Check if node_modules exists
    if [ -d "${PROJECT_ROOT}/node_modules" ]; then
        echo "  ✓ node_modules directory exists"
    else
        echo "  ⚠ node_modules not found, run: npm install"
    fi
else
    echo "  ✗ npm not found. Install Node.js: https://nodejs.org/"
fi
echo ""

# 5. Check CMake
echo "Step 5: Checking CMake..."
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1)
    echo "  ✓ CMake found: ${CMAKE_VERSION}"
    
    # Check version (need 3.27+)
    CMAKE_MAJOR=$(cmake --version | head -n1 | awk '{print $3}' | cut -d. -f1)
    CMAKE_MINOR=$(cmake --version | head -n1 | awk '{print $3}' | cut -d. -f2)
    if [ "${CMAKE_MAJOR}" -lt 3 ] || ([ "${CMAKE_MAJOR}" -eq 3 ] && [ "${CMAKE_MINOR}" -lt 27 ]); then
        echo "  ⚠ CMake 3.27+ required, found ${CMAKE_MAJOR}.${CMAKE_MINOR}"
    fi
else
    echo "  ✗ CMake not found. Install CMake: https://cmake.org/"
fi
echo ""

# 6. Check Qt6
echo "Step 6: Checking Qt6..."
if command -v qmake6 &> /dev/null || [ -n "${Qt6_DIR}" ]; then
    echo "  ✓ Qt6 found"
else
    echo "  ⚠ Qt6 not found in PATH"
    echo "     Set Qt6_DIR environment variable or install Qt6"
fi
echo ""

# 7. Create build directory structure
echo "Step 7: Creating build directories..."
mkdir -p "${PROJECT_ROOT}/build"
mkdir -p "${PROJECT_ROOT}/src-tauri/resources"
echo "  ✓ Build directories created"
echo ""

# 8. Summary
echo "=========================================="
echo "Setup Summary"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Fix any missing dependencies above"
echo "  2. Run icon setup: ./scripts/setup-icons.sh"
echo "  3. Configure CMake:"
echo "     cd build"
echo "     cmake .. -DBUILD_KELLY_CORE=ON -DBUILD_KELLY_FFI=ON"
echo "  4. Build:"
echo "     make KellyFFI"
echo "  5. Test Rust:"
echo "     cd src-tauri && cargo check --lib"
echo ""
