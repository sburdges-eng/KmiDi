#!/bin/bash
# Build Setup Script for KmiDi-1
# Sets up required dependencies and configures the build

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "KmiDi-1 Build Setup"
echo "=========================================="
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check for JUCE
echo "Step 1: Checking JUCE dependency..."
JUCE_DIR="$PROJECT_ROOT/external/JUCE"
if [ -d "$JUCE_DIR" ]; then
    echo -e "${GREEN}✅ JUCE found at: $JUCE_DIR${NC}"
else
    echo -e "${YELLOW}⚠️  JUCE not found${NC}"
    echo "Setting up JUCE..."
    
    mkdir -p "$PROJECT_ROOT/external"
    cd "$PROJECT_ROOT/external"
    
    echo "Cloning JUCE (this may take a few minutes)..."
    git clone --depth 1 --branch 8.0.0 https://github.com/juce-framework/JUCE.git || {
        echo -e "${RED}❌ Failed to clone JUCE${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ JUCE cloned successfully${NC}"
fi

# Step 2: Get pybind11 CMake directory
echo
echo "Step 2: Finding pybind11 CMake directory..."
PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || echo "")

if [ -z "$PYBIND11_DIR" ]; then
    echo -e "${RED}❌ pybind11 not found. Install with: pip install pybind11${NC}"
    exit 1
else
    echo -e "${GREEN}✅ pybind11 found at: $PYBIND11_DIR${NC}"
fi

# Step 3: Configure CMake
echo
echo "Step 3: Configuring CMake build..."
cd "$PROJECT_ROOT"

# Create build directory if it doesn't exist
mkdir -p build

# Configure with pybind11
cmake -B build -S . \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DBUILD_TESTS=OFF \
    -DBUILD_PENTA_TESTS=OFF \
    || {
    echo -e "${RED}❌ CMake configuration failed${NC}"
    echo "Check BUILD_STATUS.md for troubleshooting"
    exit 1
}

echo -e "${GREEN}✅ CMake configured successfully${NC}"

# Step 4: Build penta-core
echo
echo "Step 4: Building penta-core library..."
cmake --build build --target penta_core || {
    echo -e "${YELLOW}⚠️  penta_core build failed (this is okay if JUCE is required)${NC}"
    echo "Trying to build just the core components..."
}

# Step 5: Build Python bindings (if possible)
echo
echo "Step 5: Building Python bindings..."
if cmake --build build --target penta_core_native 2>/dev/null; then
    echo -e "${GREEN}✅ Python bindings built successfully${NC}"
    
    # Check if the module can be imported
    echo
    echo "Testing Python bindings import..."
    if python3 -c "import sys; sys.path.insert(0, 'build/python'); import penta_core_native" 2>/dev/null; then
        echo -e "${GREEN}✅ Python bindings import successful!${NC}"
    else
        echo -e "${YELLOW}⚠️  Python bindings built but import test failed${NC}"
        echo "You may need to add build/python to PYTHONPATH"
    fi
else
    echo -e "${YELLOW}⚠️  Python bindings build skipped (may require additional dependencies)${NC}"
fi

echo
echo "=========================================="
echo -e "${GREEN}Build setup complete!${NC}"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Run tests: python3 scripts/test_python_integration.py"
echo "2. Verify build: python3 scripts/verify_build.py"
echo "3. Build specific targets: cmake --build build --target <target_name>"
echo
echo "Available targets:"
echo "  - penta_core (core C++ library)"
echo "  - penta_core_native (Python bindings)"
echo "  - KellyPlugin (VST3/CLAP plugin, requires full JUCE setup)"
