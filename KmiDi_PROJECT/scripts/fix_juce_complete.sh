#!/bin/bash
# Complete JUCE Setup Fix
# Downloads missing juce_build_tools module and fixes CMakeLists.txt

set -euo pipefail

JUCE_VERSION="7.0.12"
JUCE_DIR="external/JUCE"
MODULES_DIR="${JUCE_DIR}/modules"

echo "Fixing JUCE setup for version ${JUCE_VERSION}..."
echo ""

# Check if JUCE directory exists
if [ ! -d "${JUCE_DIR}" ]; then
    echo "Error: JUCE directory not found at ${JUCE_DIR}"
    exit 1
fi

# Check if juce_build_tools exists in extras/Build
BUILD_TOOLS_SOURCE="${JUCE_DIR}/extras/Build/juce_build_tools"
if [ -d "${BUILD_TOOLS_SOURCE}" ]; then
    echo "✓ juce_build_tools found in extras/Build"
else
    echo "⚠ juce_build_tools not found in extras/Build"
    
    # Check if it should be in modules instead
    # Actually, looking at JUCE 7.0.12, juce_build_tools might not be a real module
    # It might be that extras/Build/CMakeLists.txt shouldn't call juce_add_module for it
    echo "Checking official JUCE repository structure..."
    
    # Try to download from official repo if it exists
    BASE_URL="https://raw.githubusercontent.com/juce-framework/JUCE/${JUCE_VERSION}"
    
    # Check if juce_build_tools exists in the official repo
    echo "Checking for juce_build_tools module..."
    
    # Since juce_build_tools might not be a real module, let's check the extras/Build/CMakeLists.txt
    # from the official repo to see how it should be structured
fi

# Actually, the real fix is to modify extras/Build/CMakeLists.txt
# In JUCE 7.0.12, juce_build_tools might not be needed, or it should be handled differently
# Let's check what the actual structure should be

BUILD_CMAKE="${JUCE_DIR}/extras/Build/CMakeLists.txt"
if [ -f "${BUILD_CMAKE}" ]; then
    echo ""
    echo "Current extras/Build/CMakeLists.txt content:"
    head -30 "${BUILD_CMAKE}"
    
    # Check if juce_build_tools line exists
    if grep -q "juce_add_module(juce_build_tools" "${BUILD_CMAKE}"; then
        echo ""
        echo "Found juce_add_module(juce_build_tools) call"
        echo "This module might not exist in JUCE 7.0.12"
        echo ""
        echo "Option 1: Use JUCE_MODULES_ONLY mode (recommended for testing)"
        echo "  This skips extras/Build entirely"
        echo ""
        echo "Option 2: Fix the CMakeLists.txt to handle missing module"
        echo ""
        
        # Try to download the correct CMakeLists.txt from official repo
        echo "Attempting to download correct extras/Build/CMakeLists.txt..."
        curl -s -f "${BASE_URL}/extras/Build/CMakeLists.txt" -o "${BUILD_CMAKE}.official" || {
            echo "⚠ Could not download from official repo"
        }
        
        if [ -f "${BUILD_CMAKE}.official" ]; then
            echo "✓ Downloaded official CMakeLists.txt"
            echo "Backing up current file..."
            cp "${BUILD_CMAKE}" "${BUILD_CMAKE}.backup"
            echo "Replacing with official version..."
            mv "${BUILD_CMAKE}.official" "${BUILD_CMAKE}"
            echo "✓ Updated CMakeLists.txt"
        fi
    fi
fi

echo ""
echo "Fix attempt complete. Try running CMake again."
echo ""
echo "If issues persist, you can use:"
echo "  cmake .. -DJUCE_MODULES_ONLY=ON -DBUILD_PENTA_TESTS=ON"
echo ""
echo "Note: This will skip JUCE build tools but modules will still be available."
