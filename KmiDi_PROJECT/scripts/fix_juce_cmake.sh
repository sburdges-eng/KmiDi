#!/bin/bash
# Fix JUCE CMake support files by downloading missing files from official JUCE repo

set -e

JUCE_VERSION="7.0.12"
JUCE_DIR="external/JUCE"
CMake_DIR="${JUCE_DIR}/extras/Build/CMake"

# Ensure directory exists
mkdir -p "${CMake_DIR}"

# List of required CMake files (based on error messages and JUCE structure)
FILES=(
    "JUCEModuleSupport.cmake"
    "JUCEHelperTargets.cmake"
    "JUCECheckAtomic.cmake"
    "JUCEUtils.cmake"
    "JUCEMakeBinaryData.cmake"
    "JUCEVersionUtils.cmake"
)

BASE_URL="https://raw.githubusercontent.com/juce-framework/JUCE/${JUCE_VERSION}/extras/Build/CMake"

echo "Fetching JUCE ${JUCE_VERSION} CMake support files..."

for file in "${FILES[@]}"; do
    echo "  Downloading ${file}..."
    curl -s -f "${BASE_URL}/${file}" -o "${CMake_DIR}/${file}" || {
        echo "  ⚠ Warning: Failed to download ${file}"
    }
done

echo ""
echo "Checking downloaded files..."
for file in "${FILES[@]}"; do
    if [ -f "${CMake_DIR}/${file}" ]; then
        echo "  ✓ ${file}"
    else
        echo "  ✗ ${file} (missing)"
    fi
done

echo ""
echo "Done! Try running CMake again."

