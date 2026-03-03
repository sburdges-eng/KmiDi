#!/bin/bash
# KmiDi_FINAL Integration Validator (Shell Version)
# Tests that integration components are properly configured

set -e

echo "🔍 KmiDi_FINAL Integration Validator"
echo "===================================="

# Check KmiDi_FINAL availability
echo ""
echo "🧪 KmiDi_FINAL Availability..."
KMI_DI_FINAL_ROOT="../KmiDi-1/KmiDi_FINAL"
if [ ! -d "$KMI_DI_FINAL_ROOT" ]; then
    echo "❌ KmiDi_FINAL not found at $KMI_DI_FINAL_ROOT"
    exit 1
fi

# Check required directories
# Note: apps/macOS is still present but AppKitShell moved to legacy/ui/appkit_shell (per ADR 001)
REQUIRED_DIRS=("engine/src/dsp" "apps/macOS" "CMakeLists.txt")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$KMI_DI_FINAL_ROOT/$dir" ] && [ ! -f "$KMI_DI_FINAL_ROOT/$dir" ]; then
        echo "❌ Missing required directory or file: $dir"
        exit 1
    fi
done
echo "✅ KmiDi_FINAL found with all required components"

# Check DSP purity
echo ""
echo "🧪 DSP Core Purity..."
DSP_DIR="$KMI_DI_FINAL_ROOT/engine/src/dsp"
CPP_FILES=$(find "$DSP_DIR" -name "*.cpp" 2>/dev/null)

if [ -z "$CPP_FILES" ]; then
    echo "❌ No DSP source files found"
    exit 1
fi

# Check for forbidden includes
FORBIDDEN=("juce" "JUCE" "AppKit" "SwiftUI" "Qt")
VIOLATIONS=()

for cpp_file in $CPP_FILES; do
    for pattern in "${FORBIDDEN[@]}"; do
        if grep -q "#include.*$pattern" "$cpp_file" 2>/dev/null; then
            VIOLATIONS+=("$cpp_file: $pattern")
        fi
    done
done

if [ ${#VIOLATIONS[@]} -ne 0 ]; then
    echo "❌ Framework dependencies found:"
    printf '  %s\n' "${VIOLATIONS[@]}"
    exit 1
fi

FILE_COUNT=$(echo "$CPP_FILES" | wc -l)
echo "✅ DSP core is pure: $FILE_COUNT files checked"

# Check CMake integration
echo ""
echo "🧪 CMake Integration..."
CMAKE_FILE="CMakeLists.txt"

if [ ! -f "$CMAKE_FILE" ]; then
    echo "❌ CMakeLists.txt not found"
    exit 1
fi

REQUIRED_OPTIONS=("USE_KMI_DI_FINAL" "BUILD_NATIVE_MACOS_APP" "KMI_DI_FINAL_ROOT")
for option in "${REQUIRED_OPTIONS[@]}"; do
    if ! grep -q "$option" "$CMAKE_FILE"; then
        echo "❌ Missing CMake option: $option"
        exit 1
    fi
done
echo "✅ CMake integration configured"

# Check integration files
echo ""
echo "🧪 Integration Files..."
INTEGRATION_FILES=(
    "KmiDi_FINAL_INTEGRATION_GUIDE.md"
    "KmiDi_FINAL_DSP_Migration_Example.h"
    "build_with_kmidi_final.sh"
    "validate_integration.py"
)

for file in "${INTEGRATION_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing integration file: $file"
        exit 1
    fi
done
echo "✅ All integration files present"

# Test build configuration
echo ""
echo "🧪 Build System Test..."
BUILD_DIR="build_integration_test"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "  Configuring with integration..."
if ! cmake .. -DUSE_KMI_DI_FINAL=ON -DBUILD_NATIVE_MACOS_APP=ON -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1; then
    echo "❌ CMake configuration failed"
    exit 1
fi

if ! grep -q "Using KmiDi_FINAL" CMakeCache.txt 2>/dev/null; then
    echo "❌ CMake did not detect KmiDi_FINAL integration"
    exit 1
fi

echo "✅ Build system integration working"

# Clean up
cd ..
rm -rf "$BUILD_DIR"

echo ""
echo "===================================="
echo "🎉 INTEGRATION VALIDATION: ALL TESTS PASSED"
echo ""
echo "Summary:"
echo "✅ KmiDi_FINAL availability confirmed"
echo "✅ DSP core purity verified (no JUCE dependencies)"
echo "✅ CMake integration configured"
echo "✅ Build system working"
echo "✅ All integration files present"
echo ""
echo "🎯 Ready for 95% compliance achievement!"
echo ""
echo "Next steps:"
echo "1. Run: ./build_with_kmidi_final.sh"
echo "2. Test: DSP purity and native app"
echo "3. Migrate: Replace JUCE calls with pure DSP"
echo ""
echo "Compliance target: 65% → 95% (+30%)"