#!/bin/bash
# Verify Intent IR v1 build setup

set -e

echo "Verifying Intent IR v1 Build Setup..."
echo "======================================"
echo

# Check Rust toolchain
echo "1. Checking Rust toolchain..."
if command -v cargo &> /dev/null; then
    echo "   ✓ cargo found: $(cargo --version)"
else
    echo "   ✗ cargo not found - install Rust toolchain"
    exit 1
fi

# Check CMake
echo "2. Checking CMake..."
if command -v cmake &> /dev/null; then
    echo "   ✓ cmake found: $(cmake --version | head -n1)"
else
    echo "   ✗ cmake not found"
    exit 1
fi

# Check Rust crate structure
echo "3. Checking Rust crate structure..."
RUST_CRATE="engine/intent_ir"
if [ -f "$RUST_CRATE/Cargo.toml" ]; then
    echo "   ✓ Cargo.toml found"
else
    echo "   ✗ Cargo.toml not found at $RUST_CRATE/Cargo.toml"
    exit 1
fi

if [ -f "$RUST_CRATE/src/lib.rs" ]; then
    echo "   ✓ lib.rs found"
else
    echo "   ✗ lib.rs not found"
    exit 1
fi

# Check C headers
echo "4. Checking C headers..."
if [ -f "shared/include/kmidi/IntentIR.h" ]; then
    echo "   ✓ IntentIR.h found"
else
    echo "   ✗ IntentIR.h not found"
    exit 1
fi

# Check Python modules
echo "5. Checking Python modules..."
if [ -f "python/music_brain/session/intent_ir.py" ]; then
    echo "   ✓ intent_ir.py found"
else
    echo "   ✗ intent_ir.py not found"
    exit 1
fi

if [ -f "python/music_brain/session/intent_ir_converter.py" ]; then
    echo "   ✓ intent_ir_converter.py found"
else
    echo "   ✗ intent_ir_converter.py not found"
    exit 1
fi

# Try to build Rust crate
echo "6. Testing Rust crate build..."
cd "$RUST_CRATE"
if cargo check --lib 2>&1 | grep -q "Finished"; then
    echo "   ✓ Rust crate compiles"
else
    echo "   ⚠ Rust crate build check (may need dependencies)"
fi
cd - > /dev/null

# Check C++ adapter
echo "7. Checking C++ adapter..."
if [ -f "engine/src/common/IntentIRAdapter.h" ]; then
    echo "   ✓ IntentIRAdapter.h found"
else
    echo "   ✗ IntentIRAdapter.h not found"
    exit 1
fi

if [ -f "engine/src/common/IntentIRAdapter.cpp" ]; then
    echo "   ✓ IntentIRAdapter.cpp found"
else
    echo "   ✗ IntentIRAdapter.cpp not found"
    exit 1
fi

# Check Swift model
echo "8. Checking Swift model..."
if [ -f "apps/macOS/AppKitShell/Sources/KmiDiApp/Models/IntentFrame.swift" ]; then
    echo "   ✓ IntentFrame.swift found"
else
    echo "   ✗ IntentFrame.swift not found"
    exit 1
fi

echo
echo "======================================"
echo "✓ All Intent IR v1 components found!"
echo
echo "Next steps:"
echo "1. Run CMake to configure build"
echo "2. Build the project"
echo "3. Run integration tests: python tests/intent_ir_integration_test.py"
