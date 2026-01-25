#!/bin/bash
# Unified Build Script for KmiDi
# Builds all components in correct order: C++ Core, Rust Bridge, Frontend, Python Backend

set -e # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "KmiDi Unified Build System"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build configuration
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
USE_KMI_DI_FINAL="${USE_KMI_DI_FINAL:-OFF}"

# Function to print status
print_status() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Build C++ Core
print_status "Step 1: Building C++ Core (KellyBrain + FFI)"
cd "$PROJECT_ROOT"

if [ ! -d "$BUILD_DIR" ]; then
  mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DBUILD_KELLY_CORE=ON
  -DBUILD_KELLY_FFI=ON
)

if [ "$USE_KMI_DI_FINAL" = "ON" ]; then
  CMAKE_ARGS+=(-DUSE_KMI_DI_FINAL=ON)
  print_status "Using KmiDi_FINAL integration"
fi

if ! cmake "${CMAKE_ARGS[@]}" ..; then
  print_error "CMake configuration failed"
  exit 1
fi

if ! cmake --build . --target KellyFFI -j$(sysctl -n hw.ncpu 2> /dev/null || nproc 2> /dev/null || echo 4); then
  print_error "C++ build failed"
  exit 1
fi

# Copy FFI library to Tauri resources
FFI_LIB=""
if [ -f "$BUILD_DIR/libKellyFFI.dylib" ]; then
  FFI_LIB="$BUILD_DIR/libKellyFFI.dylib"
elif [ -f "$BUILD_DIR/libKellyFFI.so" ]; then
  FFI_LIB="$BUILD_DIR/libKellyFFI.so"
elif [ -f "$BUILD_DIR/KellyFFI.dll" ]; then
  FFI_LIB="$BUILD_DIR/KellyFFI.dll"
fi

if [ -n "$FFI_LIB" ]; then
  RESOURCES_DIR="$PROJECT_ROOT/src-tauri/resources"
  mkdir -p "$RESOURCES_DIR"
  cp "$FFI_LIB" "$RESOURCES_DIR/"
  print_status "Copied FFI library to Tauri resources"
else
  print_warning "FFI library not found, Tauri build may fail"
fi

# Step 2: Build Rust Bridge (Tauri)
print_status "Step 2: Building Rust Bridge (Tauri)"
cd "$PROJECT_ROOT/src-tauri"

if ! cargo build --release; then
  print_error "Rust build failed"
  exit 1
fi

# Step 3: Build Frontend
print_status "Step 3: Building Frontend (React)"
cd "$PROJECT_ROOT"

if [ ! -d "node_modules" ]; then
  print_status "Installing npm dependencies..."
  if ! npm install; then
    print_error "npm install failed"
    exit 1
  fi
fi

if ! npm run build; then
  print_error "Frontend build failed"
  exit 1
fi

# Step 4: Verify Python Backend
print_status "Step 4: Verifying Python Backend"
cd "$PROJECT_ROOT"

# Check if Python dependencies are installed
if ! python3 -c "import fastapi" 2> /dev/null; then
  print_warning "Python dependencies may not be installed"
  print_status "Run: pip install -e '.[dev]' to install dependencies"
else
  print_status "Python dependencies verified"
fi

# Summary
echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Built components:"
echo "  ✓ C++ Core (KellyBrain + FFI)"
echo "  ✓ Rust Bridge (Tauri)"
echo "  ✓ Frontend (React)"
echo "  ✓ Python Backend (verified)"
echo ""
echo "Next steps:"
echo "  1. Run: ./scripts/start-all.sh (start all services)"
echo "  2. Run: npm run tauri dev (start desktop app)"
echo ""
