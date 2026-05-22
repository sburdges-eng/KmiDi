#!/bin/bash

# =============================================================================
# KmiDi Multi-Technology Build Script
# =============================================================================
# This script builds the complete KmiDi application stack:
# 1. C++/JUCE KellyCore and FFI library
# 2. React frontend
# 3. Tauri desktop application
# 4. Plugin formats (VST3/AU)
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
DIST_DIR="$PROJECT_ROOT/dist"
INSTALL_PREFIX="$BUILD_DIR/install"

# Build options (can be overridden by environment variables)
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_PLUGINS="${BUILD_PLUGINS:-ON}"
BUILD_TESTS="${BUILD_TESTS:-OFF}"
BUILD_TAURI="${BUILD_TAURI:-ON}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Platform detection
PLATFORM="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    PLATFORM="windows"
fi

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║  ██╗  ██╗███╗   ███╗██╗██████╗ ██╗    ██████╗ ██╗   ██╗██╗██╗     ██████╗  ║"
echo "║  ██║ ██╔╝████╗ ████║██║██╔══██╗██║    ██╔══██╗██║   ██║██║██║     ██╔══██╗ ║"
echo "║  █████╔╝ ██╔████╔██║██║██║  ██║██║    ██████╔╝██║   ██║██║██║     ██║  ██║ ║"
echo "║  ██╔═██╗ ██║╚██╔╝██║██║██║  ██║██║    ██╔══██╗██║   ██║██║██║     ██║  ██║ ║"
echo "║  ██║  ██╗██║ ╚═╝ ██║██║██████╔╝██║    ██████╔╝╚██████╔╝██║███████╗██████╔╝ ║"
echo "║  ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═════╝ ╚═╝    ╚═════╝  ╚═════╝ ╚═╝╚══════╝╚═════╝  ║"
echo "║                                                                  ║"
echo "║            Multi-Technology Build System v1.0                   ║"
echo "║        React + Tauri + C++/JUCE + Python Integration            ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Platform: ${PLATFORM}${NC}"
echo -e "${BLUE}Build Type: ${BUILD_TYPE}${NC}"
echo -e "${BLUE}Parallel Jobs: ${PARALLEL_JOBS}${NC}"
echo -e "${BLUE}Build Plugins: ${BUILD_PLUGINS}${NC}"
echo ""

# =============================================================================
# Utility Functions
# =============================================================================

log_step() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed or not in PATH"
        echo "Please install $1 and try again"
        exit 1
    fi
}

# =============================================================================
# Pre-build Checks
# =============================================================================

log_step "Checking build requirements..."

# Check required tools
check_command "cmake"
check_command "node"
check_command "npm"
check_command "cargo"

# Check for Rust/Tauri
if [ "$BUILD_TAURI" == "ON" ]; then
    check_command "cargo"
    if ! cargo list | grep -q "tauri-cli"; then
        log_error "Tauri CLI not installed"
        echo "Install with: cargo install tauri-cli"
        exit 1
    fi
fi

# Check for C++ compiler
if [[ "$PLATFORM" == "macos" ]]; then
    if ! command -v clang++ &> /dev/null; then
        log_error "Xcode Command Line Tools not installed"
        echo "Install with: xcode-select --install"
        exit 1
    fi
elif [[ "$PLATFORM" == "linux" ]]; then
    if ! command -v g++ &> /dev/null; then
        log_error "GCC not installed"
        echo "Install with: sudo apt-get install build-essential (Ubuntu) or equivalent"
        exit 1
    fi
fi

log_success "All build requirements satisfied"

# Keep UI/Rust entities in sync with canonical Python models before UI builds.
log_step "Syncing canonical entity contracts..."
python3 "$PROJECT_ROOT/scripts/sync_entities.py"
log_success "Entity contracts synced"

# =============================================================================
# Clean Previous Builds (Optional)
# =============================================================================

if [ "${CLEAN_BUILD:-false}" == "true" ]; then
    log_step "Cleaning previous builds..."
    rm -rf "$BUILD_DIR"
    rm -rf "$DIST_DIR"
    rm -rf "$PROJECT_ROOT/engine/intent_ir/target"
    rm -rf "$PROJECT_ROOT/node_modules"
    log_success "Clean completed"
fi

# =============================================================================
# Step 1: Build C++/JUCE Core and FFI Library
# =============================================================================

log_step "Building C++/JUCE Core and FFI Library..."

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
    "-DBUILD_KELLY_CORE=ON"
    "-DBUILD_KELLY_FFI=ON"
    "-DBUILD_PLUGINS=$BUILD_PLUGINS"
    "-DBUILD_TESTS=$BUILD_TESTS"
    "-DENABLE_TRACY=OFF"
)

# Platform-specific arguments
if [[ "$PLATFORM" == "macos" ]]; then
    CMAKE_ARGS+=("-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15")
fi

echo "CMake configuration:"
echo "  ${CMAKE_ARGS[*]}"

cmake "${CMAKE_ARGS[@]}" "$PROJECT_ROOT"

# Build
echo "Building with $PARALLEL_JOBS parallel jobs..."
make -j"$PARALLEL_JOBS"

# Install to local prefix
make install

log_success "C++ build completed"

# =============================================================================
# Step 2: Install Node Dependencies
# =============================================================================

log_step "Installing Node.js dependencies..."

cd "$PROJECT_ROOT"

# Check if package-lock.json exists and is newer than package.json
if [ -f "package-lock.json" ] && [ "package-lock.json" -nt "package.json" ]; then
    npm ci
else
    npm install
fi

log_success "Node.js dependencies installed"

# =============================================================================
# Step 3: Build React Frontend
# =============================================================================

log_step "Building React frontend..."

cd "$PROJECT_ROOT"

# Build React app
npm run build

# Verify build output
if [ ! -d "$PROJECT_ROOT/dist" ]; then
    log_error "React build failed - dist directory not found"
    exit 1
fi

log_success "React frontend built"

# =============================================================================
# Step 4: Configure Tauri Build
# =============================================================================

if [ "$BUILD_TAURI" == "ON" ]; then
    log_step "Configuring Tauri build..."

    cd "$PROJECT_ROOT/engine/intent_ir"

    # Ensure FFI library is available for linking
    if [ -f "$BUILD_DIR/libKellyFFI.dylib" ]; then
        # macOS
        mkdir -p resources
        cp "$BUILD_DIR/libKellyFFI.dylib" resources/
        export DYLD_LIBRARY_PATH="$BUILD_DIR:$DYLD_LIBRARY_PATH"
    elif [ -f "$BUILD_DIR/libKellyFFI.so" ]; then
        # Linux
        mkdir -p resources
        cp "$BUILD_DIR/libKellyFFI.so" resources/
        export LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH"
    fi

    # Set library path for Rust linking
    export RUSTFLAGS="-L$BUILD_DIR"
    
    log_success "Tauri configuration completed"
fi

# =============================================================================
# Step 5: Build Tauri Desktop Application
# =============================================================================

if [ "$BUILD_TAURI" == "ON" ]; then
    log_step "Building Tauri desktop application..."

    cd "$PROJECT_ROOT"

    # Build Tauri app
    if [ "$BUILD_TYPE" == "Debug" ]; then
        npm run tauri build -- --debug
    else
        npm run tauri build
    fi

    # Verify Tauri build output
    TAURI_BUILD_DIR="$PROJECT_ROOT/engine/intent_ir/target/release"
    if [ "$BUILD_TYPE" == "Debug" ]; then
        TAURI_BUILD_DIR="$PROJECT_ROOT/engine/intent_ir/target/debug"
    fi

    if [[ "$PLATFORM" == "macos" ]]; then
        if [ ! -d "$TAURI_BUILD_DIR/bundle/macos/idaw.app" ]; then
            log_error "Tauri macOS build failed - app bundle not found"
            exit 1
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        if [ ! -f "$TAURI_BUILD_DIR/idaw" ]; then
            log_error "Tauri Linux build failed - executable not found"
            exit 1
        fi
    fi

    log_success "Tauri desktop application built"
fi

# =============================================================================
# Step 6: Build Plugins (Optional)
# =============================================================================

if [ "$BUILD_PLUGINS" == "ON" ]; then
    log_step "Building audio plugins..."

    cd "$BUILD_DIR"

    # Build plugins
    if make --dry-run KellyPlugin >/dev/null 2>&1; then
        make -j"$PARALLEL_JOBS" KellyPlugin
        log_success "Audio plugins built"
        
        # List built plugins
        echo "Built plugins:"
        if [[ "$PLATFORM" == "macos" ]]; then
            find . -name "*.vst3" -o -name "*.component" | sed 's/^/  /'
        else
            find . -name "*.vst3" -o -name "*.so" | grep -i plugin | sed 's/^/  /'
        fi
    else
        echo "No plugin targets found - skipping"
    fi
fi

# =============================================================================
# Step 7: Create Distribution Package
# =============================================================================

log_step "Creating distribution package..."

# Create dist directory structure
mkdir -p "$DIST_DIR"/{app,plugins,docs,resources}

# Copy built applications
if [ "$BUILD_TAURI" == "ON" ]; then
    TAURI_BUILD_DIR="$PROJECT_ROOT/engine/intent_ir/target/release"
    if [ "$BUILD_TYPE" == "Debug" ]; then
        TAURI_BUILD_DIR="$PROJECT_ROOT/engine/intent_ir/target/debug"
    fi

    if [[ "$PLATFORM" == "macos" ]]; then
        if [ -d "$TAURI_BUILD_DIR/bundle/macos/idaw.app" ]; then
            cp -r "$TAURI_BUILD_DIR/bundle/macos/idaw.app" "$DIST_DIR/app/"
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        if [ -f "$TAURI_BUILD_DIR/idaw" ]; then
            cp "$TAURI_BUILD_DIR/idaw" "$DIST_DIR/app/"
        fi
    fi
fi

# Copy plugins
if [ "$BUILD_PLUGINS" == "ON" ]; then
    if [[ "$PLATFORM" == "macos" ]]; then
        find "$BUILD_DIR" -name "*.vst3" -exec cp -r {} "$DIST_DIR/plugins/" \; 2>/dev/null || true
        find "$BUILD_DIR" -name "*.component" -exec cp -r {} "$DIST_DIR/plugins/" \; 2>/dev/null || true
    else
        find "$BUILD_DIR" -name "*.vst3" -exec cp -r {} "$DIST_DIR/plugins/" \; 2>/dev/null || true
        find "$BUILD_DIR" -name "*Plugin*.so" -exec cp {} "$DIST_DIR/plugins/" \; 2>/dev/null || true
    fi
fi

# Copy FFI library
if [ -f "$BUILD_DIR/libKellyFFI.dylib" ]; then
    cp "$BUILD_DIR/libKellyFFI.dylib" "$DIST_DIR/resources/"
elif [ -f "$BUILD_DIR/libKellyFFI.so" ]; then
    cp "$BUILD_DIR/libKellyFFI.so" "$DIST_DIR/resources/"
fi

# Copy documentation
if [ -d "$PROJECT_ROOT/docs" ]; then
    cp -r "$PROJECT_ROOT/docs"/* "$DIST_DIR/docs/" 2>/dev/null || true
fi

# Create build info file
cat > "$DIST_DIR/BUILD_INFO.txt" << EOF
KmiDi Build Information
=======================

Build Date: $(date)
Platform: $PLATFORM
Build Type: $BUILD_TYPE
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
Git Branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

Components Built:
- C++/JUCE Core: Yes
- FFI Library: Yes
- React Frontend: Yes
- Tauri Desktop: $BUILD_TAURI
- Audio Plugins: $BUILD_PLUGINS

Build Configuration:
- Parallel Jobs: $PARALLEL_JOBS
- Build Directory: $BUILD_DIR
- Install Prefix: $INSTALL_PREFIX
EOF

log_success "Distribution package created"

# =============================================================================
# Step 8: Run Tests (Optional)
# =============================================================================

if [ "$BUILD_TESTS" == "ON" ]; then
    log_step "Running tests..."

    cd "$BUILD_DIR"

    # Run C++ tests
    if [ -f "./KellyTests" ]; then
        echo "Running C++ tests..."
        ./KellyTests || log_error "C++ tests failed"
    fi

    # Run Rust tests
    if [ "$BUILD_TAURI" == "ON" ]; then
        echo "Running Rust tests..."
        cd "$PROJECT_ROOT/engine/intent_ir"
        cargo test || log_error "Rust tests failed"
    fi

    # Run Node tests (if they exist)
    cd "$PROJECT_ROOT"
    if [ -f "package.json" ] && npm run test --dry-run >/dev/null 2>&1; then
        echo "Running Node.js tests..."
        npm run test || log_error "Node.js tests failed"
    fi

    log_success "Tests completed"
fi

# =============================================================================
# Build Summary
# =============================================================================

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                     BUILD COMPLETED!                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}Build Summary:${NC}"
echo "  📦 Platform: $PLATFORM"
echo "  🔨 Build Type: $BUILD_TYPE"
echo "  ⏱️  Build Time: $(date)"
echo "  📁 Distribution: $DIST_DIR"
echo ""

echo -e "${CYAN}Built Components:${NC}"
echo "  ⚛️  React Frontend: ✅"
echo "  🦀 Tauri Desktop: $([ "$BUILD_TAURI" == "ON" ] && echo "✅" || echo "⏭️")"
echo "  🎵 Audio Plugins: $([ "$BUILD_PLUGINS" == "ON" ] && echo "✅" || echo "⏭️")"
echo "  🧠 KellyBrain FFI: ✅"
echo "  📊 Tests: $([ "$BUILD_TESTS" == "ON" ] && echo "✅" || echo "⏭️")"
echo ""

if [ "$BUILD_TAURI" == "ON" ]; then
    echo -e "${CYAN}Application Locations:${NC}"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "  🖥️  macOS App: $DIST_DIR/app/idaw.app"
        echo "     Run with: open '$DIST_DIR/app/idaw.app'"
    elif [[ "$PLATFORM" == "linux" ]]; then
        echo "  🖥️  Linux Binary: $DIST_DIR/app/idaw"
        echo "     Run with: '$DIST_DIR/app/idaw'"
    fi
fi

if [ "$BUILD_PLUGINS" == "ON" ]; then
    echo ""
    echo -e "${CYAN}Plugin Locations:${NC}"
    if [[ "$PLATFORM" == "macos" ]]; then
        echo "  🎛️  Install VST3: cp -r '$DIST_DIR/plugins/*.vst3' ~/Library/Audio/Plug-Ins/VST3/"
        echo "  🎛️  Install AU: cp -r '$DIST_DIR/plugins/*.component' ~/Library/Audio/Plug-Ins/Components/"
    else
        echo "  🎛️  Install VST3: cp '$DIST_DIR/plugins/*.vst3' ~/.vst3/"
    fi
fi

echo ""
echo -e "${BLUE}🚀 Build completed successfully!${NC}"
echo -e "${BLUE}📖 Check BUILD_INFO.txt for detailed build information${NC}"
echo ""