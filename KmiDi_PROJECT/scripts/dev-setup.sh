#!/bin/bash

# =============================================================================
# KmiDi Development Environment Setup Script
# =============================================================================
# This script sets up a complete development environment for KmiDi:
# - Installs system dependencies (CMake, Rust, Node.js, Python)
# - Configures build environment
# - Installs language-specific dependencies
# - Sets up IDE configuration files
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
echo "║     ██╗  ██╗███╗   ███╗██╗██████╗ ██╗    ██████╗ ███████╗██╗   ██╗     ║"
echo "║     ██║ ██╔╝████╗ ████║██║██╔══██╗██║    ██╔══██╗██╔════╝██║   ██║     ║"
echo "║     █████╔╝ ██╔████╔██║██║██║  ██║██║    ██║  ██║█████╗  ██║   ██║     ║"
echo "║     ██╔═██╗ ██║╚██╔╝██║██║██║  ██║██║    ██║  ██║██╔══╝  ╚██╗ ██╔╝     ║"
echo "║     ██║  ██╗██║ ╚═╝ ██║██║██████╔╝██║    ██████╔╝███████╗ ╚████╔╝      ║"
echo "║     ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═════╝ ╚═╝    ╚═════╝ ╚══════╝  ╚═══╝       ║"
echo "║                                                                  ║"
echo "║              Development Environment Setup                      ║"
echo "║                        Platform: $PLATFORM                             ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

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

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_and_install_command() {
    local cmd="$1"
    local install_instructions="$2"
    
    if command -v "$cmd" >/dev/null 2>&1; then
        local version=$(${cmd} --version 2>/dev/null | head -n1 || echo "unknown version")
        log_success "$cmd is installed ($version)"
        return 0
    else
        log_info "$cmd not found. Installing..."
        eval "$install_instructions"
        
        if command -v "$cmd" >/dev/null 2>&1; then
            log_success "$cmd installed successfully"
            return 0
        else
            log_error "Failed to install $cmd"
            echo "Manual installation required: $install_instructions"
            return 1
        fi
    fi
}

# =============================================================================
# Step 1: Install System Dependencies
# =============================================================================

log_step "Installing system dependencies..."

if [[ "$PLATFORM" == "macos" ]]; then
    # Check for Homebrew
    if ! command -v brew >/dev/null 2>&1; then
        log_info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install dependencies via Homebrew
    check_and_install_command "cmake" "brew install cmake"
    check_and_install_command "ninja" "brew install ninja"
    check_and_install_command "node" "brew install node"
    check_and_install_command "python3" "brew install python"
    check_and_install_command "rustc" "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env"
    
    # Check for Xcode Command Line Tools
    if ! xcode-select -p >/dev/null 2>&1; then
        log_info "Installing Xcode Command Line Tools..."
        xcode-select --install
        echo "Please complete Xcode Command Line Tools installation and run this script again"
        exit 1
    fi
    
    log_success "Xcode Command Line Tools available"

elif [[ "$PLATFORM" == "linux" ]]; then
    # Detect package manager
    if command -v apt-get >/dev/null 2>&1; then
        PKG_MANAGER="apt-get"
        PKG_UPDATE="sudo apt-get update"
        PKG_INSTALL="sudo apt-get install -y"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
        PKG_UPDATE="sudo dnf check-update"
        PKG_INSTALL="sudo dnf install -y"
    elif command -v pacman >/dev/null 2>&1; then
        PKG_MANAGER="pacman"
        PKG_UPDATE="sudo pacman -Sy"
        PKG_INSTALL="sudo pacman -S --noconfirm"
    else
        log_error "Unsupported Linux distribution (no recognized package manager)"
        exit 1
    fi
    
    log_info "Using package manager: $PKG_MANAGER"
    $PKG_UPDATE
    
    # Install dependencies
    check_and_install_command "cmake" "$PKG_INSTALL cmake"
    check_and_install_command "ninja" "$PKG_INSTALL ninja-build"
    check_and_install_command "node" "$PKG_INSTALL nodejs npm"
    check_and_install_command "python3" "$PKG_INSTALL python3 python3-pip"
    check_and_install_command "rustc" "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source ~/.cargo/env"
    
    # Install build tools
    if [[ "$PKG_MANAGER" == "apt-get" ]]; then
        $PKG_INSTALL build-essential libasound2-dev libjack-jackd2-dev
    elif [[ "$PKG_MANAGER" == "dnf" ]]; then
        $PKG_INSTALL gcc gcc-c++ alsa-lib-devel jack-audio-connection-kit-devel
    elif [[ "$PKG_MANAGER" == "pacman" ]]; then
        $PKG_INSTALL base-devel alsa-lib jack2
    fi

else
    log_error "Unsupported platform: $PLATFORM"
    echo "Please manually install: CMake, Node.js, Rust, Python 3, and C++ compiler"
    exit 1
fi

log_success "System dependencies installed"

# =============================================================================
# Step 2: Install Rust Tools
# =============================================================================

log_step "Installing Rust development tools..."

# Ensure Rust is in PATH
source ~/.cargo/env 2>/dev/null || true

# Install required Rust components
rustup component add rustfmt clippy

# Install Tauri CLI
if ! cargo install --list | grep -q "tauri-cli"; then
    log_info "Installing Tauri CLI..."
    cargo install tauri-cli
fi

# Install cargo-watch for development
if ! cargo install --list | grep -q "cargo-watch"; then
    log_info "Installing cargo-watch..."
    cargo install cargo-watch
fi

log_success "Rust development tools installed"

# =============================================================================
# Step 3: Install Node.js Dependencies
# =============================================================================

log_step "Installing Node.js dependencies..."

cd "$PROJECT_ROOT"

# Update npm to latest version
npm install -g npm@latest

# Install project dependencies
npm install

# Install global development tools
npm install -g concurrently

log_success "Node.js dependencies installed"

# =============================================================================
# Step 4: Install Python Dependencies
# =============================================================================

log_step "Installing Python dependencies..."

cd "$PROJECT_ROOT"

# Create virtual environment if requirements.txt exists
if [ -f "requirements.txt" ]; then
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    log_success "Python dependencies installed in virtual environment"
else
    log_info "No requirements.txt found - skipping Python dependencies"
fi

# Install pybind11 for C++ bindings
pip3 install pybind11

# =============================================================================
# Step 5: Configure Build Environment
# =============================================================================

log_step "Configuring build environment..."

cd "$PROJECT_ROOT"

# Create build directories
mkdir -p build/{debug,release}
mkdir -p src-tauri/resources
mkdir -p dist

# Create initial CMake configuration
cd build/debug
cmake ../../ \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_KELLY_CORE=ON \
    -DBUILD_KELLY_FFI=ON \
    -DBUILD_PLUGINS=ON \
    -DBUILD_TESTS=ON \
    -DENABLE_TRACY=OFF

cd ../release
cmake ../../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_KELLY_CORE=ON \
    -DBUILD_KELLY_FFI=ON \
    -DBUILD_PLUGINS=ON \
    -DBUILD_TESTS=OFF \
    -DENABLE_TRACY=OFF

cd "$PROJECT_ROOT"

log_success "Build environment configured"

# =============================================================================
# Step 6: Create IDE Configuration Files
# =============================================================================

log_step "Creating IDE configuration files..."

# Create .vscode/settings.json
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "typescript.preferences.includePackageJsonAutoImports": "auto",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit",
    "source.organizeImports": "explicit"
  },
  "rust-analyzer.cargo.features": "all",
  "rust-analyzer.checkOnSave.command": "clippy",
  "cmake.buildDirectory": "${workspaceFolder}/build/debug",
  "cmake.installPrefix": "${workspaceFolder}/build/install",
  "cmake.configureSettings": {
    "BUILD_KELLY_CORE": "ON",
    "BUILD_KELLY_FFI": "ON",
    "BUILD_PLUGINS": "ON",
    "BUILD_TESTS": "ON"
  },
  "files.associations": {
    "*.h": "c",
    "*.hpp": "cpp"
  },
  "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
  "C_Cpp.default.intelliSenseMode": "macos-clang-x64"
}
EOF

# Create .vscode/tasks.json
cat > .vscode/tasks.json << 'EOF'
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Build All (Debug)",
      "type": "shell",
      "command": "./scripts/build-all.sh",
      "options": {
        "env": {
          "BUILD_TYPE": "Debug",
          "BUILD_TESTS": "ON"
        }
      },
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Build All (Release)",
      "type": "shell",
      "command": "./scripts/build-all.sh",
      "options": {
        "env": {
          "BUILD_TYPE": "Release",
          "BUILD_TESTS": "OFF"
        }
      },
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Start Dev Servers",
      "type": "shell",
      "command": "npm run dev:all",
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Build C++ Only",
      "type": "shell",
      "command": "cd build/debug && make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)",
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "focus": false,
        "panel": "shared"
      }
    }
  ]
}
EOF

# Create .vscode/launch.json
cat > .vscode/launch.json << 'EOF'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Tauri App",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/node_modules/@tauri-apps/cli/bin/tauri.js",
      "args": ["dev"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal"
    },
    {
      "name": "Debug C++ Tests",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/build/debug/KellyTests",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "lldb",
      "miDebuggerPath": "/usr/bin/lldb",
      "preLaunchTask": "Build C++ Only"
    }
  ]
}
EOF

log_success "IDE configuration files created"

# =============================================================================
# Step 7: Create Development Scripts
# =============================================================================

log_step "Creating development scripts..."

# Create dev-cpp.sh for C++ development
cat > scripts/dev-cpp.sh << 'EOF'
#!/bin/bash
# Watch C++ files and rebuild on changes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔨 Starting C++ development watcher..."
echo "Watching: src/**/*.{cpp,h,hpp}"
echo "Build dir: $PROJECT_ROOT/build/debug"

# Initial build
cd "$PROJECT_ROOT/build/debug"
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Watch for changes
if command -v fswatch >/dev/null 2>&1; then
    # macOS/Unix with fswatch
    fswatch -o "$PROJECT_ROOT/src" | while read f; do
        echo "🔄 C++ files changed, rebuilding..."
        make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
        echo "✅ C++ rebuild complete"
    done
elif command -v inotifywait >/dev/null 2>&1; then
    # Linux with inotify-tools
    while inotifywait -r -e modify,create,delete "$PROJECT_ROOT/src" --include '\.(cpp|h|hpp)$'; do
        echo "🔄 C++ files changed, rebuilding..."
        make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
        echo "✅ C++ rebuild complete"
    done
else
    echo "⚠️  File watching not available. Manual rebuild required."
    echo "Install fswatch (macOS) or inotify-tools (Linux) for automatic rebuilding"
    echo "Manual rebuild: cd build/debug && make"
fi
EOF

chmod +x scripts/dev-cpp.sh

# Create dev-python.sh for Python development
cat > scripts/dev-python.sh << 'EOF'
#!/bin/bash
# Start Python Music Brain API server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🐍 Starting Python Music Brain API server..."

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "📦 Virtual environment activated"
fi

# Start the Music Brain API
if [ -f "music_brain/api.py" ]; then
    python -m music_brain.api
elif [ -f "scripts/brain_server.py" ]; then
    python scripts/brain_server.py
else
    echo "❌ Python API server not found"
    echo "Expected: music_brain/api.py or scripts/brain_server.py"
    exit 1
fi
EOF

chmod +x scripts/dev-python.sh

log_success "Development scripts created"

# =============================================================================
# Step 8: Test Build System
# =============================================================================

log_step "Testing build system..."

cd "$PROJECT_ROOT"

# Test C++ build
cd build/debug
log_info "Testing C++ build..."
if make -j2 KellyCore; then
    log_success "C++ core builds successfully"
else
    log_error "C++ build failed"
fi

# Test FFI library build
if make -j2 KellyFFI; then
    log_success "FFI library builds successfully"
else
    log_error "FFI library build failed"
fi

cd "$PROJECT_ROOT"

# Test Node.js build
log_info "Testing React build..."
if npm run build; then
    log_success "React build succeeds"
else
    log_error "React build failed"
fi

# Test basic Tauri configuration
log_info "Testing Tauri configuration..."
cd src-tauri
if cargo check; then
    log_success "Tauri configuration is valid"
else
    log_error "Tauri configuration issues found"
fi

cd "$PROJECT_ROOT"

log_success "Build system tests completed"

# =============================================================================
# Step 9: Create Environment Information
# =============================================================================

log_step "Creating environment information..."

cat > DEV_ENVIRONMENT.md << EOF
# KmiDi Development Environment

## Setup Summary

**Date:** $(date)
**Platform:** $PLATFORM
**Setup Script:** scripts/dev-setup.sh

## Installed Components

### System Tools
- CMake: $(cmake --version | head -n1)
- Node.js: $(node --version)
- npm: $(npm --version)
- Rust: $(rustc --version)
- Python: $(python3 --version)

### Rust Tools
- Tauri CLI: $(cargo tauri --version 2>/dev/null || echo "not installed")
- cargo-watch: $(cargo install --list | grep cargo-watch || echo "not installed")

### IDE Configuration
- .vscode/settings.json ✅
- .vscode/tasks.json ✅  
- .vscode/launch.json ✅

## Development Workflow

### Start Development Servers
\`\`\`bash
# All services (React + Tauri + Python + C++ watcher)
npm run dev:all

# Individual services
npm run dev          # React frontend only
./scripts/dev-cpp.sh # C++ rebuild watcher
./scripts/dev-python.sh # Python API server
\`\`\`

### Build Everything
\`\`\`bash
# Full build (all technologies)
./scripts/build-all.sh

# Individual builds
npm run build:cpp    # C++ only
npm run build        # React only  
npm run tauri build  # Tauri only
\`\`\`

### Testing
\`\`\`bash
# All tests
BUILD_TESTS=ON ./scripts/build-all.sh

# Individual test suites
cd build/debug && ctest  # C++ tests
cargo test              # Rust tests
npm test                # React tests (if configured)
\`\`\`

## Debugging

### C++ Debugging
- Use VS Code with C++ extension
- Debug target: build/debug/KellyTests
- Breakpoints work in C++ code

### Rust Debugging  
- Use VS Code with rust-analyzer
- Debug Tauri commands
- RUST_LOG=debug for detailed logging

### React Debugging
- Use browser dev tools
- React DevTools extension recommended
- Console logs for Tauri command calls

## Troubleshooting

### Build Issues
- Ensure all dependencies installed: ./scripts/dev-setup.sh
- Clean build: CLEAN_BUILD=true ./scripts/build-all.sh
- Check CMake cache: rm build/CMakeCache.txt && reconfigure

### Runtime Issues  
- Check library paths: echo \$DYLD_LIBRARY_PATH (macOS) or \$LD_LIBRARY_PATH (Linux)
- Verify FFI library: ls -la src-tauri/resources/
- Check Python API: curl http://localhost:8000/health

### Plugin Issues
- Check plugin installation paths
- Verify DAW plugin scanning
- Test with simple plugin host first

## Performance

### Profiling
- Enable Tracy: ENABLE_TRACY=ON in CMake
- Use Instruments on macOS
- Valgrind on Linux

### Optimization
- Release builds: BUILD_TYPE=Release
- Enable SIMD: DAIW_ENABLE_SIMD=ON
- Profile before optimizing

EOF

log_success "Environment documentation created"

# =============================================================================
# Development Environment Summary
# =============================================================================

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║             DEVELOPMENT ENVIRONMENT READY!                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}Next Steps:${NC}"
echo ""
echo -e "${BLUE}1. Start Development:${NC}"
echo "   cd $PROJECT_ROOT"
echo "   npm run dev:all"
echo ""
echo -e "${BLUE}2. Build Everything:${NC}"
echo "   ./scripts/build-all.sh"
echo ""
echo -e "${BLUE}3. Run Tests:${NC}"
echo "   BUILD_TESTS=ON ./scripts/build-all.sh"
echo ""
echo -e "${BLUE}4. Open in IDE:${NC}"
echo "   code $PROJECT_ROOT  # VS Code"
echo ""

echo -e "${CYAN}Documentation Created:${NC}"
echo "  📖 DEV_ENVIRONMENT.md - Complete development guide"
echo "  📁 .vscode/ - IDE configuration"
echo "  🔧 scripts/dev-*.sh - Development helper scripts"
echo ""

echo -e "${YELLOW}Important:${NC}"
echo "  • Source your shell: source ~/.cargo/env (for Rust)"
echo "  • Activate Python venv: source venv/bin/activate"
echo "  • Check DEV_ENVIRONMENT.md for detailed workflow"
echo ""

log_success "Development environment setup completed successfully!"