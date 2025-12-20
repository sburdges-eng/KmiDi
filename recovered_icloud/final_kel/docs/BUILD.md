# Build Guide - Kelly MIDI Companion

Complete guide for building Kelly MIDI Companion from source.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Build Instructions](#detailed-build-instructions)
- [Build Configurations](#build-configurations)
- [Component Builds](#component-builds)
- [Troubleshooting](#troubleshooting)
- [Build Options Reference](#build-options-reference)

## Prerequisites

### System Requirements

- **Python**: 3.9 or later (3.11 recommended)
- **CMake**: 3.22 or later
- **C++ Compiler**: C++20 compatible (clang++, g++, or MSVC)
- **Build Tools**: Ninja (recommended for faster builds) or make
- **Git**: For fetching dependencies

### Platform-Specific Setup

#### macOS

```bash
# Install Xcode Command Line Tools (includes clang++)
xcode-select --install

# Install CMake and Ninja via Homebrew
brew install cmake ninja python@3.11

# Verify installation
python3 --version  # Should be 3.9+
cmake --version    # Should be 3.22+
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3-dev python3-pip git

# Verify installation
python3 --version
cmake --version
```

#### Windows

```powershell
# Install Visual Studio 2022 or later with C++ Desktop Development workload
# Install CMake from cmake.org or via chocolatey
choco install cmake ninja python3 -y

# Verify installation
python --version
cmake --version
```

### Python Dependencies

The project uses multiple Python packages. Install base dependencies:

```bash
# From project root
pip install -r requirements.txt

# Or install the main package with optional dependencies
pip install -e ".[dev]"  # Includes dev tools (pytest, black, mypy, etc.)
pip install -e ".[ml]"   # Includes ML dependencies (torch, etc.)
pip install -e ".[all]"  # All optional dependencies
```

Component-specific dependencies:

- `ml_framework/requirements.txt` - ML framework dependencies
- `python/requirements.txt` - Python utilities dependencies

## Quick Start

### Automated Build

```bash
# Full build with tests
./build_all.sh --clean --test

# Quick build (no tests, faster)
./build_quick.sh

# Build with Python bridge
./build_all.sh --python-bridge --test
```

### Manual Build

```bash
# 1. Setup environment (optional, creates venvs)
./setup_workspace.sh

# 2. Check dependencies
./scripts/check_dependencies.sh

# 3. Configure CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 4. Build
cmake --build build --config Release -j$(nproc)

# 5. Run tests (optional)
cd build && ctest --output-on-failure
```

## Detailed Build Instructions

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd "final kel"
```

### Step 2: Setup Python Environment (Recommended)

```bash
# Run the setup script
./setup_workspace.sh

# Or manually create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Step 3: Check Dependencies

```bash
./scripts/check_dependencies.sh
```

This verifies:

- Python version (>=3.9)
- CMake version (>=3.22)
- C++ compiler availability
- Optional tools (Ninja, Git)

### Step 4: Configure Build

```bash
# Basic Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# With all options
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BRIDGE=ON \
  -DBUILD_TESTS=ON \
  -DENABLE_RTNEURAL=ON

# Debug build with tests
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
```

### Step 5: Build

```bash
# Using Ninja (faster)
cmake --build build --config Release -j$(nproc)

# Using make
cmake --build build --config Release -j$(nproc)
```

On macOS/Linux, `nproc` returns CPU count. On macOS, use `sysctl -n hw.ncpu`. On Windows, omit `-j` flag or use `-j4`.

### Step 6: Verify Build

```bash
# Check build artifacts
ls -la build/KellyMidiCompanion_artefacts/Release/

# Run tests
cd build && ctest --output-on-failure
```

## Build Configurations

### Build Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Release** | Optimized build, no debug symbols | Production use |
| **Debug** | Debug symbols, no optimization | Development, debugging |
| **RelWithDebInfo** | Optimized with debug symbols | Performance testing |
| **MinSizeRel** | Optimized for size | Embedded/distribution |

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug/Release/etc.) |
| `BUILD_PYTHON_BRIDGE` | OFF | Build Python-C++ bridge module |
| `BUILD_TESTS` | ON | Build C++ unit tests |
| `ENABLE_RTNEURAL` | ON | Enable RTNeural ML inference |

### Example Build Commands

```bash
# Development build (Debug with tests)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build --config Debug -j$(nproc)

# Production build (Release, optimized)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build build --config Release -j$(nproc)

# Full build with Python bridge
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BRIDGE=ON \
  -DENABLE_RTNEURAL=ON
cmake --build build --config Release -j$(nproc)
```

## Component Builds

### Main Plugin (JUCE)

The main JUCE plugin is built by default:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Output: build/KellyMidiCompanion_artefacts/Release/
#   - VST3/ (macOS/Linux/Windows)
#   - AU/ (macOS only)
#   - Standalone/ (Standalone app)
```

### Penta-Core Library

Build the Penta-Core C++ library separately:

```bash
cmake -B build_penta-core -S src_penta-core -DCMAKE_BUILD_TYPE=Release
cmake --build build_penta-core --config Release

# Output: build_penta-core/libpenta_core.a (static library)
```

### Python-C++ Bridge

Build the Python bridge module (requires pybind11):

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BRIDGE=ON
cmake --build build --config Release

# Output: python/kelly_bridge.so (Linux/macOS) or kelly_bridge.pyd (Windows)
```

### Python Packages

Install Python packages in development mode:

```bash
# Main package (kelly)
pip install -e ".[dev]"

# ML framework
cd ml_framework && pip install -r requirements.txt && cd ..

# Python utilities
cd python && pip install -r requirements.txt && cd ..
```

## Build Order

1. **Dependencies** - Install system dependencies (CMake, Python, compiler)
2. **Python Environment** - Set up virtual environments, install Python packages
3. **External Libraries** - CMake will auto-fetch JUCE, RTNeural, pybind11, Google Test
4. **Penta-Core** - Build C++ library first (if building separately)
5. **Main Plugin** - Build JUCE plugin
6. **Python Bindings** - Build Python-C++ bridge (if enabled)
7. **Tests** - Build and run test suites

## Testing

### C++ Tests

```bash
# Build with tests enabled
cmake -B build -DBUILD_TESTS=ON
cmake --build build --config Release

# Run tests
cd build
ctest --output-on-failure

# Run specific test
./tests/KellyTests --gtest_filter="TestSuite.TestName"
```

### Python Tests

```bash
# Activate environment
source venv/bin/activate

# Run all tests
pytest tests_music-brain/ -v

# Run specific test
pytest tests_music-brain/test_groove.py -v -k "test_extract"

# With coverage
pytest tests_music-brain/ --cov=music_brain --cov-report=html
```

### ML Framework Tests

```bash
cd ml_framework
pytest tests/ -v
```

## Troubleshooting

### CMake Configuration Fails

**Issue**: CMake can't find dependencies

**Solutions**:

- Verify CMake version: `cmake --version` (needs 3.22+)
- Check compiler: `clang++ --version` or `g++ --version`
- Clear CMake cache: `rm -rf build CMakeCache.txt`

### Build Fails with Compiler Errors

**Issue**: C++20 features not recognized

**Solutions**:

- Verify C++ standard in CMakeLists.txt (should be C++20)
- Update compiler to C++20 compatible version
- Check compiler flags: `cmake -B build -DCMAKE_CXX_FLAGS="-std=c++20"`

### JUCE Not Found

**Issue**: CMake can't find JUCE

**Solutions**:

- CMake will auto-fetch JUCE if not in `external/JUCE/`
- Check network connection (GitHub access required)
- Manually clone: `git clone https://github.com/juce-framework/JUCE.git external/JUCE`

### Python Import Errors

**Issue**: Python can't import modules after install

**Solutions**:

- Verify virtual environment is activated: `which python` (should point to venv)
- Reinstall in editable mode: `pip install -e ".[dev]"`
- Check PYTHONPATH: `echo $PYTHONPATH`
- Verify package structure: `pip list | grep kelly`

### Tests Fail

**Issue**: C++ or Python tests fail

**Solutions**:

- Run tests with verbose output: `ctest --output-on-failure -V`
- Check test prerequisites (test data files, etc.)
- Verify build configuration matches test expectations
- Check for missing dependencies in test environment

### Build is Slow

**Solutions**:

- Use Ninja generator: `cmake -B build -G Ninja`
- Increase parallel jobs: `cmake --build build -j$(nproc)`
- Use ccache (if available): `export CMAKE_CXX_COMPILER_LAUNCHER=ccache`
- Build only necessary components: `cmake --build build --target KellyMidiCompanion`

### Windows-Specific Issues

**Issue**: Path issues or command not found

**Solutions**:

- Use PowerShell or Git Bash (not cmd.exe)
- Use forward slashes in paths: `/path/to/project`
- Install Visual Studio Build Tools (not just runtime)
- Use `cmake --build build --config Release` (not `make`)

## Build Options Reference

### CMake Variables

```bash
# Build type
-DCMAKE_BUILD_TYPE=Release          # Release, Debug, RelWithDebInfo, MinSizeRel

# Component options
-DBUILD_PYTHON_BRIDGE=ON            # Enable Python bridge
-DBUILD_TESTS=ON                    # Build tests
-DENABLE_RTNEURAL=ON                # Enable RTNeural ML inference

# Generator (optional)
-G Ninja                            # Use Ninja generator (faster)
-G "Unix Makefiles"                 # Use Make generator
-G Xcode                            # Generate Xcode project (macOS)

# Toolchain (optional)
-DCMAKE_C_COMPILER=clang            # Specify C compiler
-DCMAKE_CXX_COMPILER=clang++        # Specify C++ compiler

# Installation prefix (optional)
-DCMAKE_INSTALL_PREFIX=/usr/local   # Installation directory
```

### Environment Variables

```bash
# Build configuration
export CMAKE_BUILD_TYPE=Release
export BUILD_PYTHON_BRIDGE=ON
export BUILD_TESTS=ON
export ENABLE_RTNEURAL=ON

# Parallel jobs
export CMAKE_BUILD_PARALLEL_LEVEL=8  # Number of parallel compile jobs

# Compiler flags (optional)
export CXXFLAGS="-O3 -march=native"  # Optimization flags
```

## Next Steps

After building successfully:

1. **Install Plugin**: Copy VST3/AU files to your DAW's plugin directory
   - macOS: `~/Library/Audio/Plug-Ins/VST3/` or `~/Library/Audio/Plug-Ins/Components/`
   - Linux: `~/.vst3/` or system-wide `/usr/lib/vst3/`
   - Windows: `C:\Program Files\Common Files\VST3\`

2. **Verify Installation**: Open your DAW and check plugin list

3. **Run Examples**: See `examples/` directory for usage examples

4. **Read Documentation**: See `DEVELOPMENT.md` for development workflow

5. **Contribute**: See `CONTRIBUTING.md` for contribution guidelines

## Additional Resources

- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup and workflow
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [Project README](README.md) - Project overview
- [CI/CD Workflows](.github/workflows/) - Automated build configurations
