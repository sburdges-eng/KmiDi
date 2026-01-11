# Kelly - Build Instructions

## Prerequisites

### All Platforms

- CMake 3.27 or higher
- C++20 compatible compiler
- Git (for fetching dependencies)
- Qt6 (for KellyCore library)

### Platform-Specific

#### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake and Qt6 (via Homebrew)
brew install cmake qt6
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install build-essential cmake qt6-base-dev git
```

#### Windows
```powershell
# Install Visual Studio 2019 or later with C++ tools
# Install CMake from cmake.org
# Install Qt6 from qt.io or via vcpkg
```

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd kelly
```

### 2. Build Kelly Project

```bash
# Create build directory
mkdir build && cd build

# Configure (Release build with all features)
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PLUGINS=ON \
    -DBUILD_TESTS=ON

# Build (use -j for parallel compilation)
cmake --build . --config Release -j8

# Run tests
ctest --output-on-failure
```

### 3. Install

```bash
# Install to system or custom prefix
cmake --install . --prefix /usr/local
```

### 4. Run Application

```bash
# Run Kelly GUI application
./build/KellyApp
```

## Build Options

### Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Debug | Build type: Debug, Release, RelWithDebInfo |
| `BUILD_PLUGINS` | ON | Build JUCE VST3 and CLAP plugins |
| `BUILD_TESTS` | ON | Build unit tests |
| `ENABLE_TRACY` | OFF | Enable Tracy profiling |

### Example Configurations

#### Development Build
```bash
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TESTS=ON \
    -DBUILD_PLUGINS=OFF
```

#### Release Build
```bash
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PLUGINS=ON \
    -DBUILD_TESTS=ON
```

#### Application-Only Build
```bash
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build \
    -DBUILD_PLUGINS=OFF \
    -DBUILD_TESTS=OFF
```

#### Plugin-Only Build
```bash
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build \
    -DBUILD_PLUGINS=ON \
    -DBUILD_TESTS=OFF
```

## Advanced Configuration

### Custom Install Prefix

```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --install build
```

### Cross-Compilation

#### macOS Universal Binary (x86_64 + ARM64)
```bash
cmake -B build -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"
cmake --build build
```

#### Linux ARM64
```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=arm64-toolchain.cmake
cmake --build build
```

### Using System Libraries

JUCE is included as a git subdirectory in `external/JUCE/`. To use a system-installed JUCE:

```bash
# Install dependencies first
# macOS
brew install qt6

# Linux
sudo apt install qt6-base-dev

# JUCE is included in the repository, no additional installation needed
```

## Building Specific Targets

### Core Library Only
```bash
cmake --build build --target KellyCore
```

### GUI Application Only
```bash
cmake --build build --target KellyApp
```

### JUCE Plugin Only
```bash
cmake --build build --target KellyPlugin_VST3
cmake --build build --target KellyPlugin_CLAP
```

### Tests Only
```bash
cmake --build build --target KellyTests
cd build && ctest --output-on-failure
```

## Plugin Installation

### macOS

```bash
# VST3
cp -r build/KellyPlugin_artefacts/Release/VST3/KellyPlugin.vst3 \
      ~/Library/Audio/Plug-Ins/VST3/

# CLAP
cp -r build/KellyPlugin_artefacts/Release/CLAP/KellyPlugin.clap \
      ~/Library/Audio/Plug-Ins/CLAP/
```

### Linux

```bash
# VST3
cp -r build/KellyPlugin_artefacts/Release/VST3/KellyPlugin.vst3 \
      ~/.vst3/

# CLAP
cp -r build/KellyPlugin_artefacts/Release/CLAP/KellyPlugin.clap \
      ~/.clap/
```

### Windows

```powershell
# VST3
copy build\KellyPlugin_artefacts\Release\VST3\KellyPlugin.vst3 ^
     %CommonProgramFiles%\VST3\

# CLAP
copy build\KellyPlugin_artefacts\Release\CLAP\KellyPlugin.clap ^
     %CommonProgramFiles%\CLAP\
```

## Troubleshooting

### CMake Can't Find Qt6

```bash
# macOS: Specify Qt6 path explicitly
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build

# Linux: Install Qt6 development packages
sudo apt install qt6-base-dev

# Or specify Qt6_DIR manually
cmake -B build -DQt6_DIR=/path/to/qt6/lib/cmake/Qt6
```

### JUCE Build Errors

**JUCE Setup Status (Updated Dec 2024):**
- JUCE 7.0.12 is installed at `external/JUCE/`
- All required modules are present and verified
- macOS 15.0 compatibility patch applied
- Qt6 dependency installed via Homebrew

**If JUCE is missing:**
```bash
# JUCE is included as a git subdirectory
# If missing, it should be cloned automatically, or manually:
cd external
git clone --depth 1 --branch 7.0.12 https://github.com/juce-framework/JUCE.git JUCE
```

**If you see macOS 15.0 API errors:**
The project includes a patch for `CGWindowListCreateImage` deprecation in `external/JUCE/modules/juce_gui_basics/native/juce_Windowing_mac.mm`. This is already applied.

**Qt6 Setup:**
```bash
# Install Qt6 (required for KellyCore)
brew install qt6

# Configure build with Qt6 path
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build
```

### Link Errors on Linux

```bash
# Install missing libraries for JUCE
sudo apt install libasound2-dev libfreetype6-dev libx11-dev \
                 libxrandr-dev libxinerama-dev libxcursor-dev \
                 libxcomposite-dev libxcursor-dev libxext-dev
```

## Development Workflow

### Incremental Builds

```bash
# Make changes to source files
# Rebuild only changed files
cmake --build build

# Rebuild specific target
cmake --build build --target KellyCore
cmake --build build --target KellyApp
```

### Clean Build

```bash
# Clean build artifacts
cmake --build build --target clean

# Or remove build directory entirely
rm -rf build
cmake -B build
cmake --build build
```

### Rebuilding After CMakeLists.txt Changes

```bash
# Reconfigure
cmake -B build

# Rebuild
cmake --build build
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Install dependencies (macOS)
      if: runner.os == 'macOS'
      run: |
        brew install qt6 cmake
    
    - name: Install dependencies (Ubuntu)
      if: runner.os == 'Linux'
      run: |
        sudo apt update
        sudo apt install build-essential cmake qt6-base-dev
    
    - name: Configure
      run: |
        if [ "${{ runner.os }}" == "macOS" ]; then
          CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build -DCMAKE_BUILD_TYPE=Release
        else
          cmake -B build -DCMAKE_BUILD_TYPE=Release
        fi
    
    - name: Build
      run: cmake --build build --config Release
    
    - name: Test
      run: cd build && ctest --output-on-failure
```

## Performance Validation

### Release Build

```bash
# Build with optimizations
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PLUGINS=ON

cmake --build build --config Release
```

### Profile-Guided Optimization (Advanced)

```bash
# Step 1: Build with instrumentation
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build-pgo \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fprofile-generate"
cmake --build build-pgo

# Step 2: Run representative workload
./build-pgo/KellyApp

# Step 3: Rebuild with profile data
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-fprofile-use"
cmake --build build
```

## Documentation Generation

### Doxygen (C++ API)

```bash
# Install Doxygen
brew install doxygen  # macOS
sudo apt install doxygen  # Linux

# Generate docs (if Doxyfile exists)
doxygen Doxyfile

# Open docs
open docs/html/index.html
```

## Support

For build issues, please:
1. Check this document first
2. Search existing GitHub issues
3. Create a new issue with:
   - OS and version
   - CMake output
   - Compiler version
   - Full error messages

## Development Progress

### Current Status

The Kelly project is actively under development with the following components:

#### ✅ Core Library (Complete)
- **KellyCore:** Static library with emotion engine, thesaurus, groove templates, chord diagnostics, MIDI pipeline, and intent processor
- **JUCE Integration:** All required JUCE modules integrated and tested
- **Qt6 Integration:** GUI framework integrated for KellyApp

#### ✅ Build System (Complete)
- **CMake Configuration:** Fully configured with JUCE 7.0.12
- **Plugin Support:** VST3 and CLAP plugin formats supported
- **Testing Framework:** Catch2 integration for unit tests
- **macOS 15.0 Compatibility:** Patched for latest macOS APIs

#### 🚧 GUI Application (In Progress)
- **KellyApp:** Qt6-based GUI application
- Basic window structure implemented

#### 📋 Future Development
- Enhanced plugin functionality
- Additional emotion processing features
- Advanced MIDI manipulation
- Real-time audio processing

See project documentation for detailed roadmap.

---

## Next Steps

After successful build:
1. Run the Kelly GUI application: `./build/KellyApp`
2. Build and test plugins: `cmake --build build --target KellyPlugin_VST3`
3. Run unit tests: `cd build && ctest --output-on-failure`
4. Explore source code in `src/` directory
5. Check `JUCE_SETUP.md` for JUCE-specific documentation
