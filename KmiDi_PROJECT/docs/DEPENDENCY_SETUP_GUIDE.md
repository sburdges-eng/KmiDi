# Dependency Setup Guide for Critical & High Priority Features

This guide provides step-by-step instructions for setting up all dependencies required for implementing the critical and high-priority features.

---

## Table of Contents
1. [Required Dependencies](#required-dependencies)
2. [CMake Configuration](#cmake-configuration)
3. [Library Installation](#library-installation)
4. [Verification](#verification)

---

## Required Dependencies

### Already Available
- ✅ **JUCE Framework** (`external/JUCE/`) - GUI, audio, MIDI, OSC
- ✅ **CMake** 3.27+ - Build system
- ✅ **C++20 Compiler** - GCC 11+, Clang 13+, MSVC 2019+

### Need to Add
- 🔴 **readerwriterqueue** - Lock-free queue (header-only)
- 🟡 **JUCE OSC Module** - May need explicit linking
- 🟡 **JUCE DSP Module** - For FFT in GrooveEngine

---

## CMake Configuration

### Step 1: Add readerwriterqueue via FetchContent

Add to `src_penta-core/CMakeLists.txt` or root `CMakeLists.txt`:

```cmake
# Fetch readerwriterqueue library
include(FetchContent)

FetchContent_Declare(
    readerwriterqueue
    GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
    GIT_TAG v1.0.6
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(readerwriterqueue)

# Create interface library for easy linking
if(NOT TARGET readerwriterqueue)
    add_library(readerwriterqueue INTERFACE)
    target_include_directories(readerwriterqueue INTERFACE
        ${readerwriterqueue_SOURCE_DIR}
    )
endif()
```

### Step 2: Update penta_core Target

In `src_penta-core/CMakeLists.txt`, update the `target_link_libraries`:

```cmake
target_link_libraries(penta_core PUBLIC
    oscpack  # Remove if using JUCE OSC instead
    readerwriterqueue  # ADD THIS
    juce::juce_osc      # ADD THIS for OSC
    juce::juce_dsp      # ADD THIS for FFT (GrooveEngine)
    juce::juce_audio_devices  # Already present for MIDI
)
```

### Step 3: Verify JUCE Modules

Ensure JUCE modules are properly configured. Check `external/JUCE/CMakeLists.txt` or your JUCE setup:

```cmake
# JUCE modules should include:
# - juce_osc (OSC communication)
# - juce_dsp (FFT, DSP utilities)
# - juce_audio_devices (MIDI I/O)
```

---

## Library Installation

### macOS

```bash
# No system packages needed - all dependencies via CMake FetchContent
# JUCE is already in external/JUCE/

# Verify CMake can find everything:
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

### Linux (Ubuntu/Debian)

```bash
# Install build tools if needed
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git

# No additional packages needed - CMake FetchContent handles readerwriterqueue
# JUCE is already in external/JUCE/
```

### Windows

```powershell
# Install vcpkg (optional, if not using FetchContent)
# Or use CMake FetchContent (recommended)

# Verify CMake:
cmake --version  # Should be 3.27+
```

---

## Verification

### Step 1: Build Configuration

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Check for these in CMake output:
# - readerwriterqueue found/configured
# - JUCE modules: juce_osc, juce_dsp, juce_audio_devices
```

### Step 2: Build Test

```bash
# Build penta_core library
cmake --build . --target penta_core

# Should compile without errors
```

### Step 3: Verify Includes

Create a test file to verify includes work:

```cpp
// test_includes.cpp
#include "readerwriterqueue.h"  // Should work
#include <juce_osc/juce_osc.h>  // Should work
#include <juce_dsp/juce_dsp.h>  // Should work
#include "penta/osc/RTMessageQueue.h"
#include "penta/osc/OSCClient.h"
#include "penta/osc/OSCServer.h"

int main() {
    return 0;
}
```

Compile test:
```bash
g++ -std=c++20 test_includes.cpp -I/path/to/readerwriterqueue -I/path/to/JUCE/modules -o test_includes
./test_includes  # Should compile and run
```

---

## Troubleshooting

### Issue: readerwriterqueue not found

**Solution**: Ensure FetchContent is called before `target_link_libraries`:

```cmake
# Correct order:
include(FetchContent)
FetchContent_Declare(...)
FetchContent_MakeAvailable(...)

# Then later:
target_link_libraries(penta_core PUBLIC readerwriterqueue)
```

### Issue: JUCE OSC module not found

**Solution**: Verify JUCE is properly configured:

```cmake
# Check that JUCE is added as subdirectory:
add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)

# Or if using find_package:
find_package(JUCE REQUIRED)
```

### Issue: Compiler errors with readerwriterqueue

**Solution**: Ensure C++20 standard:

```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

### Issue: Link errors with JUCE modules

**Solution**: Ensure all JUCE modules are linked:

```cmake
target_link_libraries(penta_core PUBLIC
    juce::juce_osc
    juce::juce_dsp
    juce::juce_audio_devices
    juce::juce_core  # Base JUCE module
)
```

---

## Next Steps

After dependencies are set up:

1. ✅ Verify all includes compile
2. ✅ Build penta_core library successfully
3. ✅ Run existing tests to ensure nothing broke
4. 📝 Proceed with implementation per `IMPLEMENTATION_PLANS_Critical_High_Priority.md`

---

## References

- [readerwriterqueue GitHub](https://github.com/cameron314/readerwriterqueue)
- [JUCE Documentation](https://docs.juce.com/)
- [CMake FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
- [Implementation Plans](./IMPLEMENTATION_PLANS_Critical_High_Priority.md)

