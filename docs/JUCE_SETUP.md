# JUCE Setup Documentation

## Overview

This project uses JUCE 7.0.12 for audio plugin development and GUI components. JUCE is included as a git subdirectory in `external/JUCE/`.

## JUCE Version

- **Version:** 7.0.12
- **Location:** `external/JUCE/`
- **Installation Method:** Git subdirectory (cloned from JUCE GitHub repository)
- **Last Updated:** December 2024

## Required JUCE Modules

The following JUCE modules are used by this project:

- `juce_audio_basics` - Core audio functionality
- `juce_audio_devices` - Audio device management
- `juce_audio_formats` - Audio file format support
- `juce_audio_processors` - Audio processing framework
- `juce_audio_plugin_client` - Plugin format support (VST3, CLAP, AU)
- `juce_core` - Core utilities and data structures
- `juce_data_structures` - Advanced data structures
- `juce_events` - Event handling system
- `juce_graphics` - Graphics and rendering
- `juce_gui_basics` - GUI components
- `juce_osc` - OSC (Open Sound Control) communication

## Setup Verification

All required modules have been verified and are present:

```bash
# Verify JUCE installation
ls external/JUCE/modules/

# Check specific modules
ls external/JUCE/modules/juce_audio_processors/
ls external/JUCE/modules/juce_gui_basics/
```

## Build Integration

JUCE is integrated via CMake in `CMakeLists.txt`:

```cmake
# JUCE setup
add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)
```

Targets are linked using the modern JUCE CMake API:

```cmake
target_link_libraries(KellyCore PUBLIC
    juce::juce_audio_basics
    juce::juce_audio_devices
    # ... other modules
)
```

## Platform-Specific Notes

### macOS 15.0 Compatibility

JUCE 7.0.12 includes a patch for macOS 15.0 API deprecations:
- `CGWindowListCreateImage` is unavailable in macOS 15.0+
- Patch applied in `external/JUCE/modules/juce_gui_basics/native/juce_Windowing_mac.mm`
- The code gracefully handles the unavailable API by returning an empty Image

### Dependencies

- **Qt6:** Required for KellyCore library (install via `brew install qt6`)
- **CMake:** Minimum version 3.27
- **C++ Standard:** C++20

## Building with JUCE

### Basic Build

```bash
# Configure
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build

# Build
cmake --build build
```

### Plugin Build

```bash
# Configure with plugins enabled
CMAKE_PREFIX_PATH="$(brew --prefix qt6)" cmake -B build -DBUILD_PLUGINS=ON

# Build VST3 plugin
cmake --build build --target KellyPlugin_VST3

# Build CLAP plugin
cmake --build build --target KellyPlugin_CLAP
```

## Troubleshooting

### JUCE Not Found

If CMake can't find JUCE:

1. Verify JUCE is present:
   ```bash
   ls external/JUCE/CMakeLists.txt
   ```

2. If missing, clone JUCE:
   ```bash
   cd external
   git clone --depth 1 --branch 7.0.12 https://github.com/juce-framework/JUCE.git JUCE
   ```

### Build Errors with juceaide

If you see errors building `juceaide`:

- This is a JUCE build tool that's built during configuration
- The macOS 15.0 compatibility patch should resolve most issues
- If problems persist, check that you're using JUCE 7.0.12 or later

### Module Not Found Errors

If you see "module not found" errors:

1. Verify the module exists:
   ```bash
   ls external/JUCE/modules/juce_<module_name>/
   ```

2. Check CMakeLists.txt links the correct module:
   ```cmake
   target_link_libraries(YourTarget PUBLIC juce::juce_<module_name>)
   ```

## Updating JUCE

To update to a newer JUCE version:

```bash
cd external/JUCE
git fetch --tags
git checkout <new_version_tag>  # e.g., 7.0.13
```

**Note:** Always test thoroughly after updating JUCE, as API changes may require code updates.

## References

- [JUCE Documentation](https://juce.com/learn/documentation)
- [JUCE GitHub Repository](https://github.com/juce-framework/JUCE)
- [JUCE CMake API](https://github.com/juce-framework/JUCE/blob/master/docs/CMake%20API.md)
