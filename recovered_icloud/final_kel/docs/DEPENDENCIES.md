# Dependencies for iDAW Track 2: File I/O & MIDI Foundation

## Summary

This document lists all dependencies required for the File I/O and MIDI Foundation components implemented in Track 2.

---

## Required Dependencies

### 1. Standard C++ Libraries

**Already Available:**
- C++17 or later standard library
- `<filesystem>` - For file path handling
- `<fstream>` - For basic file I/O
- `<vector>`, `<string>`, `<memory>` - Standard containers
- `<algorithm>`, `<functional>` - Standard algorithms

**Status:** ✅ Available in all modern C++ compilers

---

### 2. Audio File I/O: libsndfile (RECOMMENDED)

**Purpose:** Robust audio file reading/writing for WAV, AIFF, FLAC, and more

**Current Status:** 🟡 STUB IMPLEMENTATION
- Basic WAV reading/writing implemented without libsndfile
- Only supports 32-bit float WAV format
- No support for AIFF, FLAC, or other formats
- No robust header parsing or error handling

**For Production Use:**

#### Installation

**macOS:**
```bash
brew install libsndfile
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libsndfile1-dev
```

**Windows:**
```bash
# Via vcpkg
vcpkg install libsndfile

# Or download from: http://www.mega-nerd.com/libsndfile/
```

#### CMake Integration
```cmake
find_package(SndFile REQUIRED)
target_link_libraries(your_target PRIVATE SndFile::sndfile)
```

**Why libsndfile:**
- Industry standard for audio file I/O
- Supports 20+ audio formats
- Robust error handling
- Cross-platform
- Well-maintained

**Alternatives:**
- dr_wav (single-header, WAV only)
- AudioFile (header-only C++ library)
- JUCE AudioFormatManager (if using JUCE)

---

### 3. MIDI I/O: RtMidi (RECOMMENDED)

**Purpose:** Real-time MIDI input/output device management

**Current Status:** 🟡 STUB IMPLEMENTATION
- MidiMessage and MidiSequence classes fully implemented
- MidiIO interface defined but not functional
- No actual device communication

**For Production Use:**

#### Installation

**macOS:**
```bash
brew install rtmidi
```

**Ubuntu/Debian:**
```bash
sudo apt-get install librtmidi-dev
```

**Windows:**
```bash
# Via vcpkg
vcpkg install rtmidi

# Or build from source: https://github.com/thestk/rtmidi
```

#### CMake Integration
```cmake
find_package(RtMidi REQUIRED)
target_link_libraries(your_target PRIVATE RtMidi::rtmidi)
```

**Why RtMidi:**
- Cross-platform (Linux, macOS, Windows)
- Low-latency MIDI I/O
- Simple C++ API
- Active development
- MIT license

**Alternatives:**
- PortMidi (older, C API)
- JUCE MidiInput/MidiOutput (if using JUCE)
- Platform-specific APIs (CoreMIDI, ALSA, Windows MM)

**Note:** Python code already uses `mido` which is separate and functional.

---

### 4. Sample Rate Conversion: libsamplerate (OPTIONAL)

**Purpose:** High-quality sample rate conversion

**Current Status:** 🔴 NOT IMPLEMENTED
- AudioFile::convertSampleRate() is a stub
- Returns false always

**For Production Use:**

#### Installation

**macOS:**
```bash
brew install libsamplerate
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libsamplerate0-dev
```

**Windows:**
```bash
vcpkg install libsamplerate
```

#### CMake Integration
```cmake
find_package(SampleRate REQUIRED)
target_link_libraries(your_target PRIVATE SampleRate::samplerate)
```

**Alternatives:**
- libresample
- Simple linear interpolation (low quality but fast)
- FFT-based resampling (sox resampler)

---

### 5. JSON Parsing: nlohmann/json (RECOMMENDED)

**Purpose:** Robust JSON serialization for ProjectFile

**Current Status:** 🟡 BASIC IMPLEMENTATION
- Manual JSON formatting in ProjectFile::toJSON()
- No JSON parsing in ProjectFile::fromJSON() (stub)
- Works for simple cases but not robust

**For Production Use:**

#### Installation

**Header-only - No installation required:**
```bash
# Download single header
wget https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
```

**Or via package managers:**
```bash
# macOS
brew install nlohmann-json

# Ubuntu/Debian
sudo apt-get install nlohmann-json3-dev

# vcpkg
vcpkg install nlohmann-json
```

#### CMake Integration
```cmake
find_package(nlohmann_json REQUIRED)
target_link_libraries(your_target PRIVATE nlohmann_json::nlohmann_json)
```

**Why nlohmann/json:**
- Modern C++ API
- Header-only (easy integration)
- Excellent documentation
- Wide adoption
- MIT license

**Alternatives:**
- RapidJSON (faster, more complex)
- Boost.JSON
- jsoncpp

---

## Current Implementation Status

### ✅ Fully Implemented (No Dependencies)

1. **MIDI Message Classes**
   - `MidiMessage.h/cpp` - Complete MIDI message handling
   - All message types supported (Note On/Off, CC, Pitch Bend, etc.)
   - No external dependencies

2. **MIDI Sequence**
   - `MidiSequence.h/cpp` - Time-ordered MIDI event container
   - Quantization, sorting, filtering
   - Conversion to/from NoteEvent structures
   - No external dependencies

3. **Project File (Basic)**
   - `ProjectFile.h/cpp` - Project state serialization
   - Track management
   - JSON export (manual formatting)
   - Works without external dependencies (limited functionality)

4. **Stem Exporter**
   - `StemExporter.h/cpp` - Multi-track export
   - Progress callbacks
   - Batch export
   - Works with stub AudioFile implementation

### 🟡 Stub/Limited Implementation

1. **Audio File I/O**
   - `AudioFile.h/cpp` - Basic WAV float only
   - **Needs:** libsndfile for production
   - **Priority:** HIGH

2. **MIDI Device I/O**
   - `MidiIO.h/cpp` - Interface only
   - **Needs:** RtMidi for production
   - **Priority:** MEDIUM

3. **Sample Rate Conversion**
   - `AudioFile::convertSampleRate()` - Stub only
   - **Needs:** libsamplerate
   - **Priority:** LOW (nice to have)

4. **JSON Parsing**
   - `ProjectFile::fromJSON()` - Not implemented
   - **Needs:** nlohmann/json
   - **Priority:** MEDIUM

---

## Integration Priority

### Phase 1: Critical (Do First)
1. **libsndfile** - Required for any real audio file work
2. **nlohmann/json** - Required for loading projects

### Phase 2: Important (Do Next)
3. **RtMidi** - Required for MIDI device I/O
4. **libsamplerate** - Nice to have for sample rate conversion

### Phase 3: Optional Enhancements
- Better error handling
- Format-specific optimizations
- Metadata support (BWF, iXML, etc.)

---

## CMake Integration Example

To integrate all dependencies:

```cmake
# Find packages
find_package(SndFile)
find_package(RtMidi)
find_package(SampleRate)
find_package(nlohmann_json)

# Create library with conditional compilation
add_library(idaw_fileio
    src/audio/AudioFile.cpp
    src/midi/MidiMessage.cpp
    src/midi/MidiSequence.cpp
    src/midi/MidiIO.cpp
    src/project/ProjectFile.cpp
    src/export/StemExporter.cpp
)

target_include_directories(idaw_fileio PUBLIC include)

# Link dependencies if found
if(SndFile_FOUND)
    target_link_libraries(idaw_fileio PRIVATE SndFile::sndfile)
    target_compile_definitions(idaw_fileio PRIVATE HAVE_LIBSNDFILE)
endif()

if(RtMidi_FOUND)
    target_link_libraries(idaw_fileio PRIVATE RtMidi::rtmidi)
    target_compile_definitions(idaw_fileio PRIVATE HAVE_RTMIDI)
endif()

if(SampleRate_FOUND)
    target_link_libraries(idaw_fileio PRIVATE SampleRate::samplerate)
    target_compile_definitions(idaw_fileio PRIVATE HAVE_LIBSAMPLERATE)
endif()

if(nlohmann_json_FOUND)
    target_link_libraries(idaw_fileio PRIVATE nlohmann_json::nlohmann_json)
    target_compile_definitions(idaw_fileio PRIVATE HAVE_NLOHMANN_JSON)
endif()

# Require C++17 for <filesystem>
target_compile_features(idaw_fileio PUBLIC cxx_std_17)
```

---

## Decision Rationale

### Why libsndfile over alternatives?
- Industry standard with 20+ year track record
- Supports all common formats (WAV, AIFF, FLAC, OGG, etc.)
- Excellent error handling
- C API is stable and well-documented
- Used by Audacity, Ardour, and many other DAWs

### Why RtMidi over PortMidi?
- Modern C++ API (vs C in PortMidi)
- Active development (PortMidi less maintained)
- Better cross-platform support
- Lower latency
- MIT license (PortMidi is MIT too)

### Why nlohmann/json?
- Modern C++11/14/17/20 API
- Header-only (easiest integration)
- Excellent documentation and examples
- Wide adoption in C++ community
- Great error messages

---

## Build Without Dependencies (Current State)

The current implementation can build and run without external dependencies:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**What works:**
- MIDI message handling
- MIDI sequence manipulation
- Basic WAV float reading/writing
- Project structure (export only)
- Stem export structure (limited audio support)

**What doesn't work:**
- Reading/writing non-float WAV files
- AIFF, FLAC, or other formats
- MIDI device I/O
- Sample rate conversion
- Loading project files from JSON

---

## Testing Without Dependencies

Tests can run without external libraries using:
- Generated test signals (sine waves)
- In-memory MIDI sequences
- Stub I/O operations

See test files for examples.

---

**Last Updated:** 2025-12-04
**Track:** 2 - File I/O & MIDI Foundation
**Status:** Core structures complete, production dependencies documented
