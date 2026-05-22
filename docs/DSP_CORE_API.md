# DSP Core API Documentation

**Date:** January 18, 2026
**Purpose:** Reference existing pure DSP core interface from KmiDi_FINAL
**Source:** engine/src/dsp/
**Reference:** Architectural Boundary Compliance Report

## Overview

The DSP Core is the pure audio processing engine. It contains **zero** knowledge of UI, OS, frameworks, or anything outside audio mathematics.

## Core Principle

**"If I delete JUCE tomorrow, does this file still make sense?"**

If the answer is **NO**, the file does not belong in the DSP core.

## Allowed Dependencies

### ✅ Permitted Includes

```cpp
// Standard library
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <memory>

// Audio constants only
#include "MusicConstants.h"  // Sample rates, note frequencies, etc.

// Pure math
#include <complex>  // For FFT operations
```

### ❌ Forbidden Includes

```cpp
// NEVER include these in DSP core:
#include <juce_*.h>           // JUCE framework
#include <AppKit/AppKit.h>    // macOS UI
#include <SwiftUI/SwiftUI.h>  // Swift UI
#include <QtWidgets/QtWidgets.h>  // Qt framework
#include <windows.h>          // OS-specific
#include <unistd.h>           // System calls
```

## Directory Structure (KmiDi_FINAL)

```
engine/src/dsp/
├── audio_buffer.cpp     // Audio buffer implementation
├── filters.cpp          // Filter implementations
└── simd_ops.cpp         // SIMD operations
```

**Note:** The existing DSP core consists of utility classes rather than a full engine. These provide the foundation for DSP processing without framework dependencies.

## API Contract

### Engine Interface

```cpp
namespace dsp {

// Pure parameter structure - no UI dependencies
struct Parameters {
    float cutoff = 1000.0f;
    float resonance = 0.5f;
    float gain = 1.0f;
    // ... only plain data types
};

// Pure state structure - no framework dependencies
struct State {
    float phase = 0.0f;
    float lastSample = 0.0f;
    // ... only plain data types
};

class Engine {
public:
    // Initialize with sample rate and block size
    void prepare(double sampleRate, int blockSize);
    
    // Process audio block - NO ALLOCATIONS
    void processBlock(float* const* audioChannels, 
                     int numChannels, 
                     int numSamples,
                     const Parameters& params,
                     State& state) noexcept;
    
    // Reset state
    void reset() noexcept;
    
private:
    // Internal processing - no external dependencies
    void processSample(float& sample, const Parameters& params, State& state) noexcept;
};

} // namespace dsp
```

## Real-Time Safety Rules

### ✅ Allowed in Audio Thread

- Reading/writing to pre-allocated buffers
- Mathematical operations (add, multiply, sin, cos, etc.)
- SIMD operations
- Accessing atomic variables (lock-free)
- Reading from pre-allocated parameter structures

### ❌ Forbidden in Audio Thread

- Memory allocation (`new`, `malloc`, `std::vector::push_back`)
- Mutex locks
- File I/O
- Network I/O
- System calls
- Logging (unless lock-free)
- Exceptions
- Virtual function calls (unless inlined)

## Parameter Updates

Parameters must be updated atomically or through lock-free mechanisms:

```cpp
// CORRECT: Atomic parameter update
std::atomic<float> cutoffFrequency{1000.0f};

// CORRECT: Copy parameter struct atomically
Parameters currentParams;
void updateParameters(const Parameters& newParams) {
    // Copy entire struct (small, plain data)
    currentParams = newParams;  // Atomic if Parameters is small enough
}

// WRONG: Allocating new parameter structure
void updateParameters() {
    currentParams = std::make_unique<Parameters>();  // ALLOCATION!
}
```

## State Management

State must be pre-allocated and passed by reference:

```cpp
// CORRECT: Pre-allocated state
class Engine {
    State state_;  // Pre-allocated member
    
    void processBlock(...) {
        // Use state_ directly, no allocation
    }
};

// WRONG: Allocating state during processing
void processBlock(...) {
    State* state = new State();  // ALLOCATION!
    // ...
    delete state;
}
```

## Testing Requirements

### Compilation Test

The DSP core must compile in isolation:

```bash
# Test: Compile DSP core without JUCE
g++ -std=c++20 -I./dsp -I./include \
    dsp/*.cpp \
    -o dsp_test \
    -lm  # Math library only
```

### Real-Time Test

All DSP functions must be marked `noexcept`:

```cpp
void processBlock(...) noexcept;  // Required
void processSample(...) noexcept;  // Required
```

### Performance Test

DSP core must meet real-time deadlines:

- Process 512 samples in < 10ms @ 48kHz
- No allocations during processing
- Predictable execution time

## Integration Points

### From Host Glue

```cpp
// host/juce/PluginProcessor.cpp
#include "dsp/Engine.h"  // Host glue includes DSP

void PluginProcessor::processBlock(...) {
    // Convert JUCE buffers to plain arrays
    float* channels[numChannels];
    for (int ch = 0; ch < numChannels; ++ch) {
        channels[ch] = buffer.getWritePointer(ch);
    }
    
    // Call pure DSP
    dspEngine.processBlock(channels, numChannels, 
                          numSamples, params, state);
}
```

### To UI Layer

```cpp
// ui/ParameterDisplay.cpp
// UI reads parameters (snapshot), never touches DSP directly
void updateDisplay(const Parameters& params) {
    // Display parameter values
    // UI never calls processBlock()
}
```

## Examples

### ✅ CORRECT: Pure DSP Implementation

```cpp
// dsp/Filters.h
namespace dsp {
    class LowPassFilter {
    public:
        void prepare(double sampleRate) noexcept {
            // Pre-calculate coefficients
            // No allocation
        }
        
        float processSample(float input, float cutoff) noexcept {
            // Pure math, no allocation
            float alpha = calculateAlpha(cutoff);
            // ... filter math
            return output;
        }
        
    private:
        float lastOutput = 0.0f;  // Pre-allocated state
    };
}
```

### ❌ WRONG: Contaminated DSP

```cpp
// WRONG: Includes JUCE
#include <juce_dsp/juce_dsp.h>

// WRONG: Allocates during processing
void processBlock(...) {
    std::vector<float> temp(bufferSize);  // ALLOCATION!
}

// WRONG: Uses UI framework
#include <QtWidgets/QSlider>
```

## Boundary Enforcement

### Build System Checks

Add to `CMakeLists.txt`:

```cmake
# DSP core target - no framework dependencies
add_library(dsp_core STATIC
    dsp/Engine.cpp
    dsp/Voice.cpp
    # ...
)

target_include_directories(dsp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/dsp
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Explicitly forbid JUCE in DSP core
target_compile_definitions(dsp_core PRIVATE
    DSP_CORE_PURE=1
)

# Test: Try to include JUCE (should fail)
# add_definitions(-DNO_JUCE_IN_DSP)
```

### Static Analysis

Use clang-tidy or similar to detect forbidden includes:

```bash
# Check for forbidden includes in DSP core
grep -r "juce_\|AppKit\|SwiftUI\|Qt" dsp/
# Should return zero matches
```

## Migration Path

If existing code violates boundaries:

1. **Identify contamination**: Find all forbidden includes
2. **Extract pure logic**: Move math to `dsp/` directory
3. **Create host glue**: Move framework code to `host/` directory
4. **Update includes**: Host glue includes DSP, not vice versa
5. **Test isolation**: Verify DSP compiles without frameworks

## References

- `ARCHITECTURAL_BOUNDARY_COMPLIANCE_REPORT.md` - Full boundary analysis
- `low-latency-daw.md` - Real-time audio principles
- `cpp_audio_architecture.md` - Brain/Body architecture