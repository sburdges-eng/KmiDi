# Multi-Language Architecture Guide

Overview of how Rust, C/C++, JUCE, Python, and Tauri work together in KmiDi.

## Architecture Overview

KmiDi uses a multi-language architecture optimized for different tasks:

```
┌─────────────────────────────────────────────────────────────┐
│                    KmiDi Architecture                        │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   [Python]           [C++/JUCE]          [Rust]
   ML Models          Audio Engine        Intent IR
   Training           Real-time           Validation
   Inference          Processing          Safety
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    [FFI/C Bindings]
                            │
                    [Unified Interface]
```

## Language Roles

### Python 🐍

**Purpose:** ML model training, inference, and orchestration

**Key Components:**
- `python/penta_core/ml/` - Model registry, training, inference
- `python/prrot/` - PRROT system (phoneme alignment, timbre extraction)
- `python/music_brain/` - Music intelligence layer

**Responsibilities:**
- Model training and management
- Model inference (can export to ONNX/CoreML for C++ use)
- Data processing and preparation
- High-level orchestration

**Integration Points:**
- Exports models to formats C++ can use (ONNX, CoreML, RTNeural JSON)
- Provides Python API for model access
- Can be called from C++ via Python C API or exported models

**Files:**
- `python/penta_core/ml/model_registry.py` - Model management
- `python/penta_core/ml/training_orchestrator.py` - Training coordination
- `ml/models/` - Trained model files

---

### C++ / JUCE 🎵

**Purpose:** Real-time audio processing, plugin development, core engine

**Key Components:**
- `engine/` - Core audio engine
- `plugins/` - VST3/AU plugin implementations
- JUCE framework integration

**Responsibilities:**
- Real-time audio processing (low latency)
- Audio plugin development (VST3, AU)
- MIDI processing
- Audio I/O management
- Integration with JUCE framework

**JUCE Usage:**
- Audio processing: `juce::juce_audio_processors`
- Audio formats: `juce::juce_audio_formats`
- Audio devices: `juce::juce_audio_devices`
- Core utilities: `juce::juce_core`
- Data structures: `juce::juce_data_structures`

**Integration Points:**
- Uses Rust Intent IR via FFI
- Loads models via ONNX/CoreML/RTNeural
- Provides C++ API for Python models (when needed)

**Files:**
- `engine/src/engine/IntentPipeline.h` - Intent processing
- `engine/src/common/IntentIRAdapter.cpp` - Rust FFI adapter
- `CMakeLists.txt` - Build configuration with JUCE

---

### Rust 🦀

**Purpose:** Safe, performant Intent IR validation and processing

**Key Components:**
- `engine/intent_ir/` - Intent IR v1 implementation
- FFI exports for C compatibility

**Responsibilities:**
- IntentFrame validation and clamping
- Type-safe Intent IR construction
- Memory-safe operations
- Performance-critical validation

**Features:**
- `#![no_std]` - No standard library (embedded-friendly)
- Custom allocator for system integration
- FFI exports for C/C++ interop
- Static library compilation

**Integration Points:**
- Exported as static library (`crate-type = ["staticlib"]`)
- C FFI functions for C++ integration
- Used by C++ engine via `IntentIRAdapter`

**Files:**
- `engine/intent_ir/src/lib.rs` - Main library
- `engine/intent_ir/src/ffi.rs` - C FFI exports
- `engine/intent_ir/src/validator.rs` - Validation logic
- `engine/intent_ir/src/builder.rs` - Safe construction
- `engine/intent_ir/src/types.rs` - Type definitions

**FFI Functions:**
```rust
// Validation
pub extern "C" fn validate_intent_frame_ffi(frame: *const IntentFrame) -> c_int

// Clamping
pub extern "C" fn clamp_intent_frame_ffi(frame: *mut IntentFrame)

// Version checking
pub extern "C" fn intent_frame_version_supported_ffi(version: u16) -> bool

// Builder API
pub extern "C" fn create_intent_frame_builder() -> *mut IntentFrameBuilderHandle
pub extern "C" fn IntentFrameBuilder_set_emotion(...)
pub extern "C" fn IntentFrameBuilder_set_musical_intent(...)
```

---

### Tauri (Frontend) 🖥️

**Purpose:** Desktop application UI (if used)

**Status:** Not currently in KmiDi_FINAL (may be in other directories)

**Typical Integration:**
- Rust backend (Tauri core)
- Web frontend (HTML/CSS/JS/TS)
- Can call C++ via FFI or Python via subprocess/API

**If Used:**
- Would provide desktop UI
- Could integrate with Python models via API
- Could use Rust Intent IR directly
- Could call C++ engine via FFI

---

## Data Flow

### Model Inference Flow

```
Python Model (Training/Inference)
        │
        ▼
Export to ONNX/CoreML/RTNeural
        │
        ▼
C++ Engine (Real-time)
        │
        ▼
JUCE Audio Processing
        │
        ▼
Audio Output
```

### Intent Processing Flow

```
User Input / ML Output
        │
        ▼
Python (Intent Generation)
        │
        ▼
IntentFrame (Rust)
        │
        ▼
Rust Validation (FFI)
        │
        ▼
C++ Adapter (IntentIRAdapter)
        │
        ▼
C++ Engine (IntentPipeline)
        │
        ▼
Musical Output
```

## Build System

### CMake (C++/JUCE)

**File:** `CMakeLists.txt`

**Key Features:**
- JUCE integration via `add_subdirectory(build/external/JUCE)`
- Rust integration via static library linking
- Python model integration (ONNX/CoreML loaders)
- Cross-platform build support

**Dependencies:**
- JUCE framework (audio processing)
- Rust static library (intent_ir)
- ONNX Runtime (optional, for model inference)
- CoreML (macOS, for model inference)

### Cargo (Rust)

**File:** `engine/intent_ir/Cargo.toml`

**Key Features:**
- Static library output (`crate-type = ["staticlib"]`)
- `#![no_std]` for embedded use
- Custom allocator
- cbindgen for C header generation

**Build:**
```bash
cd engine/intent_ir
cargo build --release
# Generates libintent_ir.a (static library)
```

## Integration Patterns

### Pattern 1: Python → C++ (Model Export)

**Use Case:** Use Python-trained models in C++ real-time code

**Steps:**
1. Train model in Python
2. Export to ONNX or CoreML
3. Load in C++ using ONNX Runtime or CoreML
4. Run inference in real-time audio thread

**Example:**
```cpp
// C++ code
#include <onnxruntime_cxx_api.h>

// Load ONNX model
Ort::Session session(env, "model.onnx", session_options);

// Run inference
auto outputs = session.Run(run_options, input_names, inputs, output_names, output_count);
```

### Pattern 2: Rust → C++ (FFI)

**Use Case:** Use Rust validation in C++ code

**Steps:**
1. Rust exports C-compatible functions
2. C++ includes generated header
3. C++ links Rust static library
4. C++ calls Rust functions

**Example:**
```cpp
// C++ code
#include "intent_ir_ffi.h"  // Generated by cbindgen

IntentFrame frame = {...};
if (validate_intent_frame_ffi(&frame) == 0) {
    clamp_intent_frame_ffi(&frame);
    // Use validated frame
}
```

### Pattern 3: C++ → Python (Model API)

**Use Case:** Call Python models from C++ when needed

**Options:**
1. **Export models** (preferred) - Export to ONNX/CoreML, load in C++
2. **Python C API** - Embed Python interpreter, call Python functions
3. **Subprocess** - Spawn Python process, communicate via IPC
4. **HTTP API** - Python service, C++ client

**Example (Python C API):**
```cpp
// C++ code
#include <Python.h>

PyObject* module = PyImport_ImportModule("penta_core.ml.inference");
PyObject* func = PyObject_GetAttrString(module, "run_inference");
PyObject* result = PyObject_CallFunction(func, "f", input_value);
```

## Model Integration by Language

### Python Models

**Location:** `python/penta_core/ml/`

**Formats:**
- RTNeural JSON (for C++ RTNeural library)
- ONNX (cross-platform)
- CoreML (macOS/iOS)
- PyTorch (Python-only)

**Usage:**
- Training: Python
- Inference: Python or exported to C++

### C++ Model Loading

**Options:**
1. **RTNeural** - Load JSON models, real-time inference
2. **ONNX Runtime** - Load ONNX models
3. **CoreML** - Load CoreML models (macOS)
4. **Custom C++** - Direct implementation

**Example (RTNeural):**
```cpp
#include <RTNeural.h>

RTNeural::Model<float> model;
model.parseJson("model.json");
auto output = model.forward(input);
```

## File Structure

```
KmiDi_FINAL/
├── engine/
│   ├── intent_ir/          # Rust Intent IR
│   │   ├── src/
│   │   │   ├── lib.rs       # Main library
│   │   │   ├── ffi.rs       # C FFI exports
│   │   │   ├── validator.rs # Validation
│   │   │   └── types.rs     # Type definitions
│   │   └── Cargo.toml       # Rust build config
│   └── src/                 # C++ engine
│       ├── engine/          # Core engine
│       └── common/          # Shared code
│           └── IntentIRAdapter.cpp  # Rust FFI adapter
├── python/
│   ├── penta_core/ml/       # Python ML models
│   └── prrot/               # PRROT system
├── ml/models/               # Trained models
├── CMakeLists.txt           # C++/JUCE build
└── build/external/JUCE/     # JUCE framework
```

## Development Workflow

### Working with Rust

```bash
# Build Rust library
cd engine/intent_ir
cargo build --release

# Generate C headers
cbindgen --config cbindgen.toml --crate intent_ir --output intent_ir_ffi.h

# Test Rust code
cargo test
```

### Working with C++/JUCE

```bash
# Configure build
mkdir build && cd build
cmake ..

# Build
cmake --build .

# Run tests
ctest
```

### Working with Python Models

```bash
# Train model
python -m penta_core.ml.training_orchestrator --model emotion_recognizer

# Export to ONNX
python -m penta_core.ml.export --model emotion_recognizer --format onnx

# Test inference
python scripts/test_models.py
```

## Best Practices

### Memory Management

- **Rust:** Automatic (RAII, borrow checker)
- **C++:** Manual (smart pointers recommended)
- **Python:** Automatic (GC)
- **FFI:** Careful ownership transfer

### Error Handling

- **Rust:** `Result<T, E>` types
- **C++:** Exceptions or error codes
- **Python:** Exceptions
- **FFI:** Return codes, check Rust panics

### Performance

- **Real-time audio:** C++/JUCE (low latency)
- **Model inference:** Python (training) or C++ (runtime)
- **Validation:** Rust (safe, fast)
- **UI:** Tauri/Web (if used)

## Troubleshooting

### Rust FFI Issues

```bash
# Check Rust library is built
ls engine/intent_ir/target/release/libintent_ir.a

# Verify C headers generated
ls engine/intent_ir/intent_ir_ffi.h

# Check CMake links Rust library
grep -r "intent_ir" CMakeLists.txt
```

### JUCE Build Issues

```bash
# Verify JUCE path
ls build/external/JUCE

# Check JUCE modules
grep -r "juce::" CMakeLists.txt
```

### Python Model Integration

```bash
# Verify model exports
ls ml/models/*.onnx
ls ml/models/*.mlpackage

# Test Python API
python -c "from penta_core.ml.model_registry import list_models; print(list_models())"
```

## See Also

- **Model Documentation:** `docs/MODELS_README.md`
- **Intent IR Docs:** `docs/INTENT_IR_V1_BUILD_STATUS.md`
- **Build System:** `CMakeLists.txt`
- **Rust FFI:** `engine/intent_ir/src/ffi.rs`
- **C++ Adapter:** `engine/src/common/IntentIRAdapter.cpp`

---

**Last Updated:** 2026-01-22
