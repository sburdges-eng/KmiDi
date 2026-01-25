# KmiDi System Inventory

**Date**: 2026-01-21  
**Status**: Complete System Catalog

This document catalogs all existing systems in the KmiDi project, their locations, dependencies, integration points, and build requirements.

## System Overview

KmiDi is a comprehensive music generation and therapeutic iDAW system with multiple layers:
- **Frontend**: React + Tauri desktop application
- **Backend APIs**: Python FastAPI servers
- **Core Engine**: C++ KellyBrain with Rust FFI bridge
- **MCP Servers**: Multiple Model Context Protocol servers
- **ML/Training**: Model training and inference systems
- **DAW Integrations**: Logic Pro, FL Studio, Reaper, Pro Tools

## 1. Frontend Layer

### React UI Application
- **Location**: `src/`
- **Framework**: React 19.1, TypeScript 5.8
- **Styling**: Tailwind CSS 4.x
- **Build Tool**: Vite
- **Key Components**:
  - `src/App.tsx` - Main application component
  - `src/components/` - UI components
    - Emotion selection interface
    - Intent injection panel
    - SpectoCloud visualization
    - Status indicators
- **Dependencies**: 
  - React 19.1
  - Tauri 2
  - Tailwind 4
  - TypeScript 5.8
- **Build Command**: `npm run build`
- **Dev Command**: `npm run dev`

### Tauri Desktop Bridge
- **Location**: `src-tauri/`
- **Language**: Rust
- **Framework**: Tauri 2.0
- **Key Files**:
  - `src-tauri/src/main.rs` - Entry point
  - `src-tauri/src/state.rs` - State management
  - `src-tauri/src/events.rs` - Event handling
- **Dependencies**:
  - Tauri 2.0
  - reqwest (HTTP client)
  - serde (serialization)
  - tokio (async runtime)
- **Build Command**: `cargo build --release`
- **IPC Commands**:
  - `get_emotions` - Fetch available emotions
  - `generate_music` - Generate music from intent
  - `interrogate` - Refine intent through questioning

## 2. Backend APIs

### Music Brain API
- **Location**: `music_brain/api.py`
- **Framework**: FastAPI
- **Port**: 8000 (default)
- **Endpoints**:
  - `GET /emotions` - List available emotions
  - `POST /generate` - Generate music from intent
  - `POST /interrogate` - Refine intent
  - `POST /spectocloud/render` - Render SpectoCloud visualization
  - `GET /config/humanizer` - Get humanizer configuration
- **Status**: ⚠️ Partial - `/generate` has CompleteSongIntent support but may need enhancement
- **Dependencies**:
  - FastAPI
  - uvicorn
  - All `music_brain` modules
- **Start Command**: `python -m music_brain.api` or `./scripts/start_music_brain_api.sh`

### Penta Core Server
- **Location**: `penta_core/server.py`
- **Purpose**: Music theory and rule system server
- **Features**:
  - Harmony rules
  - Counterpoint rules
  - Voice leading rules
  - Rule breaking system
- **Dependencies**: `penta_core` modules

## 3. MCP Servers

### mcp_penta_swarm
- **Location**: `mcp_penta_swarm/`
- **Purpose**: Swarm orchestration for AI agents
- **Files**:
  - `server.py` - Main server
  - `__main__.py` - Entry point
  - `README.md` - Documentation
- **Configuration**: `.env.example` provided

### mcp_workstation
- **Location**: `mcp_workstation/`
- **Purpose**: Development workflow orchestration
- **Files**:
  - `server.py` - Main server
  - `orchestrator.py` - Workflow orchestration
  - `cpp_planner.py` - C++ planning tools
  - `phases.py` - Development phases
- **Configuration**: `configs/` directory with setup guides

### daiw_mcp
- **Location**: `daiw_mcp/`
- **Purpose**: DAiW tool integrations
- **Tools**:
  - Audio analysis
  - Groove processing
  - Harmony generation
  - Intent processing
  - Teaching system
- **Files**:
  - `server.py` - Main server
  - `tools/` - Tool implementations

### mcp_todo
- **Location**: `mcp_todo/`
- **Purpose**: Task management integration
- **Files**:
  - `server.py` - Main server
  - `http_server.py` - HTTP interface

## 4. C++ Core Engine

### KellyBrain Engine
- **Location**: `src/engine/`
- **Key Files**:
  - `KellyBrain.cpp/h` - Main engine
  - `IntentPipeline.cpp/h` - Intent processing
  - `EmotionThesaurus.cpp/h` - Emotion mapping
  - `MidiGenerator.h` - MIDI generation
- **Dependencies**: 
  - C++20 standard
  - CMake 3.27+
  - Optional: JUCE (for plugin support)

### FFI Layer
- **Location**: `src/bridge/kelly_ffi.cpp`
- **Purpose**: C++ to Rust bridge
- **Features**:
  - Complete serialization of theoretical parameters
  - Emotion data (valence, arousal, dominance, complexity)
  - Melodic guidance
  - Rhythmic guidance
  - Dynamics
  - Rule breaks
- **Output**: `libKellyFFI.dylib` (macOS) or equivalent

### DSP Core
- **Location**: `src/dsp/`
- **Files**:
  - `audio_buffer.cpp` - Audio buffer management
  - `filters.cpp` - Audio filters
  - `simd_ops.cpp` - SIMD operations
- **Status**: ⚠️ May need pure DSP from KmiDi_FINAL
- **Note**: Framework contamination should be avoided for real-time safety

### Plugin System
- **Location**: `src/plugin/`
- **Formats**: VST3, CLAP
- **Key Files**:
  - `plugin_processor.cpp/h` - Plugin processor
  - `plugin_editor.cpp/h` - Plugin UI
- **Build Option**: `BUILD_PLUGINS` CMake option

## 5. Python Music Brain Systems

### Emotion Processing
- **Location**: `music_brain/emotion/`
- **Modules**:
  - `text_emotion_parser.py` - Text emotion extraction
  - `multimodal_emotion.py` - Multimodal emotion detection
  - `emotion_production.py` - Emotion to production mapping
  - `emotion_thesaurus.py` - Emotion vocabulary

### Harmony Systems
- **Location**: 
  - `music_brain/harmony_kmidi.py`
  - `penta_core/harmony/`
- **Features**:
  - Chord progression generation
  - Voice leading
  - Counterpoint
  - Neo-Riemannian transformations
  - Jazz voicings
  - Microtonal support

### Groove Systems
- **Location**: 
  - `music_brain/groove/`
  - `penta_core/groove/`
- **Features**:
  - Groove extraction
  - Humanization
  - Drum replacement
  - Polyrhythm support
  - Performance analysis

### Structure Engine
- **Location**: `music_brain/structure/`
- **Modules**:
  - `chord.py` - Chord analysis
  - `progression.py` - Progression analysis
  - `sections.py` - Section detection
  - `tension_curve.py` - Tension analysis
  - `comprehensive_engine.py` - Full structure engine

### Session System
- **Location**: `music_brain/session/`
- **Key Files**:
  - `intent_schema.py` - CompleteSongIntent definition
  - `intent_processor.py` - Intent processing
  - `intent_bridge.py` - Intent bridging
  - `teaching.py` - Teaching system
- **Features**:
  - Three-phase interrogation
  - Rule breaking system
  - Intent validation

### Voice Processing
- **Location**: `music_brain/voice/`
- **Features**:
  - Auto-tune processing
  - Voice modulation
  - Voice synthesis
  - Voice profiles

### Audio Analysis
- **Location**: `music_brain/audio/`
- **Features**:
  - Audio analysis
  - Chord detection
  - Frequency analysis
  - Feel extraction
  - Reference DNA matching

### Arrangement
- **Location**: `music_brain/arrangement/`
- **Features**:
  - Bass generation
  - Energy arc calculation
  - Arrangement templates

### Generative Systems
- **Location**: `music_brain/generative/`
- **Features**:
  - Melody generation
  - Arrangement generation
  - Pattern generation

### Export Systems
- **Location**: `music_brain/export/`
- **Features**:
  - Emotion stem export
  - Social platform export

### Visualization
- **Location**: `music_brain/visualization/`
- **Features**:
  - SpectoCloud visualization
  - Emotion trajectory visualization

## 6. DAW Integrations

### Logic Pro
- **Location**: `music_brain/daw/logic_pro.py`
- **Features**: Logic Pro project integration

### FL Studio
- **Location**: `music_brain/daw/fl_studio.py`
- **Features**: FL Studio project integration

### Reaper
- **Location**: `music_brain/daw/reaper.py`
- **Features**: Reaper project integration

### Pro Tools
- **Location**: `music_brain/daw/pro_tools.py`
- **Features**: Pro Tools project integration

### Common DAW Features
- **Location**: `music_brain/daw/`
- **Shared Modules**:
  - `markers.py` - Marker management
  - `mixer_params.py` - Mixer parameter control

## 7. ML/Training Systems

### Training Orchestrator
- **Location**: `scripts/ai_training_orchestrator.py`
- **Purpose**: Orchestrate ML model training

### Model Management
- **Location**: `config/models.yaml`
- **Models**:
  - Emotion classifier (150M params)
  - Intent mapper (300M params)
  - Harmony intelligence (500M params)
  - Text LLM fallback (1B params)
- **Memory Budgets**: Safe (1.8GB), Aggressive (4GB), Maximum (5GB)

### Dataset Systems
- **Location**: `penta_core/ml/datasets/`
- **Features**:
  - Audio dataset management
  - Dataset augmentation
  - Audio downloader

### Training Configurations
- **Location**: `training/`
- **Configs**:
  - `integrated_training_config.yaml`
  - `cuda_session/` - CUDA training configs
  - `metal_m4_session/` - Metal M4 configs
  - `mlx_session/` - MLX configs

### ML Inference
- **Location**: `penta_core/ml/`
- **Features**:
  - Model registry
  - Inference pipeline
  - GPU utilities
  - Style transfer

## 8. Collaboration Systems

### WebSocket Server
- **Location**: `music_brain/collaboration/websocket.py`
- **Purpose**: Real-time collaboration

### Version Control
- **Location**: `music_brain/collaboration/version_control.py`
- **Purpose**: Intent versioning

### Collaboration UI
- **Location**: `penta_core/collaboration/collab_ui.py`
- **Purpose**: Collaboration interface

## 9. Build Systems

### CMake Configuration
- **Location**: `CMakeLists.txt`
- **Options**:
  - `BUILD_DESKTOP` - Build desktop GUI
  - `BUILD_PLUGINS` - Build VST3/CLAP plugins
  - `BUILD_KELLY_CORE` - Build Kelly core library
  - `USE_KMI_DI_FINAL` - Use KmiDi_FINAL components
  - `BUILD_NATIVE_MACOS_APP` - Build native macOS app
- **Dependencies**: CMake 3.27+, C++20 compiler

### Cargo Configuration
- **Location**: `src-tauri/Cargo.toml`
- **Dependencies**:
  - Tauri 2.0
  - reqwest
  - serde
  - tokio
  - cpal (audio)

### npm Configuration
- **Location**: `package.json` (root)
- **Dependencies**: React, Vite, Tailwind, TypeScript

## 10. Configuration Files

### Model Configuration
- **Location**: `config/models.yaml`
- **Purpose**: Model sizing and memory budgets

### Emotion Recognizer
- **Location**: `config/emotion_recognizer.yaml`

### Harmony Predictor
- **Location**: `config/harmony_predictor.yaml`

### Groove Predictor
- **Location**: `config/groove_predictor.yaml`

### Dynamics Engine
- **Location**: `config/dynamics_engine.yaml`

## 11. Scripts and Utilities

### Development Scripts
- **Location**: `scripts/`
- **Key Scripts**:
  - `start_music_brain_api.sh` - Start API server
  - `setup-build-env.sh` - Setup build environment
  - `setup-icons.sh` - Setup Tauri icons
  - `test-ffi-minimal.sh` - Test FFI layer

### Training Scripts
- **Location**: `scripts/`
- **Key Scripts**:
  - `ai_training_orchestrator.py`
  - `train.py`
  - `prepare_datasets.py`

### DAW Integration Scripts
- **Location**: `scripts/idaw/`
- **Purpose**: iDAW integration utilities

## 12. Data Files

### Emotional Mapping
- **Location**: `music_brain/data/emotional_mapping.py`
- **Purpose**: Emotion presets and mappings

### Chord Progressions
- **Location**: `music_brain/data/chord_progressions.json`

### Scales Database
- **Location**: `music_brain/data/scales_database.json`

### Rule Breaking Database
- **Location**: `music_brain/data/rule_breaking_database.json`

## System Dependencies

### Frontend Dependencies
```
React UI
  └─ depends on: Tauri, Music Brain API (HTTP)
```

### Backend Dependencies
```
Music Brain API
  └─ depends on: All music_brain modules, penta_core
  
Penta Core Server
  └─ depends on: penta_core modules
```

### Core Engine Dependencies
```
C++ KellyBrain
  └─ depends on: CMake, C++20, Optional: JUCE
  
Rust Bridge
  └─ depends on: C++ FFI library (libKellyFFI), Music Brain API
```

### MCP Server Dependencies
```
All MCP Servers
  └─ depend on: music_brain, penta_core (various modules)
```

## Integration Points

### Data Flow
```
React UI (Frontend)
    ↕ Tauri IPC
Rust Bridge (src-tauri/)
    ↕ HTTP/REST
Music Brain API (music_brain/api.py)
    ↕ Direct calls
C++ KellyBrain (src/engine/)
    ↕ FFI
Rust State Management
    ↕ Tauri IPC
React UI (Updates)
```

### API Communication
- **Frontend → Backend**: HTTP REST API (port 8000)
- **Rust → C++**: FFI via `libKellyFFI.dylib`
- **Python → Python**: Direct module imports

## Build Order

1. **C++ Core**: Build KellyBrain and FFI library
2. **Rust Bridge**: Build Tauri application (depends on C++ FFI)
3. **Python Backend**: Install Python dependencies
4. **Frontend**: Build React application
5. **Integration**: Copy FFI library to Tauri resources

## Missing/Incomplete Systems

See `FEATURE_GAP_ANALYSIS.md` and `docs/STRUCTURE_CROSS_EXAMINATION/06_GAP_ANALYSIS_AND_RECOMMENDATIONS.md` for detailed gap analysis.

### Critical Gaps
1. Timeline UI component
2. Inspector/Browser panels
3. Full intent pipeline connection (partially implemented)
4. Audio rendering pipeline (MIDI → WAV/MP3)
5. KmiDi_FINAL integration (available but not integrated)

## References

- Architecture: `docs/ARCHITECTURE.md`
- Gap Analysis: `FEATURE_GAP_ANALYSIS.md`
- Integration Status: `FINAL_STATUS.md`
- Spec Coverage: `docs/STRUCTURE_CROSS_EXAMINATION/02_SPEC_COVERAGE_ANALYSIS.md`
