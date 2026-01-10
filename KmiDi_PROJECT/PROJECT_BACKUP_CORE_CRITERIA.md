# PROJECT BACKUP: Core Criteria

**Backup Date:** 2025-01-09
**Project:** KmiDi (Unified Music Intelligence & Audio Workstation)
**Version:** 1.0.0
**Purpose:** Comprehensive backup of all core project criteria excluding audio training data

---

## Table of Contents

1. [Project Metadata](#1-project-metadata)
2. [Repository Architecture](#2-repository-architecture)
3. [Configuration Files](#3-configuration-files)
4. [API Contracts & Schemas](#4-api-contracts--schemas)
5. [Data Schemas](#5-data-schemas)
6. [Development Workflows](#6-development-workflows)
7. [CI/CD Configuration](#7-cicd-configuration)
8. [Code Style & Conventions](#8-code-style--conventions)
9. [Key Architecture Decisions](#9-key-architecture-decisions)
10. [Documentation Structure](#10-documentation-structure)

---

## 1. Project Metadata

### Basic Information

```yaml
name: kmidi
version: 1.0.0
description: "KmiDi: Unified Music Intelligence & Audio Workstation"
readme: "KMIDI_README.md"
license: "MIT"
authors:
  - name: "Kelly Project Team"
requires_python: ">=3.9"
```

### Project Philosophy

**Core Principle:** "Interrogate Before Generate" - The tool shouldn't finish art for people. It should make them braver.

**Philosophy Statement:**
> "The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"
>
> Technical implementation serves emotional expression. The tool educates and empowers - it doesn't just generate.

### Tech Stack Summary

| Component | Technology | Version |
|-----------|------------|---------|
| Python | 3.9+ | Tested 3.9-3.13 |
| C++ | C++20 | Required |
| CMake | 3.27+ | Required |
| JUCE | 8.0.10 | Included in external/ |
| Qt | 6.x | Required for GUI |
| Node | 18+ | For React/Tauri frontend |
| Rust | Latest stable | For Tauri desktop bridge |

---

## 2. Repository Architecture

### Complete Directory Structure

```
KmiDi-1/
├── music_brain/               # Python Music Intelligence Toolkit (main package)
│   ├── tier1/                 # Pretrained models (no fine-tuning)
│   │   ├── midi_generator.py
│   │   ├── audio_generator.py
│   │   └── voice_generator.py
│   ├── tier2/                 # LoRA fine-tuning
│   │   └── lora_finetuner.py
│   ├── session/               # Intent schema, teaching, interrogation
│   ├── structure/             # Chord/progression analysis
│   ├── groove/                # Groove extraction/application
│   ├── harmony/               # Harmony engine
│   ├── melody/                # Melody generation
│   ├── audio/                 # Audio feel analysis
│   ├── data/                  # JSON/YAML data files
│   ├── cli.py                 # `daiw` CLI command
│   └── __init__.py            # Public API (v0.2.0)
│
├── mcp_workstation/           # MCP Multi-AI Workstation (orchestration)
│   ├── __init__.py            # Package exports, version 1.0.0
│   ├── cli.py                 # CLI entry point
│   ├── orchestrator.py        # Central coordinator (Workstation class)
│   ├── models.py              # Data models (AIAgent, Proposal, Phase)
│   ├── proposals.py           # Proposal management system
│   ├── phases.py              # Phase tracking
│   ├── cpp_planner.py         # C++ transition planning
│   ├── ai_specializations.py  # AI agent capabilities
│   ├── server.py              # MCP server implementation
│   └── debug.py               # Debug protocol and logging
│
├── mcp_todo/                  # MCP TODO Server (cross-AI task management)
│   ├── cli.py                 # CLI tool
│   ├── http_server.py         # HTTP API server
│   └── server.py              # MCP server implementation
│
├── mcp_penta_swarm/           # MCP Swarm Server (multi-AI aggregation)
│   └── server.py              # Aggregates GPT-4o, Claude, Gemini, Grok
│
├── daiw_mcp/                  # DAiW MCP Server
│
├── iDAW_Core/                 # JUCE Plugin Suite (C++)
│   └── plugins/               # Audio plugins (Pencil, Eraser, Palette, etc.)
│
├── src_penta-core/            # Penta-Core C++ Engines (implementation)
├── src/                       # Additional C++ source files
├── include/                   # C++ Headers (including penta/)
│   └── penta/
│       ├── common/            # RTTypes, RTLogger, RTMemoryPool, SIMDKernels
│       ├── groove/            # GrooveEngine, OnsetDetector, TempoEstimator
│       ├── harmony/           # HarmonyEngine, ChordAnalyzer, ScaleDetector
│       ├── diagnostics/       # DiagnosticsEngine, AudioAnalyzer
│       └── osc/               # OSCHub, OSCClient, OSCServer
│
├── python/penta_core/         # Python bindings for Penta-Core
│   ├── rules/                 # Music theory rules
│   └── teachers/              # Interactive teaching modules
│
├── cpp_music_brain/           # C++ Music Brain implementation
├── bindings/                  # Language bindings (Python/C++)
│
├── api/                       # FastAPI production service
│   ├── main.py                # Main API server
│   ├── requirements.txt       # API-specific dependencies
│   └── Dockerfile             # Docker configuration
│
├── scripts/                   # Standalone utility scripts
├── tools/                     # Development tools (kb_analyzer, audio_cataloger)
│
├── tests/                     # All tests
│   ├── python/                # Python unit tests
│   ├── music_brain/           # Music Brain integration tests
│   └── penta_core/            # Penta-Core tests
│
├── examples/                  # Example scripts (music_brain, penta_core)
├── benchmarks/                # Performance benchmarks
│
├── data/                      # JSON data files
│   ├── emotions/              # Emotion thesaurus data (216 nodes)
│   ├── progressions/          # Chord progression databases
│   ├── scales/                # Scale databases
│   ├── grooves/               # Groove and genre maps
│   ├── rules/                 # Rule breaking databases
│   └── music_theory/          # Music theory data
│
├── docs/                      # All documentation
│   ├── sprints/               # Sprint documentation
│   ├── summaries/             # Project summaries
│   ├── integrations/          # Integration guides
│   ├── ai_setup/              # AI assistant setup guides
│   ├── music_brain/           # Music Brain docs
│   ├── penta_core/            # Penta-Core docs
│   └── references/            # Reference materials
│
├── vault/                     # Obsidian Knowledge Base
├── Production_Workflows/      # Production workflow guides
├── Songwriting_Guides/        # Songwriting methodology guides
├── Theory_Reference/          # Music theory reference materials
├── Templates/                 # Project and document templates
│
├── deployment/                # Deployment configs (Docker, specs)
├── external/                  # External libraries (JUCE, oscpack, etc.)
├── assets/                    # SVG and image assets
├── web/                       # Web frontend (Vite, Tailwind)
├── mobile/                    # Mobile app code
├── iOS/                       # iOS-specific code
├── macOS/                     # macOS-specific code
├── plugins/                   # Audio plugins
├── output/                    # Generated output files (gitignored)
├── checkpoints/               # Model checkpoints (gitignored)
└── legacy/                    # Archived/legacy code
```

### Component Descriptions

#### music_brain/
Python-based music generation and analysis toolkit. Main package for emotion-to-music conversion.

**Key Modules:**
- `tier1/`: Pretrained models (MelodyTransformer, HarmonyPredictor, GroovePredictor)
- `tier2/`: LoRA fine-tuning (97% parameter reduction)
- `session/`: Three-phase intent schema processing
- `structure/`: Chord/progression analysis and generation
- `groove/`: Groove extraction and application
- `harmony/`: Harmony engine
- `melody/`: Melody generation
- `audio/`: Audio feel analysis

#### penta_core/ (C++)
Real-time, RT-safe audio analysis engines. High-performance C++ implementation with Python bindings.

**Key Features:**
- RT-safe (no allocations in audio callbacks)
- SIMD optimizations (AVX2 when available)
- Lock-free data structures
- Default sample rate: 44100.0 Hz

#### iDAW_Core/
JUCE-based plugin suite with art-themed naming (Pencil, Eraser, Palette, etc.)

**Dual-Heap Memory Architecture:**
- **Side A ("Work State")**: 4GB pre-allocated, no deallocation, RT-safe
- **Side B ("Dream State")**: Dynamic allocation, AI generation, UI operations
- **Communication**: Lock-free ring buffer (Side B → Side A)

#### MCP Servers
Multiple Model Context Protocol servers for multi-AI collaboration:
- `mcp_workstation/`: Multi-AI orchestration with proposal/voting system
- `mcp_todo/`: Cross-AI task management (tasks sync across all AIs)
- `mcp_penta_swarm/`: Aggregates GPT-4o, Claude, Gemini, Grok
- `daiw_mcp/`: DAiW-specific MCP server

---

## 3. Configuration Files

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "kmidi"
version = "1.0.0"
description = "KmiDi: Unified Music Intelligence & Audio Workstation"
readme = "KMIDI_README.md"
license = "MIT"
authors = [
    {name = "Kelly Project Team"},
]
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.21",
    "torch>=2.0",
    "librosa>=0.10",
    "pyyaml>=6.0",
    "scipy>=1.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "flake8>=6.0",
    "mypy>=1.0",
    "fastmcp>=2.0.0",
    "python-dotenv>=1.0.0",
    "openai>=1.0.0",
    "anthropic>=0.30.0",
    "google-generativeai>=0.5.0",
    "httpx>=0.25.0",
]
audio = [
    "soundfile>=0.12",
    "pydub>=0.25",
]
docs = [
    "sphinx>=5.0",
    "sphinx-rtd-theme>=1.0",
]
mcp = [
    "fastmcp>=2.0.0",
    "python-dotenv>=1.0.0",
    "openai>=1.0.0",
    "anthropic>=0.30.0",
    "google-generativeai>=0.5.0",
    "httpx>=0.25.0",
]
all = [
    "kmidi[dev,audio,docs,mcp]",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["music_brain*", "mcp_workstation*", "mcp_todo*", "mcp_penta_swarm*", "penta_core*"]

[project.scripts]
mcp-penta-swarm = "mcp_penta_swarm.server:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
ignore_missing_imports = true

[tool.flake8]
max-line-length = 100
extend-ignore = ["E203", "W503"]
```

### requirements.txt (Core)

Key dependencies (see pyproject.toml for full list):

**Core Scientific Computing:**
- numpy>=1.24.0,<2.0.0
- scipy>=1.10.0

**Audio Processing:**
- librosa>=0.10.0
- soundfile>=0.12.0
- audioread>=3.0.0
- resampy>=0.4.0

**PyTorch (CPU/MPS for Mac):**
- torch>=2.0.0,<2.3.0
- torchaudio>=2.0.0,<2.3.0

**ML/AI Frameworks:**
- transformers>=4.30.0
- onnx>=1.14.0
- onnxruntime>=1.15.0

**Data Processing:**
- pandas>=2.0.0
- mido>=1.3.0
- pyyaml>=6.0

**Development:**
- pytest>=7.3.0
- black>=23.3.0
- mypy>=1.3.0

### requirements-production.txt

Production-specific dependencies for API server:

```txt
# Core Dependencies
numpy>=1.21.0
torch>=2.0.0
librosa>=0.10.0
pyyaml>=6.0.0
scipy>=1.8.0

# Audio Processing
soundfile>=0.12.0
pydub>=0.25.0

# FastAPI Production Service
fastapi>=0.125.0
uvicorn[standard]>=0.30.0
slowapi>=0.1.9
pydantic>=2.0.0
python-multipart>=0.0.6

# MIDI Processing
mido>=1.2.10
pretty_midi>=0.2.10

# HTTP Client
requests>=2.31.0
httpx>=0.25.0

# Data Processing
pandas>=1.4.0

# Security & Utilities
certifi>=2024.0.0
cryptography>=41.0.0
```

### CMakeLists.txt (Root)

Key build configuration:

```cmake
cmake_minimum_required(VERSION 3.27)
project(Kelly VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_DESKTOP "Build desktop GUI application" ON)
option(BUILD_PLUGINS "Build VST3 and CLAP plugins" ON)
option(BUILD_TESTS "Build tests" OFF)
option(BUILD_KELLY_CORE "Build Kelly core library/app/plugins" ON)
option(BUILD_PENTA_TESTS "Build penta_core test suite" OFF)
option(ENABLE_TRACY "Enable Tracy profiling" OFF)
option(ENABLE_RTNEURAL "Enable RTNeural for ML inference" OFF)
option(ENABLE_ONNX_RUNTIME "Enable ONNX Runtime for ML inference" OFF)

# Dependencies
find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
add_subdirectory(external/JUCE EXCLUDE_FROM_ALL)

# Penta-Core Library
add_subdirectory(src_penta-core)

# Python Bindings (Optional)
option(BUILD_PYTHON_BINDINGS "Build Python bindings for penta-core" ON)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        add_subdirectory(bindings)
    endif()
endif()

# Main Kelly library
if(BUILD_KELLY_CORE)
    add_library(KellyCore STATIC ${KELLY_CORE_SOURCES})
    target_link_libraries(KellyCore PUBLIC
        Qt6::Core
        Qt6::Widgets
        juce::juce_audio_basics
        juce::juce_audio_devices
        juce::juce_audio_formats
        juce::juce_audio_processors
        juce::juce_core
        juce::juce_dsp
        juce::juce_osc
        readerwriterqueue
    )
endif()
```

### Environment Configuration (.env.example)

```bash
# Storage Configuration (REQUIRED for production)
KELLY_AUDIO_DATA_ROOT=/path/to/audio/data

# Alternative: External SSD mount point
# KELLY_SSD_PATH=/Volumes/Extreme SSD

# Training Configuration
# KELLY_DEVICE=auto  # auto, mps, cuda, cpu
# KELLY_TRAINING_BUDGET=100.0

# API Keys (for dataset downloading)
# FREESOUND_API_KEY=your_api_key_here

# Logging
# KELLY_LOG_LEVEL=INFO
# KELLY_LOGS_DIR=logs/training

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,http://localhost:1420
```

### Build Variants (config/*.yaml)

**build-dev-mac.yaml:**
```yaml
build: dev-mac
device: mps
python: 3.11
inference_only: true
enable_compile: true
quantize: false
logging_level: DEBUG
demo_mode: true

paths:
  data_root: ${DATA_ROOT:-data}
  models_root: ${MODELS_ROOT:-models}
  output_root: ${OUTPUT_ROOT:-output}
  checkpoints: ${MODELS_ROOT:-models}/checkpoints

performance:
  target_latency_ms: 150
  profile_enabled: true
  memory_monitor: true
  max_batch_size: 16

api:
  host: 127.0.0.1
  port: 8000
  workers: 2
  request_timeout: 60
```

**build-train-nvidia.yaml:**
- Optimized for NVIDIA GPU training
- CUDA acceleration enabled
- Larger batch sizes

**build-prod-aws.yaml:**
- AWS p3.2xlarge configuration
- Production deployment settings
- Scalable configuration

---

## 4. API Contracts & Schemas

### FastAPI Endpoints

**Base URL:** `http://127.0.0.1:8000`

#### Health & Status Endpoints

**GET /health**
- Purpose: Health check with service status
- Response: `HealthResponse` with status, version, timestamp, services, system metrics
- Rate limit: None

**GET /ready**
- Purpose: Readiness probe (Kubernetes)
- Response: `{"status": "ready", "timestamp": float}`
- Status codes: 200 (ready), 503 (not ready)

**GET /live**
- Purpose: Liveness probe (Kubernetes)
- Response: `{"status": "alive", "timestamp": float, "uptime_seconds": float}`
- Always returns 200 if process is alive

**GET /metrics**
- Purpose: Prometheus-compatible metrics endpoint
- Response: Prometheus text format
- Environment: `ENABLE_METRICS=true` required

#### Music Generation Endpoints

**GET /emotions**
- Purpose: List available emotional presets
- Response: `{"emotions": [str], "count": int}`
- Rate limit: 100/minute

**POST /generate**
- Purpose: Generate music from emotional intent
- Request body: `GenerateRequest`
  ```json
  {
    "intent": {
      "emotional_intent": "string (required)",
      "core_wound": "string (optional)",
      "core_desire": "string (optional)",
      "technical": {
        "key": "string (optional)",
        "bpm": "int (optional)",
        "progression": ["string"] (optional),
        "genre": "string (optional)"
      }
    },
    "output_format": "midi" | "audio"
  }
  ```
- Response: `{"status": "success", "result": {...}, "request_id": str, "generation_time_seconds": float}`
- Rate limit: 10/minute

**POST /interrogate**
- Purpose: Conversational intent refinement
- Request body: `InterrogateRequest`
  ```json
  {
    "message": "string (required)",
    "session_id": "string (optional)",
    "context": {} (optional)
  }
  ```
- Response: `{"status": "success", "reply": str, "session_id": str, "suggestions": [str], "request_id": str}`
- Rate limit: 30/minute

### Request/Response Models

**TechnicalIntent:**
```python
class TechnicalIntent(BaseModel):
    key: Optional[str] = None
    bpm: Optional[int] = None
    progression: Optional[list[str]] = None
    genre: Optional[str] = None
```

**EmotionalIntent:**
```python
class EmotionalIntent(BaseModel):
    core_wound: Optional[str] = None
    core_desire: Optional[str] = None
    emotional_intent: str  # Required
    technical: Optional[TechnicalIntent] = None
```

**GenerateRequest:**
```python
class GenerateRequest(BaseModel):
    intent: EmotionalIntent
    output_format: Optional[str] = "midi"
```

**InterrogateRequest:**
```python
class InterrogateRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
```

**HealthResponse:**
```python
class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded"
    version: str
    timestamp: float
    services: Dict[str, bool]
    system: Optional[Dict[str, Any]] = None  # CPU, memory metrics
```

### MCP Server Protocols

**mcp_workstation Commands:**
```bash
python -m mcp_workstation status              # Show workstation dashboard
python -m mcp_workstation register claude     # Register as Claude
python -m mcp_workstation propose claude "Title" "Desc" architecture
python -m mcp_workstation vote claude PROP_ID 1
python -m mcp_workstation phases              # Show phase progress
python -m mcp_workstation cpp                 # Show C++ transition plan
python -m mcp_workstation ai                  # Show AI specializations
python -m mcp_workstation server              # Run MCP server
```

**mcp_todo Commands:**
```bash
python -m mcp_todo.cli add "Task" --priority high --tags "code,urgent"
python -m mcp_todo.cli list --status pending
python -m mcp_todo.cli complete <id>
python -m mcp_todo.cli summary
```

**mcp_penta_swarm Tools:**
- `consult_architect` - OpenAI GPT-4o (high-level logic)
- `consult_developer` - Anthropic Claude 3.5 Sonnet (clean code)
- `consult_researcher` - Google Gemini 1.5 Pro (deep context)
- `consult_maverick` - xAI Grok Beta (creative problem-solving)
- `fetch_repo_context` - GitHub API (repository context)

---

## 5. Data Schemas

### Three-Phase Intent Schema

#### Phase 0: Core Wound/Desire (Deep Interrogation)

```python
{
    "core_event": str,           # What happened?
    "core_resistance": str,      # What holds you back from saying it?
    "core_longing": str,         # What do you want to feel?
    "core_stakes": str,          # What's at risk?
    "core_transformation": str   # How should you feel when done?
}
```

#### Phase 1: Emotional Intent

```python
{
    "mood_primary": str,                    # Dominant emotion
    "mood_secondary_tension": float,        # Internal conflict (0.0-1.0)
    "vulnerability_scale": str,             # "Low" | "Medium" | "High"
    "narrative_arc": str                    # "Climb-to-Climax" | "Slow Reveal" | "Repetitive Despair" | ...
}
```

#### Phase 2: Technical Constraints

```python
{
    "technical_genre": str,                 # Genre selection
    "technical_key": str,                   # Musical key (e.g., "C", "Am")
    "technical_mode": str,                  # Mode (e.g., "major", "minor", "dorian")
    "technical_rule_to_break": str,         # Intentional rule violation enum
    "rule_breaking_justification": str      # WHY break this rule (required!)
}
```

### Emotion Thesaurus Structure

**Schema Version:** 1.0.0
**Name:** DAiW Emotion Thesaurus
**Description:** Comprehensive 6×6×6 emotion taxonomy with intensity tiers

**Hierarchy:**
- Level 1: Base Emotions (6) - HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST
- Level 2: Sub-Emotions (6 per base = 36 total)
- Level 3: Sub-Sub-Emotions (6 per sub = 216 total)
- Intensity Tiers: 6 per sub-sub-emotion (1296 intensity nodes)
- Synonyms: 3-5 words/phrases per tier

**Total Nodes:** 216
**Total Intensity Tiers:** 1296
**Blend Zones:** 50

**Base Emotions:**
```json
[
    {"id": "I", "name": "HAPPY", "file": "happy.json", "valence": "positive"},
    {"id": "II", "name": "SAD", "file": "sad.json", "valence": "negative"},
    {"id": "III", "name": "ANGRY", "file": "angry.json", "valence": "negative"},
    {"id": "IV", "name": "FEAR", "file": "fear.json", "valence": "negative"},
    {"id": "V", "name": "SURPRISE", "file": "surprise.json", "valence": "mixed"},
    {"id": "VI", "name": "DISGUST", "file": "disgust.json", "valence": "negative"}
]
```

**Intensity Tier Definitions:**
1. **Subtle** (arousal_modifier: 0.1) - Barely perceptible, background feeling
2. **Mild** (arousal_modifier: 0.3) - Noticeable but easily managed
3. **Moderate** (arousal_modifier: 0.5) - Clearly felt, influences behavior
4. **Strong** (arousal_modifier: 0.7) - Powerful, hard to ignore
5. **Intense** (arousal_modifier: 0.9) - Overwhelming, dominates experience
6. **Overwhelming** (arousal_modifier: 1.0) - All-consuming, transcendent

**Musical Mapping Hints:**
- **Valence to Mode:**
  - Positive → major modes, lydian
  - Negative → minor modes, phrygian, locrian
  - Mixed → modal mixture, borrowed chords

- **Arousal to Tempo:**
  - Low → 40-70 BPM
  - Medium → 70-120 BPM
  - High → 120-180+ BPM

- **Intensity to Dynamics:**
  - Subtle → pp-p
  - Mild → p-mp
  - Moderate → mp-mf
  - Strong → mf-f
  - Intense → f-ff
  - Overwhelming → ff-fff with dynamic contrast

**Theoretical Foundations:**
- Primary Models: Ekman Basic Emotions, Plutchik Emotion Wheel, Russell Circumplex Model
- Music-Specific: GEMS (Geneva Emotional Music Scale), Juslin BRECVEMA Model
- Therapeutic: Affect Regulation Theory, Polyvagal Theory, DBT Emotion Regulation

### Rule-Breaking Categories

**Harmony Rules:**
- `HARMONY_AvoidTonicResolution` - Unresolved yearning (use for: Grief, longing)
- `HARMONY_ConstantTritones` - Dissonance and tension
- `HARMONY_ModalMixture` - Borrowed chords for emotional complexity

**Rhythm Rules:**
- `RHYTHM_ConstantDisplacement` - Anxiety, restlessness
- `RHYTHM_Polyrhythmic` - Complexity and conflict
- `RHYTHM_AsymmetricMeters` - Unpredictability

**Arrangement Rules:**
- `ARRANGEMENT_BuriedVocals` - Dissociation
- `ARRANGEMENT_ContrastingTextures` - Emotional shifts
- `ARRANGEMENT_UnbalancedMix` - Intentional imbalance

**Production Rules:**
- `PRODUCTION_PitchImperfection` - Emotional honesty
- `PRODUCTION_DynamicCompression` - Controlled chaos
- `PRODUCTION_SpectralDistortion` - Intensity and rawness

### Chord Progression Families

**Common Progression Families:**
- I-V-vi-IV (Pop progression)
- vi-IV-I-V (Pop progression variant)
- I-vi-IV-V (50s progression)
- ii-V-I (Jazz progression)
- I-bVII-IV (Mixolydian progression)

**Structure:** Stored in `data/chord_progression_families.json` and `data/chord_progressions_db.json`

### Genre Groove Maps

**Structure:** Stored in `data/genre_pocket_maps.json`

**Format:**
```json
{
    "genre_name": {
        "timing_deviation_ms": float,
        "velocity_variation": float,
        "swing_ratio": float,
        "ghost_notes": bool,
        "humanization_level": "subtle" | "moderate" | "strong"
    }
}
```

**Supported Genres:** funk, jazz, rock, pop, hip-hop, electronic, etc.

---

## 6. Development Workflows

### Setup Procedures

**Python Installation:**
```bash
# Core installation
pip install -e .

# With optional dependencies
pip install -e ".[dev]"      # pytest, black, flake8, mypy
pip install -e ".[audio]"    # librosa, soundfile
pip install -e ".[theory]"   # music21
pip install -e ".[ui]"       # streamlit
pip install -e ".[all]"      # Everything
```

**C++ Build:**
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja penta_core           # Build the library
ninja penta_tests          # Build tests
ctest --output-on-failure  # Run tests
```

### Testing Procedures

**Python Tests:**
```bash
# Music Brain tests
pytest tests_music-brain/ -v

# All tests with coverage
pytest tests_music-brain/ -v --cov=music_brain --cov-report=term-missing

# Specific test file
pytest tests/test_intent_processor.py -v
```

**C++ Tests:**
```bash
cd build
ctest --output-on-failure

# Run specific test suite
./penta_tests --gtest_filter="*Harmony*"
./penta_tests --gtest_filter="*Groove*"
./penta_tests --gtest_filter="*Performance*"
```

**Test Locations:**
- `tests/` - C++ unit tests (groove, harmony, simd, memory)
- `tests_music-brain/` - Python integration tests
- `tests_penta-core/` - Penta-Core C++ tests (performance, OSC, etc.)
- `DAiW-Music-Brain/tests/` - Music Brain internal tests (if exists)

### Code Style & Conventions

**Python:**
```bash
# Format
black music_brain/ tests/

# Type check
mypy music_brain/

# Lint
flake8 music_brain/ tests/
```

**Standards:**
- Line length: 100 characters
- Formatter: black
- Type hints: Required for public APIs
- Python version: Target 3.9+

**C++:**
- Standard: C++20
- Naming: PascalCase (classes), camelCase (methods), snake_case (variables)
- RT-Safety: Mark audio callbacks `noexcept`, no allocations
- Memory: Use `std::pmr` containers where possible
- SIMD: Use AVX2 optimizations with scalar fallback

**Code Patterns:**
1. Lazy imports in Python CLI for fast startup
2. Data classes with `to_dict()`/`from_dict()` serialization
3. Enums for categorical values
4. Singleton pattern for managers (MemoryManager, Workstation)
5. Lock-free ring buffers for audio/UI communication

### Common Development Tasks

**Adding a New Groove Genre:**
1. Add entry to `data/genre_pocket_maps.json`
2. Add template in `music_brain/groove/templates.py`
3. Add to CLI choices in `music_brain/cli.py`

**Adding a Rule-Breaking Option:**
1. Add enum value in `music_brain/session/intent_schema.py`
2. Add entry in `RULE_BREAKING_EFFECTS` dict
3. Implement in `intent_processor.py`

**Adding a Penta-Core Engine:**
1. Create header in `include/penta/<subsystem>/`
2. Implement in `src_penta-core/<subsystem>/`
3. Update `src_penta-core/CMakeLists.txt`
4. Add Python bindings in `python/penta_core/`
5. Add tests in `tests_penta-core/`

**Adding an iDAW_Core Plugin:**
1. Create plugin directory in `iDAW_Core/plugins/<Name>/`
2. Add `include/<Name>Processor.h`, `src/<Name>Processor.cpp`
3. Add shader files in `shaders/`
4. Register in CMakeLists.txt

**Adding a Music Theory Rule:**
1. Create rule class in `python/penta_core/rules/`
2. Add to appropriate rules module (harmony, rhythm, etc.)
3. Add teacher support in `python/penta_core/teachers/`
4. Add tests

---

## 7. CI/CD Configuration

### Main Workflows

**ci.yml** - Main CI Pipeline
- **Python Tests**: pytest with coverage (Python 3.9, 3.11)
- **C++ Build**: CMake/Ninja on Ubuntu
- **C++ Tests**: Automated test execution
- **Valgrind Memory Testing**: Leak detection (Debug build)
- **Performance Regression**: Benchmark checks (<200ms latency target)
- **Code Coverage**: Coverage reporting (Python + C++)
- **JUCE Plugin Validation**: macOS plugin validation
- **Code Quality**: black, flake8, mypy checks

**sprint_suite.yml** - Comprehensive Sprint-Based Testing
- **Sprint 1**: Core testing & quality
- **Sprint 2**: C++ build & integration
- **Sprint 3**: Documentation checks
- **Sprint 5**: Platform matrix (Linux/macOS/Windows x Python 3.9-3.13)
- **Sprint 6**: Advanced theory and AI
- **Sprint 7**: Mobile/Web (Streamlit)
- **Sprint 8**: Enterprise tests

**platform_support.yml** - Cross-Platform Testing
- Python versions: 3.9, 3.10, 3.11, 3.12, 3.13
- Platforms: Linux, macOS, Windows
- Matrix testing across all combinations

**release.yml** - Release Builds
- **Desktop Apps**: Build for macOS, Linux, Windows
- **Artifacts**: Zipped applications
- **Python Distribution**: Build wheels and source distributions
- **C++ Libraries**: Build static/dynamic libraries

### Performance Targets

- **Harmony Engine**: < 100μs latency @ 48kHz/512 samples
- **Groove Engine**: < 200μs latency @ 48kHz/512 samples
- **Overall System**: < 200ms total latency target
- **Memory**: No leaks (Valgrind clean)

### Code Coverage Targets

- **Minimum Coverage**: 80% code coverage required
- **Python**: pytest-cov with term-missing report
- **C++**: lcov with coverage.info generation

---

## 8. Code Style & Conventions

### Python Style Guide

**Formatting:**
- Tool: `black`
- Line length: 100 characters
- Target versions: py39, py310, py311

**Type Checking:**
- Tool: `mypy`
- Python version: 3.9
- Settings: `warn_return_any = true`, `ignore_missing_imports = true`

**Linting:**
- Tool: `flake8`
- Max line length: 100
- Ignore: E203, W503

**Required Practices:**
- Type hints required for public APIs
- Data classes with `to_dict()`/`from_dict()` serialization
- Lazy imports in CLI for fast startup

### C++ Style Guide

**Standard:** C++20
**Compiler Flags:**
- MSVC: `/W4`
- GCC/Clang: `-Wall -Wextra -Wpedantic`

**Naming Conventions:**
- Classes: PascalCase (e.g., `HarmonyEngine`)
- Methods: camelCase (e.g., `processAudio()`)
- Variables: snake_case (e.g., `sample_rate`)
- Constants: kConstantName (e.g., `kDefaultSampleRate`)

**RT-Safety Rules:**
1. All `processAudio()` methods marked `noexcept`
2. No memory allocation in audio callbacks
3. Use lock-free data structures for thread communication
4. Default sample rate: 44100.0 Hz

**Memory Management:**
- Use `std::pmr` containers where possible
- Pre-allocated memory pools for audio thread
- Lock-free ring buffers for audio/UI communication

**SIMD Optimizations:**
- Use AVX2 when available
- Scalar fallback for compatibility
- SIMD implementations in separate files (e.g., `ChordAnalyzerSIMD.cpp`)

### Code Patterns

1. **Lazy Imports**: Python CLI imports modules only when needed for fast startup
2. **Data Classes**: Use dataclasses with serialization methods
3. **Enums**: Use enums for categorical values (rule-breaking types, emotion categories)
4. **Singleton**: Use singleton pattern for managers (MemoryManager, Workstation)
5. **Lock-Free Communication**: Use lock-free ring buffers for audio/UI communication

---

## 9. Key Architecture Decisions

### 1. Dual-Engine Design

**Side A (C++):**
- Real-time audio processing
- Deterministic behavior
- Lock-free operations
- Pre-allocated memory (4GB)
- No deallocation during runtime
- Thread-safe for real-time audio

**Side B (Python):**
- AI generation and dynamic logic
- May block during operations
- Dynamic memory allocation allowed
- Used for AI generation and UI
- NEVER used from audio thread

**Communication:**
- Lock-free ring buffer (Side B → Side A)
- PythonBridge embeds Python interpreter in Side B
- `call_iMIDI()` - Pass knob state + text prompt, get MIDI buffer
- "Ghost Hands" - AI-suggested knob movements
- Fail-safe: Returns C Major chord on Python failure

### 2. Intent-Driven Composition

**Core Principle:**
- Emotional intent drives technical choices
- Phase 0 (why) must precede Phase 2 (how)
- Rule-breaking requires explicit justification

**Three-Phase Schema:**
1. **Phase 0**: Core Wound/Desire (deep interrogation)
2. **Phase 1**: Emotional Intent (mood, vulnerability, narrative arc)
3. **Phase 2**: Technical Constraints (genre, key, mode, rule-breaking)

**Rule-Breaking Philosophy:**
- Every rule break must have emotional justification
- Technical choices serve emotional expression
- Tool educates and empowers, doesn't just generate

### 3. Multi-AI Collaboration

**MCP Workstation System:**
- Each AI has specializations and limitations
- Proposal system with voting mechanism
- Task assignment based on AI strengths

**AI Specializations:**
- `claude` - Code architecture, real-time safety, complex debugging
- `chatgpt` - Theory analysis, explanations, documentation
- `gemini` - Cross-language patterns, multi-modal analysis
- `github_copilot` - Code completion, boilerplate generation

**Cross-AI Task Sync:**
- Tasks created in one AI appear in all others
- File-based storage: `~/.mcp_todo/todos.json`
- Rich task model: priority, tags, projects, due dates, notes, subtasks

### 4. RT-Safety Architecture

**Core Rules:**
- Audio thread never waits on UI/AI
- Lock-free communication via ring buffers
- Pre-allocated memory pools (4GB for Side A)
- No allocations in audio callbacks
- All `processAudio()` methods marked `noexcept`

**Memory Architecture:**
- **Side A**: `std::pmr::monotonic_buffer_resource` - 4GB pre-allocated
- **Side B**: `std::pmr::synchronized_pool_resource` - Dynamic allocation allowed
- Communication: Lock-free ring buffer

**Testing:**
- Valgrind for leak detection
- RT-safe assertions in debug builds
- Performance regression testing (<200ms latency target)

### 5. MCP Protocol Integration

**Standard Protocol:**
- Model Context Protocol (MCP) for AI tool integration
- Multiple MCP servers for different purposes
- Cross-AI task synchronization

**MCP Servers:**
- `mcp_workstation/`: Multi-AI orchestration with proposals/voting
- `mcp_todo/`: Cross-AI task management
- `mcp_penta_swarm/`: Multi-AI aggregation (GPT-4o, Claude, Gemini, Grok)
- `daiw_mcp/`: DAiW-specific MCP server

---

## 10. Documentation Structure

### Documentation Directories

**docs/**
- `sprints/` - Sprint documentation
- `summaries/` - Project summaries
- `integrations/` - Integration guides
- `ai_setup/` - AI assistant setup guides
- `music_brain/` - Music Brain documentation
- `penta_core/` - Penta-Core documentation
- `references/` - Reference materials

**Production_Workflows/**
- Production workflow guides
- Best practices for music production
- Mixing and mastering workflows

**Songwriting_Guides/**
- Intent schema documentation
- Rule-breaking guides
- Songwriting methodology

**Theory_Reference/**
- Music theory reference materials
- Harmonic analysis guides
- Rhythm and groove reference

**vault/**
- Obsidian-compatible knowledge base
- `Songwriting_Guides/` - Intent schema, rule-breaking guides
- `Songs/` - Song-specific project files
- Uses `[[wiki links]]` for cross-referencing

**Templates/**
- Project templates
- Document templates
- Intent schema templates

### Key Documentation Files

**Root Level:**
- `PROJECT_MAIN.md` - Main project entry point (this file's companion)
- `PROJECT_QUICK_START.md` - Quick reference guide
- `PROJECT_BACKUP_CORE_CRITERIA.md` - This comprehensive backup
- `CLAUDE.md` - Comprehensive AI assistant guide
- `KMIDI_README.md` - Repository overview
- `README.md` - Current project README
- `SETUP_GUIDE.md` - Setup instructions
- `QUICK_START.md` - Quick start guide

**Implementation Guides:**
- `IMPLEMENTATION_PLAN.md` - 24-week phased roadmap
- `IMPLEMENTATION_ALTERNATIVES.md` - Route A/B/C comparison
- `BUILD_VARIANTS.md` - Hardware-specific builds

**Architecture:**
- `docs/ARCHITECTURE.md` - System architecture
- `docs/iDAW_IMPLEMENTATION_GUIDE.md` - Complete architecture guide
- `docs/TIER123_MAC_IMPLEMENTATION.md` - Detailed Mac implementation

---

## Excluded from Backup

The following are explicitly excluded from this backup (as they contain audio training data):

- Audio files (`.wav`, `.mp3`, `.flac`, `.aiff`, `.ogg`, etc.)
- Training datasets:
  - MAESTRO v3.0 raw data
  - CREMA-D raw data
  - RAVDESS raw data
  - TESS raw data
  - GTZAN dataset
- Generated outputs (`output/`, `checkpoints/`)
- Model binary files (`.bin`, `.mlmodel`, `.pth`, `.ckpt`, `.onnx`, `.h5`)
- Large data files in `datasets/`, `training/` directories
- External audio libraries and samples

**Note:** This backup contains only the structure, schemas, and configuration for data files. Actual audio data and trained models are not included and should be backed up separately if needed.

---

## Backup Metadata

- **Created:** 2025-01-09
- **Project Version:** 1.0.0
- **Backup Format:** Markdown (comprehensive documentation)
- **Companion File:** `PROJECT_BACKUP_CORE_CRITERIA.json` (structured data)
- **Purpose:** Complete restoration of project structure, configuration, and core criteria
- **Excludes:** Audio training data, model binaries, generated outputs

---

**END OF BACKUP**
