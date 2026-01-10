# KmiDi: Unified Music Intelligence & Audio Workstation

**Version:** 1.0.0
**Status:** Active Development
**Last Updated:** 2025-01-09

---

## Project Philosophy

**"Interrogate Before Generate"** - The tool shouldn't finish art for people. It should make them braver.

KmiDi is a therapeutic interactive Digital Audio Workstation (iDAW) that translates emotions into music using a unique three-phase intent system:

1. **Wound** → Identify the emotional trigger
2. **Emotion** → Map to the 216-node emotion thesaurus
3. **Rule-breaks** → Express through intentional musical violations

> "The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"

Technical implementation serves emotional expression. The tool educates and empowers - it doesn't just generate.

---

## What is KmiDi?

KmiDi is a unified repository structure consolidating:
- **music_brain**: Python-based music intelligence and AI-assisted composition
- **penta_core**: C++ real-time audio processing engines
- **iDAW_Core**: JUCE-based plugin suite
- **mcp_workstation**: Multi-AI orchestration for collaborative development

### Tech Stack

| Component | Technology |
|-----------|------------|
| Brain | Python 3.11 (music21, librosa, mido, torch) |
| Body | C++20 (JUCE 8, Qt 6, CMake) |
| Plugins | CLAP 1.2, VST3 3.7 |
| Audio | CoreAudio (macOS), ASIO (Windows), JACK (Linux) |
| API | FastAPI (REST), WebSocket (real-time) |
| Frontend | React (Tauri), Streamlit (web), Qt (desktop) |

---

## Quick Start

### Prerequisites

- **Python**: 3.9+ (tested 3.9-3.13)
- **Node**: 18+ (for React/Tauri frontend)
- **Rust**: Latest stable (for Tauri desktop bridge)
- **C++**: C++20 compatible compiler
- **CMake**: 3.27+
- **JUCE**: 8.0.10 (included in `external/`)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd KmiDi-1

# Install Python package with dependencies
pip install -e ".[dev]"      # Development dependencies
pip install -e ".[audio]"    # Audio processing libraries
pip install -e ".[all]"      # All optional dependencies

# Install Node dependencies (for frontend)
npm install

# Build C++ libraries (optional, for real-time engines)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja penta_core
```

### Running the Application

#### Option 1: API Server (Backend Only)

```bash
# Start FastAPI server
python -m music_brain.api
# Or use the production API
cd api && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Test the API
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/emotions
```

#### Option 2: Desktop App (React + Tauri)

```bash
# Terminal 1: Start API server
python -m music_brain.api  # or: cd api && uvicorn main:app --reload

# Terminal 2: Launch desktop app
npm run tauri dev
```

#### Option 3: Web Interface (Streamlit)

```bash
# Start API server first, then:
streamlit run streamlit_app.py
```

### First Music Generation

```bash
# Using the API
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "I feel peaceful and calm, like a quiet morning",
      "core_wound": "fear of being forgotten",
      "core_desire": "to feel seen"
    },
    "output_format": "midi"
  }'
```

---

## Repository Architecture

This is a **monorepo** containing multiple subsystems:

```
KmiDi-1/
├── music_brain/               # Python Music Intelligence Toolkit (main package)
│   ├── tier1/                 # Pretrained models (no fine-tuning)
│   ├── tier2/                 # LoRA fine-tuning
│   ├── session/               # Intent schema, teaching, interrogation
│   ├── structure/             # Chord/progression analysis
│   ├── groove/                # Groove extraction/application
│   ├── harmony/               # Harmony engine
│   ├── melody/                # Melody generation
│   └── audio/                 # Audio feel analysis
│
├── mcp_workstation/           # MCP Multi-AI Workstation (orchestration)
├── mcp_todo/                  # MCP TODO Server (cross-AI task management)
├── mcp_penta_swarm/           # MCP Swarm Server (multi-AI aggregation)
├── daiw_mcp/                  # DAiW MCP Server
│
├── iDAW_Core/                 # JUCE Plugin Suite (C++)
│   └── plugins/               # Audio plugins (Pencil, Eraser, Palette, etc.)
│
├── src_penta-core/            # Penta-Core C++ Engines (implementation)
├── src/                       # Additional C++ source files
├── include/                   # C++ Headers (including penta/)
├── python/penta_core/         # Python bindings for Penta-Core
│
├── api/                       # FastAPI production service
├── scripts/                   # Standalone utility scripts
├── tools/                     # Development tools
│
├── tests/                     # All tests (C++, Python, music_brain, penta_core)
├── examples/                  # Example scripts
├── benchmarks/                # Performance benchmarks
│
├── data/                      # JSON data files (emotions, progressions, scales, grooves)
├── docs/                      # All documentation
├── vault/                     # Obsidian Knowledge Base
├── Production_Workflows/      # Production workflow guides
├── Songwriting_Guides/        # Songwriting methodology guides
└── Theory_Reference/          # Music theory reference materials
```

---

## Core Modules

### 1. music_brain/ - Music Intelligence

Python-based music generation and analysis toolkit.

**Key Features:**
- Three-phase intent schema (Phase 0: Wound/Desire, Phase 1: Emotion, Phase 2: Technical)
- 216-node emotion thesaurus (6×6×6 structure, 1296 intensity tiers)
- Intent-driven composition with rule-breaking support
- Groove extraction and application
- Chord progression analysis and generation

**CLI Commands (`daiw`):**
```bash
daiw extract drums.mid            # Extract groove from MIDI
daiw apply --genre funk track.mid # Apply genre groove template
daiw analyze --chords song.mid    # Analyze chord progression
daiw diagnose "F-C-Am-Dm"         # Diagnose harmonic issues
daiw intent new --title "My Song" # Create intent template
daiw intent suggest grief         # Suggest rules to break
daiw teach rulebreaking           # Interactive teaching mode
```

**Architecture:**
- **Tier 1**: Pretrained models (no fine-tuning) - ready now
- **Tier 2**: LoRA fine-tuning (97% parameter reduction)
- **Tier 3**: Full training (future, Phase 3)

### 2. penta_core/ - Real-time Engines

C++ implementations of high-performance, RT-safe audio analysis engines.

**C++ Headers (`include/penta/`):**
- `common/` - RTTypes, RTLogger, RTMemoryPool, SIMDKernels
- `groove/` - GrooveEngine, OnsetDetector, TempoEstimator, RhythmQuantizer
- `harmony/` - HarmonyEngine, ChordAnalyzer, ScaleDetector, VoiceLeading
- `diagnostics/` - DiagnosticsEngine, AudioAnalyzer, PerformanceMonitor
- `osc/` - OSCHub, OSCClient, OSCServer, OSCMessage, RTMessageQueue

**Python Bindings:**
```python
from penta_core import PentaCore, HarmonyEngine, GrooveEngine

core = PentaCore(sample_rate=48000.0)
core.process(audio_buffer, midi_notes=[(60, 100), (64, 100)])
state = core.get_state()  # chord, scale, groove, diagnostics
```

**RT-Safety Rules:**
1. All `processAudio()` methods are marked `noexcept`
2. No memory allocation in audio callbacks
3. Use lock-free data structures for thread communication
4. Default sample rate: 44100.0 Hz

### 3. iDAW_Core/ - JUCE Plugin Suite

Art-themed audio plugins built on JUCE 8.

**Plugins:**
| Plugin | Description | Priority |
|--------|-------------|----------|
| **Pencil** | Sketching/drafting audio ideas | HIGH |
| **Eraser** | Audio removal/cleanup | HIGH |
| **Palette** | Tonal coloring/mixing | MID |
| **Smudge** | Audio blending/smoothing | MID |
| **Press** | Dynamics/compression | HIGH |

**Dual-Heap Memory Architecture:**
- **Side A ("Work State")**: 4GB pre-allocated, no deallocation, RT-safe
- **Side B ("Dream State")**: Dynamic allocation, AI generation, UI operations
- **Communication**: Lock-free ring buffer (Side B → Side A)

### 4. MCP Servers

**mcp_workstation/**: Multi-AI orchestration
```bash
python -m mcp_workstation status
python -m mcp_workstation propose claude "Title" "Desc" architecture
python -m mcp_workstation vote claude PROP_ID 1
```

**mcp_todo/**: Cross-AI task management
```bash
python -m mcp_todo.cli add "Task" --priority high
python -m mcp_todo.cli list --status pending
```

**mcp_penta_swarm/**: Multi-AI aggregation (GPT-4o, Claude, Gemini, Grok)

---

## Three-Phase Intent Schema

### Phase 0: Core Wound/Desire (Deep Interrogation)
- `core_event` - What happened?
- `core_resistance` - What holds you back from saying it?
- `core_longing` - What do you want to feel?
- `core_stakes` - What's at risk?
- `core_transformation` - How should you feel when done?

### Phase 1: Emotional Intent
- `mood_primary` - Dominant emotion
- `mood_secondary_tension` - Internal conflict (0.0-1.0)
- `vulnerability_scale` - Low/Medium/High
- `narrative_arc` - Climb-to-Climax, Slow Reveal, Repetitive Despair, etc.

### Phase 2: Technical Constraints
- `technical_genre` - Genre selection
- `technical_key` - Musical key
- `technical_mode` - Mode (major, minor, dorian, etc.)
- `technical_rule_to_break` - Intentional rule violation
- `rule_breaking_justification` - WHY break this rule (required!)

### Rule-Breaking Categories

| Category | Examples | Effect |
|----------|----------|--------|
| Harmony | `HARMONY_AvoidTonicResolution` | Unresolved yearning |
| Rhythm | `RHYTHM_ConstantDisplacement` | Anxiety, restlessness |
| Arrangement | `ARRANGEMENT_BuriedVocals` | Dissociation |
| Production | `PRODUCTION_PitchImperfection` | Emotional honesty |

---

## API Reference

### Endpoints

**Health & Status:**
- `GET /health` - Health check with service status
- `GET /ready` - Readiness probe (Kubernetes)
- `GET /live` - Liveness probe (Kubernetes)
- `GET /metrics` - Prometheus-compatible metrics

**Music Generation:**
- `GET /emotions` - List available emotional presets
- `POST /generate` - Generate music from emotional intent
- `POST /interrogate` - Conversational intent refinement

### Example: Generate Music

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "intent": {
      "emotional_intent": "I feel anxious but hopeful, like things will get better",
      "core_wound": "fear of being forgotten",
      "core_desire": "to feel seen",
      "technical": {
        "key": "C",
        "bpm": 90,
        "progression": ["I", "V", "vi", "IV"],
        "genre": "indie"
      }
    },
    "output_format": "midi"
  }'
```

**Response:**
```json
{
  "status": "success",
  "result": {
    "affect": {
      "primary": "anxious_hopeful",
      "secondary": "anticipation",
      "intensity": 0.7
    },
    "plan": {
      "root_note": "C",
      "mode": "major",
      "tempo_bpm": 90,
      "length_bars": 16,
      "chord_symbols": ["C", "G", "Am", "F"],
      "complexity": "moderate"
    }
  },
  "request_id": "abc123",
  "generation_time_seconds": 0.45
}
```

---

## Development Workflow

### Running Tests

```bash
# Python tests
pytest tests/ -v
pytest tests_music-brain/ -v --cov=music_brain --cov-report=term-missing

# C++ tests
cd build
ctest --output-on-failure
./penta_tests --gtest_filter="*Harmony*"
```

### Code Style

**Python:**
```bash
black music_brain/ tests/          # Format
mypy music_brain/                  # Type check
flake8 music_brain/ tests/         # Lint
```

- Line length: 100 characters
- Python version: 3.9+
- Type hints required for public APIs

**C++:**
- Standard: C++20
- Naming: PascalCase (classes), camelCase (methods), snake_case (variables)
- RT-Safety: Mark audio callbacks `noexcept`, no allocations

### CI/CD

Workflows in `.github/workflows/`:
- `ci.yml` - Main CI pipeline (Python tests, C++ builds, memory testing)
- `sprint_suite.yml` - Comprehensive sprint-based testing
- `platform_support.yml` - Cross-platform testing (3.9-3.13, Linux/macOS/Windows)
- `release.yml` - Desktop app builds and releases

---

## Common Development Tasks

### Adding a New Groove Genre
1. Add entry to `data/grooves/genre_pocket_maps.json`
2. Add template in `music_brain/groove/templates.py`
3. Add to CLI choices in `music_brain/cli.py`

### Adding a Rule-Breaking Option
1. Add enum value in `music_brain/session/intent_schema.py`
2. Add entry in `RULE_BREAKING_EFFECTS` dict
3. Implement in `intent_processor.py`

### Adding a Penta-Core Engine
1. Create header in `include/penta/<subsystem>/`
2. Implement in `src_penta-core/<subsystem>/`
3. Update `src_penta-core/CMakeLists.txt`
4. Add Python bindings in `python/penta_core/`
5. Add tests in `tests_penta-core/`

---

## Key Architecture Decisions

### 1. Dual-Engine Design
- **Side A (C++)**: Real-time audio, deterministic, lock-free
- **Side B (Python)**: AI generation, dynamic, may block

### 2. Intent-Driven Composition
- Emotional intent drives technical choices
- Phase 0 (why) must precede Phase 2 (how)
- Rule-breaking requires explicit justification

### 3. RT-Safety
- Audio thread never waits on UI/AI
- Lock-free communication via ring buffers
- Pre-allocated memory pools

### 4. MCP Protocol
- Standard protocol for AI tool integration
- Multiple MCP servers for different purposes
- Cross-AI task synchronization

---

## Documentation Structure

| Location | Purpose |
|----------|---------|
| `docs/` | All documentation (architecture, guides, references) |
| `Production_Workflows/` | Production workflow guides |
| `Songwriting_Guides/` | Songwriting methodology |
| `Theory_Reference/` | Music theory reference |
| `vault/` | Obsidian knowledge base |
| `PROJECT_QUICK_START.md` | Quick reference guide |
| `PROJECT_BACKUP_CORE_CRITERIA.md` | Comprehensive backup of core criteria |

---

## Troubleshooting

### Python Import Errors
```bash
pip install -e .
python --version  # Requires 3.9+
```

### C++ Build Failures
```bash
cmake --version  # Requires 3.27+
g++ --version    # Requires C++20 support
```

### API Server Not Starting
```bash
# Check if port 8000 is available
lsof -i :8000

# Check Music Brain imports
python -c "from music_brain.structure.comprehensive_engine import TherapySession"
```

### Audio Thread Issues
- Verify no allocations in `processBlock()`
- Check `isAudioThread()` assertions
- Use `assertNotAudioThread()` before blocking operations

---

## Resources

- **Quick Start Guide**: See `PROJECT_QUICK_START.md`
- **Comprehensive Backup**: See `PROJECT_BACKUP_CORE_CRITERIA.md`
- **Implementation Plan**: See `IMPLEMENTATION_PLAN.md` (24-week roadmap)
- **Build Variants**: See `BUILD_VARIANTS.md` (hardware configs)
- **AI Assistant Guide**: See `CLAUDE.md` (comprehensive developer guide)

---

## License

MIT

---

**Repository**: KmiDi (consolidated monorepo)
**Status**: Active Development
**Version**: 1.0.0
