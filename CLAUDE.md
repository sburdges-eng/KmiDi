# CLAUDE.md - AI Assistant Guide for iDAW

> Comprehensive guide for AI assistants working with the iDAW (Intelligent Digital Audio Workstation) monorepo.

## Project Philosophy

**"Interrogate Before Generate"** - The tool shouldn't finish art for people. It should make them braver.

iDAW is a multi-component music production platform that combines:

- Real-time C++ audio processing (JUCE-based)
- Python-based music intelligence and AI-assisted composition
- Multi-AI collaboration orchestration
- Intent-driven creative workflows

---

## Repository Architecture

This is a **monorepo** containing multiple subsystems:

```
iDAW/
├── music_brain/               # Python Music Intelligence Toolkit (main package)
├── mcp_workstation/           # MCP Multi-AI Workstation (orchestration)
├── mcp_todo/                  # MCP TODO Server (cross-AI task management)
├── mcp_penta_swarm/           # MCP Swarm Server (multi-AI aggregation)
├── daiw_mcp/                  # DAiW MCP Server
│
├── iDAW_Core/                 # JUCE Plugin Suite (C++)
├── src_penta-core/            # Penta-Core C++ Engines (implementation)
├── src/                       # Additional C++ source files
├── include/                   # C++ Headers (including penta/)
├── cpp_music_brain/           # C++ Music Brain implementation
├── bindings/                  # Language bindings (Python/C++)
│
├── python/penta_core/         # Python bindings for Penta-Core
├── scripts/                   # Standalone utility scripts
├── tools/                     # Development tools (kb_analyzer, audio_cataloger)
│
├── tests/                     # All tests (C++, Python, music_brain, penta_core)
│   ├── python/                # Python unit tests
│   ├── music_brain/           # Music Brain integration tests
│   └── penta_core/            # Penta-Core tests
├── examples/                  # Example scripts (music_brain, penta_core)
├── benchmarks/                # Performance benchmarks
│
├── data/                      # JSON data files
│   ├── emotions/              # Emotion thesaurus data
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
├── output/                    # Generated output files
└── legacy/                    # Archived/legacy code
```

---

## 1. MCP Multi-AI Workstation (`mcp_workstation/`)

Orchestration system for multi-AI collaboration on iDAW development.

### Key Files
| File | Purpose |
|------|---------|
| `__init__.py` | Package exports, version 1.0.0 |
| `cli.py` | CLI entry point for workstation commands |
| `orchestrator.py` | Central coordinator (`Workstation` class) |
| `models.py` | Data models (AIAgent, Proposal, Phase, etc.) |
| `proposals.py` | Proposal management system |
| `phases.py` | Phase tracking for iDAW development |
| `cpp_planner.py` | C++ transition planning |
| `ai_specializations.py` | AI agent capabilities and task assignment |
| `server.py` | MCP server implementation |
| `debug.py` | Debug protocol and logging |

### CLI Commands
```bash
# From project root
python -m mcp_workstation status              # Show workstation dashboard
python -m mcp_workstation register claude     # Register as Claude
python -m mcp_workstation propose claude "Title" "Desc" architecture
python -m mcp_workstation vote claude PROP_ID 1
python -m mcp_workstation phases              # Show phase progress
python -m mcp_workstation cpp                 # Show C++ transition plan
python -m mcp_workstation ai                  # Show AI specializations
python -m mcp_workstation server              # Run MCP server
```

### AI Agents
- `claude` - Code architecture, real-time safety, complex debugging
- `chatgpt` - Theory analysis, explanations, documentation
- `gemini` - Cross-language patterns, multi-modal analysis
- `github_copilot` - Code completion, boilerplate generation

---

## 2. MCP TODO Server (`mcp_todo/`)

Cross-AI task management server. Tasks created in one AI are instantly available in all others.

### Supported AI Assistants
- Claude (Desktop & Code), Cursor, VSCode + Copilot (MCP stdio)
- ChatGPT (HTTP/OpenAPI), Gemini (Function Calling), OpenAI API

### Key Features
- **Cross-AI Sync**: Tasks created in Claude appear in ChatGPT, Cursor, etc.
- **Rich Task Model**: Priority, tags, projects, due dates, notes, subtasks
- **File-based Storage**: JSON storage at `~/.mcp_todo/todos.json`
- **CLI Tool**: `python -m mcp_todo.cli`
- **HTTP API**: `python -m mcp_todo.http_server`

### CLI Commands
```bash
python -m mcp_todo.cli add "Task" --priority high --tags "code,urgent"
python -m mcp_todo.cli list --status pending
python -m mcp_todo.cli complete <id>
python -m mcp_todo.cli summary
```

---

## 3. Penta-Core MCP Swarm (`penta_core_music-brain/`)

An MCP server that aggregates the top 5 AI platforms into a single "Swarm" toolset.

### AI Tools
| Tool | Backend | Purpose |
|------|---------|---------|
| `consult_architect` | OpenAI GPT-4o | High-level logic and design patterns |
| `consult_developer` | Anthropic Claude 3.5 Sonnet | Clean code and refactoring |
| `consult_researcher` | Google Gemini 1.5 Pro | Deep context analysis |
| `consult_maverick` | xAI Grok Beta | Creative problem-solving and red teaming |
| `fetch_repo_context` | GitHub API | Repository context fetching |

---

## 4. DAiW-Music-Brain (Python Toolkit)

Music production intelligence library for groove extraction, chord analysis, and intent-based generation.

### Directory Structure
```
DAiW-Music-Brain/
├── music_brain/
│   ├── __init__.py           # Public API (v0.2.0)
│   ├── cli.py                # `daiw` CLI command
│   ├── data/                 # JSON/YAML data files
│   ├── groove/               # Groove extraction/application
│   ├── structure/            # Chord/progression analysis
│   ├── session/              # Intent schema, teaching, interrogation
│   ├── audio/                # Audio feel analysis
│   ├── daw/                  # DAW integration (Logic Pro)
│   └── utils/                # Utility functions
├── tests/
└── pyproject.toml
```

### CLI Commands (`daiw`)
```bash
daiw extract drums.mid            # Extract groove from MIDI
daiw apply --genre funk track.mid # Apply genre groove template
daiw analyze --chords song.mid    # Analyze chord progression
daiw diagnose "F-C-Am-Dm"         # Diagnose harmonic issues
daiw intent new --title "My Song" # Create intent template
daiw intent suggest grief         # Suggest rules to break
daiw teach rulebreaking           # Interactive teaching mode
```

### Three-Phase Intent Schema
1. **Phase 0: Core Wound/Desire** - `core_event`, `core_resistance`, `core_longing`
2. **Phase 1: Emotional Intent** - `mood_primary`, `vulnerability_scale`, `narrative_arc`
3. **Phase 2: Technical Constraints** - `technical_genre`, `technical_key`, `technical_rule_to_break`

### Rule-Breaking Categories
| Category | Examples | Effect |
|----------|----------|--------|
| Harmony | `HARMONY_AvoidTonicResolution` | Unresolved yearning |
| Rhythm | `RHYTHM_ConstantDisplacement` | Anxiety, restlessness |
| Arrangement | `ARRANGEMENT_BuriedVocals` | Dissociation |
| Production | `PRODUCTION_PitchImperfection` | Emotional honesty |

---

## 5. Penta-Core (C++ Real-time Engines)

High-performance, RT-safe audio analysis engines.

### C++ Headers (`include/penta/`)
```
include/penta/
├── common/           # RTTypes, RTLogger, RTMemoryPool, SIMDKernels
├── groove/           # GrooveEngine, OnsetDetector, TempoEstimator, RhythmQuantizer
├── harmony/          # HarmonyEngine, ChordAnalyzer, ScaleDetector, VoiceLeading
├── diagnostics/      # DiagnosticsEngine, AudioAnalyzer, PerformanceMonitor
└── osc/              # OSCHub, OSCClient, OSCServer, OSCMessage, RTMessageQueue
```

### C++ Implementation (`src_penta-core/`)
- Static library: `penta_core`
- Dependencies: `oscpack`, `readerwriterqueue` (lock-free queue)
- SIMD optimizations: AVX2 when available (`ChordAnalyzerSIMD.cpp`)

### Python Bindings (`python/penta_core/`)
```python
from penta_core import PentaCore, HarmonyEngine, GrooveEngine, DiagnosticsEngine, OSCHub

# Integrated workflow
core = PentaCore(sample_rate=48000.0)
core.process(audio_buffer, midi_notes=[(60, 100), (64, 100)])
state = core.get_state()  # chord, scale, groove, diagnostics
```

### RT-Safety Rules
1. All `processAudio()` methods are marked `noexcept`
2. No memory allocation in audio callbacks
3. Use lock-free data structures for thread communication
4. `kDefaultSampleRate = 44100.0`

### Rules System (`python/penta_core/rules/`)
| Module | Purpose |
|--------|---------|
| `base.py` | Base rule classes and interfaces |
| `harmony_rules.py` | Harmony and chord progression rules |
| `counterpoint_rules.py` | Voice leading and counterpoint rules |
| `rhythm_rules.py` | Rhythm and timing rules |
| `voice_leading.py` | Voice leading analysis |
| `species.py` | Species counterpoint rules |
| `emotion.py` | Emotional expression rules |

### Teachers (`python/penta_core/teachers/`)
- `rule_breaking_teacher.py` - Interactive rule-breaking instruction
- `voice_leading_rules.py` - Voice leading teaching
- `counterpoint_rules.py` - Counterpoint teaching
- `rule_reference.py` - Rule reference documentation

---

## 6. iDAW_Core (JUCE Plugin Suite)

Art-themed audio plugins built on JUCE 8.

### Plugins
| Plugin | Description | Shader | Priority |
|--------|-------------|--------|----------|
| **Pencil** | Sketching/drafting audio ideas | Graphite | HIGH |
| **Eraser** | Audio removal/cleanup | ChalkDust | HIGH |
| **Palette** | Tonal coloring/mixing | Watercolor | MID |
| **Smudge** | Audio blending/smoothing | Scrapbook | MID |
| **Press** | Dynamics/compression | Heartbeat | HIGH |
| **Trace** | Pattern following/automation | Spirograph | LOW |
| **Parrot** | Sample playback/mimicry | Feather | LOW |
| **Stencil** | Sidechain/ducking effect | Cutout | LOW |
| **Chalk** | Lo-fi/bitcrusher effect | Dusty | LOW |
| **Brush** | Modulated filter effect | Brushstroke | LOW |
| **Stamp** | Stutter/repeater effect | RubberStamp | LOW |

### Dual-Heap Memory Architecture
```
Side A ("Work State"):
  - std::pmr::monotonic_buffer_resource
  - 4GB pre-allocated at startup
  - NO deallocation during runtime
  - Thread-safe for real-time audio

Side B ("Dream State"):
  - std::pmr::synchronized_pool_resource
  - Dynamic allocation allowed
  - Used for AI generation and UI
  - May block - NEVER use from audio thread

Communication: Lock-free ring buffer (Side B → Side A)
```

### PythonBridge
- Embeds Python interpreter in Side B (non-audio thread)
- `call_iMIDI()` - Pass knob state + text prompt, get MIDI buffer
- "Ghost Hands" - AI-suggested knob movements
- Fail-safe: Returns C Major chord on Python failure

---

## 7. Data Files

### Location: `Data_Files/`
| File | Purpose |
|------|---------|
| `chord_progression_families.json` | Chord progression family definitions |
| `chord_progressions_db.json` | Database of common progressions |
| `common_progressions.json` | Frequently used progressions |
| `genre_mix_fingerprints.json` | Genre mixing characteristics |
| `genre_pocket_maps.json` | Genre-specific groove pocket maps |

---

## Development Setup

### Python Installation
```bash
# Core installation
pip install -e .

# With optional dependencies
pip install -e ".[dev]"      # pytest, black, flake8, mypy
pip install -e ".[audio]"    # librosa, soundfile
pip install -e ".[theory]"   # music21
pip install -e ".[ui]"       # streamlit
pip install -e ".[desktop]"  # streamlit + pywebview
pip install -e ".[build]"    # + pyinstaller
pip install -e ".[all]"      # Everything
```

### C++ Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja penta_core           # Build the library
ninja penta_tests          # Build tests
ctest --output-on-failure  # Run tests
```

### Requirements
- **Python**: 3.9+ (tested 3.9-3.13)
- **C++**: C++17 standard
- **CMake**: 3.22+
- **JUCE**: 8.0.10

---

## Running Tests

### Python Tests
```bash
# Music Brain tests
pytest tests_music-brain/ -v

# DAiW-Music-Brain internal tests
pytest DAiW-Music-Brain/tests/ -v

# All tests with coverage
pytest tests_music-brain/ -v --cov=music_brain --cov-report=term-missing
```

### C++ Tests
```bash
cd build
ctest --output-on-failure

# Run specific test suite
./penta_tests --gtest_filter="*Harmony*"
./penta_tests --gtest_filter="*Groove*"
./penta_tests --gtest_filter="*Performance*"
```

### Test Files
| Location | Tests |
|----------|-------|
| `tests/` | C++ unit tests (groove, harmony, simd, memory) |
| `tests_music-brain/` | Python integration tests |
| `tests_penta-core/` | Penta-Core C++ tests (performance, OSC, etc.) |
| `DAiW-Music-Brain/tests/` | Music Brain internal tests |

---

## CI/CD Workflows

### Main Workflows (`.github/workflows/`)

| Workflow | Purpose |
|----------|---------|
| `ci.yml` | Python tests, C++ builds, memory testing (Valgrind), performance regression |
| `sprint_suite.yml` | Comprehensive sprint-based testing |
| `platform_support.yml` | Cross-platform Python testing (3.9-3.13, Linux/macOS/Windows) |
| `release.yml` | Build desktop apps (macOS/Linux/Windows), Python dist, C++ libraries |

### CI Jobs
1. **Python Tests** - pytest with coverage (Python 3.9, 3.11)
2. **C++ Build** - CMake/Ninja on Ubuntu and macOS
3. **Memory Testing** - Valgrind for leak detection
4. **Performance Testing** - Benchmark regression checks (<200ms latency target)
5. **Desktop Builds** - PyInstaller for macOS/Linux/Windows

### Sprint Suite Jobs
1. **Sprint 1** - Core testing & quality
2. **Sprint 2** - C++ build & integration
3. **Sprint 3** - Documentation checks
4. **Sprint 5** - Platform matrix (Linux/macOS/Windows x Python 3.9-3.13)
5. **Sprint 6** - Advanced theory and AI
6. **Sprint 7** - Mobile/Web (Streamlit)
7. **Sprint 8** - Enterprise tests

---

## Code Style & Conventions

### Python
```bash
# Format
black music_brain/ tests/

# Type check
mypy music_brain/

# Lint
flake8 music_brain/ tests/
```

- **Line length**: 100 characters
- **Formatter**: black
- **Type hints**: Required for public APIs
- **Python version**: Target 3.9+

### C++
- **Standard**: C++17
- **Naming**: PascalCase for classes, camelCase for methods, snake_case for variables
- **RT-Safety**: Mark audio callbacks `noexcept`, no allocations
- **Memory**: Use `std::pmr` containers where possible
- **SIMD**: Use AVX2 optimizations with scalar fallback

### Code Patterns

1. **Lazy imports** in Python CLI for fast startup
2. **Data classes** with `to_dict()`/`from_dict()` serialization
3. **Enums** for categorical values
4. **Singleton** pattern for managers (MemoryManager, Workstation)
5. **Lock-free ring buffers** for audio/UI communication

---

## Key Architecture Decisions

### 1. Dual-Engine Design
- **Side A (C++)**: Real-time audio, deterministic, lock-free
- **Side B (Python)**: AI generation, dynamic, may block

### 2. Intent-Driven Composition
- Emotional intent drives technical choices
- Phase 0 (why) must precede Phase 2 (how)
- Rule-breaking requires explicit justification

### 3. Multi-AI Collaboration
- Each AI has specializations and limitations
- Proposal system with voting
- Task assignment based on AI strengths

### 4. RT-Safety
- Audio thread never waits on UI/AI
- Lock-free communication via ring buffers
- Pre-allocated memory pools

### 5. MCP Protocol
- Standard protocol for AI tool integration
- Multiple MCP servers for different purposes
- Cross-AI task synchronization

---

## Common Development Tasks

### Adding a New Groove Genre
1. Add entry to `Data_Files/genre_pocket_maps.json`
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

### Adding an iDAW_Core Plugin
1. Create plugin directory in `iDAW_Core/plugins/<Name>/`
2. Add `include/<Name>Processor.h`, `src/<Name>Processor.cpp`
3. Add shader files in `shaders/`
4. Register in CMakeLists.txt

### Adding a Music Theory Rule
1. Create rule class in `python/penta_core/rules/`
2. Add to appropriate rules module (harmony, rhythm, etc.)
3. Add teacher support in `python/penta_core/teachers/`
4. Add tests

---

## Vault (Knowledge Base)

Obsidian-compatible markdown files in `vault/`:
```
vault/
├── Songwriting_Guides/    # Intent schema, rule-breaking guides
└── Songs/                 # Song-specific project files
```

Additional guides:

- `Production_Workflows/` - Production workflow documentation
- `Songwriting_Guides/` - Songwriting methodology
- `Theory_Reference/` - Music theory reference

Uses `[[wiki links]]` for cross-referencing.

---

## Troubleshooting

### Python Import Errors
```bash
pip install -e .
python --version  # Requires 3.9+
```

### C++ Build Failures
```bash
# Check CMake version
cmake --version  # Requires 3.22+

# Check compiler
g++ --version    # Requires C++17 support

# Check for AVX2 support (optional)
grep avx2 /proc/cpuinfo
```

### Audio Thread Issues
- Verify no allocations in `processBlock()`
- Check `isAudioThread()` assertions
- Use `assertNotAudioThread()` before blocking operations

### Test Failures
```bash
pytest -v --tb=long  # Verbose output with full tracebacks
```

### MCP Server Issues
```bash
# Test MCP TODO server
python -m mcp_todo.server --help

# Test Workstation server
python -m mcp_workstation.server --help
```

---

## Data Flow

```
User Intent --> Intent Schema --> Intent Processor --> Musical Elements
                                                    |-> GeneratedProgression
                                                    |-> GeneratedGroove
                                                    `-> GeneratedArrangement

Text Prompt --> PythonBridge --> Ring Buffer --> Audio Engine
(Side B)                                         (Side A)

AI Proposal --> Voting --> Approved --> Task Assignment --> Implementation

MCP TODO: Claude <--> JSON Storage <--> ChatGPT/Cursor/Gemini
```

---

## Meta Principle

> "The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"

Technical implementation serves emotional expression. The tool educates and empowers - it doesn't just generate.
