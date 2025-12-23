# CLAUDE AGENT GUIDE - Complete DAiW-Music-Brain Reference

> **Comprehensive guide for AI assistants working with DAiW (Digital Audio intelligent Workstation)**
> 
> **Version:** 0.4.0 | **Last Updated:** 2025-01-XX
> 
> This document contains everything a Claude agent needs to effectively work with the DAiW-Music-Brain codebase.

---

## Table of Contents

1. [Project Philosophy & Core Principles](#project-philosophy--core-principles)
2. [Architecture Overview](#architecture-overview)
3. [Module Reference](#module-reference)
4. [MCP Tools Reference](#mcp-tools-reference)
5. [API Reference](#api-reference)
6. [Data Structures & Schemas](#data-structures--schemas)
7. [Development Workflows](#development-workflows)
8. [Common Patterns & Best Practices](#common-patterns--best-practices)
9. [Troubleshooting & Debugging](#troubleshooting--debugging)
10. [Recent Changes & Known Issues](#recent-changes--known-issues)

---

## Project Philosophy & Core Principles

### Core Philosophy

**"Interrogate Before Generate"** - The tool shouldn't finish art for people. It should make them braver.

### Key Principles

1. **Emotional intent drives technical decisions** - Never generate without understanding the "why"
2. **Rules are broken intentionally** - Every rule break requires emotional justification
3. **Human imperfection is valued** - Lo-fi, pitch drift, room noise are features, not bugs
4. **Phase 0 must come first** - Technical decisions (Phase 2) can't be made without emotional clarity (Phase 0)
5. **Teaching over finishing** - The tool should educate and empower, not just generate

### Meta Principle

> "The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"

**When working on this codebase, remember: the technical implementation serves the emotional expression, never the other way around.**

---

## Architecture Overview

### Project Structure

```
DAiW-Music-Brain/
├── music_brain/              # Main Python package (v0.4.0)
│   ├── __init__.py          # Package exports
│   ├── cli.py               # CLI entry point (`daiw` command)
│   │
│   ├── groove/              # Groove extraction & application
│   │   ├── extractor.py     # extract_groove(), GrooveTemplate
│   │   ├── applicator.py    # apply_groove()
│   │   ├── templates.py     # Genre templates (funk, jazz, rock, etc.)
│   │   └── groove_engine.py # Humanization engine
│   │
│   ├── structure/           # Harmonic analysis
│   │   ├── chord.py         # Chord, ChordProgression, analyze_chords()
│   │   ├── progression.py   # diagnose_progression(), generate_reharmonizations()
│   │   ├── sections.py      # Section detection
│   │   └── comprehensive_engine.py # Therapy-to-music pipeline
│   │
│   ├── session/             # Intent schema & teaching
│   │   ├── intent_schema.py # CompleteSongIntent, rule-breaking enums
│   │   ├── intent_processor.py # process_intent(), IntentProcessor
│   │   ├── teaching.py      # RuleBreakingTeacher
│   │   ├── interrogator.py  # SongInterrogator
│   │   └── generator.py     # Generation utilities
│   │
│   ├── harmony/             # Harmony generation
│   │   ├── harmony_generator.py # HarmonyGenerator, ChordVoicing
│   │   └── __init__.py      # Exports
│   │
│   ├── audio/               # Audio analysis suite
│   │   ├── analyzer.py      # AudioAnalyzer, detect_bpm(), detect_key()
│   │   ├── chord_detection.py # ChordDetector, detect_chords_from_audio()
│   │   ├── frequency.py     # FrequencyAnalyzer, FFT analysis
│   │   ├── theory_analyzer.py # TheoryAnalyzer, scales/modes/arpeggios
│   │   ├── feel.py          # analyze_feel(), AudioFeatures
│   │   └── reference_dna.py # Reference track analysis
│   │
│   ├── realtime/            # Real-time MIDI processing
│   │   ├── midi_processor.py # RealtimeMidiProcessor, MidiProcessorConfig
│   │   ├── transformers.py  # MIDI transformation callbacks
│   │   ├── engine.py        # RealtimeEngine, RealtimeClock
│   │   └── scheduler.py     # EventScheduler
│   │
│   ├── effects/             # Guitar effects modulator
│   │   ├── base.py          # Effect, ModulationSource, LFO, Envelope
│   │   ├── effects.py       # 28+ effect implementations
│   │   └── guitar_fx.py     # GuitarFXEngine, GuitarFXChain, Presets
│   │
│   ├── emidi/               # EMIDI format (C++ implementation)
│   │   ├── emidi.h          # EMIDI format definitions
│   │   └── emidi.cpp        # Serialization/deserialization
│   │
│   ├── utils/               # Utilities
│   │   ├── midi_io.py       # MIDI file handling
│   │   ├── instruments.py   # Instrument mappings
│   │   └── ppq.py           # PPQ normalization, quantize_ticks()
│   │
│   ├── daw/                 # DAW integration
│   │   ├── logic.py         # Logic Pro integration
│   │   └── markers.py       # Emotional structure markers
│   │
│   ├── text/                # Text generation
│   │   └── lyrical_mirror.py # generate_lyrical_fragments()
│   │
│   └── data/                # JSON/YAML data files
│       ├── chord_progressions.json
│       ├── genre_pocket_maps.json
│       ├── song_intent_examples.json
│       ├── song_intent_schema.yaml
│       └── scales_database.json
│
├── daiw_mcp/                # Model Context Protocol server
│   ├── server.py            # MCP server implementation
│   └── tools/               # MCP tool modules
│       ├── harmony_tools.py # 6 harmony tools
│       ├── groove_tools.py  # 5 groove tools
│       ├── intent_tools.py  # 4 intent tools
│       ├── audio_tools.py   # 6 audio analysis tools
│       └── teaching_tools.py # 3 teaching tools
│
├── vault/                   # Knowledge base (Obsidian-compatible)
│   ├── Songwriting_Guides/
│   ├── Theory_Reference/
│   ├── Production_Workflows/
│   └── Templates/
│
├── tests/                   # Test suite
│   ├── test_basic.py
│   ├── test_cli_commands.py
│   ├── test_audio_analyzer.py
│   └── ...
│
├── examples/                # Example files
│   ├── realtime_midi_demo.py
│   └── midi/
│
├── app.py                   # Streamlit UI application
├── launcher.py              # Native desktop app launcher
├── pyproject.toml           # Package configuration
└── requirements.txt         # Core dependencies
```

### Data Flow

```
User Input → CompleteSongIntent → IntentProcessor → Generated Elements
                                                    ├── GeneratedProgression
                                                    ├── GeneratedGroove
                                                    ├── GeneratedArrangement
                                                    └── GeneratedProduction
```

---

## Module Reference

### Core Modules

#### `music_brain.groove`
**Purpose:** Extract and apply groove patterns from MIDI files

**Key Functions:**
- `extract_groove(midi_path, quantize_resolution=16) -> GrooveTemplate`
- `apply_groove(midi_path, genre=None, output=None, intensity=0.5) -> str`

**Key Classes:**
- `GrooveTemplate` - Timing/velocity patterns extracted from MIDI

**Genre Templates:** funk, jazz, rock, hiphop, edm, latin, reggae, blues

#### `music_brain.structure`
**Purpose:** Harmonic analysis and chord progression diagnosis

**Key Functions:**
- `analyze_chords(midi_path, quantize_beats=0.5) -> ChordProgression`
- `diagnose_progression(progression, key=None) -> Dict`
- `generate_reharmonizations(progression, style="jazz", count=3) -> List[Dict]`

**Key Classes:**
- `Chord` - Single chord with root, quality, extensions
- `ChordProgression` - Complete progression with analysis

#### `music_brain.session`
**Purpose:** Intent-based song generation and teaching

**Key Functions:**
- `process_intent(intent: CompleteSongIntent) -> Dict`
- `validate_intent(intent: CompleteSongIntent) -> List[str]`
- `suggest_rule_break(emotion: str) -> List[Dict]`

**Key Classes:**
- `CompleteSongIntent` - Three-phase intent schema
- `IntentProcessor` - Converts intent to musical elements
- `RuleBreakingTeacher` - Interactive teaching

#### `music_brain.harmony`
**Purpose:** Generate harmony from emotional intent

**Key Functions:**
- `generate_midi_from_harmony(harmony, output_path, tempo_bpm=120) -> None`

**Key Classes:**
- `HarmonyGenerator` - Generates chord progressions and voicings
- `HarmonyResult` - Generated harmony with voicings
- `ChordVoicing` - MIDI voicing data

#### `music_brain.audio`
**Purpose:** Comprehensive audio analysis

**Key Functions:**
- `detect_bpm(audio_data, sr) -> BPMDetectionResult`
- `detect_key(audio_data, sr) -> KeyDetectionResult`
- `detect_chords_from_audio(audio_file, window_size=0.5) -> ChordProgressionDetection`
- `analyze_feel(audio_file) -> AudioFeatures`

**Key Classes:**
- `AudioAnalyzer` - Main audio analysis interface
- `ChordDetector` - Chord detection from audio
- `FrequencyAnalyzer` - FFT and pitch analysis
- `TheoryAnalyzer` - Scales, modes, arpeggios, triads

**Note:** Audio analysis requires `librosa`. Gracefully degrades if not available.

#### `music_brain.realtime`
**Purpose:** Real-time MIDI input/output processing

**Key Classes:**
- `RealtimeMidiProcessor` - Core real-time MIDI processor
- `MidiProcessorConfig` - Configuration for processor
- `RealtimeEngine` - Higher-level real-time engine
- `RealtimeClock` - Timing/clock management
- `EventScheduler` - Event scheduling

**Transformers:**
- `create_transpose_transformer(semitones) -> MidiTransformCallback`
- `create_velocity_scale_transformer(scale) -> MidiTransformCallback`
- `create_humanize_transformer(amount) -> MidiTransformCallback`
- `create_arpeggiator_transformer(pattern) -> MidiTransformCallback`
- `create_chord_generator_transformer(chords) -> MidiTransformCallback`

#### `music_brain.effects`
**Purpose:** Comprehensive guitar effects modulator

**Key Classes:**
- `GuitarFXEngine` - Main effects engine
- `GuitarFXChain` - Ordered effect chain
- `GuitarFXPreset` - Effect and modulation settings
- `Effect` - Base class for all effects
- `ModulationSource` - LFO, Envelope, StepSequencer, etc.

**Effect Categories:**
- **Distortion:** Distortion, Overdrive, Fuzz (8 circuit types)
- **Modulation:** Chorus, Flanger, Phaser, Tremolo, Vibrato, Rotary, RingMod, Univibe
- **Time:** Delay (12 types), Reverb (15 algorithms)
- **Dynamics:** Compressor (5 modes), NoiseGate
- **Filter:** EQ, Wah, MultimodeFilter
- **Pitch:** PitchShift, Harmonizer, Octaver
- **Amp:** AmpSim (15 models), CabinetIR
- **Special:** Looper, Granular, Shimmer, Freeze, Slicer, Bitcrusher

**Modulation Sources:**
- LFO (14 waveforms)
- EnvelopeFollower
- StepSequencer
- RandomSource
- EmotionSource (emotion-to-parameter mapping)
- MIDI_CC_Source
- ExpressionSource

#### `music_brain.daw`
**Purpose:** DAW integration utilities

**Key Functions:**
- Logic Pro MIDI export/import
- Emotional structure markers export

---

## MCP Tools Reference

The MCP (Model Context Protocol) server provides 24 tools across 5 categories:

### Harmony Tools (6 tools)
**File:** `daiw_mcp/tools/harmony_tools.py`

1. **`analyze_progression`** - Analyze chord progression for harmonic characteristics
2. **`generate_harmony`** - Generate harmony from intent or parameters
3. **`diagnose_chords`** - Diagnose harmonic issues
4. **`suggest_reharmonization`** - Suggest chord substitutions
5. **`find_key`** - Detect key from progression (returns float confidence 0.0-1.0)
6. **`voice_leading`** - Optimize voice leading with parallel motion detection

### Groove Tools (5 tools)
**File:** `daiw_mcp/tools/groove_tools.py`

1. **`extract_groove`** - Extract groove characteristics from MIDI
2. **`apply_groove`** - Apply genre groove template
3. **`analyze_pocket`** - Analyze timing pocket
4. **`humanize_midi`** - Add human feel with complexity/vulnerability
5. **`quantize_smart`** - Smart quantization preserving feel (strength 0.0-1.0)

### Intent Tools (4 tools)
**File:** `daiw_mcp/tools/intent_tools.py`

1. **`create_intent`** - Create song intent template
2. **`process_intent`** - Process intent → music
3. **`validate_intent`** - Validate intent schema
4. **`suggest_rulebreaks`** - Suggest emotional rule-breaks

### Audio Analysis Tools (6 tools)
**File:** `daiw_mcp/tools/audio_tools.py`

1. **`detect_bpm`** - Detect tempo from audio (returns float confidence 0.0-1.0)
2. **`detect_key`** - Detect key from audio (returns float confidence 0.0-1.0)
3. **`analyze_audio_feel`** - Analyze groove feel from audio
4. **`extract_chords`** - Extract chords from audio
5. **`detect_scale`** - Detect scales/modes from audio
6. **`analyze_theory`** - Complete music theory analysis

### Teaching Tools (3 tools)
**File:** `daiw_mcp/tools/teaching_tools.py`

1. **`explain_rulebreak`** - Explain rule-breaking technique
2. **`get_progression_info`** - Get progression details
3. **`emotion_to_music`** - Map emotion to musical parameters

**Important:** All confidence values are consistently returned as floats (0.0-1.0) for type consistency.

---

## API Reference

### Public API Exports (`music_brain/__init__.py`)

```python
# Groove
from music_brain import extract_groove, apply_groove, GrooveTemplate

# Structure
from music_brain import analyze_chords, detect_sections, ChordProgression

# Audio (feel)
from music_brain import analyze_feel, AudioFeatures

# Audio (analysis)
from music_brain import (
    AudioAnalyzer, AudioAnalysis,
    ChordDetector, FrequencyAnalyzer, TheoryAnalyzer,
    detect_key, detect_bpm, detect_chords_from_audio,
    detect_scale, detect_mode, analyze_harmony,
    SCALES, MODE_CHARACTERISTICS
)

# Harmony
from music_brain import HarmonyGenerator, HarmonyResult, generate_midi_from_harmony

# Real-time MIDI
from music_brain import (
    RealtimeMidiProcessor, MidiProcessorConfig,
    RealtimeEngine, RealtimeClock, EventScheduler
)

# Comprehensive Engine
from music_brain import (
    AffectAnalyzer, TherapySession, HarmonyPlan, render_plan_to_midi
)

# Text/Lyrical
from music_brain import generate_lyrical_fragments
```

---

## Data Structures & Schemas

### Three-Phase Intent Schema

#### Phase 0: Core Wound/Desire
```python
SongRoot(
    core_event: str,           # What happened?
    core_resistance: str,       # What holds you back?
    core_longing: str,          # What do you want to feel?
    core_stakes: CoreStakes,    # What's at risk?
    core_transformation: str    # How should you feel when done?
)
```

#### Phase 1: Emotional Intent
```python
SongIntent(
    mood_primary: str,                    # Dominant emotion
    mood_secondary_tension: float,        # Internal conflict (0.0-1.0)
    imagery_texture: str,                 # Visual/tactile quality
    vulnerability_scale: VulnerabilityScale, # Low/Medium/High
    narrative_arc: NarrativeArc           # Climb-to-Climax, Slow Reveal, etc.
)
```

#### Phase 2: Technical Constraints
```python
TechnicalConstraints(
    technical_genre: str,
    technical_tempo_range: Tuple[int, int],
    technical_key: str,
    technical_mode: str,
    technical_groove_feel: str,
    technical_rule_to_break: str,         # Intentional rule violation
    rule_breaking_justification: str      # WHY break this rule (required!)
)
```

### Rule-Breaking Categories

| Category | Examples | Effect |
|----------|----------|--------|
| **Harmony** | `HARMONY_AvoidTonicResolution`, `HARMONY_ModalInterchange` | Unresolved yearning, bittersweet color |
| **Rhythm** | `RHYTHM_ConstantDisplacement`, `RHYTHM_TempoFluctuation` | Anxiety, organic breathing |
| **Arrangement** | `ARRANGEMENT_BuriedVocals`, `ARRANGEMENT_ExtremeDynamicRange` | Dissociation, dramatic impact |
| **Production** | `PRODUCTION_PitchImperfection`, `PRODUCTION_ExcessiveMud` | Emotional honesty, claustrophobia |

---

## Development Workflows

### Installation

```bash
# Clone and install as editable package
git clone https://github.com/yourusername/DAiW-Music-Brain.git
cd DAiW-Music-Brain
pip install -e .

# With optional dependencies
pip install -e ".[dev]"      # pytest, black, flake8, mypy
pip install -e ".[audio]"    # librosa, soundfile
pip install -e ".[theory]"   # music21
pip install -e ".[ui]"       # streamlit
pip install -e ".[desktop]"  # streamlit + pywebview
pip install -e ".[mcp]"       # mcp (Model Context Protocol)
pip install -e ".[build]"     # pyinstaller
pip install -e ".[all]"       # Everything
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test class
pytest tests/test_basic.py::TestImports -v
```

### Code Style

- **Line length:** 100 characters (configured in `pyproject.toml`)
- **Formatter:** black
- **Linter:** flake8, mypy

```bash
# Format code
black music_brain/ tests/

# Type check
mypy music_brain/

# Lint
flake8 music_brain/ tests/
```

### CLI Usage

**Complete Command Reference:**

```bash
# Groove operations
daiw extract <midi_file> [-o output.json]              # Extract groove from MIDI
daiw apply --genre <genre> <midi_file> [-o output.mid]  # Apply groove template
daiw humanize <midi_file> [options]                    # Apply humanization

# Humanization options:
#   --preset <name>        # Emotional preset (use --list-presets)
#   --style <style>         # tight, natural, loose, drunk
#   --complexity <0.0-1.0>  # Timing chaos
#   --vulnerability <0.0-1.0> # Dynamic fragility
#   -o, --output <file>     # Output MIDI file

# Chord analysis
daiw analyze --chords <midi_file>                      # Analyze chord progression
daiw diagnose <progression>                            # Diagnose harmonic issues
daiw reharm <progression> [--style <style>]            # Generate reharmonizations
#   --style: jazz, pop, rnb, classical, experimental

# Audio analysis
daiw analyze-audio <audio_file> [options]             # Analyze audio file
#   --bpm                   # Detect BPM only
#   --key                   # Detect key only
#   --chords                # Detect chords only
#   --feel                  # Analyze feel/groove
#   --all                   # Full analysis (default)
#   --max-duration <sec>    # Limit analysis duration

daiw compare-audio <file1> <file2>                    # Compare two audio files
daiw batch-analyze <files...> [options]               # Batch analyze multiple files
daiw export-features <audio_file> -o <out> [options]  # Export features to JSON/CSV
#   --format json|csv       # Output format
#   --include-segments      # Include segment analysis
#   --include-chords        # Include chord detection

# Intent-based generation
daiw generate [options]                                # Generate harmony
#   -i, --intent-file <file> # Intent JSON file
#   -k, --key <key>          # Key (e.g., C, F, Bb)
#   -m, --mode <mode>        # major, minor, dorian, etc.
#   -p, --pattern <pattern>  # Roman numeral pattern (e.g., "I-V-vi-IV")
#   -o, --output <file>      # Output MIDI file
#   -t, --tempo <bpm>        # Tempo in BPM

daiw intent new [--title <title>]                     # Create intent template
daiw intent process <file>                            # Generate from intent
daiw intent suggest <emotion>                         # Suggest rules to break
daiw intent list                                       # List all rule-breaking options
daiw intent validate <file>                           # Validate intent file

# Teaching
daiw teach <topic>                                     # Interactive teaching mode
#   topic: rulebreaking (interactive lessons on rule-breaking)
```

**Available Genres for Groove Application:**
- funk, jazz, rock, hiphop, edm, latin, reggae, blues

**Humanization Styles:**
- `tight` - Minimal drift, confident (complexity=0.1, vulnerability=0.2)
- `natural` - Human feel, balanced (complexity=0.4, vulnerability=0.5)
- `loose` - Relaxed, laid back (complexity=0.6, vulnerability=0.6)
- `drunk` - Maximum chaos, fragile (complexity=0.9, vulnerability=0.8)

### Desktop Application

```bash
# Streamlit in browser (development)
streamlit run app.py

# Native window (requires pywebview)
python launcher.py

# Build executable
pip install -e ".[build]"
pyinstaller daiw.spec --clean --noconfirm
```

---

## Common Patterns & Best Practices

### 1. Lazy Imports in CLI
Heavy modules are imported lazily to speed up CLI startup:
```python
def get_audio_module():
    from music_brain.audio import AudioAnalyzer
    return AudioAnalyzer
```

### 2. Graceful Degradation
Audio modules should work even without `librosa`:
```python
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

# Constants stored as Python lists, converted to np.array only when needed
MAJOR_PROFILE_VALUES = [6.35, 2.23, ...]  # Not np.array at module level
```

### 3. Data Classes for Structured Data
Use dataclasses with serialization:
```python
@dataclass
class MyClass:
    field: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MyClass':
        return cls(**data)
```

### 4. Type Consistency in MCP Tools
Always return consistent data types:
```python
# ✅ Good: Float confidence
"confidence": 0.8

# ❌ Bad: String confidence
"confidence": "high"
```

### 5. Error Handling
Provide helpful error messages:
```python
if not LIBROSA_AVAILABLE:
    raise ImportError("librosa required for BPM detection. Install with: pip install librosa")
```

### 6. Module Exports
Each subpackage exports its public API via `__all__`:
```python
__all__ = [
    "PublicClass",
    "public_function",
    # ...
]
```

---

## Troubleshooting & Debugging

### Import Errors
- Ensure package is installed: `pip install -e .`
- Check Python version: `python --version` (requires 3.9+)
- Verify dependencies: `pip list | grep mido`

### MIDI File Issues
- Verify mido is installed: `pip install mido`
- Check file exists and is valid MIDI
- Use `mido.MidiFile(filename)` to validate

### Audio Analysis Issues
- Install librosa: `pip install librosa soundfile numpy scipy`
- Check file format is supported (WAV, MP3, AIFF, etc.)
- Verify sample rate compatibility

### MCP Server Issues
- Check MCP is installed: `pip install mcp`
- Verify tool modules have `register_tools(server)` function
- Check server logs for registration errors

### Test Failures
- Run with verbose: `pytest -v`
- Check data files exist in `music_brain/data/`
- Verify test fixtures are set up correctly

### Build Issues
If built app opens and immediately closes:
1. Edit `daiw.spec`: change `console=False` to `console=True`
2. Rebuild: `pyinstaller daiw.spec --clean --noconfirm`
3. Run from terminal to see error messages
4. Add missing modules to `hiddenimports` list in spec file

---

## Recent Changes & Known Issues

### Recent Fixes (2025-01-XX)

1. **Fixed BPM detection tempo candidates** - Replaced incorrect `aggregate=None` approach with proper tempogram-based peak detection using `librosa.feature.tempogram()` for meaningful alternative tempo candidates
2. **Fixed confidence type inconsistency** - All MCP tools now return float confidence (0.0-1.0) instead of strings for type consistency
3. **Implemented voice_leading tool** - Full voice leading optimization with:
   - Multiple voicing options (inversions, octaves)
   - Greedy algorithm to minimize voice movement
   - Parallel fifths/octaves detection
   - Quality score calculation
4. **Implemented quantize_smart tool** - Feel-preserving quantization with:
   - Strength parameter (0.0-1.0) for partial quantization
   - Blends original timing with quantized grid
   - Statistics on quantized vs preserved notes
5. **Fixed graceful degradation** - Audio modules store constants as Python lists at module level, convert to `np.array` only inside functions after checking `librosa` availability

### Known Issues

1. **Test coverage** - MCP tools need comprehensive test coverage (pending)
2. **Voice leading** - Currently uses greedy algorithm; could be improved with dynamic programming for optimal voicing selection
3. **Quantize smart** - Time delta calculation could be more sophisticated for better feel preservation (currently uses simple sum of previous message times)

### Graceful Degradation

- **Audio modules:** Work without `librosa` (constants stored as Python lists, converted to `np.array` only when needed)
- **MCP tools:** Fall back to `AUDIO_CATALOGER_AVAILABLE` if `AUDIO_AVAILABLE` is False
- **Type consistency:** All confidence values normalized to floats (0.0-1.0) across all tools
- **Import errors:** Provide helpful error messages with installation instructions

### Implementation Details

**BPM Detection Fix:**
- Old: Used `librosa.beat.tempo(..., aggregate=None)` which returns per-frame tempo estimates
- New: Uses `librosa.feature.tempogram()` to find peaks in tempo distribution, providing meaningful alternatives (half/double tempo, etc.)

**Voice Leading Implementation:**
- Generates multiple voicing options per chord (close position, open position, inversions)
- Uses greedy algorithm to select voicings minimizing total voice movement
- Detects parallel fifths (7 semitones) and parallel octaves (0 semitones)
- Calculates quality score (0.0-1.0) based on parallel motion and large leaps

**Smart Quantization:**
- Uses `quantize_ticks()` from `music_brain.utils.ppq` for grid alignment
- Blends original timing with quantized timing: `blended = original * (1 - strength) + quantized * strength`
- Preserves feel by only partially quantizing based on strength parameter

---

## Key Files to Understand

### Entry Points
- `music_brain/cli.py` - CLI implementation, all commands
- `music_brain/__init__.py` - Public API exports
- `daiw_mcp/server.py` - MCP server implementation

### Core Logic
- `music_brain/session/intent_schema.py` - The heart of the intent system
- `music_brain/session/intent_processor.py` - Converts intent to musical elements
- `music_brain/groove/templates.py` - Genre groove definitions
- `music_brain/structure/progression.py` - Chord parsing and diagnosis
- `music_brain/audio/analyzer.py` - Audio analysis with graceful degradation

### Data Files
- `music_brain/data/genre_pocket_maps.json` - Genre timing characteristics
- `music_brain/data/song_intent_schema.yaml` - Schema definition
- `music_brain/data/chord_progressions.json` - Common progressions
- `music_brain/data/scales_database.json` - Scale definitions

---

## When Adding Features

### General Guidelines
1. Consider the "Interrogate Before Generate" philosophy
2. Rule-breaking should always have emotional justification
3. Add tests for new functionality
4. Update `__all__` exports if adding public API
5. Keep CLI startup fast (use lazy imports)
6. Maintain type consistency (especially in MCP tools)

### When Modifying Intent Schema
1. Update both `intent_schema.py` and `song_intent_schema.yaml`
2. Ensure `to_dict()` / `from_dict()` handle new fields
3. Add validation in `validate_intent()`
4. Update vault documentation in `vault/Songwriting_Guides/`

### When Adding Rule-Breaking Options
1. Add enum value in appropriate class (`HarmonyRuleBreak`, etc.)
2. Add entry in `RULE_BREAKING_EFFECTS` dict
3. Implement processor function in `intent_processor.py`
4. Update CLI help text if needed

### When Adding MCP Tools
1. Create tool module in `daiw_mcp/tools/`
2. Implement `register_tools(server)` function
3. Use consistent data types (float for confidence, etc.)
4. Add error handling with helpful messages
5. Auto-discovery will register the tool automatically

### When Adding Audio Analysis
1. Store constants as Python lists (not np.array) at module level
2. Convert to np.array only inside functions after checking availability
3. Provide graceful degradation if librosa is not available
4. Return consistent data types (float confidence, etc.)

---

## Vault (Knowledge Base)

The `vault/` directory is an Obsidian-compatible knowledge base containing:
- **Songwriting_Guides/** - Intent schema docs, rule-breaking guides
- **Theory_Reference/** - Music theory reference materials
- **Production_Workflows/** - Production technique guides
- **Templates/** - Task boards and templates
- **Data_Files/** - Supporting data

These markdown files use Obsidian-style `[[wiki links]]` for cross-referencing.

---

## Quick Reference

### Version Information
- **Package Version:** 0.4.0
- **Python Support:** 3.9, 3.10, 3.11, 3.12
- **License:** MIT

### Dependencies
- **Core:** `mido>=1.2.10`, `numpy>=1.21.0`
- **Dev:** `pytest>=7.0.0`, `black>=22.0.0`, `flake8>=4.0.0`, `mypy>=0.900`
- **UI:** `streamlit>=1.28.0`
- **Desktop:** `pywebview>=4.0.0`
- **MCP:** `mcp>=0.1.0`
- **Optional:** `librosa`, `soundfile`, `music21`

### Project Status
- **Phase 1:** ✅ Complete (100%)
- **Phase 2:** ✅ Complete (100%)
- **Phase 3:** 60% Complete
  - ✅ Real-time MIDI processing
  - ✅ Audio analysis module
  - ✅ MCP tool coverage (24 tools)
  - ⏳ DAW plugin integration (planned)

---

## Final Notes

This guide is a living document. When making significant changes:
1. Update this guide
2. Update `DEVELOPMENT_ROADMAP.md` if applicable
3. Update `CHANGELOG.md` if it exists
4. Update relevant module docstrings

**Remember:** The technical implementation serves the emotional expression, never the other way around.

---

*End of CLAUDE AGENT GUIDE*

