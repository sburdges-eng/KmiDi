# PRROT/PARROT Voice-Instrument Compiler - Architecture Documentation

## Overview

PRROT/PARROT is a production-grade voice-instrument compiler that analyzes reference voice audio to extract parametric voice profiles, then generates MIDI, phoneme timing, pitch curves, and articulation control data for DAW vocal synthesizers.

**Key Principle**: PRROT/PARROT is explicitly a voice-instrument compiler, NOT a text-to-speech engine or audio cloning system. It outputs control data (MIDI, phonemes, pitch curves), not final audio.

## Three-Tier Architecture

### Tier C: Embedded Articulation Intelligence (C++/Rust Core)

**Location**: `engine/src/prrot/`

**Purpose**: RT-safe, deterministic, always-on phoneme segmentation, articulation analysis, and MIDI shaping.

**Constraints**:
- No Python
- No ML inference
- No disk I/O
- No dynamic memory allocation in audio callbacks
- Pre-allocated buffers only

**Components**:
- `PhonemeSegmenter`: RT-safe phoneme segmentation (rule-based + DSP)
- `ArticulationAnalyzer`: Vowel/consonant classification, onset/offset timing
- `EnvelopeGenerator`: Articulation envelope generation
- `SpectralAnalyzer`: Pitch-independent spectral analysis (FFT with pre-allocated buffers)
- `BreathDetector`: Breath and noise estimation
- `VarianceModeler`: Articulation variance modeling and prominence curves
- `MidiShaper`: MIDI probability shaping
- `PRROTEngine`: Main embedded engine API

### Tier B: Offline Analysis & Reasoning (Python Worker)

**Location**: `python/prrot/`

**Purpose**: Deep phoneme alignment, speaker-specific articulation analysis, timbre embeddings, prosody analysis.

**Constraints**:
- Single worker on 16GB systems
- Short-lived (exits after job completion)
- One model per job
- ~3B parameters Q4 quantization preferred
- Up to 7B only when DAW idle

**Components**:
- `worker.py`: Disposable worker process (main entry)
- `phoneme_aligner.py`: Deep phoneme alignment (ML)
- `articulation_analyzer.py`: Speaker-specific articulation analysis
- `timbre_embeddings.py`: Non-reconstructive timbre embeddings
- `prosody_analyzer.py`: Prosody tendency analysis
- `lyric_planner.py`: Lyric-to-phoneme planning
- `articulation_inference.py`: Parse descriptive articulation text
- `instrument_affinity.py`: Map articulation → instrument suggestions

### Tier A: Optional Cloud/Large Model (Future)

**Purpose**: Same job schemas and data formats as Tier B, but allows larger models, more memory, fine-tuning.

**Constraint**: Local system must function fully without Tier A.

## Data Structures

### Voice Profile

Structured voice profile containing:
- Phoneme inventory (CMU phoneme set)
- Phoneme duration distributions
- Vowel sustain characteristics
- Consonant attack/release profiles
- Transition statistics
- Breath and noise weighting
- Vibrato tendencies
- Pitch stability profiles
- Articulation variance ranges
- Prominence tendencies
- Phoneme bias table

**Explicitly NOT**: Sentence-level waveforms, reusable audio chunks

### Phoneme Control Data

Output structure containing:
- Phoneme sequence with timing
- Per-phoneme timing (start, duration, onset, offset)
- Stress and emphasis markers
- Pitch targets (MIDI note + cents)
- Vibrato flags and parameters
- Breath markers
- MIDI notes
- Automation envelopes
- Articulation envelopes

## RT Safety Guarantees

All Tier C components are RT-safe:
- No dynamic memory allocation in audio callbacks
- Pre-allocated buffers only
- No Python, ML, or disk I/O in callbacks
- Deterministic execution
- Loaded at startup, remains in memory

## Memory Management (16GB Constraint)

- Tier B worker loads exactly one model per job
- Model quantization (Q4) enforced for 16GB systems
- Worker exits fully after job completion (no persistent processes)
- External SSD used only for reference audio, profiles, models, caches, outputs (never swap/paging)

## External SSD Directory Structure

Base path: Configurable via `PRROT_EXTERNAL_SSD_PATH` environment variable

```
prrot/
├── reference_audio/      # Raw voice reference audio
├── voice_profiles/       # Extracted voice profiles (JSON)
├── models/              # Cached ML models (quantized)
├── jobs/                # Job artifacts
├── caches/              # Cached embeddings and features
└── outputs/             # Generated control data outputs
```

## License Compliance

All dependencies must be Apache 2.0, MIT, or BSD. No proprietary services or telemetry.
