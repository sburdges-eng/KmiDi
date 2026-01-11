# Complete DAW Documentation & Information
## Compiled from all user-uploaded DAW data - Including ALL Audio Files

**Date Created:** 2025-11-28
**Purpose:** Comprehensive collection of all DAW-related information, tools, workflows, documentation, and audio resources

**IMPORTANT:** This is a compilation document. All original files remain untouched and preserved.

---

# Table of Contents

1. [Audio File Inventory](#audio-file-inventory)
2. [iDAW System Overview](#idaw-system-overview)
3. [DAiW Music Brain System](#daiw-music-brain-system)
4. [Logic Pro X Settings & Workflow](#logic-pro-x-settings--workflow)
5. [DAW Integration Guide](#daw-integration-guide)
6. [Harmony & Chord Systems](#harmony--chord-systems)
7. [Groove & Rhythm Systems](#groove--rhythm-systems)
8. [Audio Production Philosophy](#audio-production-philosophy)
9. [Complete File Locations Reference](#complete-file-locations-reference)

---

# Audio File Inventory

## WAV Files Overview

### Production Audio Files (AudioVault)

**Location:** `/Users/seanburdges/Music/AudioVault/`
**Total Size:** 1.0MB
**Organization:** Two versions - Raw and Refined

#### Refined Demo Kit (`/Music/AudioVault/refined/Demo_Kit/`)
Complete drum kit with processed samples:

| File | Size | Type | Notes |
|------|------|------|-------|
| Crash_01.wav | 129K | Cymbal | Crash cymbal |
| HiHat_Closed_01.wav | 13K | HiHat | Closed hi-hat variant 1 |
| HiHat_Closed_02.wav | 8.7K | HiHat | Closed hi-hat variant 2 |
| HiHat_Open_01.wav | 43K | HiHat | Open hi-hat |
| Kick_01.wav | 43K | Kick | Kick drum variant 1 |
| Kick_02.wav | 52K | Kick | Kick drum variant 2 |
| Snare_01.wav | 26K | Snare | Snare drum variant 1 |
| Snare_02.wav | 34K | Snare | Snare drum variant 2 |
| Tom_High_01.wav | 34K | Tom | High tom |
| Tom_Mid_01.wav | 34K | Tom | Mid tom |
| Tom_Low_01.wav | 34K | Tom | Low tom |

**Last Modified:** November 25, 2024

#### Raw Demo Kit (`/Music/AudioVault/raw/Demo_Kit/`)
Unprocessed versions of the same samples (same file names and structure)

### Guitar Test Files

**Location:** `/Users/seanburdges/Music/TEST/Audio Files/`

- `GUITAR L#06.wav` - Left channel guitar recording
- `GUITAR R#11.wav` - Right channel guitar recording #11
- `GUITAR R#12.wav` - Right channel guitar recording #12
- `TEST.wav` - Test audio file
- `TEST.mp3` - Test audio (MP3 format)
- `TEST.m4a` - Test audio (M4A format)

### JUCE Framework Sample Files

**Locations:** Multiple directories in `/Users/seanburdges/Downloads/JUCE/` and `/Downloads/JUCE 4/`

Standard JUCE example audio files (repeated across multiple build directories):

- `guitar_amp.wav` - Guitar amp simulation example
- `cello.wav` - Cello sample for audio examples
- `reverb_ir.wav` - Reverb impulse response
- `cassette_recorder.wav` - Cassette recorder effect sample

### Acoustic Guitar Sample Library

**Location:** `/Users/seanburdges/Desktop/clusterfuck/Project 3/Samples/Steel String Acoustic/`

Complete sampled steel string acoustic guitar with note mapping:

**Format:** All files in AIFF format (.aif)
**Naming Convention:** `50B-[position][string][variation]-[note][octave].aif`

Examples:

- `50B-1GA1-D2.aif` - D note, octave 2
- `50B-1GA1-E1.aif` through `50B-1GA1-E4.aif` - E notes across octaves
- `50B-1GA2-A1.aif` through `50B-1GA2-A3.aif` - A notes across octaves
- `50B-1GA3-D2.aif`, `50B-1GA3-D3.aif` - D note variations
- `50B-1GA3-E1.aif` through `50B-1GA3-E4.aif` - E note variations
- `50B-1GA3-F#1.aif`, `50B-1GA3-F#2.aif` - F# variations
- `50B-1GA4-G2.aif` - G note
- `50B-1GA5-B2.aif`, `50B-1GA5-C2.aif` - B and C notes
- `50B-2GA1-06.aif` through `50B-2GA1-11.aif` - Alternative samples

**Total Coverage:** Full chromatic range with multiple velocity layers and round-robins

## Google Drive Sample Library

**Base Location:** `~/Google Drive/My Drive/audio_vault/`

### Drums
- **Drum Tornado 2023** (including Acoustic variations)
- **Drum Empire 2020** (+ add-ons)
- **Studio 2018/2009/2008 Percussion**

### Organized Elements
- **Kick** folder - Kick drum samples
- **Snare** folder - Snare drum samples
- **HiHat** folder - Hi-hat samples
- **Percussion** folder - Misc percussion

### Synths & Keys
- **Synthwave** (organized by BPM subfolders)
- **Plucks & Keys**
- **Pads & Strings**

### Bass
- **Bass** folder - Bass samples and loops

**Estimated Total:** ~8GB of samples

---

# iDAW System Overview

## Core Philosophy

**iDAW (intelligent Digital Audio Workspace)** - A complete music generation system that creates finished songs from emotional intent.

### Three Core Principles:
1. **"Interrogate Before Generate"** - Understand the emotion first, then translate to music
2. **"Imperfection is Intentional"** - Lo-fi aesthetic treats flaws as authenticity
3. **"Every Rule-Break Needs Justification"** - Breaking music theory rules requires emotional reasoning

## Three Operating Modes

### MODE 1: GENERATE
**Input:** Emotional description, vernacular, or technical specs
**Output:** Complete Python script that generates MIDI + renders audio with samples

**Process:**
1. Parse input (vernacular → technical, emotion → parameters)
2. Suggest rule-breaks if emotional intent warrants
3. Generate complete arrangement (drums, bass, chords, melody)
4. Map instruments to user's sample library
5. Output Python script user runs locally

**Example input:** "fat laid back grief song in F, 82 BPM, lo-fi bedroom"

**Sample Mapping:** The system automatically maps to your local sample library:
- Drums → AudioVault Demo Kit or Drum Tornado/Empire
- Bass → Bass folder samples
- Pads → Pads & Strings folder
- Keys → Plucks & Keys folder

### MODE 2: CRITIQUE
**Input:** User's song idea, MIDI file description, or lyrics
**Output:** Three-perspective analysis

**The Three Critics:**

1. **Quality Checker** - Technical assessment
   - Timing naturalness (robotic vs human?)
   - Velocity dynamics (static vs expressive?)
   - Harmonic coherence (does it make sense?)
   - Phrase continuity (does it breathe?)
   - Score: 0-100

2. **Interpretation Critic** - Intent matching
   - Does the music match the stated emotion?
   - Cliché detection (is this the 4-chord progression again?)
   - Vernacular misinterpretation check
   - Suggested fixes

3. **Arbiter** - Final judgment
   - Weighs both perspectives against artistic intent
   - Prevents over-correction (killing expression)
   - Prevents under-correction (letting garbage through)
   - Final verdict: PASS / REVISE / RETHINK

### MODE 3: ANALYZE
**Input:** Description of existing song (user's or reference)
**Output:** Deep breakdown using all databases

**Analysis includes:**
- Key, mode, tempo
- Chord progression (with meme theory name if applicable)
- Rule-breaks identified (intentional or accidental)
- Emotional mapping (what emotions does this evoke?)
- Production notes (vernacular description of sound)
- Sample suggestions from user's library to recreate

## Vernacular Translation

| They say | You interpret |
|----------|---------------|
| "fat" | Full low-mids, light saturation |
| "laid back" | Behind-beat timing, relaxed groove |
| "crispy" | Bright presence, high-end clarity |
| "punchy" | Fast attack, transient emphasis |
| "lo-fi" | Degraded, vintage, imperfect |
| "breathing" | Rubato, tempo fluctuation |
| "boots and cats" | 4/4 kick-hat pattern |
| "boom bap" | Hip-hop groove pattern |

## Rule-Break Suggestions by Emotion

| Emotion | Suggest | Musical Effect |
|---------|---------|----------------|
| grief | STRUCTURE_NonResolution | Yearning without closure |
| bittersweet | HARMONY_ModalInterchange | Happy-sad ambiguity |
| power | HARMONY_ParallelMotion | Bold, defiant strength |
| anxiety | RHYTHM_ConstantDisplacement | Off-kilter unease |
| vulnerability | PRODUCTION_PitchImperfection | Raw emotional honesty |
| chaos | HARMONY_Polytonality | Multiple competing keys |

---

# DAiW Music Brain System

## Project Overview

DAiW-Music-Brain is a CLI toolkit and Python library for:

- **Groove extraction & application** - Extract timing/velocity patterns from MIDI, apply genre templates
- **Chord & harmony analysis** - Roman numeral analysis, key detection, borrowed chord identification
- **Intent-based song generation** - Three-phase deep interrogation system for emotionally-driven composition
- **Intentional rule-breaking** - Structured approach to breaking music theory "rules" for emotional effect
- **Interactive teaching** - Lessons on production philosophy and music theory concepts

## The Three-Phase Intent Schema

### Phase 0: Core Wound/Desire (Deep interrogation)
- `core_event` - What happened?
- `core_resistance` - What holds you back from saying it?
- `core_longing` - What do you want to feel?
- `core_stakes` - What's at risk?
- `core_transformation` - How should you feel when done?

### Phase 1: Emotional Intent (Validated by Phase 0)
- `mood_primary` - Dominant emotion
- `mood_secondary_tension` - Internal conflict (0.0-1.0)
- `vulnerability_scale` - Low/Medium/High
- `narrative_arc` - Climb-to-Climax, Slow Reveal, Repetitive Despair, etc.

### Phase 2: Technical Constraints (Implementation)
- `technical_genre`, `technical_key`, `technical_mode`
- `technical_rule_to_break` - Intentional rule violation
- `rule_breaking_justification` - WHY break this rule (required!)

## Rule-Breaking Categories

### Harmony Rules
| Rule | Effect | Use When |
|------|--------|----------|
| HARMONY_AvoidTonicResolution | Unresolved yearning | Grief, longing |
| HARMONY_ModalInterchange | Bittersweet color | Making hope feel earned |
| HARMONY_ParallelMotion | Power, defiance | Anger, punk energy |
| HARMONY_Polytonality | Chaos, conflict | Multiple competing emotions |

### Rhythm Rules
| Rule | Effect | Use When |
|------|--------|----------|
| RHYTHM_ConstantDisplacement | Off-kilter anxiety | Before a dramatic shift |
| RHYTHM_TempoFluctuation | Organic breathing | Intimacy, vulnerability |

### Arrangement Rules
| Rule | Effect | Use When |
|------|--------|----------|
| ARRANGEMENT_BuriedVocals | Dissociation, texture | Dreams, distance |
| ARRANGEMENT_ExtremeDynamicRange | Dramatic impact | Emotional peaks/valleys |

### Production Rules
| Rule | Effect | Use When |
|------|--------|----------|
| PRODUCTION_PitchImperfection | Emotional honesty | Raw vulnerability |
| PRODUCTION_ExcessiveMud | Claustrophobia | Overwhelming emotion |

## CLI Commands

```bash
# Groove operations
daiw extract drums.mid                    # Extract groove from MIDI
daiw apply --genre funk track.mid         # Apply genre groove template

# Chord analysis
daiw analyze --chords song.mid            # Analyze chord progression
daiw diagnose "F-C-Am-Dm"                 # Diagnose harmonic issues
daiw reharm "F-C-Am-Dm" --style jazz      # Generate reharmonizations

# Intent-based generation
daiw intent new --title "My Song"         # Create intent template
daiw intent process my_intent.json        # Generate from intent
daiw intent suggest grief                 # Suggest rules to break
daiw intent list                          # List all rule-breaking options

# Teaching
daiw teach rulebreaking                   # Interactive teaching mode
```

---

# Logic Pro X Settings & Workflow

## Recommended Project Settings

### New Project Defaults
```
Sample Rate: 48 kHz (or 44.1 for CD-only projects)
Bit Depth: 24-bit
Frame Rate: (leave default unless for video)
```

### Audio Preferences
```
Output Device: PreSonus AudioBox iTwo
Input Device: PreSonus AudioBox iTwo
Buffer Size: 128 (recording) / 256-512 (mixing)
Process Buffer Range: Medium or Large
Multithreading: Playback & Live Tracks
```

## Essential Keyboard Shortcuts

### Transport
| Shortcut | Action |
|----------|--------|
| Space | Play/Stop |
| Enter | Go to beginning |
| R | Record |
| C | Cycle on/off |

### Tracks
| Shortcut | Action |
|----------|--------|
| M | Mute track |
| S | Solo track |
| A | Show/hide automation |
| Cmd+T | New track |
| Cmd+D | Duplicate region |

### Windows & Views
| Shortcut | Action |
|----------|--------|
| X | Mixer |
| Y | Library |
| E | Editors |
| P | Piano Roll |
| B | Smart Controls |

## Stock Plugins Guide

### Essential Plugins

**Channel EQ** - Your go-to EQ
- Clean, transparent
- Built-in analyzer
- 8 bands

**Compressor Circuit Types:**
| Type | Character | Best For |
|------|-----------|----------|
| Platinum Digital | Clean, precise | Surgical control |
| Studio VCA | Punchy, snappy | Drums, general |
| Vintage Opto | Smooth, slow | Vocals, bass |

**ChromaVerb** - Algorithmic reverb
- Visual feedback
- Great algorithms
- Built-in damping EQ

**Alchemy** - THE synth in Logic
- Subtractive, Wavetable, Granular, Sampling, Additive
- Best for: Any synth sound

## Mix Bus Chain (Stock Only)

1. **Gain** — Trim input level
2. **Channel EQ** — Gentle shaping
3. **Compressor** (Vintage VCA) — Glue, 1-3dB GR
4. **Limiter** — Final peak control

---

# DAW Integration Guide

## Importing iDAW Generated Files into Logic

### Step-by-Step Process

1. **Open Logic Pro X**
2. **Create new project** at target BPM (e.g., 82 BPM for Kelly song)
3. **File → Import → MIDI File**
4. **Select** generated .mid file from `~/Music/iDAW_Output/`
5. **Assign instruments** to tracks:
   - Lead Track → Keys/Piano category
   - Pad Track → Pads category
   - Bass Track → Bass Synth category

### Example: Kelly Song Import

**Files to import:**
- `iDAW_20251127_182312_neutral.mid` - MIDI arrangement
- `iDAW_20251127_182312_neutral.idaw.json` - Metadata
- `iDAW_20251127_182312_neutral.README.md` - Import instructions

**Project Settings:**
- BPM: 82
- Key: F major
- Emotion: Neutral (customizable)

**Sample Assignment:**
- Lead → Plucks & Keys folder
- Pad → Pads & Strings folder
- Bass → Bass folder

### Assigning Local Samples

After MIDI import, assign your local WAV files:

1. **For drums:**
   - Use AudioVault Demo Kit (refined or raw)
   - Or use Drum Tornado 2023 from Google Drive

2. **For melodic instruments:**
   - Acoustic guitar → Steel String Acoustic samples
   - Pads → Pads & Strings folder
   - Bass → Bass sample folder

3. **Load in EXS24/Sampler:**
   - Drag WAV files directly
   - Or use Quick Sampler for single shots

---

# Harmony & Chord Systems

## Harmony Generator

**Purpose:** Translates emotional intent → chord progressions → MIDI files

**Features:**
- Implements modal interchange (borrowing chords from parallel keys)
- Supports avoid-resolution rule-breaking
- Generates playable MIDI with proper voicings
- Works with CompleteSongIntent schema

**Example - Kelly Song:**
```python
from music_brain.harmony import HarmonyGenerator

generator = HarmonyGenerator()
harmony = generator.generate_from_intent(kelly_intent)
# Result: F - C - Dm - Bbm (modal interchange applied)
# Bbm = borrowed from F minor, creates "bittersweet darkness"
```

## Chord Diagnostics

**Purpose:** Comprehensive chord analysis tool

**Features:**
- Performs Roman numeral analysis
- Detects borrowed chords and modal interchange
- Identifies rule-breaking patterns
- Provides emotional function analysis
- Generates reharmonization suggestions

**Example Output:**
```
Progression: F-C-Bbm-F
Key: F major
Roman Numerals: I-V-iv-I

CHORD      ROMAN      DIATONIC   EMOTIONAL FUNCTION
F          I          ✓          home, resolution
C          V          ✓          dominant, tension
Bbm        iv         ✗          bittersweet darkness, borrowed sadness
F          I          ✓          home, resolution (changed by grief)

RULE BREAKS DETECTED:
  • HARMONY_ModalInterchange: Bbm (iv) - parallel minor
  • Emotional justification: "Makes hope feel earned"
```

---

# Groove & Rhythm Systems

## Core Concepts

### Timing Deviations
- **Swing**: How far offbeats deviate from perfect 16ths
  - 50% = straight (quantized)
  - 58-62% = funk/soul swing
  - 66% = triplet swing (jazz)

- **Push/Pull**: Per-instrument timing
  - Kick pushes +15ms = "ahead of beat" (drives)
  - Snare pulls -5ms = "lays back" (relaxed)
  - Hihat pulls -10ms = "tightens up" (precise)

### Velocity Patterns
- **Dynamic Range**: Min/max velocity
- **Accents**: Notes 25%+ louder than average
- **Humanization**: Slight random variations (±5)

## Genre Pocket Maps

| Genre | BPM | Swing | Kick | Snare | HiHat | Character |
|-------|-----|-------|------|-------|-------|-----------|
| Funk | 95 | 58% | +15ms | -8ms | -10ms | Laid back groove |
| Boom-bap | 92 | 54% | +12ms | -5ms | -15ms | Hip-hop pocket |
| Dilla | 88 | 62% | +20ms | -12ms | -18ms | Heavy swing |
| Straight | 120 | 50% | 0ms | 0ms | 0ms | Quantized + humanize |
| Trap | 140 | 51% | +3ms | -2ms | -5ms | Minimal swing |

## Groove Extractor

**Purpose:** Extract timing/velocity patterns from existing MIDI drums

**Example:**
```python
from music_brain.groove import GrooveExtractor

extractor = GrooveExtractor()
groove = extractor.extract_from_midi_file("questlove_drums.mid")

# Output:
# Swing: 57.2% (funk swing)
# kick pushes 18.5ms
# snare lays back 8.2ms
# hihat pulls 12.1ms
# Genre hint: funk/soul pocket
```

## Groove Applicator

**Purpose:** Apply groove templates to MIDI patterns

**Example:**
```python
from music_brain.groove import GrooveApplicator

applicator = GrooveApplicator()
funk = applicator.get_genre_template('funk')

applicator.apply_groove(
    "my_drums_quantized.mid",
    "my_drums_funky.mid",
    funk,
    intensity=1.0  # 0.0-1.0 (100% = full groove)
)
```

### Custom Groove for Kelly Song

```python
kelly_groove = GrooveTemplate(
    name="kelly_lofi_pocket",
    tempo_bpm=82,
    swing_percentage=52,  # Minimal swing = intimacy
    push_pull={
        'kick': 8,      # Slightly pushes
        'snare': -3,    # Barely lays back
        'hihat': -10    # Tight
    },
    velocity_map={
        'kick': 95,     # Not too loud
        'snare': 100,   # Natural
        'hihat': 60     # Quiet, bedroom level
    },
    accent_pattern=[0, 4, 8, 12]  # Every beat
)
```

---

# Audio Production Philosophy

## Core Principles

### 1. Interrogate Before Generate
- Never generate without understanding the emotional "why"
- Technical decisions must serve emotional intent
- Deep questioning reveals true creative direction

### 2. Intentional Imperfection
- Lo-fi aesthetic treats flaws as authenticity
- Voice cracks, timing drift, buried vocals can serve emotion
- Imperfection isn't laziness—it's intentional humanization
- "The wrong note played with conviction is the right note"

### 3. Rule-Breaking Requires Justification
- Every violation of music theory must have emotional reasoning
- Rules exist to be understood, then strategically broken
- Document WHY you break each rule

### 4. Emotional Intent Drives Technical Choices
- Not "what chords sound good?"
- But "what emotional truth am I expressing?"
- Technique serves feeling, never the reverse

### 5. Systematic Humanization
- Feel isn't random—it's systematic deviation from the grid
- Great pockets have consistent, intentional timing variations
- Per-instrument timing creates groove
- "The grid is just a suggestion. The pocket is where life happens"

## Production Workflow Philosophy

### For Lo-Fi Bedroom Aesthetic:
- **Minimal groove** = intimacy
- **Slight humanization** = authenticity
- **No perfection** = lo-fi aesthetic
- **Intentional imperfection** = emotional truth

### For Emotional Honesty:
- `PRODUCTION_PitchImperfection` serves vulnerability
- `ARRANGEMENT_BuriedVocals` creates dissociation
- `HARMONY_ModalInterchange` creates bittersweet ambiguity
- `RHYTHM_TempoFluctuation` mimics organic breathing

## Kelly Song Case Study

**Intent:** Processing grief from finding someone you loved after they chose to leave

**Harmonic Choice:**
- Progression: F - C - Dm - Bbm
- Rule-break: HARMONY_ModalInterchange (Bbm in F major)
- Justification: "Bbm makes hope feel earned and bittersweet"
- Effect: First 3 chords sound like love, Bbm is THE REVEAL (grief speaking)

**Groove Choice:**
- Tempo: 82 BPM
- Swing: 52% (minimal, mostly straight)
- Push/Pull: Kick +8ms, Snare -3ms, Hihat -10ms
- Intensity: 75% (mostly straight, slight humanization)
- Effect: Intimacy without robotic perfection

**Sample Selection:**
- Drums: AudioVault Demo Kit (refined) for clean bedroom sound
- Or: Drum Tornado Acoustic for more organic feel
- Pads: Pads & Strings (swells, warm)
- No bass: Keep it sparse and intimate

**Production:**
- Lo-fi bedroom emo aesthetic
- Keep it raw and unpolished
- Intentional register breaks in vocals
- Buried vocal moments for emotional distance
- Use ChromaVerb for subtle room ambience

---

# Complete File Locations Reference

## iDAW Application Files

### Main Application
- `/Applications/iDAW.app` - Native macOS application (current)
- `/Users/seanburdges/Desktop/iDAW.app` - Desktop version
- `/Users/seanburdges/Downloads/iDAW.app` - Downloaded version
- `/Users/seanburdges/Downloads/iDAW_v1.0.04/iDAW.app` - Version 1.0.04

### iDAW Output Files
**Location:** `/Users/seanburdges/Music/iDAW_Output/`

**MIDI Files:**
- `iDAW_20251127_173218_hope.mid`
- `iDAW_20251127_173526_hope.mid`
- `iDAW_20251127_173701_grief.mid` through `iDAW_20251127_173734_grief.mid`
- `iDAW_20251127_175358_neutral.mid` through `iDAW_20251127_182526_neutral.mid`

**Metadata Files (JSON):**
- `iDAW_20251127_182312_neutral.idaw.json`
- `iDAW_20251127_182414_neutral.idaw.json`
- `iDAW_20251127_182418_neutral.idaw.json`
- `iDAW_20251127_182441_neutral.idaw.json`
- `iDAW_20251127_182459_neutral.idaw.json`
- `iDAW_20251127_182526_neutral.idaw.json`

**README Files:**
- `iDAW_20251127_182312_neutral.README.md`
- `iDAW_20251127_182414_neutral.README.md`
- `iDAW_20251127_182418_neutral.README.md`
- `iDAW_20251127_182441_neutral.README.md`
- `iDAW_20251127_182459_neutral.README.md`
- `iDAW_20251127_182526_neutral.README.md`

### iDAW Source Code (Multiple Versions)
**v1.0.00:**
- `/Users/seanburdges/Downloads/idaw_v1.0.00/`
- `/Users/seanburdges/Desktop/idaw_v1.0.00/`

**v1.0.03:**
- `/Users/seanburdges/Downloads/idaw_v1.0.03/`

**v1.0.04:**
- `/Users/seanburdges/Desktop/idaw_v1.0.04/`
- `/Users/seanburdges/Downloads/iDAW_v1.0.04/`

**Installed Version:**
- `/Users/seanburdges/Applications/iDAW/`

**Key Files:**
- `iDAW_GPT_Instructions.md` - System instructions
- `idaw_complete_pipeline.py` - Main generation pipeline
- `idaw_listener_public.py` - Event listener
- `idaw_ableton_ui.py` - Ableton integration UI
- `idaw_menubar.py` - Menu bar interface
- `idaw_ui.py` - Main UI
- `idaw_launcher.py` - Application launcher
- `idaw_library_integration.py` - Sample library integration

## DAiW Music Brain Files

### Main Project Directories
- `/Users/seanburdges/Desktop/DAiW-Music-Brain/` - Primary development
- `/Users/seanburdges/.cursor/worktrees/DAiW-Music-Brain/hgg/` - Cursor worktree (version hgg)
- `/Users/seanburdges/.cursor/worktrees/DAiW-Music-Brain/hgd/` - Cursor worktree (version hgd)
- `/Users/seanburdges/.cursor/worktrees/DAiW-Music-Brain/iry/` - Cursor worktree (version iry)
- `/Users/seanburdges/.cursor/worktrees/DAiW-Music-Brain/uyc/` - Cursor worktree (version uyc)
- `/Users/seanburdges/.cursor/worktrees/DAiW-Music-Brain/uya/` - Cursor worktree (version uya)
- `/Users/seanburdges/.cursor/worktrees/DAiW-Music-Brain/iug/` - Cursor worktree (version iug)

### Key Documentation Files
**Main Docs:**
- `README.md` - Project overview
- `CLAUDE.md` - AI assistant guide
- `DEVELOPMENT_ROADMAP.md` - Development plan
- `LICENSE` - MIT License

**Integration & Delivery:**
- `docs/INTEGRATION_GUIDE.md` - Integration instructions
- `docs/DAIW_INTEGRATION.md` - Full integration script
- `docs/DELIVERY_SUMMARY.md` - Delivery summary (version 1)
- `docs/DELIVERY_SUMMARY_V2.md` - Delivery summary (version 2)
- `docs/FINAL_SESSION_SUMMARY.md` - Session summary (version 1)
- `docs/FINAL_SESSION_SUMMARY_V2.md` - Session summary (version 2)
- `docs/downloads_README.md` - Downloads documentation
- `docs/idaw_example_README.md` - iDAW example documentation

**Groove Module:**
- `docs/GROOVE_MODULE_GUIDE.md` - Groove system guide (version 1)
- `docs/GROOVE_MODULE_GUIDE_V2.md` - Groove system guide (version 2)

### Python Modules
**Core:**
- `music_brain/cli.py` - CLI interface
- `music_brain/__init__.py` - Package init

**Groove:**
- `music_brain/groove/extractor.py` - Groove extraction
- `music_brain/groove/applicator.py` - Groove application
- `music_brain/groove/groove_engine.py` - Groove engine

**Structure:**
- `music_brain/structure/chord.py` - Chord analysis
- `music_brain/structure/comprehensive_engine.py` - Comprehensive engine
- `music_brain/structure/progression.py` - Progression analysis
- `music_brain/structure/sections.py` - Section analysis
- `music_brain/structure/tension_curve.py` - Tension analysis

**Session:**
- `music_brain/session/generator.py` - Session generator
- `music_brain/session/intent_processor.py` - Intent processing
- `music_brain/session/intent_schema.py` - Intent schema
- `music_brain/session/interrogator.py` - Song interrogator
- `music_brain/session/teaching.py` - Teaching module

**DAW Integration:**
- `music_brain/daw/__init__.py` - DAW init
- `music_brain/daw/logic.py` - Logic Pro integration
- `music_brain/daw/markers.py` - Marker management

**Utils:**
- `music_brain/utils/__init__.py` - Utils init
- `music_brain/utils/instruments.py` - Instrument mappings
- `music_brain/utils/midi_io.py` - MIDI I/O
- `music_brain/utils/ppq.py` - PPQ normalization

**Audio:**
- `music_brain/audio/__init__.py` - Audio init
- `music_brain/audio/feel.py` - Feel analysis
- `music_brain/audio/reference_dna.py` - Reference DNA

### Data Files
**Scales & Theory:**
- `music_brain/data/scales_database.json` - Scales database
- `music_brain/data/chord_progressions.json` - Chord progressions
- `music_brain/data/chord_progression_families.json` - Progression families

**Intent & Emotion:**
- `music_brain/data/song_intent_schema.yaml` - Intent schema
- `music_brain/data/song_intent_examples.json` - Intent examples
- `music_brain/data/genre_pocket_maps.json` - Genre groove maps

**Rule-Breaking:**
- `data/rule_breaks.json` - Rule-breaking database
- `data/emotional_mapping.py` - Emotional mapping
- `data/harmony_generator.py` - Harmony generation
- `data/chord_diagnostics.py` - Chord diagnostics

**Example Data:**
- `data/idaw_examples/iDAW_20251127_182312_neutral.idaw.json`
- `happy.json`, `sad.json`, `angry.json`, `fear.json`, `disgust.json`, `surprise.json`, `blends.json`
- `emotion_thesaurus.py`

### Vault (Knowledge Base)
**Songwriting Guides:**
- `vault/Songwriting_Guides/song_intent_schema.md`
- `vault/Songwriting_Guides/rule_breaking_practical.md`
- `vault/Songwriting_Guides/rule_breaking_masterpieces.md`

**Production Workflows:**
- `vault/Production_Workflows/cpp_audio_architecture.md`
- `vault/Production_Workflows/hybrid_development_roadmap.md`
- `vault/Production_Workflows/juce_getting_started.md`
- `vault/Production_Workflows/juce_survival_kit.md`
- `vault/Production_Workflows/osc_bridge_python_cpp.md`

**Kelly Song Project:**
- `vault/Songs/when-i-found-you-sleeping/research/genre-research.md`
- `vault/Songs/when-i-found-you-sleeping/performance/vowel-guide.md`
- `vault/Songs/when-i-found-you-sleeping/performance/timestamped-sheet.md`
- `vault/Songs/when-i-found-you-sleeping/performance/pitch-correction.md`
- `vault/Songs/when-i-found-you-sleeping/lyrics/version-history.md`
- `vault/Songs/when-i-found-you-sleeping/lyrics/v13-with-chords.md`
- `vault/Songs/when-i-found-you-sleeping/creative/freeze-expansion.md`

**Templates:**
- `vault/Templates/DAiW_Task_Board.md`

## Audio Files Complete List

### Production Samples (AudioVault)
**Refined Kit:** `/Users/seanburdges/Music/AudioVault/refined/Demo_Kit/`
- Crash_01.wav (129K)
- HiHat_Closed_01.wav (13K), HiHat_Closed_02.wav (8.7K)
- HiHat_Open_01.wav (43K)
- Kick_01.wav (43K), Kick_02.wav (52K)
- Snare_01.wav (26K), Snare_02.wav (34K)
- Tom_High_01.wav (34K), Tom_Mid_01.wav (34K), Tom_Low_01.wav (34K)

**Raw Kit:** `/Users/seanburdges/Music/AudioVault/raw/Demo_Kit/`
- (Same file structure as refined)

### Test Files
**Location:** `/Users/seanburdges/Music/TEST/`
- Audio Files/GUITAR L#06.wav
- Audio Files/GUITAR R#11.wav
- Audio Files/GUITAR R#12.wav
- TEST.wav
- TEST.mp3
- TEST.m4a

### Acoustic Guitar Library
**Location:** `/Users/seanburdges/Desktop/clusterfuck/Project 3/Samples/Steel String Acoustic/`
- 30+ .aif files covering full chromatic range
- Multiple octaves and velocity layers
- Format: `50B-[position][string][variation]-[note][octave].aif`

### JUCE Example Audio
**Locations:** Multiple directories in JUCE/JUCE 4
- guitar_amp.wav
- cello.wav
- reverb_ir.wav
- cassette_recorder.wav

## Logic Pro Settings & Configurations

### User Settings
- `/Users/seanburdges/Desktop/Music-Brain-Vault/Gear/Logic Pro Settings.md`
- `/Users/seanburdges/Desktop/clusterfuck/Music-Brain-Vault 2/Gear/Logic Pro Settings.md`
- `/Users/seanburdges/Desktop/clusterfuck/Music-Brain-Vault 2/Gear/Logic Pro Stock Plugins Guide.md`
- `/Users/seanburdges/Desktop/clusterfuck/DAiW_ORIGINAL/Music-Brain-Vault/Gear/Logic Pro Settings.md`

### Project Setup Scripts
- `/Users/seanburdges/Desktop/clusterfuck/create_logic_template.scpt`
- `/Users/seanburdges/Music/Kelly_Song_Project/setup_logic_project.py`
- `/Users/seanburdges/Music/Kelly_Song_Project/LOGIC_PRO_SETUP.txt`

### Templates & Guides
- `/Users/seanburdges/Desktop/clusterfuck/Documents_To_Organize/Music_Production/Logic_Template_Setup_Guide.txt`
- `/Users/seanburdges/Desktop/clusterfuck/Documents_To_Organize/Music_Production/Kelly_Song_Logic_Template.txt`
- `/Users/seanburdges/Desktop/clusterfuck/Documents_To_Organize/Music_Production/LOGIC_PRO_SETUP.txt`

### Databases
- `/Users/seanburdges/Music/Audio Music Apps/Databases/LogicLoopsDatabaseV10.db`

## Google Drive Sample Library
**Base Path:** `~/Google Drive/My Drive/audio_vault/`

**Structure:**
```
audio_vault/
├── Drums/
│   ├── Drum Tornado 2023/ (incl. Acoustic)
│   ├── Drum Empire 2020/ (+ add-ons)
│   ├── Studio 2018/
│   ├── Studio 2009/
│   └── Studio 2008 Percussion/
├── Elements/
│   ├── Kick/
│   ├── Snare/
│   ├── HiHat/
│   └── Percussion/
├── Synths/
│   ├── Synthwave/ (organized by BPM)
│   ├── Plucks & Keys/
│   └── Pads & Strings/
└── Bass/
```

**Total Size:** ~8GB

---

# Quick Reference Tables

## Emotion → Music Parameter Mapping

| Emotion | BPM | Mode | Swing | Rule-Break | Sample Choice |
|---------|-----|------|-------|------------|---------------|
| Grief | 60-82 | Minor/Dorian | 50-52% | ModalInterchange | Acoustic, sparse |
| Anxiety | 100-140 | Phrygian | 50-54% | ConstantDisplacement | Electronic, busy |
| Nostalgia | 70-90 | Mixolydian | 54-58% | ModalInterchange | Warm pads, vintage |
| Calm | 60-80 | Major/Lydian | 50% | None | Soft pads, gentle |
| Anger | 120-180 | Minor | 50% | ParallelMotion | Distorted, heavy |
| Bittersweet | 70-95 | Major+iv | 52-56% | ModalInterchange | Mix clean & dark |

## Sample Assignment Guide

| Track Type | First Choice | Alternative | Notes |
|------------|-------------|-------------|-------|
| Kick | AudioVault/Kick_01.wav | Drum Tornado Kick | Use Kick_02 for variation |
| Snare | AudioVault/Snare_01.wav | Drum Empire Snare | Snare_02 for ghost notes |
| HiHat | AudioVault/HiHat_Closed_01.wav | Drum Tornado HiHats | Use Open for accents |
| Acoustic Guitar | Steel String samples | Logic Vintage Electric Piano | Full chromatic mapping |
| Pads | Google Drive/Pads & Strings | Alchemy presets | Choose by emotion |
| Bass | Google Drive/Bass folder | Logic Retro Synth | Match genre feel |
| Synth Lead | Google Drive/Plucks & Keys | Logic Alchemy | Use for melodic lines |

## Genre → Groove Template

| Genre | Template | BPM | Swing | Push/Pull | Velocity | Use For |
|-------|----------|-----|-------|-----------|----------|---------|
| Funk | funk.json | 95 | 58% | K+15/S-8/H-10 | 60-115 | Groovy, laid back |
| Hip-Hop | boom_bap.json | 92 | 54% | K+12/S-5/H-15 | 70-120 | Classic boom-bap |
| Dilla | dilla.json | 88 | 62% | K+20/S-12/H-18 | 65-120 | Heavy swing feel |
| Electronic | straight.json | 120 | 50% | K0/S0/H0+humanize | 80-110 | Modern production |
| Trap | trap.json | 140 | 51% | K+3/S-2/H-5 | 75-125 | Minimal swing |
| Lo-Fi | kelly_groove | 82 | 52% | K+8/S-3/H-10 | 60-100 | Bedroom intimacy |

---

# Command Reference

## iDAW Commands

### Launch Application
```bash
open /Applications/iDAW.app
```

### Import into Logic
1. Open Logic Pro X
2. Create new project at target BPM
3. File → Import → MIDI File
4. Navigate to `~/Music/iDAW_Output/`
5. Select .mid file
6. Assign instruments from sample library

## DAiW Commands

### Groove Operations
```bash
# Extract groove from MIDI
daiw extract <midi_file>

# Apply genre template
daiw apply --genre funk <input.mid> <output.mid>
daiw apply --genre boom_bap <input.mid> <output.mid>
daiw apply --genre dilla <input.mid> <output.mid>

# Apply custom intensity
daiw apply --genre funk --intensity 0.75 <input.mid> <output.mid>
```

### Chord Analysis
```bash
# Analyze MIDI chords
daiw analyze --chords <midi_file>

# Diagnose progression
daiw diagnose "F-C-Am-Dm"
daiw diagnose "F-C-Bbm-F" --key F --mode major

# Generate reharmonizations
daiw reharm "F-C-Am-Dm" --style jazz
daiw reharm "F-C-Am-Dm" --style classical
```

### Intent Processing
```bash
# Create new intent
daiw intent new --title "My Song" --output my_intent.json

# Process intent to generate music
daiw intent process my_intent.json

# Get rule-break suggestions
daiw intent suggest grief
daiw intent suggest anxiety
daiw intent suggest bittersweet

# List all options
daiw intent list

# Validate intent file
daiw intent validate my_intent.json
```

### Teaching Mode
```bash
# Interactive teaching
daiw teach rulebreaking
daiw teach <topic>
```

## Logic Pro Commands

### Import & Setup
```bash
# Launch Logic
open -a "Logic Pro"

# Create project at specific BPM (via AppleScript)
osascript -e 'tell application "Logic Pro" to set tempo to 82'
```

### Sample Management
```bash
# Open Audio MIDI Setup
open "/System/Applications/Utilities/Audio MIDI Setup.app"

# List Logic plugins
defaults read com.apple.logic10
```

---

# Workflow Examples

## Complete Production Workflow: Kelly Song

### 1. Generate Intent-Based Harmony
```bash
# Create intent JSON
cat > kelly_intent.json << EOF
{
  "song_root": {
    "core_event": "Finding someone I loved after they chose to leave",
    "core_resistance": "Fear of making it about me"
  },
  "song_intent": {
    "mood_primary": "Grief",
    "vulnerability_scale": "High"
  },
  "technical_constraints": {
    "technical_key": "F",
    "technical_mode": "major",
    "technical_rule_to_break": "HARMONY_ModalInterchange",
    "rule_breaking_justification": "Bbm makes hope feel earned"
  }
}
EOF

# Process intent
daiw intent process kelly_intent.json
# Output: F - C - Dm - Bbm progression
```

### 2. Apply Lo-Fi Groove
```python
from music_brain.groove import GrooveTemplate, GrooveApplicator

kelly_groove = GrooveTemplate(
    name="kelly_lofi",
    tempo_bpm=82,
    swing_percentage=52,
    push_pull={'kick': 8, 'snare': -3, 'hihat': -10},
    velocity_map={'kick': 95, 'snare': 100, 'hihat': 60}
)

applicator = GrooveApplicator()
applicator.apply_groove(
    "kelly_drums_programmed.mid",
    "kelly_drums_final.mid",
    kelly_groove,
    intensity=0.75
)
```

### 3. Import into Logic
```bash
# 1. Open Logic Pro X
# 2. New project at 82 BPM
# 3. Import harmony MIDI
# 4. Import drums MIDI
# 5. Assign samples:
#    - Kick → AudioVault/Kick_01.wav
#    - Snare → AudioVault/Snare_01.wav
#    - HiHat → AudioVault/HiHat_Closed_01.wav
# 6. Add pads from Pads & Strings folder
# 7. Record acoustic guitar
# 8. Record vocals with intentional imperfection
```

### 4. Mix with Lo-Fi Aesthetic
**Vocal Chain:**
1. Channel EQ (HPF at 80Hz)
2. Compressor (Vintage Opto, 3:1 ratio)
3. Channel EQ (slight presence boost)
4. ChromaVerb (small room, short decay)
5. **Intentionally bury in mix at emotional moments**

**Drum Bus:**
1. Channel EQ (shape)
2. Compressor (Studio VCA, light)
3. **Keep dynamics natural**

**Master Bus:**
1. Gain (trim)
2. Channel EQ (subtle)
3. Compressor (Vintage VCA, 1-2dB GR for glue)
4. Limiter (-1dB ceiling, minimal limiting)

---

# Meta Principles (Summary)

> **"Interrogate Before Generate"**
> The tool shouldn't finish art for people. It should make them braver.

> **"Imperfection is Intentional"**
> Lo-fi aesthetic treats flaws as authenticity.

> **"Every Rule-Break Needs Justification"**
> When you break music theory rules, you explain WHY it serves the emotional intent.

> **"The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"**
> The technical implementation serves the emotional expression, never the other way around.

> **"The wrong note played with conviction is the right note."**
> Commitment to emotional truth supersedes technical correctness.

> **"The grid is just a suggestion. The pocket is where life happens."**
> Feel isn't random—it's systematic deviation from perfection.

---

# Appendix: File Inventory Summary

## Total Files Found

### Audio Files (WAV)
- **Production samples:** 30 files (AudioVault Demo Kit)
- **Test recordings:** 4 files (Guitar test files)
- **JUCE examples:** 28 files (repeated across directories)
- **Total WAV files:** ~60 unique files

### Audio Files (Other Formats)
- **AIFF samples:** 30+ files (Steel String Acoustic Guitar)
- **MP3/M4A:** 2 files (Test files)

### MIDI Files
- **iDAW generated:** 15+ files
- **Example files:** Multiple in DAiW project

### Documentation Files
- **Markdown docs:** 50+ files
- **Python source:** 100+ files
- **JSON data:** 30+ files
- **Configuration:** 10+ files

### Applications
- **iDAW:** 4 versions/locations
- **DAiW:** Multiple development versions

## Storage Breakdown
- **AudioVault:** 1.0MB (local samples)
- **Google Drive audio_vault:** ~8GB (sample library)
- **JUCE examples:** ~200MB
- **DAiW projects:** ~50MB
- **Documentation:** ~10MB

---

**END OF COMPLETE DAW DOCUMENTATION**

*This comprehensive document combines information from:*
- iDAW GPT Instructions & Documentation
- DAiW Music Brain System & All Modules
- Logic Pro X Settings, Shortcuts & Plugin Guides
- Complete Audio File Inventory (WAV, AIFF, MP3, M4A)
- Harmony, Groove & Chord Analysis Systems
- Integration Guides & Production Workflows
- Philosophy Documents & Teaching Materials
- Kelly Song Project Documentation
- All File Locations & Organization

*Created: 2025-11-28*
*All original source files remain unchanged and preserved.*
*This is a reference compilation document only.*
