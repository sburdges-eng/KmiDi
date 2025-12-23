# DAiW Music Brain - Main Documentation
## Verified & Accurate as of 2025-11-28

---

## Repository

**GitHub:** `https://github.com/sburdges-eng/DAiW-Music-Brain.git`  
**Branch:** main  
**Status:** Up to date with origin, has uncommitted Phase 2 work

---

## What This Is

**DAiW (Digital Audio intelligent Workstation)** is a Python CLI toolkit and library for:

- **Intent-based music generation** — Translate emotions into chord progressions and MIDI
- **Groove extraction & humanization** — Extract timing/velocity patterns, apply genre templates
- **Chord analysis & diagnostics** — Roman numeral analysis, borrowed chord detection
- **Rule-breaking with justification** — Intentional music theory violations for emotional effect
- **Interactive teaching** — Lessons on production philosophy and music theory

---

## Core Philosophy

> **"Interrogate Before Generate"** — Understand the emotion first, then translate to music.

> **"Imperfection is Intentional"** — Lo-fi aesthetic treats flaws as authenticity.

> **"Every Rule-Break Needs Justification"** — Breaking music theory rules requires emotional reasoning.

---

## Verified Working Features

### CLI Commands (all tested ✅)

```bash
# Diagnose chord progressions
daiw diagnose "F-C-Am-Dm"
# Output: Key estimate, Roman numerals, suggestions

# Get rule-breaking suggestions for emotions
daiw intent suggest grief
daiw intent suggest anxiety
daiw intent suggest bittersweet

# Generate MIDI from parameters
daiw generate --key F --mode major --pattern "I-V-vi-IV" --tempo 82 -o output.mid

# Apply groove templates
daiw apply --genre funk input.mid output.mid
daiw apply --genre boom_bap input.mid output.mid
daiw apply --genre dilla input.mid output.mid

# Humanize drums (Drunken Drummer)
daiw humanize input.mid output.mid --style lofi_depression
daiw humanize input.mid output.mid --style defiant_punk

# Extract groove from MIDI
daiw extract drums.mid

# Analyze MIDI
daiw analyze song.mid

# Teaching mode
daiw teach borrowed_chords
daiw teach modal_mixture
```

### Test Status

**35/35 basic tests passing** (verified 2025-11-28)

Tests cover:
- Imports and module structure
- Groove templates and application
- Chord parsing and progression analysis
- Teaching module
- Interrogator system
- Drum humanization presets

---

## Project Structure

```
DAiW-Music-Brain/
├── music_brain/           # Core Python package
│   ├── cli.py             # CLI interface (49KB)
│   ├── groove/            # Groove extraction/application
│   │   ├── extractor.py
│   │   ├── applicator.py
│   │   ├── templates.py
│   │   └── groove_engine.py
│   ├── structure/         # Chord/progression analysis
│   │   ├── chord.py
│   │   ├── progression.py
│   │   ├── comprehensive_engine.py
│   │   └── tension_curve.py
│   ├── session/           # Intent processing
│   │   ├── intent_schema.py
│   │   ├── intent_processor.py
│   │   ├── interrogator.py
│   │   ├── teaching.py
│   │   └── generator.py
│   ├── harmony/           # Harmony generation
│   │   └── harmony_generator.py
│   ├── audio/             # Audio analysis (requires numpy/librosa)
│   │   ├── analyzer.py
│   │   ├── chord_detection.py
│   │   └── feel.py
│   ├── daw/               # DAW integration
│   │   ├── logic.py
│   │   └── markers.py
│   ├── realtime/          # Real-time MIDI (Phase 2)
│   ├── effects/           # Audio effects (Phase 2)
│   └── data/              # JSON data files
│       ├── genre_pocket_maps.json
│       ├── chord_progressions.json
│       ├── song_intent_schema.yaml
│       └── humanize_presets.json
│
├── data/                  # Additional data files
│   ├── harmony_generator.py
│   ├── chord_diagnostics.py
│   ├── groove_extractor.py
│   ├── emotional_mapping.py
│   ├── rule_breaks.json
│   └── scales/
│
├── tests/                 # Test suite (35+ tests)
├── examples/              # Example scripts
│   ├── kelly_song_example.py
│   ├── therapy_prompts_example.py
│   └── midi/              # Example MIDI files
│
├── vault/                 # Knowledge base (Obsidian-compatible)
│   ├── Songwriting_Guides/
│   │   ├── rule_breaking_masterpieces.md
│   │   ├── rule_breaking_practical.md
│   │   └── song_intent_schema.md
│   ├── Songs/
│   │   └── when-i-found-you-sleeping/  # Kelly song project
│   └── Templates/
│
├── docs/                  # Documentation
├── daiw_mcp/              # MCP server (Phase 2)
├── app.py                 # Streamlit UI (102KB)
└── launcher.py            # Desktop launcher
```

---

## Three-Phase Intent Schema

### Phase 0: Core Wound/Desire (Deep Interrogation)
- `core_event` — What happened?
- `core_resistance` — What holds you back from saying it?
- `core_longing` — What do you want to feel?
- `core_stakes` — What's at risk?
- `core_transformation` — How should you feel when done?

### Phase 1: Emotional Intent
- `mood_primary` — Dominant emotion
- `mood_secondary_tension` — Internal conflict (0.0-1.0)
- `vulnerability_scale` — Low/Medium/High
- `narrative_arc` — Climb-to-Climax, Slow Reveal, Repetitive Despair, etc.

### Phase 2: Technical Constraints
- `technical_genre`, `technical_key`, `technical_mode`
- `technical_rule_to_break` — Intentional rule violation
- `rule_breaking_justification` — WHY break this rule (required!)

---

## Rule-Breaking Categories

### Harmony Rules
| Rule | Effect | Use When |
|------|--------|----------|
| `HARMONY_AvoidTonicResolution` | Unresolved yearning | Grief, longing |
| `HARMONY_ModalInterchange` | Bittersweet color | Making hope feel earned |
| `HARMONY_ParallelMotion` | Power, defiance | Anger, punk energy |
| `HARMONY_UnresolvedDissonance` | Lingering tension | Grief, open questions |

### Rhythm Rules
| Rule | Effect | Use When |
|------|--------|----------|
| `RHYTHM_ConstantDisplacement` | Off-kilter anxiety | Before a dramatic shift |
| `RHYTHM_TempoFluctuation` | Organic breathing | Intimacy, vulnerability |

### Production Rules
| Rule | Effect | Use When |
|------|--------|----------|
| `PRODUCTION_PitchImperfection` | Emotional honesty | Raw vulnerability |
| `ARRANGEMENT_BuriedVocals` | Dissociation, texture | Dreams, distance |

---

## Genre Groove Templates

| Genre | BPM | Swing | Kick | Snare | HiHat | Character |
|-------|-----|-------|------|-------|-------|-----------|
| Funk | 95 | 58% | +15ms | -8ms | -10ms | Laid back groove |
| Boom-bap | 92 | 54% | +12ms | -5ms | -15ms | Hip-hop pocket |
| Dilla | 88 | 62% | +20ms | -12ms | -18ms | Heavy swing |
| Straight | 120 | 50% | 0ms | 0ms | 0ms | Quantized + humanize |
| Trap | 140 | 51% | +3ms | -2ms | -5ms | Minimal swing |

---

## Kelly Song Case Study

**Song:** "When I Found You Sleeping"  
**Theme:** Processing grief from finding someone you loved after they chose to leave

**Generated Harmony:**
```
Key: F major
Progression: F - C - Dm - Bbm
Rule break: HARMONY_ModalInterchange
Why: Bbm makes hope feel earned and bittersweet; 
     grief expressed through borrowed darkness
```

**Diagnostic Output:**
```
CHORD      ROMAN      DIATONIC   EMOTIONAL FUNCTION
F          I          ✓          home, resolution
C          V          ✓          dominant, tension seeking resolution
Bbm        iv         ✗          bittersweet darkness, borrowed sadness
F          I          ✓          home, resolution
```

---

## Installation & Usage

```bash
# Clone
git clone https://github.com/sburdges-eng/DAiW-Music-Brain.git
cd DAiW-Music-Brain

# Install
pip install -e .

# Or install dependencies manually
pip install mido pyyaml numpy pytest

# Run tests
pytest tests/test_basic.py -v

# Use CLI
python -m music_brain.cli --help
python -m music_brain.cli diagnose "F-C-Am-Dm"
python -m music_brain.cli generate --key F --mode major --pattern "I-V-vi-IV" -o output.mid

# Run Streamlit UI
pip install streamlit
streamlit run app.py
```

---

## Dependencies

**Required:**
- Python 3.9+
- mido (MIDI I/O)
- pyyaml (config files)

**Optional:**
- numpy (audio analysis)
- librosa (advanced audio)
- streamlit (web UI)
- pytest (testing)

---

## Development Status

### Phase 1 (Complete ✅)
- CLI implementation
- Intent schema system
- Groove extraction/application
- Chord diagnostics
- Teaching module
- Basic MIDI generation
- 35+ passing tests

### Phase 2 (In Progress)
- Real-time MIDI processing
- Audio analyzer tools
- MCP server integration
- Effects processing
- Therapy prompts system

### Uncommitted Work
- `daiw_mcp/` — MCP server for AI integration
- `music_brain/realtime/` — Real-time engine
- `music_brain/effects/` — Audio effects
- `music_brain/harmony/` — Expanded harmony generation
- Phase 2 documentation and summaries

---

## Related Projects

- **iDAW** — Streamlit-based UI with Ableton-style interface (separate app)
- **Music Brain Vault** — Obsidian knowledge base (integrated in `/vault`)

---

## Meta Principles

> **"The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"**

> **"The wrong note played with conviction is the right note."**

> **"The grid is just a suggestion. The pocket is where life happens."**

---

*Documentation verified against actual codebase: 2025-11-28*  
*35/35 tests passing | CLI functional | MIDI generation working*
