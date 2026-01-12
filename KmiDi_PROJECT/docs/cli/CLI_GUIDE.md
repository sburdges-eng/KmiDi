# KmiDi CLI Guide

> Complete command-line interface reference for the `daiw` tool

**Command**: `daiw`
**Version**: 1.0.0

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Reference](#quick-reference)
3. [Groove Commands](#groove-commands)
4. [Harmony Commands](#harmony-commands)
5. [Audio Commands](#audio-commands)
6. [Intent Commands](#intent-commands)
7. [Generation Commands](#generation-commands)
8. [Teaching Mode](#teaching-mode)
9. [Common Workflows](#common-workflows)
10. [Configuration](#configuration)
11. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
# Install from project root
cd KmiDi_PROJECT
pip install -e .

# Verify installation
daiw --version
daiw --help
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `daiw extract <file>` | Extract groove from MIDI |
| `daiw apply --genre <g> <file>` | Apply groove template |
| `daiw humanize <file>` | Humanize MIDI timing |
| `daiw analyze --chords <file>` | Analyze chord progression |
| `daiw diagnose <prog>` | Diagnose harmonic issues |
| `daiw reharm <prog>` | Generate reharmonizations |
| `daiw generate` | Generate music from params |
| `daiw intent new` | Create intent template |
| `daiw intent suggest <emotion>` | Get rule-breaking suggestions |
| `daiw teach <topic>` | Interactive learning |

---

## Groove Commands

### daiw extract

Extract groove characteristics from a MIDI file.

```bash
daiw extract <midi_file> [-o output.json]
```

**Options**:
- `-o, --output <file>` - Output JSON file (default: `<input>_groove.json`)

**Examples**:

```bash
# Basic extraction
daiw extract drums.mid

# Specify output file
daiw extract drums.mid -o my_groove.json

# Extract from subdirectory
daiw extract samples/funk_drums.mid
```

**Output Format**:

```json
{
  "timing_offsets": [0.02, -0.01, 0.03, ...],
  "velocity_curve": [100, 95, 110, ...],
  "swing_factor": 0.23,
  "pocket_depth": 0.15,
  "ppq": 480
}
```

---

### daiw apply

Apply a genre groove template to a MIDI file.

```bash
daiw apply --genre <genre> <midi_file> [-o output.mid]
```

**Available Genres**:

| Genre | Description |
|-------|-------------|
| `funk` | Classic funk pocket |
| `boom_bap` | 90s hip-hop feel |
| `dilla` | J Dilla-style drunk drums |
| `trap` | Modern trap hi-hats |
| `jazz` | Jazz swing feel |
| `rock` | Straight rock pocket |
| `latin` | Latin rhythmic feel |

**Examples**:

```bash
# Apply funk groove
daiw apply --genre funk track.mid

# Apply Dilla feel with output
daiw apply --genre dilla drums.mid -o dilla_drums.mid
```

---

### daiw humanize

Add human feel to MIDI with emotional presets.

```bash
daiw humanize <midi_file> [options] [-o output.mid]
```

**Options**:

| Option | Description |
|--------|-------------|
| `--preset <name>` | Emotional preset |
| `--style <style>` | Feel style: tight, natural, loose, drunk |
| `--complexity <0-1>` | Timing chaos (default: 0.5) |
| `--vulnerability <0-1>` | Dynamic fragility (default: 0.5) |
| `--list-presets` | Show available presets |

**Presets**:

| Preset | Description |
|--------|-------------|
| `lofi_depression` | Lo-fi, fragile, hesitant |
| `defiant_punk` | Aggressive, rushed |
| `natural` | Subtle humanization |
| `drunk_jazz` | J Dilla / loose jazz |

**Examples**:

```bash
# List available presets
daiw humanize --list-presets

# Apply lo-fi preset
daiw humanize input.mid --preset lofi_depression

# Custom humanization
daiw humanize input.mid --style loose --complexity 0.7 --vulnerability 0.8

# Output to specific file
daiw humanize input.mid --preset natural -o humanized.mid
```

---

## Harmony Commands

### daiw analyze --chords

Analyze chord progression in a MIDI file.

```bash
daiw analyze --chords <midi_file>
```

**Output**:
- Detected chords with timestamps
- Key/mode detection
- Roman numeral analysis
- Harmonic function analysis

**Example**:

```bash
daiw analyze --chords song.mid
```

**Sample Output**:

```
=== Chord Analysis ===
Key: F Major (confidence: 0.89)

Progression:
  Bar 1: F        (I)   - Tonic
  Bar 2: C        (V)   - Dominant
  Bar 3: Am       (iii) - Mediant
  Bar 4: Dm       (vi)  - Submediant

Harmonic Function: T-D-T-T (Classic pop progression)
```

---

### daiw diagnose

Diagnose harmonic issues in a chord progression.

```bash
daiw diagnose <progression>
```

**Progression Format**: Hyphen or space-separated chord names

**Examples**:

```bash
# Hyphen-separated
daiw diagnose "F-C-Am-Dm"

# Space-separated
daiw diagnose "C G Am F"

# With extensions
daiw diagnose "Cmaj7-Dm7-G7-Cmaj7"
```

**Sample Output**:

```
=== Harmonic Diagnosis ===
Key Estimate: F Major

Analysis:
  F  → I   (Tonic)
  C  → V   (Dominant)
  Am → iii (Mediant)
  Dm → vi  (Submediant)

Issues: None detected
Suggestions:
  - Consider adding passing tones between C and Am
  - Try a ii-V (Gm-C) approach to the tonic
```

---

### daiw reharm

Generate reharmonization alternatives.

```bash
daiw reharm <progression> [--style <style>] [--count <n>]
```

**Styles**:

| Style | Description |
|-------|-------------|
| `jazz` | Jazz substitutions (tritone subs, etc.) |
| `pop` | Contemporary pop alternatives |
| `rnb` | R&B extensions and alterations |
| `classical` | Classical voice leading |
| `experimental` | Unusual substitutions |

**Examples**:

```bash
# Jazz reharmonization
daiw reharm "F-C-Am-Dm" --style jazz

# Multiple suggestions
daiw reharm "C-G-Am-F" --style pop --count 5
```

---

## Audio Commands

### daiw analyze-audio

Analyze audio file for musical characteristics.

```bash
daiw analyze-audio <audio_file> [options]
```

**Options**:

| Option | Description |
|--------|-------------|
| `--bpm` | Detect BPM only |
| `--key` | Detect key only |
| `--chords` | Detect chords only |
| `--feel` | Analyze groove/feel |
| `--all` | Full analysis (default) |
| `--max-duration <sec>` | Limit analysis duration |

**Supported Formats**: WAV, MP3, FLAC, M4A, OGG

**Examples**:

```bash
# Full analysis
daiw analyze-audio song.wav

# BPM detection only
daiw analyze-audio song.wav --bpm

# Key and chords
daiw analyze-audio song.wav --key --chords

# Limit to first 60 seconds
daiw analyze-audio long_song.wav --max-duration 60
```

---

### daiw export-features

Export audio features to JSON or CSV.

```bash
daiw export-features <audio_file> -o <output> [options]
```

**Options**:

| Option | Description |
|--------|-------------|
| `--format json\|csv` | Output format (default: json) |
| `--include-segments` | Include segment analysis |
| `--include-chords` | Include chord detection |

**Examples**:

```bash
# Export to JSON
daiw export-features song.wav -o features.json

# Export to CSV
daiw export-features song.wav -o features.csv --format csv

# With chord analysis
daiw export-features song.wav -o features.json --include-chords
```

---

## Intent Commands

### daiw intent new

Create a new song intent template.

```bash
daiw intent new --title <title> [--output <file>]
```

**Examples**:

```bash
# Create template
daiw intent new --title "My Song"

# Specify output file
daiw intent new --title "Grief Song" --output grief_intent.json
```

**Output Template**:

```json
{
  "title": "Grief Song",
  "phase_0": {
    "core_wound": "",
    "core_resistance": "",
    "core_longing": ""
  },
  "phase_1": {
    "mood_primary": "",
    "vulnerability_scale": 5,
    "narrative_arc": ""
  },
  "phase_2": {
    "technical_genre": "",
    "technical_key": "",
    "technical_rule_to_break": ""
  }
}
```

---

### daiw intent suggest

Get rule-breaking suggestions based on emotion.

```bash
daiw intent suggest <emotion>
```

**Examples**:

```bash
daiw intent suggest grief
daiw intent suggest anxiety
daiw intent suggest bittersweet
daiw intent suggest defiance
```

**Sample Output**:

```
=== Rule-Breaking Suggestions for "grief" ===

Recommended Rules to Break:
  1. HARMONY_AvoidTonicResolution
     Effect: Unresolved yearning, perpetual longing

  2. RHYTHM_ConstantDisplacement
     Effect: Anxiety, inability to find footing

  3. ARRANGEMENT_BuriedVocals
     Effect: Dissociation, numbness

  4. PRODUCTION_PitchImperfection
     Effect: Raw emotional honesty

Why: Grief often resists resolution. Breaking these rules
creates music that mirrors the emotional experience.
```

---

### daiw intent list

List all available rule-breaking options.

```bash
daiw intent list
```

---

### daiw intent validate

Validate an intent file against the schema.

```bash
daiw intent validate <intent_file>
```

---

### daiw intent process

Process intent JSON to generate musical elements.

```bash
daiw intent process <intent_file>
```

---

## Generation Commands

### daiw generate

Generate music from parameters or intent file.

```bash
daiw generate [options]
```

**Options**:

| Option | Description |
|--------|-------------|
| `-i, --intent-file <file>` | Intent JSON file |
| `-k, --key <key>` | Key (C, F, Bb, etc.) |
| `-m, --mode <mode>` | Mode (major, minor, dorian, etc.) |
| `-p, --pattern <pattern>` | Roman numeral pattern |
| `-t, --tempo <bpm>` | Tempo (40-200) |
| `-o, --output <file>` | Output MIDI file |

**Examples**:

```bash
# Basic generation
daiw generate --key F --mode major --pattern "I-V-vi-IV" --tempo 82 -o output.mid

# From intent file
daiw generate --intent-file my_intent.json -o output.mid

# Quick generation
daiw generate -k C -m minor -t 70 -o sad.mid
```

---

## Teaching Mode

### daiw teach

Interactive lessons on music theory and production.

```bash
daiw teach <topic> [--quick]
```

**Topics**:

| Topic | Description |
|-------|-------------|
| `rulebreaking` | Philosophy of breaking music rules |
| `borrowed_chords` | Modal interchange and borrowing |
| `voice_leading` | Voice leading principles |
| `counterpoint` | Counterpoint fundamentals |
| `modes` | Modal theory |
| `tension` | Harmonic tension and release |

**Examples**:

```bash
# Full interactive lesson
daiw teach rulebreaking

# Quick summary
daiw teach borrowed_chords --quick

# List all topics
daiw teach --list
```

---

## Common Workflows

### Workflow 1: Humanize a Beat

```bash
# 1. Analyze original
daiw analyze --chords original.mid

# 2. Extract groove reference
daiw extract reference_drums.mid -o reference.json

# 3. Apply humanization
daiw humanize original.mid --preset lofi_depression -o humanized.mid
```

### Workflow 2: Intent-Based Creation

```bash
# 1. Create intent template
daiw intent new --title "Loss" --output loss.json

# 2. Edit loss.json with your emotional intent

# 3. Get suggestions
daiw intent suggest grief

# 4. Validate
daiw intent validate loss.json

# 5. Generate
daiw generate --intent-file loss.json -o output.mid
```

### Workflow 3: Analyze and Improve Progression

```bash
# 1. Diagnose current progression
daiw diagnose "F-C-Am-Dm"

# 2. Get reharmonization ideas
daiw reharm "F-C-Am-Dm" --style jazz --count 3

# 3. Generate new version
daiw generate --key F --mode major --pattern "Fmaj7-C7-Am7-Dm7" -o improved.mid
```

### Workflow 4: Audio to MIDI Analysis

```bash
# 1. Analyze audio
daiw analyze-audio reference.wav --all

# 2. Export features
daiw export-features reference.wav -o features.json --include-chords

# 3. Use detected parameters
daiw generate --key A --mode minor --tempo 95 -o inspired.mid
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KELLY_AUDIO_DATA_ROOT` | Audio data directory | `~/.kelly/audio-data` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `KELLY_DEVICE` | ML device (auto/cpu/cuda/mps) | `auto` |

### Config File

Create `~/.daiw/config.yaml`:

```yaml
defaults:
  output_dir: ~/Music/DAiW
  tempo: 82
  key: C

audio:
  sample_rate: 48000
  buffer_size: 512

logging:
  level: INFO
  file: ~/.daiw/logs/daiw.log
```

---

## Troubleshooting

### Command Not Found

```bash
# Verify installation
pip show kmidi

# Reinstall
pip install -e .
```

### Import Errors

```bash
# Install missing dependencies
pip install -r requirements-production.txt

# Check Python version
python --version  # Need 3.9+
```

### MIDI Analysis Fails

```bash
# Check file exists
ls -la file.mid

# Check file format
file song.mid  # Should show "Standard MIDI"
```

### Audio Analysis Fails

```bash
# Check audio format
ffprobe song.wav

# Install audio dependencies
pip install librosa soundfile
```

---

## Getting Help

```bash
# General help
daiw --help

# Command-specific help
daiw extract --help
daiw generate --help
daiw intent --help

# Version info
daiw --version
```

---

## See Also

- [API Reference](../api/API_REFERENCE.md)
- [GUI Manual](../gui/GUI_MANUAL.md)
- [Quick Start Guide](../QUICKSTART_GUIDE.md)

---

*Last Updated: 2026-01-11*
