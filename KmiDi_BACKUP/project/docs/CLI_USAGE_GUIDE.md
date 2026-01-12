# KmiDi CLI Usage Guide

**Command**: `daiw`  
**Version**: 1.0.0

## Overview

The `daiw` command-line interface provides tools for music analysis, generation, and manipulation. It's the primary way to interact with the KmiDi Music Brain toolkit.

## Installation

```bash
# Install the package
pip install -e .

# Verify installation
daiw --help
```

## Command Reference

### Groove Operations

#### Extract Groove

Extract groove characteristics from a MIDI file.

```bash
daiw extract <midi_file> [-o output.json]
```

**Options**:
- `-o, --output <file>` - Output JSON file (default: `<input>_groove.json`)

**Example**:
```bash
daiw extract drums.mid
daiw extract drums.mid -o my_groove.json
```

**Output**: JSON file with groove characteristics (timing, velocity, swing).

#### Apply Groove Template

Apply a genre groove template to a MIDI file.

```bash
daiw apply --genre <genre> <midi_file> [-o output.mid]
```

**Genres**:
- `funk` - Funk groove
- `boom_bap` - Boom bap hip-hop
- `dilla` - J Dilla style
- `trap` - Trap music
- `jazz` - Jazz swing

**Example**:
```bash
daiw apply --genre funk track.mid
daiw apply --genre boom_bap drums.mid -o funky_drums.mid
```

#### Humanize MIDI

Add human feel to MIDI with emotional presets.

```bash
daiw humanize <midi_file> [options] [-o output.mid]
```

**Options**:
- `--preset <name>` - Emotional preset (use `--list-presets` to see all)
- `--style <style>` - Style: `tight`, `natural`, `loose`, `drunk`
- `--complexity <0.0-1.0>` - Timing chaos (default: 0.5)
- `--vulnerability <0.0-1.0>` - Dynamic fragility (default: 0.5)
- `--list-presets` - List available presets
- `-o, --output <file>` - Output MIDI file

**Presets**:
- `lofi_depression` - Lo-fi depression style
- `defiant_punk` - Defiant punk style
- `natural` - Natural humanization

**Example**:
```bash
daiw humanize input.mid --preset lofi_depression
daiw humanize input.mid --style loose --complexity 0.7
daiw humanize --list-presets
```

### Chord Analysis

#### Analyze Chords

Analyze chord progression in a MIDI file.

```bash
daiw analyze --chords <midi_file>
```

**Example**:
```bash
daiw analyze --chords song.mid
```

**Output**: Chord progression analysis with Roman numerals, key detection, and harmonic function.

#### Diagnose Progression

Diagnose harmonic issues in a chord progression.

```bash
daiw diagnose <progression>
```

**Progression Format**: Space or comma-separated chord names (e.g., "F C Am Dm")

**Example**:
```bash
daiw diagnose "F-C-Am-Dm"
daiw diagnose "C G Am F"
```

**Output**: Key estimate, Roman numeral analysis, and suggestions for improvement.

#### Reharmonize

Generate reharmonization alternatives.

```bash
daiw reharm <progression> [--style <style>]
```

**Styles**:
- `jazz` - Jazz reharmonization
- `pop` - Pop reharmonization
- `rnb` - R&B reharmonization
- `classical` - Classical reharmonization
- `experimental` - Experimental reharmonization

**Example**:
```bash
daiw reharm "F-C-Am-Dm" --style jazz
daiw reharm "C G Am F" --style pop
```

### Audio Analysis

#### Analyze Audio File

Analyze audio file for BPM, key, chords, and feel.

```bash
daiw analyze-audio <audio_file> [options]
```

**Options**:
- `--bpm` - Detect BPM only
- `--key` - Detect key only
- `--chords` - Detect chords only
- `--feel` - Analyze feel/groove only
- `--all` - Full analysis (default)
- `--max-duration <sec>` - Limit analysis duration

**Example**:
```bash
daiw analyze-audio song.wav
daiw analyze-audio song.wav --bpm
daiw analyze-audio song.wav --key --chords
```

#### Compare Audio Files

Compare two audio files for similarity.

```bash
daiw compare-audio <file1> <file2>
```

**Example**:
```bash
daiw compare-audio original.wav remix.wav
```

#### Batch Analyze

Analyze multiple audio files.

```bash
daiw batch-analyze <files...> [options]
```

**Example**:
```bash
daiw batch-analyze song1.wav song2.wav song3.wav
```

#### Export Features

Export audio features to JSON or CSV.

```bash
daiw export-features <audio_file> -o <out> [options]
```

**Options**:
- `--format json|csv` - Output format (default: json)
- `--include-segments` - Include segment analysis
- `--include-chords` - Include chord detection

**Example**:
```bash
daiw export-features song.wav -o features.json
daiw export-features song.wav -o features.csv --format csv
```

### Intent-Based Generation

#### Create Intent Template

Create a new song intent template.

```bash
daiw intent new --title <title> [--output <file>]
```

**Example**:
```bash
daiw intent new --title "My Song" --output my_intent.json
```

**Output**: JSON template with three-phase intent schema (Core Wound/Desire, Emotional Intent, Technical Constraints).

#### Process Intent

Process intent JSON to generate musical elements.

```bash
daiw intent process <intent_file>
```

**Example**:
```bash
daiw intent process my_intent.json
```

#### Suggest Rule-Breaking

Get suggestions for rules to break based on emotion.

```bash
daiw intent suggest <emotion>
```

**Example**:
```bash
daiw intent suggest grief
daiw intent suggest anxiety
daiw intent suggest bittersweet
```

**Output**: Suggested rule-breaking options with emotional justification.

#### List Rules

List all available rule-breaking options.

```bash
daiw intent list
```

#### Validate Intent

Validate an intent file against the schema.

```bash
daiw intent validate <intent_file>
```

### Music Generation

#### Generate Harmony

Generate harmony from parameters or intent.

```bash
daiw generate [options]
```

**Options**:
- `-i, --intent-file <file>` - Intent JSON file
- `-k, --key <key>` - Key (e.g., C, F, Bb)
- `-m, --mode <mode>` - Mode: `major`, `minor`, `dorian`, etc.
- `-p, --pattern <pattern>` - Roman numeral pattern (e.g., "I-V-vi-IV")
- `-t, --tempo <bpm>` - Tempo in BPM
- `-o, --output <file>` - Output MIDI file

**Example**:
```bash
daiw generate --key F --mode major --pattern "I-V-vi-IV" --tempo 82 -o output.mid
daiw generate --intent-file my_intent.json -o output.mid
```

### Teaching Mode

#### Interactive Teaching

Interactive lessons on music theory and production.

```bash
daiw teach <topic>
```

**Topics**:
- `rulebreaking` - Rule-breaking philosophy
- `borrowed_chords` - Borrowed chord usage
- `voice_leading` - Voice leading principles
- `counterpoint` - Counterpoint rules

**Options**:
- `--quick` - Quick summary mode

**Example**:
```bash
daiw teach rulebreaking
daiw teach borrowed_chords --quick
```

### General MIDI Analysis

#### Analyze MIDI File

Full analysis of a MIDI file.

```bash
daiw analyze <midi_file>
```

**Example**:
```bash
daiw analyze song.mid
```

**Output**: Comprehensive analysis including chords, tempo, key, structure.

## Common Workflows

### Workflow 1: Extract and Apply Groove

```bash
# Extract groove from reference track
daiw extract reference_drums.mid -o reference_groove.json

# Apply groove to your track
daiw apply --genre funk my_track.mid -o funky_track.mid
```

### Workflow 2: Analyze and Improve Chord Progression

```bash
# Analyze existing progression
daiw diagnose "F-C-Am-Dm"

# Get reharmonization suggestions
daiw reharm "F-C-Am-Dm" --style jazz

# Generate new progression
daiw generate --key F --mode major --pattern "I-V-vi-IV" -o new_progression.mid
```

### Workflow 3: Intent-Based Song Creation

```bash
# Create intent template
daiw intent new --title "Grief Song" --output grief_intent.json

# Edit grief_intent.json with your emotional intent...

# Get rule-breaking suggestions
daiw intent suggest grief

# Process intent to generate music
daiw intent process grief_intent.json

# Validate intent
daiw intent validate grief_intent.json
```

### Workflow 4: Audio Analysis Pipeline

```bash
# Analyze audio file
daiw analyze-audio song.wav

# Export features for further analysis
daiw export-features song.wav -o features.json --include-chords

# Compare with reference
daiw compare-audio song.wav reference.wav
```

## Configuration

### Environment Variables

See `docs/ENVIRONMENT_CONFIGURATION.md` for all configuration options.

Key variables:
- `KELLY_AUDIO_DATA_ROOT` - Audio data directory
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

### Config Files

Configuration can be set via:
- Environment variables
- `.env` file in project root
- Command-line arguments

## Troubleshooting

### Command Not Found

```bash
# Verify installation
pip show music_brain

# Reinstall if needed
pip install -e .
```

### MIDI File Not Found

```bash
# Check file path
ls -la <midi_file>

# Use absolute path if needed
daiw extract /full/path/to/file.mid
```

### Audio Analysis Fails

```bash
# Check audio file format (supports: wav, mp3, flac, m4a)
file song.wav

# Check file permissions
chmod 644 song.wav
```

### Import Errors

```bash
# Install missing dependencies
pip install -r requirements-production.txt

# Check Python version (requires 3.9+)
python --version
```

## Examples

### Quick Examples

```bash
# Extract groove
daiw extract drums.mid

# Apply funk groove
daiw apply --genre funk track.mid

# Diagnose progression
daiw diagnose "C G Am F"

# Generate music
daiw generate --key C --mode major --pattern "I-V-vi-IV" -o song.mid

# Analyze audio
daiw analyze-audio song.wav --bpm

# Get rule-breaking suggestions
daiw intent suggest grief
```

## Advanced Usage

### Combining Commands

```bash
# Extract, analyze, and generate in one workflow
daiw extract reference.mid && \
daiw analyze reference.mid && \
daiw generate --key F --mode major -o output.mid
```

### Scripting

```bash
#!/bin/bash
# Batch process multiple files
for file in *.mid; do
    daiw extract "$file" -o "${file%.mid}_groove.json"
done
```

### Python Integration

```python
from music_brain.groove import extract_groove, apply_groove
from music_brain.harmony import analyze_chords

# Use CLI functions programmatically
groove = extract_groove("drums.mid")
chords = analyze_chords("song.mid")
```

## Getting Help

```bash
# General help
daiw --help

# Command-specific help
daiw extract --help
daiw generate --help
daiw intent --help
```

## See Also

- [API Documentation](API_DOCUMENTATION.md)
- [Environment Configuration](ENVIRONMENT_CONFIGURATION.md)
- [Quick Start Guide](QUICKSTART_GUIDE.md)
