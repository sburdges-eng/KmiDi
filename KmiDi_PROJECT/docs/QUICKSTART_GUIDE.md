# KmiDi Quick Start Guide

**Get started in 10 minutes**

## For Therapists & Musicians

### Step 1: Installation (2 minutes)

```bash
# Clone the repository
git clone <repository-url>
cd KmiDi-1

# Install dependencies
pip install -r requirements-production.txt

# Install the package
pip install -e .
```

### Step 2: Basic Usage (3 minutes)

#### Generate Music from Emotion

```bash
# Using the CLI
daiw generate --key F --mode major --pattern "I-V-vi-IV" --tempo 72 -o my_song.mid

# Or use the API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"intent": {"emotional_intent": "grief hidden as love"}}'
```

#### Analyze Existing Music

```bash
# Analyze a MIDI file
daiw analyze song.mid

# Diagnose a chord progression
daiw diagnose "F-C-Am-Dm"

# Analyze audio file
daiw analyze-audio song.wav --bpm
```

### Step 3: Intent-Based Creation (5 minutes)

```bash
# Create an intent template
daiw intent new --title "My Emotional Song" --output my_intent.json

# Edit my_intent.json with your emotional intent:
# - Core wound/desire
# - Emotional intent
# - Technical preferences

# Get rule-breaking suggestions
daiw intent suggest grief

# Process the intent
daiw intent process my_intent.json
```

## For Developers

### Step 1: Setup (2 minutes)

```bash
# Clone and install
git clone <repository-url>
cd KmiDi-1
pip install -e ".[dev]"
```

### Step 2: Run API Server (1 minute)

```bash
# Start the API
cd api
python -m uvicorn api.main:app --reload

# Or use Docker
docker-compose up -d
```

### Step 3: Test (2 minutes)

```bash
# Health check
curl http://localhost:8000/health

# List emotions
curl http://localhost:8000/emotions

# Generate music
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"intent": {"emotional_intent": "calm"}}'
```

### Step 4: Use in Your Code (5 minutes)

```python
from music_brain.emotion_api import MusicBrain

# Initialize
brain = MusicBrain()

# Generate from text
result = brain.generate_from_text("I'm feeling sad and melancholic")

# Access results
print(f"Tempo: {result.musical_params.tempo_suggested}")
print(f"Key: {result.musical_params.key_suggested}")
```

## Common Use Cases

### Use Case 1: Music Therapy Session

```bash
# Client describes emotion
emotion = "grief hidden as love"

# Generate music
daiw generate --intent-file intent.json -o therapy_session.mid

# Apply humanization for emotional authenticity
daiw humanize therapy_session.mid --preset lofi_depression -o final.mid
```

### Use Case 2: Songwriting Assistant

```bash
# Analyze reference track
daiw analyze-audio reference.wav

# Extract groove
daiw extract reference.mid -o reference_groove.json

# Create new progression
daiw generate --key F --mode major --pattern "I-V-vi-IV" -o new_song.mid

# Apply reference groove
daiw apply --genre funk new_song.mid -o final.mid
```

### Use Case 3: Music Analysis

```bash
# Full analysis
daiw analyze song.mid

# Specific analysis
daiw analyze-audio song.wav --bpm --key --chords

# Compare tracks
daiw compare-audio original.wav remix.wav
```

## Next Steps

- **CLI Usage**: See [CLI Usage Guide](CLI_USAGE_GUIDE.md)
- **API Reference**: See [API Documentation](API_DOCUMENTATION.md)
- **Configuration**: See [Environment Configuration](ENVIRONMENT_CONFIGURATION.md)
- **Deployment**: See [Deployment Guide](DEPLOYMENT_GUIDE.md)

## Getting Help

```bash
# CLI help
daiw --help
daiw <command> --help

# API docs
# Visit http://localhost:8000/docs
```

## Troubleshooting

**Command not found?**
```bash
pip install -e .
```

**Import errors?**
```bash
pip install -r requirements-production.txt
```

**API not starting?**
```bash
# Check port
lsof -i :8000

# Check logs
tail -f logs/api.log
```
