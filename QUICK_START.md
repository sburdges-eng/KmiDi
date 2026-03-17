# Quick Start

## Repo build and dev (canonical)

From the KmiDi repo root:

1. **Setup:** `./scripts/dev-setup.sh`
2. **Run:** `npm run dev:all` (React + Tauri + Music Brain API), or run separately: `npm run dev`, `npm run dev:tauri`, `npm run dev:python`
3. **Full builds:** See [README.md](README.md) for the two V1 pipelines (penta_core + PyInstaller vs KellyFFI + Tauri) and [docs/FULL_STACK_BUILD.md](docs/FULL_STACK_BUILD.md) for native integration.

---

## Kelly Companion — Python usage

The Kelly Companion system is already integrated into KmiDi. No additional installation needed.

## Basic Usage

### 1. Emotion Thesaurus

```python
from music_brain.kelly_companion.core.emotion_thesaurus import EmotionThesaurus

thesaurus = EmotionThesaurus()
emotion = thesaurus.find_emotion("joyful")
print(emotion)
```

### 2. Musical Engines

```python
from music_brain.kelly_companion.engines import BassEngine, MelodyEngine

bass_engine = BassEngine()
melody_engine = MelodyEngine()

# Generate bass line
bass_line = bass_engine.generate_bass_line(key="C", tempo=120)

# Generate melody
melody = melody_engine.generate_melody(key="C", emotion="joyful")
```

### 3. Harmony System

```python
from music_brain.kelly_companion.utils.harmony_system import HarmonySystem
from music_brain.kelly_companion.utils.harmony_deps import ChordDetector, KeyAnalyzer

# Detect chords
detector = ChordDetector()
chords = detector.detect_chords(midi_notes)

# Analyze key
analyzer = KeyAnalyzer()
key = analyzer.analyze_key(chords)

# Generate harmony
harmony = HarmonySystem()
progression = harmony.generate_progression(key="C", emotion="joyful")
```

### 4. Intent Processing

```python
from music_brain.kelly_companion.session import IntentProcessor

processor = IntentProcessor()
intent = processor.process_intent("I want to create a joyful, uplifting song")
```

### 5. Data Access

```python
from music_brain.kelly_companion.data import EMOTIONS_DIR, CHORDS_DIR
import json

# Load emotion data
with open(EMOTIONS_DIR / "joy.json") as f:
    joy_data = json.load(f)

# Load chord progressions
with open(CHORDS_DIR / "chord_progressions.json") as f:
    progressions = json.load(f)
```

## Available Engines

1. **ArrangementEngine** - Song arrangement
2. **BassEngine** - Bass line generation
3. **CounterMelodyEngine** - Counter-melody
4. **DynamicsEngine** - Dynamic control
5. **FillEngine** - Fills and transitions
6. **MelodyEngine** - Melody generation
7. **Orchestration** - Orchestration
8. **PadEngine** - Pad textures
9. **RhythmEngine** - Rhythm patterns
10. **StringEngine** - String arrangements
11. **TensionEngine** - Tension/resolution
12. **TransitionEngine** - Transitions
13. **VariationEngine** - Variations

## Available Data

- **Emotions:** anger.json, joy.json, sad.json, fear.json, disgust.json, surprise.json
- **Chords:** chord_progressions.json, chord_progression_families.json, etc.
- **Genres:** genre_pocket_maps.json, genre_mix_fingerprints.json
- **Intent:** song_intent_examples.json, song_intent_schema.yaml

## Documentation

See `COMPLETE_MIGRATION_SUMMARY.md` for full details.
