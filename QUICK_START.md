# Quick Start

Status: current quick-start reference aligned to actual repo scripts
Last updated: 2026-06-08

## Repo build and dev

From the KmiDi repo root:

1. Setup:

```bash
./scripts/dev-setup.sh
```

2. Run the active combined stack:

```bash
npm run dev:all
```

This starts:
- React frontend on `http://localhost:1420`
- Music Brain API on `http://localhost:8000`

3. Or run services separately:

```bash
npm run dev
npm run dev:python
```

4. Verify basic health:

```bash
npx tsc --noEmit
python3 -m pytest tests/unit/test_api_schema.py
```

Important clarification:
- `package.json` does not currently define `npm run dev:tauri`.
- Treat older references to that command as historical/legacy drift, not current runnable truth.

## Native/plugin builds

For KellyFFI / plugin / native work, see:
- `BUILD.md`
- `docs/FULL_STACK_BUILD.md`
- `docs/DEVELOPMENT.md`

Minimal example:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_KELLY_CORE=ON \
  -DBUILD_KELLY_FFI=ON \
  -DKMIDI_BUILD_JUCE_UI=ON \
  -DBUILD_PLUGINS=ON

cmake --build build --target KellyFFI -j8
```

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
```
