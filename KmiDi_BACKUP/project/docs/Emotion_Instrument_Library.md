# Emotion Instrument Library

## Overview

The Emotion Instrument Library is a curated collection of audio samples organized by emotion and instrument type. It provides a structured way to select appropriate audio samples based on the emotional intent of a musical composition.

## Source Location

**Google Drive Path:** `/Users/seanburdges/Library/CloudStorage/GoogleDrive-sburdges@gmail.com/My Drive/GOOGLE KELLY INFO/iDAW_Samples/Emotion_Instrument_Library/`

This directory serves as the source of truth for the emotion instrument library. The catalog is generated from this location and stored in the project at `data/emotion_instrument_library_catalog.json`.

## Directory Structure

```
Emotion_Instrument_Library/
├── base/
│   ├── ANGRY/
│   │   ├── drums/
│   │   ├── guitar/
│   │   ├── piano/
│   │   └── vocals/
│   ├── DISGUST/
│   ├── FEAR/
│   ├── HAPPY/
│   ├── SAD/
│   └── SURPRISE/
└── sub/
    ├── ANXIETY/
    ├── CONTENTMENT/
    ├── DESPAIR/
    ├── DISAPPOINTMENT/
    ├── EXCITEMENT/
    ├── GRIEF/
    ├── HOPE/
    ├── HOSTILITY/
    ├── HURT/
    ├── INDIGNATION/
    ├── IRRITATION/
    ├── JEALOUSY/
    ├── JOY/
    ├── LOVE/
    ├── LONELINESS/
    ├── MELANCHOLY/
    ├── PRIDE/
    ├── RAGE/
    ├── TERROR/
    └── VENGEANCE/
```

### Base Emotions

The `base/` directory contains the six primary emotions from Ekman's basic emotion model:

- **ANGRY** - Anger, aggression, frustration
- **DISGUST** - Revulsion, contempt, distaste
- **FEAR** - Anxiety, terror, apprehension
- **HAPPY** - Joy, happiness, elation
- **SAD** - Sadness, sorrow, melancholy
- **SURPRISE** - Astonishment, shock, amazement

### Sub-Emotions

The `sub/` directory contains more specific emotional states that are variations or combinations of base emotions:

- **ANXIETY** - Worry, nervousness, unease
- **CONTENTMENT** - Satisfaction, peace, calm
- **DESPAIR** - Hopelessness, desperation
- **EXCITEMENT** - Enthusiasm, anticipation, energy
- **GRIEF** - Deep sorrow, mourning, loss
- **HOPE** - Optimism, expectation, aspiration
- **HOSTILITY** - Aggression, antagonism
- **HURT** - Emotional pain, wounded feelings
- **IRRITATION** - Annoyance, frustration
- **JEALOUSY** - Envy, possessiveness
- **JOY** - Happiness, delight, elation
- **LOVE** - Affection, warmth, care
- **LONELINESS** - Isolation, solitude, emptiness
- **MELANCHOLY** - Sadness, pensiveness, nostalgia
- **PRIDE** - Self-satisfaction, accomplishment
- **RAGE** - Intense anger, fury
- **TERROR** - Extreme fear, horror
- **VENGEANCE** - Revenge, retribution

### Instrument Categories

Each emotion directory contains four instrument subdirectories:

- **drums/** - Percussive samples (kicks, snares, hi-hats, loops)
- **guitar/** - Guitar samples (acoustic, electric, riffs, strums)
- **piano/** - Piano/keyboard samples (acoustic, electric, chords, melodies)
- **vocals/** - Vocal samples (phrases, expressions, textures)

## Catalog Generation

The catalog is generated using the `catalog_emotion_library.py` script:

```bash
python scripts/catalog_emotion_library.py
```

This script:
1. Scans the Google Drive emotion instrument library directory
2. Extracts metadata for each audio file (filename, size, path, sample ID)
3. Organizes files by emotion and instrument
4. Generates statistics (total files, sizes)
5. Saves the catalog as JSON at `data/emotion_instrument_library_catalog.json`

### Catalog Structure

The generated catalog has the following structure:

```json
{
  "metadata": {
    "generated": "2026-01-10T01:45:27.317735",
    "source_path": "/path/to/GOOGLE KELLY INFO",
    "schema_version": "1.0.0",
    "statistics": {
      "total_files": 295,
      "total_size_mb": 80.35,
      "emotions": {
        "ANGRY": {
          "files": 17,
          "size_mb": 5.7,
          "level": "base"
        }
      }
    }
  },
  "emotions": {
    "ANGRY": {
      "level": "base",
      "instruments": {
        "drums": [
          {
            "filename": "27836_InduMetal_Drums002_hearted.wav.mp3",
            "path": "/full/path/to/file.mp3",
            "size_bytes": 123456,
            "size_mb": 0.12,
            "extension": ".mp3",
            "sample_id": "27836",
            "modified": "2025-01-01T12:00:00"
          }
        ],
        "guitar": [...],
        "piano": [...],
        "vocals": [...]
      }
    }
  }
}
```

## Usage

### Using the Instrument Selector

The `InstrumentSelector` class in `scripts/idaw_library_integration.py` can use the catalog to find samples:

```python
from scripts.idaw_library_integration import InstrumentSelector, LibraryScanner
from pathlib import Path

# Initialize scanner and selector
scanner = LibraryScanner()
selector = InstrumentSelector(scanner)

# Get samples for an emotion and instrument
samples = selector.get_samples_for_emotion("ANGRY", "drums")
# Returns: ["/path/to/27836_InduMetal_Drums002_hearted.wav.mp3", ...]

# Or use lowercase (automatically normalized)
samples = selector.get_samples_for_emotion("grief", "piano")
```

### Programmatic Access

You can also load the catalog directly:

```python
import json
from pathlib import Path

catalog_path = Path("data/emotion_instrument_library_catalog.json")
with open(catalog_path, 'r') as f:
    catalog = json.load(f)

# Access emotion data
angry_drums = catalog["emotions"]["ANGRY"]["instruments"]["drums"]
for sample in angry_drums:
    print(f"{sample['filename']} - {sample['size_mb']} MB")
```

## Sample ID Extraction

Sample filenames often contain numeric IDs at the beginning (e.g., `27836_InduMetal_Drums002_hearted.wav.mp3`). The catalog extracts these IDs for reference, though they may not always be present.

## Adding New Samples

To add new samples to the library:

1. **Organize by emotion**: Place samples in the appropriate emotion directory (`base/` or `sub/`)
2. **Organize by instrument**: Place samples in the correct instrument subdirectory (`drums/`, `guitar/`, `piano/`, or `vocals/`)
3. **Regenerate catalog**: Run `python scripts/catalog_emotion_library.py` to update the catalog

### Naming Conventions

- Use descriptive filenames that include:
  - Sample ID (if available)
  - Instrument/type description
  - Emotional context (optional)
- Examples:
  - `27836_InduMetal_Drums002_hearted.wav.mp3`
  - `394820_Distorted_Wah_Growl_1.wav.mp3`
  - `621385_Screaming_No_WITHOUT_reverb.wav.mp3`

## Statistics

As of the last catalog generation:

- **Total Files**: 295 audio samples
- **Total Size**: 80.35 MB
- **Base Emotions**: 6 (ANGRY, DISGUST, FEAR, HAPPY, SAD, SURPRISE)
- **Sub-Emotions**: 22 additional emotional states
- **Instrument Categories**: 4 (drums, guitar, piano, vocals)

## Integration with Music Brain

The emotion instrument library integrates with the Music Brain intent schema:

- **Phase 1 (Emotional Intent)**: The `mood_primary` field maps to emotion directories
- **Phase 2 (Technical Constraints)**: Instrument selection uses the catalog to find appropriate samples
- **Rule Breaking**: Some production rules (like `PRODUCTION_Distortion`) may influence which samples are selected

## Related Documentation

- [Song Intent Schema](../music_brain/data/song_intent_schema.yaml) - Defines emotional intent structure
- [Intent Schema Python Module](../music_brain/session/intent_schema.py) - Python implementation
- [Library Integration Script](../scripts/idaw_library_integration.py) - Instrument selection logic

## Future Enhancements

- [ ] Add metadata tags (tempo, key, genre) to samples
- [ ] Support for emotion intensity levels (1-6 scale)
- [ ] Integration with Emotion Scale Library
- [ ] Automatic sample recommendation based on song intent
- [ ] Sample preview/playback functionality
