# Kelly MIDI Project

This folder contains music intelligence modules copied from iDAW and renamed for use with the Kelly MIDI Companion plugin.

## Structure

```
Kelly_MIDI_Project/
├── kellymidicompanion/              # Main Python modules
│   ├── kellymidicompanion_data/    # Emotion mappings, intent examples
│   ├── kellymidicompanion_groove/  # Groove extraction and application
│   ├── kellymidicompanion_session/ # Intent processing and teaching
│   └── kellymidicompanion_emotion_api.py  # Emotion-to-music API
├── kellymidicompanion_data/        # Chord progressions, genre data
├── docs/                           # Documentation
└── examples/                       # Example files (to be added)
```

## Key Modules

### Intent Processing (`kellymidicompanion_session/`)
- **kellymidicompanion_intent_schema.py**: Three-phase intent system (Wound → Emotion → Rule-Breaks)
- **kellymidicompanion_intent_processor.py**: Generates musical elements from intent
- **kellymidicompanion_interrogator.py**: Deep questioning system
- **kellymidicompanion_teaching.py**: Rule-breaking education module

### Groove Engine (`kellymidicompanion_groove/`)
- **kellymidicompanion_extractor.py**: Extract groove patterns from MIDI
- **kellymidicompanion_applicator.py**: Apply grooves to MIDI files
- **kellymidicompanion_templates.py**: Genre-based groove templates
- **kellymidicompanion_groove_engine.py**: "Drunken Drummer" humanization

### Emotion API (`kellymidicompanion_emotion_api.py`)
- Clean interface for emotion-to-music generation
- Declarative and fluent API styles
- Maps emotional intent to musical parameters

## Data Files

### `kellymidicompanion_data/`
- `chord_progression_families.json`: Progression family definitions
- `chord_progressions_db.json`: Database of common progressions
- `common_progressions.json`: Frequently used progressions
- `genre_mix_fingerprints.json`: Genre mixing characteristics
- `genre_pocket_maps.json`: Genre-specific groove pocket maps

### `kellymidicompanion/kellymidicompanion_data/`
- Emotion JSON files: `anger.json`, `joy.json`, `sad.json`, `fear.json`, etc.
- `song_intent_schema.yaml`: Complete intent schema definition
- `song_intent_examples.json`: Example intents
- `kellymidicompanion_emotional_mapping.py`: Emotion-to-music parameter mappings

## Usage

All imports have been updated to use `kellymidicompanion` naming:

```python
from kellymidicompanion.kellymidicompanion_session.kellymidicompanion_intent_schema import (
    CompleteSongIntent,
    HarmonyRuleBreak,
    suggest_rule_break,
)

from kellymidicompanion.kellymidicompanion_groove.kellymidicompanion_extractor import (
    extract_groove,
    GrooveTemplate,
)
```

## Philosophy

> "Interrogate Before Generate" — The tool shouldn't finish art for people; it should make them braver.

All modules follow the three-phase intent system:
1. **Phase 0**: Core Wound/Desire (what hurts?)
2. **Phase 1**: Emotional Intent (map to 216-node thesaurus)
3. **Phase 2**: Technical Constraints (which rules to break and why)

## Integration with Kelly MIDI Companion

These modules can be integrated into the C++ JUCE plugin via:
- Python bindings (using pybind11 or similar)
- JSON data files (loaded at runtime)
- OSC communication (for real-time interaction)

## Notes

- All files have been renamed from `music_brain`/`DAiW` to `kellymidicompanion`
- All imports have been updated accordingly
- Documentation references have been updated to "Kelly MIDI Companion"
- Original file structure and functionality preserved

