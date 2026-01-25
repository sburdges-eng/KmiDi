# Harmony System & Dependencies Search Report

**Date:** 2026-01-21
**Status:** 🔍 SEARCH COMPLETE

## Files Found

### 1. harmony_system.py
**Location:** `/Users/seanburdges/My Mac/Desktop/kelly-midi-companion/harmony_system.py`
**Size:** 14,575 bytes (403 lines)
**Status:** ✅ FOUND

**Description:**
Comprehensive intelligent harmony system that integrates:
- Chord detection (rule-based + ML hybrid)
- Key + Scale analysis
- Probabilistic harmony engine
- Chord memory + long-term context

**Dependencies Required:**
The file uses relative imports:
```python
from .chord_detector import ChordDetector, PolyphonicScorer, JazzChordAnalyzer, ChordMatch
from .key_analyzer import KeyScaleAnalyzer, KeyAnalyzer, KeyEstimate, Mode, ModalInterchangeDetector, SecondaryDominantDetector
from .harmony_engine import ProbabilisticHarmonyEngine, Genre, CadenceDetector, VoiceLeadingAnalyzer, TensionResolutionAnalyzer
from .chord_memory import ChordMemorySystem, SectionType, EmotionalArc, ChordEvent
```

**Missing Dependencies:**
❌ `chord_detector.py` - NOT FOUND in kelly-midi-companion directory
❌ `key_analyzer.py` - NOT FOUND in kelly-midi-companion directory
❌ `harmony_engine.py` - NOT FOUND in kelly-midi-companion directory
❌ `chord_memory.py` - NOT FOUND in kelly-midi-companion directory

### 2. chord_detection.py
**Locations Found:**
- `/Users/seanburdges/MISC CODE/chord_detection.py` (410 lines)
- `/Users/seanburdges/KmiDi-1/KmiDi_FINAL/python/music_brain/audio/chord_detection.py` (410+ lines)

**Status:** ✅ FOUND (but different implementation)

**Description:**
Audio-based chord detection using chromagram analysis and template matching.
Contains `ChordDetector` class but NOT the same as what harmony_system.py expects.

**Key Differences:**
- MISC CODE version: Uses librosa, simpler implementation
- harmony_system.py expects: `ChordDetector`, `PolyphonicScorer`, `JazzChordAnalyzer`, `ChordMatch` classes
- Current chord_detection.py: Only has `ChordDetector` class, missing other classes

### 3. audio_analyzer_starter.py
**Locations Found:** Multiple (28+ copies in various archive/backup locations)
**Primary Location:** `/Users/seanburdges/MISC CODE/audio_analyzer_starter.py`
**Status:** ✅ FOUND (but marked as obsolete)

**Description:**
Minimal Phase 2 starter implementation for audio analysis.
- Basic tempo (BPM) detection
- Basic key detection
- Frequency balance (8-band)
- Dynamic range

**Note:** This is indeed a starter/minimal version, superseded by full `analyzer.py` in KmiDi-1.

## Analysis

### harmony_system.py Dependencies

The `harmony_system.py` file requires 4 dependency modules that are NOT present:

1. **chord_detector.py** - Expected classes:
   - `ChordDetector` (exists in chord_detection.py but different implementation)
   - `PolyphonicScorer` (NOT FOUND)
   - `JazzChordAnalyzer` (NOT FOUND)
   - `ChordMatch` (NOT FOUND)

2. **key_analyzer.py** - Expected classes:
   - `KeyScaleAnalyzer` (NOT FOUND)
   - `KeyAnalyzer` (NOT FOUND)
   - `KeyEstimate` (NOT FOUND)
   - `Mode` (NOT FOUND)
   - `ModalInterchangeDetector` (NOT FOUND)
   - `SecondaryDominantDetector` (NOT FOUND)

3. **harmony_engine.py** - Expected classes:
   - `ProbabilisticHarmonyEngine` (NOT FOUND)
   - `Genre` (NOT FOUND)
   - `CadenceDetector` (NOT FOUND)
   - `VoiceLeadingAnalyzer` (NOT FOUND)
   - `TensionResolutionAnalyzer` (NOT FOUND)

4. **chord_memory.py** - Expected classes:
   - `ChordMemorySystem` (NOT FOUND)
   - `SectionType` (NOT FOUND)
   - `EmotionalArc` (NOT FOUND)
   - `ChordEvent` (NOT FOUND)

## Related Files Found in KmiDi-1

✅ **Existing Harmony-Related Files:**
- `KmiDi_FINAL/python/music_brain/audio/chord_detection.py` - ChordDetector class
- `KmiDi_FINAL/python/music_brain/harmony.py` - Harmony functionality
- `KmiDi_FINAL/python/music_brain/structure/chord.py` - Chord structure
- `KmiDi_FINAL/python/music_brain/generative/chord_generator.py` - Chord generation
- `KmiDi_FINAL/python/penta_core/ml/chord_predictor.py` - ML chord prediction
- `KmiDi_FINAL/python/penta_core/rules/harmony_rules.py` - Harmony rules

## Recommendations

### Option 1: Migrate harmony_system.py with Stub Dependencies
1. ✅ Migrate `harmony_system.py` to `music_brain/harmony/harmony_system.py`
2. ⚠️ Create stub modules for missing dependencies:
   - `music_brain/harmony/chord_detector.py` (stub)
   - `music_brain/harmony/key_analyzer.py` (stub)
   - `music_brain/harmony/harmony_engine.py` (stub)
   - `music_brain/harmony/chord_memory.py` (stub)
3. ⚠️ Gradually implement or adapt from existing KmiDi-1 modules

### Option 2: Adapt harmony_system.py to Use Existing Modules
1. ✅ Migrate `harmony_system.py`
2. ✅ Refactor imports to use existing KmiDi-1 modules:
   - Use `music_brain.audio.chord_detection.ChordDetector`
   - Create adapters for missing functionality
   - Implement missing classes using existing infrastructure

### Option 3: Search for Dependencies Elsewhere
1. 🔍 Search other directories (RECOVERY_OPS, Archive, etc.)
2. 🔍 Check if dependencies exist with different names
3. 🔍 Look for complete harmony module packages

## Next Steps

1. ⚠️ **Search for dependencies in other locations** (RECOVERY_OPS, Archive, etc.)
2. ⚠️ **Check if similar functionality exists in KmiDi-1** that can be adapted
3. ⚠️ **Decide on migration strategy** (stub vs. adapt vs. wait for dependencies)

## Status

**harmony_system.py:** ✅ FOUND but dependencies missing
**audio_analyzer_starter.py:** ✅ FOUND (obsolete, multiple copies)
**chord_detection.py:** ✅ FOUND (different implementation)
**Dependencies:** ❌ NOT FOUND in expected locations
