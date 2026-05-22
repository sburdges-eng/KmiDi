# MISC CODE Utility Scripts Review

**Date:** 2026-01-21
**Status:** ✅ REVIEW COMPLETE

## Summary

Reviewed missing utility scripts from `/Users/seanburdges/MISC CODE` and `/Users/seanburdges/My Mac/Desktop/kelly-midi-companion` to determine if they should be integrated into KmiDi.

## Files Reviewed

### 1. `audio_tools.py` (MISC CODE)
**Type:** MCP (Model Context Protocol) server tool wrapper
**Purpose:** Provides MCP tools for audio analysis (detect_bpm, detect_key, analyze_audio_feel, extract_chords, detect_scale, analyze_theory)
**Status:** ⚠️ **PARTIALLY DUPLICATE**
- **Similar to:** `KmiDi_FINAL/python/daiw_mcp/tools/audio_analysis.py`
- **Difference:** This is an MCP tool wrapper that imports from `music_brain.audio`
- **Recommendation:** Check if `daiw_mcp/tools/audio_analysis.py` has the same MCP tools. If not, this could be a newer/better MCP integration.

### 2. `audio_analyzer_starter.py` (MISC CODE)
**Type:** Basic audio analysis starter/minimal implementation
**Purpose:** Phase 2 quick start - basic tempo, key detection, frequency analysis (8-band), dynamic range
**Status:** ⚠️ **OBSOLETE/STARTER VERSION**
- **Similar to:** `KmiDi_FINAL/python/music_brain/audio/analyzer.py` (full implementation)
- **Difference:** This is a minimal "starter" version for Phase 2. The full `analyzer.py` in KmiDi is more complete.
- **Recommendation:** **DO NOT MIGRATE** - This is an older/starter version superseded by the full analyzer.

### 3. `theory_analyzer.py` (MISC CODE)
**Type:** Comprehensive music theory analyzer
**Purpose:** Deep analysis of scales, modes, arpeggios, triads, harmonic patterns
**Features:**
- Scale detection (major, minor, pentatonic, blues, modes, exotic scales)
- Mode detection with emotional characteristics
- Triad and seventh chord detection
- Arpeggio pattern detection
- Interval analysis
- Melodic contour analysis
- Harmonic complexity scoring
- Audio file analysis via librosa
**Status:** ✅ **UNIQUE - SHOULD MIGRATE**
- **Similar to:** No direct equivalent found in KmiDi
- **Difference:** This is a comprehensive standalone theory analyzer
- **Recommendation:** **MIGRATE** - This appears to be a sophisticated theory analysis module that doesn't exist in KmiDi. Should go in `music_brain/theory/` or similar.

### 4. `harmony_system.py` (Desktop/kelly-midi-companion)
**Type:** Intelligent Harmony System - unified integration module
**Purpose:** Integrates chord detector, key analyzer, harmony engine, chord memory
**Features:**
- Chord detection (rule-based + ML hybrid)
- Key + scale analysis
- Probabilistic harmony engine
- Chord memory + long-term context
- Roman numeral analysis
- Modal interchange detection
- Secondary dominant detection
- Voice leading analysis
- Tension/resolution analysis
- Emotional arc tracking
**Status:** ⚠️ **NEEDS VERIFICATION**
- **Similar to:** `KmiDi_FINAL/python/music_brain/harmony.py` and other harmony modules
- **Difference:** This appears to be a unified integration layer that combines multiple modules
- **Dependencies:** Imports from `.chord_detector`, `.key_analyzer`, `.harmony_engine`, `.chord_memory` (relative imports)
- **Recommendation:** **REVIEW DEPENDENCIES** - Check if the dependency modules exist. If they do, this could be a valuable integration layer. If not, it may be incomplete.

### 5. Other MISC CODE files (frequency.py, effects.py, etc.)
**Status:** ⚠️ **REVIEW NEEDED**
- `frequency.py` - FFT, pitch detection, harmonic analysis utilities
- `effects.py` - Individual effect implementations (distortion, delay, reverb, filters, etc.)
- Other files: `modulator.py`, `auto_tune.py`, `synthesizer.py`, `neural_voice.py`, `guitar_fx.py`

**Recommendation:** These appear to be utility modules. Need to check if similar functionality exists in KmiDi's `music_brain/effects/` or other locations.

## Recommendations

### High Priority - Should Migrate:
1. **`theory_analyzer.py`** - Comprehensive theory analysis not found elsewhere
   - **Location:** `music_brain/theory/theory_analyzer.py` or `music_brain/analysis/theory_analyzer.py`

### Medium Priority - Needs Investigation:
2. **`harmony_system.py`** - Check if dependencies exist, then migrate if complete
   - **Location:** `music_brain/harmony/harmony_system.py` or `music_brain/harmony/integrated.py`
3. **`audio_tools.py`** - Compare with existing MCP tools, migrate if newer/better
   - **Location:** `daiw_mcp/tools/audio_tools.py` (if different from existing)

### Low Priority - Review Only:
4. **`audio_analyzer_starter.py`** - Obsolete starter version, do not migrate
5. **Other utility files** - Review individually to see if they add value

## Next Steps

1. ✅ Migrate `theory_analyzer.py` - High value, no duplicates
2. ⚠️ Investigate `harmony_system.py` dependencies
3. ⚠️ Compare `audio_tools.py` with existing MCP tools
4. ⚠️ Review other utility files (frequency.py, effects.py, etc.) for unique functionality
