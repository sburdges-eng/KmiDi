# Duplicate Code Analysis Report

**Date:** 2026-01-23  
**Phase:** 2.1.1 - Code Organization

## Summary

This document identifies duplicate code patterns across the codebase and provides recommendations for consolidation.

## Identified Duplicates

### 1. Emotion Thesaurus Modules

**Locations:**
- `music_brain/kelly/core/emotion_thesaurus.py` (447 lines)
- `music_brain/kelly_companion/core/emotion_thesaurus.py` (758 lines)
- `music_brain/emotion/emotion_thesaurus.py` (exists)
- `music_brain/emotion_kmidi/emotion_thesaurus.py` (identical to emotion/)

**Status:**
- `kelly_companion` version is the most complete (758 lines, includes JSON loading, word indexing)
- `kelly` version is simpler (447 lines, basic implementation)
- `emotion` and `emotion_kmidi` appear identical

**Recommendation:**
- **Active:** Use `music_brain/kelly_companion/core/emotion_thesaurus.py`
- **Deprecate:** `music_brain/kelly/core/emotion_thesaurus.py` (mark as deprecated)
- **Consolidate:** `music_brain/emotion/` and `music_brain/emotion_kmidi/` are identical - keep one, remove other

### 2. Groove Engine Modules

**Locations:**
- `music_brain/groove_kmidi/groove_engine.py` (14 files total in directory)
- `music_brain/kelly_companion/groove/groove_engine.py` (4 files total in directory)

**Status:**
- `groove_kmidi` has more features (14 files: humanizers, drum analysis, fan feedback, etc.)
- `kelly_companion/groove` has core functionality (4 files: engine, extractor, applicator, templates)
- Files differ (confirmed via diff)

**Recommendation:**
- **Active:** Use `music_brain/kelly_companion/groove/` for core functionality
- **Evaluate:** Review `groove_kmidi` features - migrate useful ones to `kelly_companion/groove` if needed
- **Deprecate:** Mark `groove_kmidi` as deprecated after migration

### 3. Kelly vs Kelly Companion

**Locations:**
- `music_brain/kelly/` (6 files: CLI, core modules)
- `music_brain/kelly_companion/` (89 files: comprehensive implementation)

**Status:**
- `kelly_companion` is the active, comprehensive implementation
- `kelly` appears to be an older/simpler version
- Current imports use `kelly_companion`

**Recommendation:**
- **Active:** Use `music_brain/kelly_companion/`
- **Deprecate:** Mark `music_brain/kelly/` as deprecated
- **Migration:** Move any unique functionality from `kelly/` to `kelly_companion/` if needed

## Action Items

### Immediate Actions
1. ✅ Document duplicates (this file)
2. ⏳ Create deprecation warnings in old modules
3. ⏳ Create shared utilities module structure
4. ⏳ Update imports to use canonical modules

### Consolidation Plan
1. **Emotion modules:** Keep `kelly_companion/core/emotion_thesaurus.py`, deprecate others
2. **Groove modules:** Keep `kelly_companion/groove/`, evaluate `groove_kmidi` features
3. **Kelly modules:** Keep `kelly_companion/`, deprecate `kelly/`

## Files Requiring Import Updates

Based on grep analysis, these files import from deprecated modules:
- `music_brain/emotion/emotion_production.py` - imports from `music_brain.emotion.emotion_thesaurus`

## Next Steps

1. Add deprecation warnings to old modules
2. Create migration guide
3. Update imports gradually
4. Remove deprecated modules after migration period
