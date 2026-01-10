# Emotion Thesaurus File Organization

**Date**: 2025-01-08  
**Purpose**: Document the multiple `emotion_thesaurus.py` files and their purposes

---

## Overview

There are multiple `emotion_thesaurus.py` files in the repository. This document clarifies which ones are active and which are legacy.

---

## Active Files

### 1. ✅ `music_brain/emotion/emotion_thesaurus.py` (MAIN)
**Status**: ✅ **ACTIVE - PRIMARY IMPLEMENTATION**

**Purpose**: Main emotion thesaurus implementation for music_brain package

**Features**:
- 6×6×6 emotion taxonomy
- Intensity tiers support
- Emotional blends support
- Synonym matching
- Loads JSON data from `data/emotion_thesaurus/` directory

**Used By**:
- `music_brain/cultural/cross_cultural_music.py`
- `music_brain/production/dynamics_engine.py`
- `music_brain/kelly.py`
- `music_brain/emotion/text_emotion_parser.py`
- `music_brain/emotion/emotion_production.py`
- `music_brain/emotion/__init__.py` (exported)

**Import Pattern**:
```python
from music_brain.emotion.emotion_thesaurus import EmotionThesaurus, EmotionMatch, BlendMatch
```

---

### 2. ✅ `music_brain/emotion_thesaurus.py` (WRAPPER)
**Status**: ✅ **ACTIVE - BACKWARD COMPATIBILITY WRAPPER**

**Purpose**: Provides backward compatibility for old import paths

**Content**: Single line wrapper that re-exports from `emotion/emotion_thesaurus.py`

```python
from .emotion.emotion_thesaurus import EmotionMatch, EmotionThesaurus, BlendMatch
```

**Used By**: Old code that imports from `music_brain.emotion_thesaurus` directly

**Import Pattern**:
```python
from music_brain.emotion_thesaurus import EmotionThesaurus  # Still works
```

**Recommendation**: Keep this file for backward compatibility. New code should use `music_brain.emotion.emotion_thesaurus`.

---

### 3. ✅ `src/kelly/core/emotion_thesaurus.py` (ALTERNATIVE IMPLEMENTATION)
**Status**: ✅ **ACTIVE - KELLY-SPECIFIC IMPLEMENTATION**

**Purpose**: Different emotion thesaurus implementation for Kelly project

**Features**:
- VAD (Valence-Arousal-Dominance) dimensions
- Plutchik's emotion wheel (8 categories)
- 216-node emotion thesaurus
- Different data structure and API

**Used By**:
- `src/kelly/core/intent_processor.py`
- `src/kelly/core/__init__.py`
- `src/kelly/cli.py`
- `src/kelly/__init__.py`

**Import Pattern**:
```python
from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionNode, EmotionCategory
```

**Note**: This is a **different implementation** from the music_brain one. It uses different data structures and is specifically for the Kelly project.

**Recommendation**: Keep this file - it's actively used by Kelly codebase and has a different purpose.

---

## Legacy/Duplicate Files

### 4. ⚠️ `kelly/core/emotion_thesaurus.py`
**Status**: ⚠️ **DUPLICATE - CHECK IF NEEDED**

**Location**: Root-level `kelly/` directory

**Status**: Appears to be a duplicate of `src/kelly/core/emotion_thesaurus.py`

**Used By**: Same files as `src/kelly/core/emotion_thesaurus.py` (may be shadowed)

**Recommendation**: 
- Check if this is a symlink or duplicate
- If duplicate, consider removing
- If symlink, document it

**Action Needed**: Verify if this is actually used or if imports resolve to `src/kelly/` version.

---

### 5. ❌ `data/emotion_thesaurus/emotion_thesaurus.py`
**Status**: ❓ **UNKNOWN - VERIFY PURPOSE**

**Location**: `data/emotion_thesaurus/` directory

**Note**: This directory should contain **JSON data files**, not Python code.

**Recommendation**: 
- Verify if this is actually a Python file or misnamed data file
- If it's a Python file, it should probably be moved to `music_brain/emotion/` or removed
- If it's data, it should be renamed to `.json`

**Action Needed**: Inspect file contents and determine if it's code or data.

---

## Summary

| File | Status | Purpose | Action |
|------|--------|---------|--------|
| `music_brain/emotion/emotion_thesaurus.py` | ✅ Active | Main implementation | Keep |
| `music_brain/emotion_thesaurus.py` | ✅ Active | Backward compat wrapper | Keep |
| `src/kelly/core/emotion_thesaurus.py` | ✅ Active | Kelly-specific impl | Keep |
| `kelly/core/emotion_thesaurus.py` | ⚠️ Duplicate? | Unknown | Verify & remove if duplicate |
| `data/emotion_thesaurus/emotion_thesaurus.py` | ❓ Unknown | Unknown | Verify purpose |

---

## Recommended Import Patterns

### For music_brain code:
```python
# Preferred (explicit path)
from music_brain.emotion.emotion_thesaurus import EmotionThesaurus, EmotionMatch, BlendMatch

# Also works (backward compat)
from music_brain.emotion_thesaurus import EmotionThesaurus
```

### For Kelly code:
```python
from kelly.core.emotion_thesaurus import EmotionThesaurus, EmotionNode, EmotionCategory
```

### Should NOT use:
```python
# Don't import directly from data directory
from data.emotion_thesaurus.emotion_thesaurus import ...  # Wrong!
```

---

## Migration Notes

If you need to consolidate:

1. **Don't remove** `music_brain/emotion/emotion_thesaurus.py` - it's the main implementation
2. **Keep** `music_brain/emotion_thesaurus.py` wrapper for backward compatibility
3. **Keep** `src/kelly/core/emotion_thesaurus.py` - it's a different implementation for Kelly
4. **Verify and potentially remove** `kelly/core/emotion_thesaurus.py` if duplicate
5. **Verify** `data/emotion_thesaurus/emotion_thesaurus.py` - should probably not be a Python file in data directory

---

**Last Updated**: 2025-01-08  
**Maintained By**: Development Team
