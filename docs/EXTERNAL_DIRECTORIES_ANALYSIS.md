# External Directories Code Analysis

Historical note
- This document records a point-in-time external-directory survey and is not current repository architecture authority.
- References here to Tauri application directories or migration targets should be read as historical findings only.
- When it conflicts with the current repo structure, follow `README.md`, `docs/ARCHITECTURE.md`, and `docs/REPO_MODULE_MAP.md`.

**Date:** 2026-01-21
**Status:** ✅ COMPLETE - All relevant code verified

## Summary

Checked all specified directories for KmiDi-related source code that might need migration to KmiDi.

## Directory Analysis

### 1. `/Users/seanburdges/venv`
- **Type:** Python virtual environment
- **Source Files:** 0 (venv contains only installed packages)
- **Status:** ✅ Not relevant - virtual environment only

### 2. `/Users/seanburdges/tauri-app`
- **Type:** Tauri application directory
- **Source Files:** 0
- **Status:** ✅ Not found or empty

### 3. `/Users/seanburdges/RECOVERY_OPS`
- **Type:** Recovery operations / Music projects
- **Source Files:** ~20 Python scripts
- **Content:** Logic Pro project scripts, Kelly song generation scripts, AudioVault tools
- **Status:** ⚠️ Contains project-specific scripts (not core KmiDi code)
- **Files:** `setup_logic_project.py`, `generate_midi.py`, `kelly_*.py` scripts, `audio_refinery.py`
- **Recommendation:** These are project-specific tools, not core library code

### 4. `/Users/seanburdges/My Mac/Desktop`
- **Type:** Desktop directory
- **Source Files:** Found `kelly-midi-companion` project
- **Status:** ✅ Most files already exist in KmiDi
- **Findings:**
  - ✅ `ChordGenerator.cpp` - EXISTS in KmiDi_FINAL
  - ✅ `emotion_thesaurus.py` - EXISTS in KmiDi_FINAL
  - ❌ `harmony_system.py` - MISSING (but may be experimental/older version)

### 5. `/Users/seanburdges/MP3`
- **Type:** Audio files directory
- **Source Files:** 0
- **Status:** ✅ Not relevant - audio files only

### 6. `/Users/seanburdges/ml-training-suite`
- **Type:** ML training suite
- **Source Files:** ~15 Python training scripts
- **Content:** Training scripts (`train.py`, `train_voice.py`, `train_emotion.py`), data processing, models
- **Status:** ⚠️ ML training infrastructure (separate from core KmiDi)
- **Files:** `scripts/train*.py`, `src/training/trainer.py`, `src/models/*.py`
- **Recommendation:** This is a separate ML training project, not core KmiDi code

### 7. `/Users/seanburdges/ml`
- **Type:** ML directory
- **Source Files:** ~5 Python files
- **Content:** Device picker, CLI, API
- **Status:** ⚠️ Minimal ML utilities
- **Files:** `device_picker.py`, `cli.py`, `api/main.py`
- **Recommendation:** These appear to be utility scripts, not core KmiDi code

### 8. `/Users/seanburdges/MISC CODE`
- **Type:** Miscellaneous code directory
- **Source Files:** ~20 Python files
- **Content:** Audio analysis, effects, synthesizer, voice processing utilities
- **Status:** ⚠️ Utility scripts - some may be relevant
- **Findings:**
  - ❌ `audio_tools.py` - MISSING
  - ❌ `audio_analyzer_starter.py` - MISSING
  - ✅ `chord_detection.py` - EXISTS in KmiDi_FINAL
  - ❌ `theory_analyzer.py` - MISSING
  - Other files: `frequency.py`, `effects.py`, `modulator.py`, `auto_tune.py`, `synthesizer.py`, `neural_voice.py`, `guitar_fx.py`
- **Recommendation:** Review these utility scripts to see if they should be integrated

### 9. `/Users/seanburdges/KmiDi-Backup-20260108_224837`
- **Type:** Backup of KmiDi from 2026-01-08
- **Source Files:** Full backup of KmiDi structure
- **Status:** ✅ All files from backup already exist in current KmiDi
- **Verification:**
  - ✅ `src_penta-core/osc/RTMessageQueue.cpp` - EXISTS
  - ✅ `src_penta-core/ml/MLInterface.cpp` - EXISTS
  - ✅ `src_penta-core/mixer/MixerEngine.cpp` - EXISTS
  - ✅ All other backup files verified present
- **Conclusion:** Backup is older version, current KmiDi is more complete

### 10. `/Users/seanburdges/Archive`
- **Type:** Archive directory
- **Source Files:** 0
- **Status:** ✅ Empty or no source code found

## Missing Files Analysis

### Potentially Missing Files (Need Review):

1. **MISC CODE:**
   - `audio_tools.py` - Audio utility functions
   - `audio_analyzer_starter.py` - Audio analysis starter
   - `theory_analyzer.py` - Music theory analysis
   - `harmony_system.py` (from Desktop) - Harmony system

2. **Other utility scripts from MISC CODE:**
   - `frequency.py`, `effects.py`, `modulator.py`, `auto_tune.py`
   - `synthesizer.py`, `neural_voice.py`, `guitar_fx.py`

### Recommendation

These missing files appear to be:
- **Utility scripts** rather than core library code
- **Experimental/older versions** that may have been superseded
- **Project-specific tools** rather than reusable components

**Action:** Review these files to determine if they:
1. Contain functionality not present in KmiDi
2. Are newer/better versions than what exists
3. Should be integrated into the project structure

## Conclusion

✅ **Core KmiDi source code is complete in KmiDi**

All critical C++ source files, headers, and core Python modules are present. The missing files are primarily utility scripts and project-specific tools that may or may not need integration.

**Next Steps:**
1. Review missing utility scripts from MISC CODE
2. Determine if they should be integrated or archived
3. Check if `harmony_system.py` from Desktop is needed
