# AUDIO_MIDI_DATA Directory Review

**Date:** 2025-01-09
**Location:** `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA`
**Total Size:** 52GB

---

## 📊 Current Status

### Directory Structure

The `AUDIO_MIDI_DATA` directory contains:
- `kelly-audio-data/` - Main data directory (52GB)
- `SSD_Transfer/` - Transfer directory

### Data Inventory

| Category | Expected Location | Files Found | Status | Notes |
|----------|-------------------|-------------|--------|-------|
| **Grooves** | `raw/grooves/groove_midi/` | 1,150 MIDI files | ✅ **COMPLETE** | Groove MIDI dataset present |
| **Chord Progressions** | `raw/chord_progressions/` | 10,180 files | ✅ **COMPLETE** | Lakh MIDI dataset likely present |
| **Melodies** | `raw/melodies/` | 0 files | ❌ **MISSING** | MAESTRO dataset not present |
| **Emotions** | `raw/emotions/` | 0 files | ❌ **MISSING** | RAVDESS, CREMA-D, TESS datasets not present |
| **Processed Emotions** | `processed/emotions/ravdess/` | WAV files found | ⚠️ **PARTIAL** | Processed RAVDESS exists but raw not found |
| **Training Logs** | `logs/training/` | Multiple runs | ✅ **PRESENT** | Training logs for various models |

---

## ✅ What's Available

### 1. Groove MIDI Dataset
- **Location:** `kelly-audio-data/raw/grooves/groove_midi/groove/`
- **Files:** 1,150 MIDI files
- **Format:** Expressive drum performances
- **Examples Found:**
  - `drummer8/session2/12_funk_81_beat_4-4.mid`
  - `drummer8/session2/25_latin_84_beat_4-4.mid`
  - `drummer8/session2/2_funk_92_beat_4-4.mid`
  - Various genres: funk, latin, rock, afrobeat, dance-disco, etc.

### 2. Chord Progressions (Lakh MIDI)
- **Location:** `kelly-audio-data/raw/chord_progressions/`
- **Files:** 10,180 files
- **Status:** Likely contains Lakh MIDI dataset for harmony/chord progression training

### 3. Processed Emotion Data
- **Location:** `kelly-audio-data/processed/emotions/ravdess/processed/`
- **Files:** WAV files found in emotion categories (happy, etc.)
- **Note:** Processed files exist but raw source files may be missing

### 4. Training Logs
- **Location:** `kelly-audio-data/logs/training/`
- **Models Trained:**
  - `emotionrecognizer` - Multiple training runs (Dec 26, 2025)
  - `melodytransformer` - Training runs (Dec 26, 2025)
  - `groovepredictor` - Multiple training runs (Dec 26, 2025)
  - `emotionnodeclassifier` - Training runs (Dec 26, 2025)

---

## ❌ Missing Datasets

### 1. Melody Datasets (High Priority)

#### MAESTRO v3.0
- **Expected Location:** `kelly-audio-data/raw/melodies/maestro/`
- **Purpose:** Piano MIDI with dynamics and timing (200+ hours)
- **Download:** `https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip`
- **Status:** ❌ Not found in `raw/melodies/`
- **Impact:** Melody generation training may be incomplete

#### MusicNet
- **Expected Location:** `kelly-audio-data/raw/melodies/musicnet/`
- **Purpose:** Classical music with note annotations (168GB)
- **Status:** ❌ Not found
- **Note:** Very large dataset, may be intentionally excluded

### 2. Emotion Datasets (High Priority)

#### RAVDESS (Raw)
- **Expected Location:** `kelly-audio-data/raw/emotions/ravdess/`
- **Purpose:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **Status:** ⚠️ **PARTIAL** - Processed files exist, but raw source files not found
- **Impact:** Can use processed files, but cannot reprocess raw data

#### CREMA-D
- **Expected Location:** `kelly-audio-data/raw/emotions/cremad/`
- **Purpose:** Crowd-sourced Emotional Multimodal Actors Dataset
- **Status:** ❌ Not found
- **Download:** `https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip`

#### TESS
- **Expected Location:** `kelly-audio-data/raw/emotions/tess/`
- **Purpose:** Toronto Emotional Speech Set
- **Status:** ❌ Not found
- **Download:** Kaggle dataset `ejlok1/toronto-emotional-speech-set-tess`

#### GTZAN
- **Expected Location:** `kelly-audio-data/raw/emotions/gtzan/`
- **Purpose:** Music genre classification (10 genres)
- **Status:** ❌ Not found

### 3. Optional Large Datasets

These datasets are very large and may be intentionally excluded:

- **FMA (Free Music Archive)** - 7.2GB (small) to 900GB (full)
- **MTG-Jamendo** - ~1TB
- **NSynth Full** - ~30GB
- **MUSDB18** - ~10GB compressed

---

## 🔍 Codebase Expectations

### Expected Datasets (from `scripts/prepare_datasets.py`)

Based on the codebase, these datasets are expected:

1. **Emotion Recognition:**
   - RAVDESS (required for emotion recognizer training)
   - CREMA-D (optional)
   - TESS (optional)
   - GTZAN (genre classification)

2. **Melody Generation:**
   - MAESTRO v3 (high priority for melody transformer)
   - MusicNet (optional, very large)

3. **Groove/Rhythm:**
   - Groove MIDI ✅ (present)

4. **Harmony/Chords:**
   - Lakh MIDI ✅ (likely present in chord_progressions)

5. **Processed Data:**
   - Processed RAVDESS ✅ (present)
   - Mel spectrograms (not checked)
   - Embeddings (not checked)

---

## 🎯 Recommendations

### High Priority (Required for Training)

1. **Download MAESTRO v3.0 MIDI:**
   ```bash
   # Use prepare_datasets.py
   python scripts/prepare_datasets.py --dataset maestro --download
   ```
   - Required for melody transformer training
   - Estimated size: ~200MB compressed

2. **Verify/Download RAVDESS Raw:**
   ```bash
   python scripts/prepare_datasets.py --dataset emotion_ravdess --download
   ```
   - Required for emotion recognizer training
   - May already exist but not in expected location

3. **Download CREMA-D (Optional but Recommended):**
   ```bash
   python scripts/prepare_datasets.py --dataset emotion_cremad --download
   ```
   - Expands emotion recognition training data

### Medium Priority

4. **Download TESS:**
   - Additional emotion recognition dataset
   - Requires Kaggle API setup

5. **Download GTZAN:**
   - Genre classification dataset
   - Useful for genre-based generation

### Low Priority

6. **Large Datasets (Only if needed):**
   - MusicNet (168GB) - Only if classical music training needed
   - FMA/MTG-Jamendo - Only for massive dataset experiments

---

## 📝 Action Items

### Immediate Actions

- [ ] Verify MAESTRO dataset location or download it
- [ ] Check if RAVDESS raw files exist elsewhere in the system
- [ ] Download missing emotion datasets (CREMA-D, TESS)
- [ ] Verify Lakh MIDI dataset is complete (10,180 files)

### Verification Steps

- [ ] Run `python scripts/prepare_datasets.py --list` to see configured datasets
- [ ] Check if datasets are in alternative locations:
  - `/Volumes/Extreme SSD/kmidi_audio_data/`
  - `/Volumes/sbdrive/kmidi_audio_data/`
  - `~/Music/`
- [ ] Verify processed data matches raw data requirements

### Documentation

- [ ] Update dataset inventory when missing datasets are found/downloaded
- [ ] Document alternative dataset locations if found
- [ ] Note any intentionally excluded large datasets

---

## 🔗 Related Files

- **Storage Config:** `configs/storage.py` - Defines expected data structure
- **Dataset Script:** `scripts/prepare_datasets.py` - Downloads and prepares datasets
- **Missing Data Report:** `MISSING_DATA_REPORT.md` - Previous analysis

---

## Summary

**Overall Status:** ⚠️ **PARTIALLY COMPLETE**

**Strengths:**
- ✅ Groove MIDI dataset complete (1,150 files)
- ✅ Chord progression dataset present (10,180 files)
- ✅ Processed emotion data available
- ✅ Training logs preserved

**Gaps:**
- ❌ Melody datasets missing (MAESTRO not found)
- ❌ Raw emotion datasets mostly missing (RAVDESS raw not found)
- ⚠️ Processed data exists but raw source verification needed

**Priority:** Download MAESTRO and verify RAVDESS raw files for complete training pipeline support.
