# Audio Dataset Download Progress

**Date:** 2025-01-09
**Status:** In Progress

---

## ✅ Successfully Downloaded

### 1. MAESTRO v3.0 ✅
- **Status:** Downloaded and extracted
- **Location:** `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/melodies/maestro-v3.0.0/`
- **Size:** ~55.7 MB compressed
- **Purpose:** Piano MIDI with dynamics and timing (200+ hours) - Required for melody transformer training
- **Impact:** ✅ Melody training can now proceed

### 2. CREMA-D ✅
- **Status:** Downloaded and extracted
- **Location:** `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/cremad/`
- **Size:** ~22.3 MB compressed
- **Purpose:** Crowd-sourced Emotional Multimodal Actors Dataset
- **Impact:** ✅ Expands emotion recognition training data

---

## ⚠️ Failed Downloads

### 3. GTZAN ❌
- **Status:** Download failed (404 Error)
- **URL:** `https://mirg.city.ac.uk/datasets/gtzan/genres.tar.gz`
- **Issue:** URL appears to be incorrect or dataset moved
- **Action Required:** Find alternative GTZAN download source or skip if not critical
- **Alternative Sources:**
  - Kaggle: `https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification`
  - Alternative mirror sites

---

## 📋 Remaining Tasks

### High Priority (Required)

#### RAVDESS Raw Files
- **Status:** ⏳ Pending (Requires Kaggle API)
- **Priority:** 🔴 High
- **Required for:** Emotion recognition training
- **Action:**
  ```bash
  # Setup Kaggle API first
  pip install kaggle
  # Get API token from https://www.kaggle.com/settings
  # Save to ~/.kaggle/kaggle.json

  # Then download
  kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
  unzip ravdess-emotional-speech-audio.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/ravdess/
  ```

#### TESS Dataset
- **Status:** ⏳ Pending (Requires Kaggle API)
- **Priority:** 🟡 Medium
- **Required for:** Emotion recognition training expansion
- **Action:**
  ```bash
  kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
  unzip toronto-emotional-speech-set-tess.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/tess/
  ```

### Medium Priority (Optional)

#### GTZAN (Alternative Source)
- **Status:** ⏳ Pending (Need alternative URL)
- **Priority:** 🟡 Medium
- **Purpose:** Genre classification
- **Action:** Find and download from alternative source (Kaggle, etc.)

---

## 📊 Progress Summary

| Dataset | Status | Priority | Impact |
|---------|--------|----------|--------|
| MAESTRO v3.0 | ✅ Complete | 🔴 High | Melody training enabled |
| CREMA-D | ✅ Complete | 🟡 Medium | Emotion data expanded |
| RAVDESS | ⏳ Pending | 🔴 High | Emotion training incomplete |
| TESS | ⏳ Pending | 🟡 Medium | Emotion data expansion |
| GTZAN | ❌ Failed | 🟡 Medium | Genre classification unavailable |

**Completion:** 2/5 datasets (40%)
**Critical Completion:** 1/2 high-priority datasets (50%)

---

## 🎯 Next Steps

1. **Setup Kaggle API** for RAVDESS and TESS downloads
   - Install: `pip install kaggle`
   - Get API token: https://www.kaggle.com/settings
   - Configure: Save to `~/.kaggle/kaggle.json`

2. **Download RAVDESS** (Critical)
   - Required for emotion recognition training
   - Use Kaggle API once configured

3. **Download TESS** (Recommended)
   - Expands emotion training dataset
   - Use Kaggle API once configured

4. **Fix GTZAN download** (Optional)
   - Find alternative source or skip if not needed

5. **Verify all datasets**
   - Run dataset verification script
   - Check file counts and integrity
   - Update documentation

---

## 📝 Notes

- MAESTRO download was successful and critical for melody training
- CREMA-D adds valuable emotion training data
- GTZAN URL appears to be outdated - may need alternative source
- Processed RAVDESS files exist but raw source needed for retraining
- All downloads are being stored in RECOVERY_OPS/AUDIO_MIDI_DATA structure

---

## 🔗 Related Files

- `MISSING_AUDIO_DATASETS_REPORT.md` - Initial audit report
- `AUDIO_MIDI_DATA_REVIEW.md` - Directory structure review
- `download_missing_datasets.py` - Download script
