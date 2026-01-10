# Missing Audio Datasets Report

**Date:** 2025-01-09
**System Scan:** Complete computer search for audio datasets

---

## 📊 Summary

**Total Locations Checked:** 8+
**Datasets Found:** 2 core datasets
**Missing Datasets:** 6 key training datasets

---

## ✅ Found Datasets

### 1. Groove MIDI Dataset
- **Location:** `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/grooves/groove_midi`
- **Files:** 1,189 MIDI files
- **Size:** 0.01 GB
- **Status:** ✅ Complete

### 2. Lakh MIDI Dataset
- **Location:** `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/chord_progressions/lakh`
- **Files:** 10,180 files
- **Size:** 0.40 GB
- **Status:** ✅ Complete
- **Note:** Also found in `/Users/seanburdges/audio/datasets/lakh_midi/`

### 3. Additional Datasets in `/Users/seanburdges/audio/datasets/`
- **Total Files:** 241,380 files
- **Total Size:** 27.85 GB
- **Contains:**
  - Lakh MIDI (full dataset)
  - Genius Lyrics dataset
  - M4Singer dataset
  - MoodyLyrics dataset
  - Wasabi dataset
  - Training splits (emotion, melody, groove, harmony, dynamics)
  - Model checkpoints

---

## ❌ Missing Critical Datasets

### High Priority (Required for Training)

#### 1. MAESTRO v3.0
- **Purpose:** Piano MIDI with dynamics and timing (200+ hours)
- **Required for:** Melody transformer training
- **Expected Location:** `raw/melodies/maestro/` or `raw/melodies/maestro-v3.0.0/`
- **Status:** ❌ NOT FOUND
- **Download URL:** `https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip`
- **Size:** ~200MB compressed, ~500MB uncompressed

#### 2. RAVDESS (Raw)
- **Purpose:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **Required for:** Emotion recognition training
- **Expected Location:** `raw/emotions/ravdess/`
- **Status:** ❌ NOT FOUND (processed version may exist)
- **Download:** Kaggle dataset `uwrfkaggler/ravdess-emotional-speech-audio`
- **Note:** Processed RAVDESS files were found in `processed/emotions/ravdess/` but raw source files are missing

#### 3. CREMA-D
- **Purpose:** Crowd-sourced Emotional Multimodal Actors Dataset
- **Required for:** Emotion recognition training (expands dataset)
- **Expected Location:** `raw/emotions/cremad/`
- **Status:** ❌ NOT FOUND
- **Download URL:** `https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip`

#### 4. TESS
- **Purpose:** Toronto Emotional Speech Set
- **Required for:** Emotion recognition training
- **Expected Location:** `raw/emotions/tess/`
- **Status:** ❌ NOT FOUND
- **Download:** Kaggle dataset `ejlok1/toronto-emotional-speech-set-tess`

### Medium Priority

#### 5. GTZAN
- **Purpose:** Music genre classification (10 genres)
- **Expected Location:** `raw/emotions/gtzan/`
- **Status:** ❌ NOT FOUND
- **Download URL:** `https://mirg.city.ac.uk/datasets/gtzan/genres.tar.gz`

### Low Priority (Very Large)

#### 6. MusicNet
- **Purpose:** Classical music with note annotations
- **Expected Location:** `raw/melodies/musicnet/`
- **Status:** ❌ NOT FOUND
- **Size:** 168GB (may be intentionally excluded)
- **Download URL:** `https://zenodo.org/record/5120004/files/musicnet.tar.gz`

---

## 📁 Locations Checked

| Location | Exists | Datasets Found |
|----------|--------|----------------|
| `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data` | ✅ | Groove MIDI, Lakh MIDI |
| `/Volumes/Extreme SSD/kelly-audio-data` | ❌ | Not mounted |
| `/Volumes/sbdrive/kmidi_audio_data` | ❌ | Not mounted |
| `/Volumes/Extreme SSD/kmidi_audio_data` | ❌ | Not mounted |
| `/Users/seanburdges/audio` | ✅ | 27.85 GB (241,380 files) - Various datasets |
| `/Users/seanburdges/Music/AudioVault` | ✅ | Production samples (Demo Kit) |
| `/Users/seanburdges/.kelly/audio-data` | ⚠️ | Fallback location (empty) |
| `/Users/seanburdges/BASIC STRUCTURE FOR miDiKompanion` | ✅ | Empty |

---

## 🎯 Recommendations

### Immediate Actions (High Priority)

1. **Download MAESTRO v3.0** (Required for melody training)
   ```bash
   # Option 1: Use prepare_datasets.py
   python scripts/prepare_datasets.py --dataset maestro --download

   # Option 2: Manual download
   wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
   unzip maestro-v3.0.0-midi.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/melodies/
   ```

2. **Download RAVDESS Raw** (Required for emotion training)
   ```bash
   # Requires Kaggle API setup
   kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
   unzip ravdess-emotional-speech-audio.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/ravdess/
   ```

3. **Download CREMA-D** (Recommended)
   ```bash
   wget https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip
   unzip master.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/
   ```

4. **Download TESS** (Recommended)
   ```bash
   kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
   unzip toronto-emotional-speech-set-tess.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/tess/
   ```

### Optional Actions (Medium Priority)

5. **Download GTZAN** (For genre classification)
   ```bash
   wget https://mirg.city.ac.uk/datasets/gtzan/genres.tar.gz
   tar -xzf genres.tar.gz -C /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/
   ```

### Large Datasets (Low Priority - Only if Needed)

6. **MusicNet** - 168GB - Only download if classical music training is specifically needed

---

## 📝 Alternative Locations

### Processed Data Available

The following processed datasets were found but may need raw source verification:

- **Processed RAVDESS:** Found in `processed/emotions/ravdess/`
  - Status: WAV files present
  - Issue: Raw source files not found
  - Impact: Can use processed files but cannot reprocess

### Additional Resources Found

- **Training Splits:** Found in `/Users/seanburdges/audio/datasets/training_splits/`
  - Contains: emotion, melody, groove, harmony, dynamics splits
- **Model Checkpoints:** Found in `/Users/seanburdges/audio/datasets/checkpoints/`
- **Other Datasets:** Genius Lyrics, M4Singer, MoodyLyrics, Wasabi

---

## 🔍 Verification Steps

1. **Check if datasets are on external SSD** (if mounted)
   - External SSDs may not be currently mounted
   - Check `/Volumes/` when SSDs are connected

2. **Verify processed datasets match raw requirements**
   - Check if processed RAVDESS is sufficient or if raw is needed

3. **Check environment variables**
   ```bash
   echo $KELLY_AUDIO_DATA_ROOT
   echo $KELLY_SSD_PATH
   echo $AUDIO_DATA_ROOT
   ```

4. **Run dataset preparation script**
   ```bash
   python scripts/prepare_datasets.py --list
   python scripts/prepare_datasets.py --dataset all --download
   ```

---

## 📊 Priority Matrix

| Dataset | Priority | Status | Impact if Missing |
|---------|----------|--------|-------------------|
| MAESTRO | 🔴 High | ❌ Missing | Melody transformer cannot train |
| RAVDESS | 🔴 High | ⚠️ Partial | Emotion recognizer may have limited data |
| CREMA-D | 🟡 Medium | ❌ Missing | Emotion dataset expansion |
| TESS | 🟡 Medium | ❌ Missing | Emotion dataset expansion |
| GTZAN | 🟡 Medium | ❌ Missing | Genre classification unavailable |
| MusicNet | 🟢 Low | ❌ Missing | Classical music training unavailable |

---

## Summary

**Current Status:** ⚠️ **PARTIALLY COMPLETE**

**Critical Missing:**
- MAESTRO (required for melody training)
- RAVDESS raw files (required for emotion training)

**Next Steps:**
1. Download MAESTRO v3.0 immediately
2. Download RAVDESS raw files (or verify processed version is sufficient)
3. Download CREMA-D and TESS for expanded emotion training
4. Verify external SSD mounts if datasets should be there
