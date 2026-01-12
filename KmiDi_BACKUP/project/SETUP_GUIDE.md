# Dataset Setup Guide

**Date:** 2025-01-09
**Status:** Setup in Progress

---

## ✅ Completed Setup Steps

### 1. Kaggle Package Installed ✅
- **Status:** ✓ Installed successfully
- **Package:** `kaggle`
- **Location:** Python packages

### 2. Datasets Downloaded ✅

#### MAESTRO v3.0 ✅
- **Location:** `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/melodies/maestro-v3.0.0/`
- **Status:** Complete
- **Purpose:** Melody transformer training
- **Impact:** ✅ Melody training enabled

#### CREMA-D ✅
- **Location:** `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/cremad/`
- **Status:** Complete
- **Purpose:** Emotion recognition training
- **Impact:** ✅ Emotion data expanded

---

## ⏳ Pending Setup Steps

### 3. Kaggle API Credentials Setup ⏳

**Required for:** RAVDESS and TESS downloads

#### Step-by-Step Instructions:

1. **Get Kaggle API Token**
   - Go to: https://www.kaggle.com/settings
   - Scroll to "API" section
   - Click "Create New Token" button
   - This will download a `kaggle.json` file

2. **Install Credentials**
   ```bash
   # Create Kaggle directory if it doesn't exist
   mkdir -p ~/.kaggle

   # Move the downloaded kaggle.json file
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

   # Set secure permissions (required by Kaggle API)
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Verify Setup**
   ```bash
   # Test Kaggle API connection
   kaggle datasets list --max-size 10
   ```

4. **Download Remaining Datasets**
   Once credentials are configured, run:
   ```bash
   python setup_kaggle_datasets.py
   ```

   Or download manually:
   ```bash
   # RAVDESS (High Priority)
   kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
   unzip ravdess-emotional-speech-audio.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/ravdess/

   # TESS (Medium Priority)
   kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
   unzip toronto-emotional-speech-set-tess.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/tess/
   ```

---

## 📋 Quick Setup Checklist

- [x] Install Kaggle package
- [x] Download MAESTRO v3.0
- [x] Download CREMA-D
- [ ] Setup Kaggle API credentials (`~/.kaggle/kaggle.json`)
- [ ] Download RAVDESS dataset
- [ ] Download TESS dataset
- [ ] Verify all datasets are complete
- [ ] Update documentation

---

## 🎯 Next Actions

### Immediate (Required)
1. **Setup Kaggle Credentials** (see instructions above)
2. **Download RAVDESS** (High Priority - Required for emotion training)

### Recommended
3. **Download TESS** (Medium Priority - Expands emotion dataset)
4. **Verify all datasets** are complete and accessible

### Optional
5. **Fix GTZAN** (Alternative download source if needed)

---

## 📊 Current Status

| Component | Status | Priority |
|-----------|--------|----------|
| Kaggle Package | ✅ Complete | Required |
| MAESTRO v3.0 | ✅ Complete | 🔴 High |
| CREMA-D | ✅ Complete | 🟡 Medium |
| Kaggle Credentials | ⏳ Pending | Required |
| RAVDESS | ⏳ Pending | 🔴 High |
| TESS | ⏳ Pending | 🟡 Medium |
| GTZAN | ❌ Failed | 🟡 Medium |

**Overall Progress:** 3/7 components complete (43%)

---

## 🔧 Troubleshooting

### Kaggle API Issues

**Error: "401 Unauthorized"**
- Check that `~/.kaggle/kaggle.json` exists
- Verify username and key are correct
- Ensure file permissions are 600: `chmod 600 ~/.kaggle/kaggle.json`

**Error: "403 Forbidden"**
- Verify you accepted the dataset terms on Kaggle website
- Check that dataset is publicly available

**Error: "Package not found"**
- Install: `pip install kaggle`
- Verify: `python -m pip list | grep kaggle`

### Download Issues

**Large Dataset Downloads**
- Use stable internet connection
- Downloads may take time (RAVDESS ~1GB, TESS ~400MB)
- Check available disk space

**Extraction Issues**
- Ensure sufficient disk space (2-3x compressed size)
- Verify ZIP file integrity: `unzip -t filename.zip`

---

## 📝 Notes

- All datasets are stored in: `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/`
- Raw datasets go in `raw/` subdirectories
- Processed datasets go in `processed/` subdirectories
- Download scripts available: `download_missing_datasets.py`, `setup_kaggle_datasets.py`

---

## 🔗 Related Files

- `DOWNLOAD_PROGRESS.md` - Download status tracking
- `MISSING_AUDIO_DATASETS_REPORT.md` - Initial audit report
- `AUDIO_MIDI_DATA_REVIEW.md` - Directory structure review
- `setup_kaggle_datasets.py` - Kaggle download script
- `download_missing_datasets.py` - General download script
