# Kaggle API Setup Instructions for SeanBurdges

**Username:** SeanBurdges
**Profile:** https://www.kaggle.com/seanburdges

---

## Quick Setup Steps

### 1. Get Your API Token

1. **Visit your Kaggle settings:**
   - Direct link: https://www.kaggle.com/seanburdges/settings
   - Or: https://www.kaggle.com/settings → Your Account → API

2. **Create API Token:**
   - Scroll to "API" section
   - Click **"Create New Token"** button
   - This will automatically download `kaggle.json` to your Downloads folder

### 2. Install Credentials

After downloading `kaggle.json`, run these commands:

```bash
# Create Kaggle directory if needed
mkdir -p ~/.kaggle

# Move the downloaded file
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# Set secure permissions (required by Kaggle API)
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Verify Setup

Test your Kaggle API connection:

```bash
# Test connection
kaggle datasets list --max-size 10

# Or test with Python
python3 -c "import kaggle; print('Kaggle API configured correctly!')"
```

### 4. Download Datasets

Once credentials are verified, download the remaining datasets:

```bash
# Run the setup script
python3 setup_kaggle_datasets.py

# Or download manually:
# RAVDESS (High Priority)
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
unzip ravdess-emotional-speech-audio.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/ravdess/

# TESS (Medium Priority)
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/raw/emotions/tess/
```

---

## What the kaggle.json File Looks Like

The file should contain your username and API key:

```json
{
  "username": "seanburdges",
  "key": "your_api_key_here"
}
```

**Location:** `~/.kaggle/kaggle.json`
**Permissions:** Must be `600` (read/write for owner only)

---

## Troubleshooting

### "401 Unauthorized" Error
- Verify `~/.kaggle/kaggle.json` exists
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Ensure username and key are correct

### "403 Forbidden" Error
- Make sure you've accepted the dataset terms on Kaggle website
- Visit the dataset page and click "Accept" on terms

### Token Expired
- Create a new token at https://www.kaggle.com/seanburdges/settings
- Replace the old `kaggle.json` file

### File Not Found
- Check Downloads folder: `ls ~/Downloads/kaggle.json`
- Or search: `find ~ -name "kaggle.json" -type f`

---

## Expected Downloads

### RAVDESS Dataset
- **Size:** ~1 GB
- **Files:** Audio WAV files with emotion labels
- **Location:** `raw/emotions/ravdess/`
- **Purpose:** Emotion recognition training

### TESS Dataset
- **Size:** ~400 MB
- **Files:** Audio WAV files with emotion labels
- **Location:** `raw/emotions/tess/`
- **Purpose:** Emotion recognition training expansion

---

## Next Steps After Setup

1. ✅ Setup Kaggle API credentials
2. ⏳ Download RAVDESS dataset
3. ⏳ Download TESS dataset
4. ⏳ Verify all datasets are complete
5. ⏳ Run dataset verification script

---

## Quick Reference

**Your Kaggle Profile:** https://www.kaggle.com/seanburdges
**API Settings:** https://www.kaggle.com/seanburdges/settings
**Credentials File:** `~/.kaggle/kaggle.json`
**Setup Script:** `setup_kaggle_datasets.py`
