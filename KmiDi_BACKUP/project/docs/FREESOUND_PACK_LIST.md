# Freesound Pack Download List
## Lo-Fi Bedroom Emo / Indie Folk Starter Kit

**Created:** 2025-11-25
**For:** Kelly Song Production + DAiW Music Brain

---

## Priority 1: Essential Drums (Lo-Fi Aesthetic)

### Acoustic Drum One-Shots
- **Pack:** "Acoustic Drum Kit Clean Samples"
  - URL: https://freesound.org/people/afleetingspeck/packs/14856/
  - Contains: Clean kick, snare, hi-hat, toms
  - Perfect for: Natural bedroom recordings

- **Pack:** "Vintage Drum Machine Samples"
  - URL: https://freesound.org/people/suburban_grilla/packs/6/
  - Contains: Lo-fi electronic drums
  - Perfect for: Tape-saturated vibe

- **Pack:** "Brush Drum Samples"
  - URL: https://freesound.org/people/Mhistorically/packs/28587/
  - Contains: Soft brush hits
  - Perfect for: Intimate verses

### Room/Ambience
- **Pack:** "Room Tones and Ambiences"
  - URL: https://freesound.org/people/klankbeeld/packs/10855/
  - Contains: Bedroom room tone, tape hiss
  - Perfect for: Lo-fi texture layers

---

## Priority 2: Acoustic Guitars

### Fingerpicking
- **Pack:** "Acoustic Guitar Single Notes"
  - URL: https://freesound.org/people/MTG/packs/8647/
  - Contains: Individual string plucks
  - Perfect for: Custom patterns like 1-5-6-4-3-2

- **Pack:** "Guitar Harmonics"
  - URL: https://freesound.org/people/ERH/packs/3336/
  - Contains: Harmonic overtones
  - Perfect for: Ambient fills

### Strumming
- **Pack:** "Acoustic Guitar Strums"
  - URL: https://freesound.org/people/HerbertBoland/packs/18534/
  - Contains: Various strum patterns
  - Perfect for: Chorus energy

---

## Priority 3: Bass

### Acoustic/Electric Bass
- **Pack:** "Fender Jazz Bass Samples"
  - URL: https://freesound.org/people/FrusciMike/packs/13115/
  - Contains: Plucked bass notes (E-G)
  - Perfect for: Root note support

- **Pack:** "Upright Bass Pizzicato"
  - URL: https://freesound.org/people/No_Go/packs/619/
  - Contains: Upright bass plucks
  - Perfect for: Jazzy/intimate bass

---

## Priority 4: Vocal Textures

### Breathy/Lo-Fi Vocals
- **Pack:** "Vocal Samples Human Voice"
  - URL: https://freesound.org/people/pushtobreak/packs/14808/
  - Contains: Ahhs, ohhs, breaths
  - Perfect for: Background vocal layers

- **Pack:** "Whisper Samples"
  - URL: https://freesound.org/people/HerbertBoland/packs/30032/
  - Contains: Whispered phrases
  - Perfect for: Intimate doubles

---

## Priority 5: FX & Ambience

### Tape/Vinyl FX
- **Pack:** "Vinyl Crackle and Noise"
  - URL: https://freesound.org/people/OldFritz/packs/17926/
  - Contains: Vinyl noise, tape wow/flutter
  - Perfect for: Lo-fi warmth

- **Pack:** "Cassette Tape Artifacts"
  - URL: https://freesound.org/people/j1987/packs/15003/
  - Contains: Tape stop, warble
  - Perfect for: Transitions

### Nature Ambience
- **Pack:** "Forest Ambience Pack"
  - URL: https://freesound.org/people/klankbeeld/packs/5246/
  - Contains: Birds, wind, rustling
  - Perfect for: Outdoor bedroom vibe

---

## Batch Download Instructions

### Option 1: Manual Download
1. Visit each URL
2. Click "Download Pack"
3. Login/create Freesound account (free)
4. Save to ~/Downloads/Freesound_Packs/

### Option 2: Automated (using freesound-python)
```bash
# Install Freesound Python API
pip install freesound-python

# Run download script
python ~/Music/Samples/freesound_downloader.py
```

---

## Organization After Download

Run the auto-organizer:
```bash
python ~/Music/Samples/organize_samples.py
```

This will:
1. Extract all zips
2. Rename files to: `[BPM]_[Key]_[Type]_[Description]_[Number].wav`
3. Sort into proper folders
4. Generate sample database JSON

---

## Freesound API Key Setup

1. Go to: https://freesound.org/apiv2/apply/
2. Create API key (free)
3. Save to: `~/.freesound_api_key`
```bash
echo "YOUR_API_KEY_HERE" > ~/.freesound_api_key
chmod 600 ~/.freesound_api_key
```

---

## Alternative: Quick Start Packs (Pre-Curated)

If you want to skip Freesound hunting:

### Splice (Subscription)
- "LANDR Lo-Fi Hip-Hop"
- "Bedroom Pop Essentials"
- "Indie Folk Guitars"

### Free Alternatives
- **Bedroom Producers Blog:** https://bedroomproducersblog.com/free-samples/
- **99Sounds:** https://99sounds.org/
- **Reverb Drum Machines:** https://reverb.com/software/samples-and-loops

---

## Total Download Size Estimate
- Priority 1-5: ~2-3 GB
- Full collection: ~5-7 GB

---

## Next Steps

Once downloaded:
1. Run `organize_samples.py` to sort everything
2. Update Sample Library Index with favorites
3. Create Logic Pro sampler instruments
4. Build first lo-fi drum kit

---

*For Kelly song template: Focus on Priority 1 (Drums) and Priority 2 (Guitars)*
