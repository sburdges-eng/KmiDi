# Audio Assets

This directory contains audio assets organized from OneDrive integration.

## Guitar Samples

**Location:** `guitar_samples/`
**Source:** OneDrive `daiw_complete/GUITAR*.wav`

Guitar samples are available in OneDrive but not copied to project due to size. To use:

```bash
# Copy specific samples as needed
cp "/Users/seanburdges/Library/CloudStorage/OneDrive-Personal/daiw_complete/GUITAR L#01.wav" assets/audio/guitar_samples/
cp "/Users/seanburdges/Library/CloudStorage/OneDrive-Personal/daiw_complete/GUITAR R#01.wav" assets/audio/guitar_samples/
```

**Available Samples:**
- GUITAR L#01 through L#05 (left channel)
- GUITAR R#01 through R#15 (right channel)
- Some duplicate files with timestamps

## MIDI Files

**Location:** `midi/`
**Source:** OneDrive `daiw_complete/audio_vault/output/`

- `i_feel_broken.mid` - Generated MIDI file from intent processing

## Audio Vault Structure

OneDrive contains an audio vault with:
- `raw/` - Unprocessed samples
- `refined/` - Processed samples
- `output/` - Generated MIDI files
- `kits/` - Kit mapping files

To access full audio vault:
```bash
# OneDrive location
/Users/seanburdges/Library/CloudStorage/OneDrive-Personal/daiw_complete/audio_vault/
```
