# Data Location Update - Files Moved from External SSD

**Date**: 2025-01-09  
**Status**: ✅ **CONFIGURATION UPDATED**

---

## Summary

All audio/MIDI data files have been moved from external SSD to local storage. Configuration files have been updated to reflect the new location.

---

## New Location

**Primary Data Location**:
```
/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/
```

**Previous Location** (no longer in use):
```
/Volumes/Extreme SSD/kelly-audio-data/
```

---

## Configuration Changes

### Environment Variables

**Set this environment variable** (recommended):
```bash
export KELLY_AUDIO_DATA_ROOT=/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data
```

**Or add to `.env` file**:
```bash
KELLY_AUDIO_DATA_ROOT=/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data
```

---

## Files Updated

### Configuration Files
- ✅ `configs/storage.py` - Updated platform paths, added new location to auto-detection
- ✅ `env.example` - Updated default path to new location
- ✅ `config/emotion_recognizer.yaml` - Updated default path
- ✅ `config/harmony_predictor.yaml` - Updated default path
- ✅ `config/groove_predictor.yaml` - Updated default path
- ✅ `config/melody_transformer.yaml` - Updated default path
- ✅ `config/instrument_recognizer.yaml` - Updated default path
- ✅ `config/dynamics_engine.yaml` - Updated default path
- ✅ `config/emotion_node_classifier.yaml` - Updated default path
- ✅ `config/music_foundation_base.yaml` - Updated default path
- ✅ `config/music_foundation_dry_run.yaml` - Updated default path
- ✅ `config/dryrun_ssl.yaml` - Updated default path
- ✅ `config/build-m4-local-inference.yaml` - Updated default paths
- ✅ `config/build-dev-mac.yaml` - Updated documentation

### Scripts
- ✅ `check_datasets.py` - Updated to prioritize new location

### Documentation
- ✅ `env.example` - Updated with new path and notes
- ✅ Created this document

---

## Path Resolution Priority

The storage configuration now resolves paths in this order:

1. **KELLY_AUDIO_DATA_ROOT** (explicit environment variable - **recommended**)
2. **KELLY_SSD_PATH/kelly-audio-data** (if KELLY_SSD_PATH points to parent directory)
3. **Auto-detected paths**:
   - `/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA` (new location - checked first)
   - Legacy SSD mounts (`/Volumes/Extreme SSD`, etc.) - kept for backward compatibility
4. **Fallback**: `~/.kelly/audio-data` (always writable)

---

## Backward Compatibility

**Legacy SSD paths are still checked** as fallbacks, but will only be used if:
- The new location doesn't exist
- No environment variable is set
- Legacy SSD is actually mounted

**Recommendation**: Set `KELLY_AUDIO_DATA_ROOT` explicitly to avoid any confusion.

---

## Verification

### Check Current Configuration
```bash
# From project root
python3 -c "from configs.storage import get_storage_config; config = get_storage_config(); print(f'Root: {config.audio_data_root}'); print(f'Source: {config.source}'); print(f'Type: {config.storage_type}')"
```

### Verify New Location Exists
```bash
ls -la /Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data/
```

### Run Dataset Check
```bash
python check_datasets.py
```

---

## Migration Notes

- ✅ All configuration files updated
- ✅ Auto-detection updated to check new location first
- ✅ Legacy paths kept as fallbacks
- ✅ Environment variable support maintained
- ✅ No breaking changes (backward compatible)

**Action Required**: Set `KELLY_AUDIO_DATA_ROOT` environment variable for best results.

---

## Related Files

- **Storage Config**: `configs/storage.py`
- **Environment Template**: `env.example`
- **Dataset Checker**: `check_datasets.py`
- **Data Review**: `AUDIO_MIDI_DATA_REVIEW.md`

---

**Last Updated**: 2025-01-09  
**Updated By**: AI Assistant
