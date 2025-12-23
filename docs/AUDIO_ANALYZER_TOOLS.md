# Audio Analyzer Tools - Implementation Summary

**Date:** 2025-01-XX  
**Status:** ✅ Complete

---

## Overview

Three new CLI tools have been created using the `AudioAnalyzer` class from `music_brain.audio.analyzer`. These tools extend DAiW's audio analysis capabilities for practical use cases.

---

## Tool 1: `compare-audio`

**Command:** `daiw compare-audio <file1> <file2>`

**Purpose:** Compare two audio files side-by-side for BPM, key, feel, and features.

### Features:
- Duration comparison
- BPM/tempo comparison with confidence scores
- Key detection comparison (match/difference)
- Feel analysis comparison (tempo, dynamic range, swing)
- Overall similarity score
- Detailed feature comparison (optional `--detailed` flag)
- JSON export support

### Usage:
```bash
# Basic comparison
daiw compare-audio song1.wav song2.wav

# Detailed comparison with JSON export
daiw compare-audio song1.wav song2.wav --detailed -o comparison.json
```

### Output Example:
```
Comparing audio files:
  File 1: song1.wav
  File 2: song2.wav

Analyzing file 1...
Analyzing file 2...

=== Audio Comparison ===

⏱️  Duration:
  File 1: 180.50s
  File 2: 185.20s
  Difference: 4.70s

🎵 Tempo (BPM):
  File 1: 120.5 BPM (confidence: 95.00%)
  File 2: 121.2 BPM (confidence: 92.00%)
  Difference: 0.7 BPM

🎹 Key:
  File 1: C major (confidence: 88.00%)
  File 2: C major (confidence: 85.00%)
  Match: ✓ Yes

🎭 Feel Analysis:
  File 1 Tempo: 120.5 BPM
  File 2 Tempo: 121.2 BPM
  File 1 Dynamic Range: 12.5 dB
  File 2 Dynamic Range: 13.1 dB
  File 1 Swing: 0.45
  File 2 Swing: 0.48

  Overall Similarity: 87.50%
  Tempo Similarity: 96.50%
  Swing Similarity: 97.00%
  Energy Similarity: 78.00%
```

---

## Tool 2: `batch-analyze`

**Command:** `daiw batch-analyze <files...> [options]`

**Purpose:** Analyze multiple audio files in batch, useful for cataloging libraries or processing folders.

### Features:
- Process multiple files or entire directories
- Recursive directory scanning (`--recursive`)
- JSON or CSV output format
- Progress tracking
- Error handling (continues on errors)
- Configurable max duration per file
- Output to file or stdout

### Usage:
```bash
# Analyze specific files
daiw batch-analyze song1.wav song2.wav song3.wav

# Analyze directory (non-recursive)
daiw batch-analyze /path/to/audio/files/

# Analyze directory recursively
daiw batch-analyze /path/to/audio/ --recursive

# Export to CSV
daiw batch-analyze *.wav --format csv -o analysis.csv

# Export to JSON
daiw batch-analyze *.wav --format json -o analysis.json
```

### Output Format (CSV):
```csv
file,filename,duration_seconds,sample_rate,bpm,bpm_confidence,key,key_confidence,feel_tempo,dynamic_range_db,swing_estimate,spectral_centroid_mean,...
/path/song1.wav,song1.wav,180.5,44100,120.5,0.95,C major,0.88,120.5,12.5,0.45,2500.0,...
/path/song2.wav,song2.wav,185.2,44100,121.2,0.92,C major,0.85,121.2,13.1,0.48,2550.0,...
```

### Output Format (JSON):
```json
[
  {
    "file": "/path/song1.wav",
    "filename": "song1.wav",
    "duration_seconds": 180.5,
    "sample_rate": 44100,
    "bpm": 120.5,
    "bpm_confidence": 0.95,
    "key": "C major",
    "key_confidence": 0.88,
    "feel_tempo": 120.5,
    "dynamic_range_db": 12.5,
    "swing_estimate": 0.45,
    "spectral_centroid_mean": 2500.0,
    ...
  },
  ...
]
```

---

## Tool 3: `export-features`

**Command:** `daiw export-features <audio_file> -o <output_file> [options]`

**Purpose:** Export comprehensive audio features to JSON or CSV for further analysis or integration.

### Features:
- Complete feature extraction
- BPM with alternatives
- Key detection with confidence
- Feel analysis
- Feature summary (spectral, MFCC, chroma, etc.)
- Optional segment analysis (`--include-segments`)
- Optional chord detection (`--include-chords`)
- JSON or CSV format

### Usage:
```bash
# Export to JSON (default)
daiw export-features song.wav -o features.json

# Export to CSV
daiw export-features song.wav -o features.csv --format csv

# Include segments
daiw export-features song.wav -o features.json --include-segments

# Include chords
daiw export-features song.wav -o features.json --include-chords

# Full export (segments + chords)
daiw export-features song.wav -o full_analysis.json --include-segments --include-chords
```

### Output Structure (JSON):
```json
{
  "file": "/path/song.wav",
  "filename": "song.wav",
  "duration_seconds": 180.5,
  "sample_rate": 44100,
  "bpm": {
    "value": 120.5,
    "confidence": 0.95,
    "alternatives": [121.0, 120.0, 122.0]
  },
  "key": {
    "key": "C",
    "mode": "major",
    "full_key": "C major",
    "confidence": 0.88
  },
  "feel": {
    "tempo_bpm": 120.5,
    "dynamic_range_db": 12.5,
    "swing_estimate": 0.45,
    "groove_regularity": 0.82,
    ...
  },
  "features": {
    "spectral_centroid_mean": 2500.0,
    "spectral_bandwidth_mean": 1800.0,
    "rms_mean": 0.45,
    "mfcc_means": [0.1, 0.2, ...],
    "chroma_means": [0.08, 0.09, ...],
    ...
  },
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 45.0,
      "duration": 45.0,
      "energy": 0.45,
      "label": "segment_1"
    },
    ...
  ],
  "chords": {
    "chords": [...],
    "sequence": ["C", "F", "Am", "G"],
    "unique_chords": ["C", "F", "Am", "G"],
    "estimated_key": "C major",
    "confidence": 0.75
  }
}
```

---

## Implementation Details

### Code Location
- **File:** `music_brain/cli.py`
- **Functions:**
  - `cmd_compare_audio(args)` - Lines ~475-580
  - `cmd_batch_analyze(args)` - Lines ~583-740
  - `cmd_export_features(args)` - Lines ~743-865

### Dependencies
- `music_brain.audio.analyzer.AudioAnalyzer`
- `music_brain.audio.feel.compare_feel` (for comparison)
- `music_brain.audio.chord_detection.ChordDetector` (optional)
- `librosa` (required for audio analysis)
- Standard library: `json`, `csv`, `pathlib`, `glob`

### Error Handling
- File existence checks
- Import error handling (librosa availability)
- Per-file error handling in batch mode (continues on errors)
- Graceful degradation when features unavailable

### Performance Considerations
- Batch mode processes files sequentially (can be parallelized)
- Max duration limit prevents long processing times
- Progress indicators for batch operations

---

## Use Cases

### 1. Audio Library Cataloging
```bash
# Catalog entire music library
daiw batch-analyze ~/Music/ --recursive --format csv -o music_library.csv
```

### 2. Reference Track Analysis
```bash
# Compare your track to reference
daiw compare-audio my_track.wav reference.wav --detailed -o comparison.json
```

### 3. Feature Extraction for ML
```bash
# Export features for machine learning
daiw export-features dataset/*.wav -o features.json --include-segments
```

### 4. Quality Control
```bash
# Batch check BPM/key consistency
daiw batch-analyze album/*.wav --format csv -o album_analysis.csv
```

### 5. Remix/Production Analysis
```bash
# Compare original vs remix
daiw compare-audio original.wav remix.wav --detailed
```

---

## Integration with DAiW Workflow

These tools integrate with DAiW's workflow:

1. **Audio Analysis → Intent Generation**
   ```bash
   # Analyze reference track
   daiw export-features reference.wav -o ref_features.json
   # Use features to inform intent generation
   ```

2. **Batch Processing → Catalog Building**
   ```bash
   # Build audio catalog
   daiw batch-analyze audio_vault/ --recursive -o catalog.json
   # Use catalog for reference track selection
   ```

3. **Comparison → Quality Assurance**
   ```bash
   # Compare generated MIDI (rendered to audio) vs reference
   daiw compare-audio generated.wav reference.wav
   ```

---

## Future Enhancements

Potential improvements:
- Parallel processing for batch mode
- Progress bars for long operations
- Filtering options (by BPM range, key, etc.)
- Statistical summaries for batch results
- Integration with DAiW's audio cataloger
- Real-time analysis mode

---

## Testing

To test the tools:

```bash
# Test compare-audio
daiw compare-audio test1.wav test2.wav

# Test batch-analyze
daiw batch-analyze test*.wav --format json

# Test export-features
daiw export-features test.wav -o test_features.json --include-segments
```

---

**Last Updated:** 2025-01-XX  
**Status:** ✅ All 3 tools implemented and ready for use

