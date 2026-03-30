# PRROT/PARROT Quick Start Guide

## Overview

PRROT/PARROT is a voice-instrument compiler that extracts parametric voice characteristics from reference audio and generates MIDI/control data for DAW integration.

## Building

### C++ Tier C Components

The PRROT core library is automatically built when `BUILD_KMIDI_CORE=ON`:

```bash
cd KmiDi_FINAL
mkdir -p build && cd build
cmake .. -DBUILD_KMIDI_CORE=ON
cmake --build . -j$(sysctl -n hw.ncpu)
```

The `prrot_core` static library will be built and linked to `KellyCore`.

### Python Tier B Components

Python components are ready to use. Install dependencies:

```bash
pip install numpy psutil  # Core dependencies
# Optional: pip install librosa soundfile  # For audio processing
```

## Usage

### Tier C: Embedded Engine (C++)

```cpp
#include "prrot/PRROTEngine.h"

// Initialize engine
prrot::PRROTEngine engine;
engine.initialize();

// Load voice profile (non-RT operation)
prrot::VoiceProfile profile;
// ... populate profile from JSON or extraction ...
engine.loadVoiceProfile(profile);

// Process audio segment (RT-safe)
float* audio_samples = ...;
size_t num_samples = ...;
float sample_rate = 44100.0f;

prrot::PhonemeControlData control_data = engine.processAudioSegment(
    audio_samples, num_samples, sample_rate, 120.0f
);

// Use control_data: phoneme_sequence, pitch_targets, midi_notes, etc.
```

### Tier B: Python Worker

#### 1. Create a Voice Profile Extraction Job

```python
from prrot.job_schema import VoiceProfileExtractionJob
from prrot.job_schema import save_job_to_file
from pathlib import Path

job = VoiceProfileExtractionJob(
    job_id="extract_001",
    reference_audio_paths=[
        "/path/to/reference_audio/sample_001.wav",
        "/path/to/reference_audio/sample_002.wav"
    ],
    profile_id="my_voice",
    profile_name="My Voice Profile"
)

job_path = Path("/path/to/jobs/extract_001.json")
save_job_to_file(job, job_path)
```

#### 2. Run Worker

```bash
python -m prrot.worker /path/to/jobs/extract_001.json \
    --external-ssd /Volumes/ExternalSSD/prrot
```

#### 3. Generate Control Data from Lyrics

```python
from prrot.job_schema import ControlDataGenerationJob, save_job_to_file

job = ControlDataGenerationJob(
    job_id="generate_001",
    voice_profile_id="my_voice",
    lyrics="Hello world, this is a test",
    melody_midi_notes=[60, 62, 64, 65, 64, 62, 60],
    melody_timing=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    tempo_bpm=120.0
)

save_job_to_file(job, Path("/path/to/jobs/generate_001.json"))
```

```bash
python -m prrot.worker /path/to/jobs/generate_001.json
```

## External SSD Setup

Set the external SSD path via environment variable:

```bash
export PRROT_EXTERNAL_SSD_PATH=/Volumes/ExternalSSD/prrot
```

Or pass it to the worker:

```bash
python -m prrot.worker job.json --external-ssd /Volumes/ExternalSSD/prrot
```

The system will create the following directory structure:

```
prrot/
├── reference_audio/      # Raw voice reference audio
├── voice_profiles/       # Extracted voice profiles (JSON)
├── models/              # Cached ML models (quantized)
├── jobs/                # Job artifacts
├── caches/              # Cached embeddings and features
└── outputs/             # Generated control data outputs
```

## Memory Management (16GB Systems)

The worker automatically monitors memory usage:

- Maximum worker memory: 8GB
- Warning threshold: 6GB
- Model quantization: Q4 (reduces memory by ~75%)
- Worker exits after job completion

## RT Safety

**Important**: Tier C components are RT-safe and can be used in audio callbacks:

- ✅ Pre-allocated buffers only
- ✅ No dynamic allocation
- ✅ No Python, ML, or disk I/O
- ✅ Deterministic execution

**Do NOT**:
- ❌ Call Python worker from audio callback
- ❌ Load models in audio callback
- ❌ Perform file I/O in audio callback
- ❌ Allocate memory in audio callback

## Output Format

Control data is output as JSON with the following structure:

```json
{
  "phoneme_sequence": [
    {
      "phoneme": 0,
      "start_time_ms": 0.0,
      "duration_ms": 100.0,
      "is_stressed": false,
      "stress_level": 0.0
    }
  ],
  "pitch_targets": [
    {
      "time_ms": 0.0,
      "midi_note": 60,
      "cents_offset": 0.0,
      "confidence": 1.0
    }
  ],
  "midi_notes": [
    {
      "note_number": 60,
      "velocity": 100,
      "start_time_ms": 0.0,
      "duration_ms": 100.0,
      "channel": 0
    }
  ],
  "automation_envelopes": [],
  "breath_markers": []
}
```

## Integration with DAW

1. Load voice profile from JSON
2. Generate control data from lyrics + melody
3. Export MIDI file from `midi_notes`
4. Export automation envelopes
5. Import into DAW and use with vocal synthesizer plugins

## Troubleshooting

### Memory Issues

If worker fails with memory errors:
- Check available system memory
- Ensure only one worker runs at a time
- Use Q4 quantized models
- Close other applications

### External SSD Not Found

- Verify SSD is mounted
- Check path permissions
- Set `PRROT_EXTERNAL_SSD_PATH` environment variable

### Model Loading Errors

- Verify model file exists in `models/` directory
- Check model format (Q4 quantization required)
- Ensure sufficient memory available

## Next Steps

1. Integrate actual ML models (3B parameter Q4)
2. Test with real reference audio
3. Generate control data for DAW integration
4. Fine-tune voice profiles
5. Optimize for production use
