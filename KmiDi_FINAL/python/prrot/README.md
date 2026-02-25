# PRROT/PARROT Python Tier B Components

## Overview

This directory contains the Python Tier B components for PRROT/PARROT - the offline analysis and reasoning layer responsible for deep phoneme alignment, speaker-specific articulation analysis, timbre embeddings, and prosody analysis.

## Architecture

**Tier B**: Disposable Python worker processes that:
- Load exactly one ML model per job (Q4 quantized, ~3B parameters)
- Process voice profile extraction or control data generation jobs
- Write structured JSON outputs (never audio)
- Exit fully after job completion to reclaim memory

## 16GB Mac Safety

All components are designed for safe operation on 16GB Mac systems:

- **Single Worker**: Process lock prevents concurrent workers
- **Memory Monitoring**: 8GB worker limit, 10GB system reserve required
- **Q4 Quantization**: Required and enforced for model loading
- **Process Lifecycle**: Guaranteed cleanup and exit after job completion
- **External SSD**: Used for storage only, never as swap/paging

## Components

### Core Modules

- `worker.py` - Main disposable worker process
- `job_schema.py` - Job input/output schemas (JSON)
- `voice_profile.py` - Voice profile data structures (Python/C++ compatible)

### Analysis Modules

- `phoneme_aligner.py` - Deep phoneme alignment (ML model integration ready)
- `articulation_analyzer.py` - Speaker-specific articulation analysis
- `timbre_embeddings.py` - Non-reconstructive timbre embedding extraction
- `prosody_analyzer.py` - Prosody tendency analysis
- `lyric_planner.py` - Lyric-to-phoneme planning and control data generation

### Inference Modules

- `articulation_inference.py` - Parse descriptive articulation text or analyze non-speech audio
- `instrument_affinity.py` - Map articulation profiles to instrument suggestions

### Utilities

- `utils/memory_monitor.py` - Memory monitoring for 16GB constraint compliance
- `utils/process_manager.py` - Single worker process enforcement
- `utils/external_ssd.py` - External SSD path management (USB 2.0 optimized)
- `model_manager.py` - Model loading with memory constraints

## Usage

### Run Worker

```bash
# Basic usage
python -m prrot.worker /path/to/job.json

# With external SSD path
python -m prrot.worker /path/to/job.json --external-ssd /Volumes/ExternalSSD/prrot
```

### Create Job

```python
from prrot.job_schema import VoiceProfileExtractionJob, save_job_to_file
from pathlib import Path

job = VoiceProfileExtractionJob(
    job_id="extract_001",
    reference_audio_paths=["/path/to/sample1.wav", "/path/to/sample2.wav"],
    profile_id="my_voice",
    profile_name="My Voice Profile"
)

save_job_to_file(job, Path("/path/to/jobs/extract_001.json"))
```

### Load Voice Profile

```python
from prrot.voice_profile import VoiceProfile
from pathlib import Path

profile = VoiceProfile.load(Path("/path/to/profile.json"))
print(f"Profile: {profile.profile_name}")
```

## Memory Constraints

### 16GB Mac Systems

- **Worker Maximum**: 8GB (hard limit)
- **Warning Threshold**: 6GB
- **System Reserve**: 10GB minimum required before starting
- **Model Size**: ~1.5-2GB (Q4 quantized 3B parameter model)
- **Single Worker**: Only one worker can run at a time

### Verification

The system automatically:
1. Checks system memory before starting (requires 10GB+)
2. Checks process memory before model load
3. Validates memory after model load
4. Monitors memory during processing
5. Forces garbage collection after job completion

## External SSD Configuration

Set via environment variable or command-line argument:

```bash
export PRROT_EXTERNAL_SSD_PATH=/Volumes/ExternalSSD/prrot
```

Or pass to worker:

```bash
python -m prrot.worker job.json --external-ssd /Volumes/ExternalSSD/prrot
```

## Dependencies

### Required

```bash
pip install psutil numpy
```

### Optional (for audio processing)

```bash
pip install librosa soundfile
```

### Optional (for phoneme conversion)

```bash
pip install g2p_en
```

## Safety Features

- ✅ Single worker process lock
- ✅ Memory monitoring and limits
- ✅ Model manager (Q4 enforcement)
- ✅ Process lifecycle management
- ✅ Automatic cleanup on exit
- ✅ Stale process detection

## Job Types

1. **Voice Profile Extraction** (`extract_voice_profile`)
   - Extracts parametric voice profile from reference audio
   - Outputs: Voice profile JSON

2. **Control Data Generation** (`generate_control_data`)
   - Generates MIDI/control data from voice profile + lyrics
   - Outputs: Phoneme sequence, MIDI, automation envelopes

3. **Articulation Analysis** (`analyze_articulation`)
   - Analyzes articulation from text or audio
   - Outputs: Articulation profile, instrument affinity

## Output Format

All outputs are JSON-based:
- Voice profiles: JSON format (C++/Python compatible)
- Control data: JSON format (DAW-compatible)
- Job artifacts: JSON format (versioned schemas)

## License

Part of the PRROT/PARROT voice-instrument compiler system.
All dependencies must be Apache 2.0, MIT, or BSD.
