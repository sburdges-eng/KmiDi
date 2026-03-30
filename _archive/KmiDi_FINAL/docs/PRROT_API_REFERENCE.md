# PRROT/PARROT API Reference

## Tier C: C++ Embedded Engine API

### PRROTEngine

**Location**: `engine/src/prrot/PRROTEngine.h`

Main embedded engine for RT-safe voice analysis and control data generation.

#### Initialization

```cpp
#include "prrot/PRROTEngine.h"

prrot::PRROTEngine engine;
bool initialized = engine.initialize();
```

#### Load Voice Profile

```cpp
prrot::VoiceProfile profile;
// ... populate profile ...
bool loaded = engine.loadVoiceProfile(profile);
```

#### Process Audio Segment (RT-Safe)

```cpp
float* audio_samples = ...;
size_t num_samples = ...;
float sample_rate = 44100.0f;

prrot::PhonemeControlData control_data = engine.processAudioSegment(
    audio_samples,
    num_samples,
    sample_rate,
    120.0f  // tempo_bpm
);
```

#### Generate Control Data from Phonemes

```cpp
std::vector<prrot::PhonemeTiming> phoneme_sequence = ...;
std::vector<prrot::PitchTarget> pitch_targets = ...;

prrot::PhonemeControlData control_data = engine.generateControlData(
    phoneme_sequence,
    pitch_targets,
    120.0f  // tempo_bpm
);
```

### VoiceProfile

**Location**: `engine/src/prrot/VoiceProfile.h`

Voice profile data structure containing parametric voice characteristics.

#### Access Phoneme Duration

```cpp
prrot::PhonemeType phoneme = prrot::PhonemeType::AH;
prrot::PhonemeDurationStats stats = profile.getPhonemeDuration(phoneme);
float mean_duration_ms = stats.mean_ms;
```

#### Access Consonant Profile

```cpp
prrot::ConsonantProfile consonant = profile.getConsonantProfile(prrot::PhonemeType::B);
float attack_time = consonant.attack_time_ms;
float strength = consonant.strength_scalar;
```

#### Check Validity

```cpp
if (profile.isValid()) {
    // Use profile
}
```

### PhonemeControlData

**Location**: `engine/src/prrot/PhonemeControlData.h`

Output control data structure for DAW integration.

#### Access Phoneme Sequence

```cpp
for (const auto& phoneme_timing : control_data.phoneme_sequence) {
    float start_ms = phoneme_timing.start_time_ms;
    float duration_ms = phoneme_timing.duration_ms;
    prrot::PhonemeType phoneme = phoneme_timing.phoneme;
}
```

#### Access MIDI Notes

```cpp
for (const auto& note : control_data.midi_notes) {
    int midi_note = note.note_number;
    int velocity = note.velocity;
    float start_ms = note.start_time_ms;
    float duration_ms = note.duration_ms;
}
```

#### Get Automation Value

```cpp
float dynamics_value = control_data.getAutomationValue(
    prrot::AutomationType::Dynamics,
    1000.0f  // time_ms
);
```

#### Get Vibrato Parameters

```cpp
prrot::VibratoParameters vibrato = control_data.getVibratoAt(1000.0f);
if (vibrato.enabled) {
    float rate = vibrato.rate_hz;
    float depth = vibrato.depth_cents;
}
```

## Tier B: Python Worker API

### Worker Process

**Location**: `python/prrot/worker.py`

Disposable worker process for ML-based analysis.

#### Command-Line Usage

```bash
python -m prrot.worker /path/to/job.json
python -m prrot.worker /path/to/job.json --external-ssd /Volumes/ExternalSSD/prrot
```

#### Programmatic Usage

```python
from prrot.worker import PRROTWorker
from pathlib import Path

worker = PRROTWorker(
    job_path=Path("/path/to/job.json"),
    external_ssd_path=Path("/Volumes/ExternalSSD/prrot")
)

success = worker.process()
```

### Job Schemas

**Location**: `python/prrot/job_schema.py`

#### VoiceProfileExtractionJob

```python
from prrot.job_schema import VoiceProfileExtractionJob, save_job_to_file

job = VoiceProfileExtractionJob(
    job_id="extract_001",
    reference_audio_paths=["sample1.wav", "sample2.wav"],
    profile_id="my_voice",
    profile_name="My Voice Profile",
    transcripts=["Hello world", "This is a test"]  # Optional
)

save_job_to_file(job, Path("job.json"))
```

#### ControlDataGenerationJob

```python
from prrot.job_schema import ControlDataGenerationJob

job = ControlDataGenerationJob(
    job_id="generate_001",
    voice_profile_id="my_voice",
    lyrics="Hello world, this is a test",
    melody_midi_notes=[60, 62, 64, 65, 64, 62, 60],
    melody_timing=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    tempo_bpm=120.0
)
```

#### ArticulationAnalysisJob

```python
from prrot.job_schema import ArticulationAnalysisJob

# From descriptive text
job = ArticulationAnalysisJob(
    job_id="analyze_001",
    descriptive_text="lips pressed together, buzzing air"
)

# From audio file
job = ArticulationAnalysisJob(
    job_id="analyze_002",
    audio_path="/path/to/audio.wav"
)
```

### Voice Profile (Python)

**Location**: `python/prrot/voice_profile.py`

#### Load Voice Profile

```python
from prrot.voice_profile import VoiceProfile
from pathlib import Path

profile = VoiceProfile.load(Path("profile.json"))
```

#### Save Voice Profile

```python
profile = VoiceProfile(
    profile_id="my_voice",
    profile_name="My Voice",
    created_timestamp=int(time.time())
)

profile.save(Path("profile.json"))
```

#### Access Profile Data

```python
# Phoneme inventory
phonemes = profile.phoneme_inventory

# Duration statistics
duration_stats = profile.phoneme_durations.get(PhonemeType.AH)

# Vowel sustain
sustain_mult = profile.vowel_sustain["sustain_multiplier"]

# Vibrato characteristics
vibrato_rate = profile.vibrato["rate_mean_hz"]
```

### Model Manager

**Location**: `python/prrot/model_manager.py`

#### Load Model with Memory Checks

```python
from prrot.model_manager import ModelManager
from pathlib import Path

manager = ModelManager()

model = manager.load_model(
    model_path=Path("/path/to/model_q4.bin"),
    model_type="phoneme_aligner",
    quantization="Q4"  # Required for 16GB
)

if model is None:
    print("Model load failed (memory constraint?)")
```

#### Check Memory Before Load

```python
can_load, warning = manager.check_memory_before_load(model_size_gb=2.0)

if not can_load:
    print(f"Cannot load: {warning}")
elif warning:
    print(f"Warning: {warning}")
```

#### Unload Model

```python
manager.unload_model()  # Unload all models
```

### Memory Monitor

**Location**: `python/prrot/utils/memory_monitor.py`

#### Check Memory Limits

```python
from prrot.utils.memory_monitor import MemoryMonitor

monitor = MemoryMonitor()

within_limit, warning = monitor.check_memory_limit()

if not within_limit:
    print(f"Memory limit exceeded: {warning}")
elif warning:
    print(f"Warning: {warning}")
```

#### Get Memory Statistics

```python
stats = monitor.get_memory_stats()

print(f"Process memory: {stats['process_memory_gb']:.2f}GB")
print(f"System available: {stats['system_available_gb']:.2f}GB")
print(f"Within limit: {stats['within_limit']}")
```

### Process Manager

**Location**: `python/prrot/utils/process_manager.py`

#### Acquire Process Lock

```python
from prrot.utils.process_manager import ProcessManager

manager = ProcessManager()

if not manager.acquire_lock():
    print("Another worker is running")
    sys.exit(1)

try:
    # Do work
    ...
finally:
    manager.release_lock()
```

#### Check System Memory

```python
has_memory, message = manager.check_system_memory()

if not has_memory:
    print(f"Insufficient memory: {message}")
```

### External SSD Manager

**Location**: `python/prrot/utils/external_ssd.py`

#### Get Paths

```python
from prrot.utils.external_ssd import ExternalSSDManager

ssd = ExternalSSDManager(base_path=Path("/Volumes/ExternalSSD/prrot"))

# Reference audio path
audio_path = ssd.get_reference_audio_path("profile_001")

# Voice profile path
profile_path = ssd.get_voice_profile_path("profile_001")

# Model path
model_path = ssd.get_model_path("phoneme_aligner_q4.bin")

# Job path
job_path = ssd.get_job_path("job_001")
```

#### List Reference Audio

```python
audio_files = ssd.list_reference_audio_files("profile_001")
for audio_file in audio_files:
    print(audio_file)
```

#### Batch File Operations

```python
# Batch read (optimized for USB 2.0)
file_paths = [Path("file1.wav"), Path("file2.wav")]
contents = ssd.batch_read_files(file_paths)

# Batch write
ssd.batch_write_files(file_paths, contents)
```

### Lyric Planner

**Location**: `python/prrot/lyric_planner.py`

#### Generate Control Data

```python
from prrot.lyric_planner import LyricPlanner
from prrot.voice_profile import VoiceProfile

profile = VoiceProfile.load(Path("profile.json"))
planner = LyricPlanner(profile)

control_data = planner.generate_control_data(
    lyrics="Hello world",
    melody_midi_notes=[60, 62, 64],
    melody_timing=[0.0, 0.5, 1.0],
    tempo_bpm=120.0
)
```

### Articulation Inference

**Location**: `python/prrot/articulation_inference.py`

#### Parse Descriptive Text

```python
from prrot.articulation_inference import ArticulationInference

inference = ArticulationInference()

profile = inference.parse_descriptive_text(
    "lips pressed together, buzzing air"
)

print(f"Excitation type: {profile.excitation_type}")
print(f"Airflow continuity: {profile.airflow_continuity}")
```

#### Analyze Audio

```python
profile = inference.analyze_audio(
    audio_path=Path("audio.wav")
)
```

### Instrument Affinity Mapper

**Location**: `python/prrot/instrument_affinity.py`

#### Map Articulation to Instruments

```python
from prrot.instrument_affinity import InstrumentAffinityMapper
from prrot.articulation_inference import ArticulationInference

inference = ArticulationInference()
mapper = InstrumentAffinityMapper()

articulation = inference.parse_descriptive_text(
    "lips pressed together, buzzing air"
)

affinity = mapper.map_articulation(articulation)

# Primary suggestion
primary = affinity.primary_suggestion
print(f"Primary: {primary.instrument_name} ({primary.confidence:.2f})")

# All suggestions
for suggestion in affinity.suggestions:
    print(f"{suggestion.instrument_name}: {suggestion.confidence:.2f}")
```

## Error Handling

### Memory Errors

```python
try:
    model = manager.load_model(model_path)
except RuntimeError as e:
    if "memory" in str(e).lower():
        print("Insufficient memory for model")
```

### Process Lock Errors

```python
if not manager.acquire_lock():
    # Another worker is running
    sys.exit(1)
```

### Job Processing Errors

The worker automatically marks jobs as failed and logs errors:

```python
# Job status checked in job JSON
job.status  # "failed"
job.error_message  # Error description
```

## Best Practices

1. **Always check memory before operations**
   ```python
   can_load, warning = manager.check_memory_before_load(2.0)
   if not can_load:
       return
   ```

2. **Use process manager for worker processes**
   ```python
   @ensure_single_worker
   def my_worker_function():
       ...
   ```

3. **Always unload models after use**
   ```python
   try:
       model = manager.load_model(...)
       # Use model
   finally:
       manager.unload_model()
   ```

4. **Check job status after processing**
   ```python
   job = load_job_from_file(job_path)
   if job.status == "failed":
       print(f"Job failed: {job.error_message}")
   ```

## Examples

See `docs/PRROT_QUICK_START.md` for complete usage examples.
