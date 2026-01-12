# Voice Learning System

Parrot can learn voice characteristics from voice samples and use them for synthesis.

## Overview

The voice learning system allows you to:
1. **Store voice samples** - Record or import voice audio
2. **Extract features** - Analyze voice characteristics
3. **Learn profiles** - Build voice models from multiple samples
4. **Use learned voices** - Synthesize with learned characteristics
5. **Update profiles** - Improve profiles with more samples

## Quick Start

```python
from music_brain.voice import Parrot

parrot = Parrot()

# Record voice samples
sample1 = parrot.add_voice_sample(parrot.record_voice(3.0), text="Hello")
sample2 = parrot.add_voice_sample(parrot.record_voice(3.0), text="World")

# Learn voice profile
profile = parrot.learn_voice_profile("my_voice", [sample1, sample2])

# Use learned voice
audio = parrot.sing_with_learned_voice(
    lyrics="Hello world",
    melody=[60, 62, 64],
    profile_name="my_voice"
)
```

## Features

### Voice Sample Storage

Samples are stored in `~/.parrot/voice_samples/samples/`:

- Audio files: `{sample_id}.wav`
- Metadata: `{sample_id}.json`

### Feature Extraction

The system extracts:

- **Pitch characteristics**: Mean pitch, range, variance
- **Spectral features**: Brightness, rolloff, MFCCs
- **Voice quality**: Breathiness, formant emphasis
- **Temporal features**: Duration, energy, zero-crossing rate

### Profile Learning

Profiles aggregate features from multiple samples:

- Weighted averaging by sample count
- Incremental updates supported
- Profiles saved to `~/.parrot/voice_samples/profiles/`

## API Reference

### Adding Samples

```python
sample_id = parrot.add_voice_sample(
    audio,                    # Audio array
    sample_id=None,          # Auto-generated if None
    text=None,               # Optional transcript
    metadata=None            # Optional metadata dict
)
```

### Learning Profiles

```python
profile = parrot.learn_voice_profile(
    profile_name,            # Name for profile
    sample_ids=None          # List of IDs, or None for all
)
```

### Using Learned Voices

```python
# Method 1: Direct use
audio = parrot.sing_with_learned_voice(
    lyrics="Hello",
    melody=[60, 62, 64],
    profile_name="my_voice"
)

# Method 2: Get characteristics and use
characteristics = parrot.load_voice_profile("my_voice")
audio = parrot.sing_with_voice(lyrics, melody, characteristics)
```

### Profile Management

```python
# List profiles
profiles = parrot.list_voice_profiles()

# Update profile
updated = parrot.update_voice_profile("my_voice", [new_sample_id1, new_sample_id2])

# Load profile characteristics
characteristics = parrot.load_voice_profile("my_voice")
```

## Storage Structure

```
~/.parrot/voice_samples/
├── samples/
│   ├── sample_20241218_123456_123456.wav
│   ├── sample_20241218_123456_123456.json
│   └── ...
└── profiles/
    ├── my_voice.json
    └── ...
```

## Best Practices

1. **Sample Quality**: Use clear recordings (3-10 seconds each)
2. **Sample Variety**: Include different pitches, emotions, styles
3. **Sample Count**: 3-10 samples per profile recommended
4. **Transcripts**: Include text transcripts for better alignment
5. **Metadata**: Add metadata (emotion, style) for organization

## Example Workflow

```python
# 1. Record multiple samples
samples = []
for i in range(5):
    print(f"Recording sample {i+1}/5...")
    audio = parrot.record_voice(3.0)
    sample_id = parrot.add_voice_sample(
        audio,
        text=f"Sample {i+1}",
        metadata={"recording_session": "session_1"}
    )
    samples.append(sample_id)

# 2. Learn profile
profile = parrot.learn_voice_profile("my_voice", samples)
print(f"Learned from {profile.sample_count} samples")

# 3. Use for synthesis
audio = parrot.sing_with_learned_voice(
    lyrics="This is my learned voice",
    melody=[60, 62, 64, 65, 67],
    profile_name="my_voice"
)

# 4. Improve with more samples
new_sample = parrot.add_voice_sample(parrot.record_voice(3.0))
parrot.update_voice_profile("my_voice", [new_sample])
```

## Advanced Usage

### Direct Access to Learning Manager

```python
from music_brain.voice import VoiceLearningManager

manager = VoiceLearningManager()

# Add sample
sample_id = manager.add_sample(audio, text="Hello")

# Learn profile
profile = manager.learn_profile("my_voice", [sample_id])

# Get characteristics
characteristics = manager.get_profile_characteristics("my_voice")
```

### Feature Extraction

```python
from music_brain.voice import VoiceLearner

learner = VoiceLearner()
features = learner.extract_features(audio)

# Features include:
# - mean_pitch, pitch_range, pitch_variance
# - brightness, spectral_centroid_mean
# - mfcc_mean, mfcc_std
# - chroma_mean
# - breathiness, formant_emphasis
```

## Limitations

- Current implementation uses statistical features (not neural voice cloning)
- Quality depends on sample quality and quantity
- Best results with 5-10 diverse samples
- Real-time voice cloning requires neural models (future enhancement)

## Future Enhancements

- Neural voice cloning integration
- Real-time voice adaptation
- Multi-speaker profiles
- Emotion/style-specific profiles
- Automatic sample quality assessment
