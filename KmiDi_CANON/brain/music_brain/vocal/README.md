# Parrot Vocal Synthesizer

The **Parrot** function is a vocal synthesizer that learns voice characteristics from uploaded audio and can mimic voices after prolonged exposure.

## Features

- **Voice Learning**: Analyzes uploaded audio to learn:
  - Vowel formants (F1, F2, F3) and transitions
  - Accent characteristics
  - Pitch contours and vibrato
  - Timbre and spectral characteristics
  - Speaking/singing style

- **Adaptive Learning**: Improves mimicry accuracy with more exposure time
- **Voice Synthesis**: Generates vocal audio using learned voice models
- **Multiple Voices**: Train and store multiple voice models

## How It Works

### Learning Process

1. **Formant Analysis**: Detects vowel formants (F1, F2, F3) to identify vowel types
2. **Pitch Analysis**: Extracts pitch contours, vibrato rate, and depth
3. **Timbre Analysis**: Analyzes spectral centroid, rolloff, and bandwidth
4. **Accent Detection**: Learns vowel shifts and intonation patterns
5. **Progressive Learning**: Updates model with weighted averages as more audio is analyzed

### Confidence System

- **Minimum Exposure**: 30 seconds of audio required for basic learning
- **Confidence Score**: Increases with exposure time (0.0-1.0)
- **Threshold**: 0.7 confidence required for synthesis

## Usage

### CLI Commands

```bash
# Train Parrot on a voice
daiw parrot train <audio_file> --name <voice_name> [-o model.json]

# Synthesize vocal audio
daiw parrot synthesize <text> --voice <voice_name> [-o output.wav]

# List trained voices
daiw parrot list

# Show voice model info
daiw parrot info <voice_name>
```

### Python API

```python
from music_brain.vocal import ParrotVocalSynthesizer, train_parrot, synthesize_vocal

# Train on audio
model = train_parrot("voice_sample.wav", "my_voice")

# Synthesize
audio = synthesize_vocal("Hello world", "my_voice", "output.wav")

# Or use the class directly
parrot = ParrotVocalSynthesizer()
parrot.train_parrot("voice_sample.wav", "my_voice")
audio = parrot.synthesize_vocal("Hello world", "my_voice")
```

## Voice Characteristics Learned

### Formant Data
- **F1 (First Formant)**: Vowel height (low = high vowel, high = low vowel)
- **F2 (Second Formant)**: Vowel frontness (high = front, low = back)
- **F3 (Third Formant)**: Vowel rounding

### Vowel Classification
- A (ah) - Low, central
- E (eh) - Mid, front
- I (ee) - High, front
- O (oh) - Mid, back
- U (oo) - High, back
- Schwa (uh) - Mid, central

### Pitch Characteristics
- Average pitch (Hz)
- Pitch range (min/max Hz)
- Vibrato rate (Hz)
- Vibrato depth (cents)

### Timbre Characteristics
- Spectral centroid (brightness)
- Spectral rolloff (high-frequency content)
- Spectral bandwidth (timbre width)

### Accent Characteristics
- Vowel formant shifts from standard
- Pitch range and intonation patterns
- Rhythm timing characteristics
- Consonant emphasis

## Learning Parameters

```python
from music_brain.vocal import ParrotConfig, ParrotVocalSynthesizer

config = ParrotConfig(
    min_exposure_time=30.0,      # Minimum seconds to learn
    learning_rate=0.1,            # Adaptation speed (0.0-1.0)
    confidence_threshold=0.7,     # Minimum confidence for synthesis
    formant_window_size=0.025,    # Analysis window (seconds)
    pitch_hop_length=512,         # Pitch detection hop length
    vowel_detection_threshold=0.6 # Vowel detection confidence
)

parrot = ParrotVocalSynthesizer(config=config)
```

## Progressive Learning

Parrot improves with prolonged exposure:

1. **Initial Learning** (< 30s): Low confidence, basic characteristics
2. **Building Confidence** (30-60s): Improving accuracy
3. **High Confidence** (> 60s): Full mimicry capability

The learning rate controls how quickly the model adapts:
- **Low (0.05)**: Slow adaptation, stable model
- **Medium (0.1)**: Balanced learning
- **High (0.3)**: Fast adaptation, may be less stable

## Requirements

- `librosa` - Audio analysis
- `soundfile` - Audio I/O
- `numpy` - Numerical operations
- `scipy` - Signal processing

Install with:
```bash
pip install librosa soundfile numpy scipy
```

## Limitations

- Current synthesis is simplified (formant synthesis would be more realistic)
- Requires sufficient audio data for good results
- Works best with clear, single-voice audio
- Accent learning improves with diverse vowel samples

## Enhanced Features (v2.0)

### ✅ Implemented

- **Full Formant Synthesis**: Proper formant-based vocal synthesis using learned voice characteristics
- **Phoneme Conversion**: Text-to-phoneme conversion with stress markers and timing
- **Prosody Analysis**: Rhythm, intonation, and speaking rate detection
- **Voice Quality Analysis**: Jitter, shimmer, and harmonic-to-noise ratio
- **Emotion Control**: Emotion presets (happy, sad, angry, excited, calm) with formant/pitch modifications
- **Voice Blending**: Blend two voice models together with adjustable ratio
- **Batch Training**: Train on multiple audio files simultaneously
- **Enhanced Learning**: Progressive learning with jitter/shimmer and prosody analysis

### Advanced Capabilities

- **Formant Synthesis Engine**: Uses proper formant filters (F1, F2, F3) for realistic vowel synthesis
- **Glottal Pulse Model**: Rosenberg glottal pulse model for natural excitation
- **Consonant Synthesis**: Noise-based fricatives, burst plosives, and formant-like nasals
- **Prosody Application**: Applies learned rhythm and intonation patterns to synthesized speech
- **Emotion Expression**: Modifies formants and pitch based on emotion presets
- **Voice Morphing**: Blend characteristics from multiple voices

## Future Enhancements

- Real-time voice morphing
- Advanced phoneme dictionary (CMU Pronouncing Dictionary integration)
- Neural vocoder integration for even more realistic synthesis
- Multi-language support
- Singing voice synthesis

