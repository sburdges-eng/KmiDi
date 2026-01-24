# Parrot - Singing Voice Synthesizer

Parrot is a unified singing voice synthesizer with voice mimicking and instrument conversion capabilities.

## Features

- **Singing Synthesis**: Generate singing from lyrics and MIDI melody
- **Voice Mimicking**: Record and mimic user voice characteristics
- **Note Extraction**: Extract MIDI notes from sung audio
- **Instrument Conversion**: Convert sung notes to different instruments (piano, guitar, strings, etc.)
- **Dual Backends**: Formant synthesis (fast preview) and neural synthesis (production quality)

## Quick Start

```python
from music_brain.voice import Parrot, create_parrot

# Create Parrot instance
parrot = create_parrot(backend="formant")

# Sing from lyrics and melody
lyrics = "Hello world"
melody = [60, 62, 64, 65, 64, 62, 60]  # MIDI notes
audio = parrot.sing(lyrics, melody, tempo_bpm=120)

# Save output
parrot.save("output.wav", audio)
```

## Voice Recording and Mimicking

```python
# Record voice
recorded = parrot.record_voice(duration_seconds=3.0)

# Extract voice characteristics
characteristics = parrot.extract_voice(recorded)

# Sing with mimicked voice
audio = parrot.sing_with_voice(
    lyrics="Hello world",
    melody=[60, 62, 64],
    voice_characteristics=characteristics
)
```

## Voice Learning from Samples

```python
# Add voice samples for learning
sample_id1 = parrot.add_voice_sample(audio1, text="Hello world")
sample_id2 = parrot.add_voice_sample(audio2, text="How are you")
sample_id3 = parrot.add_voice_sample(audio3, text="I'm fine")

# Learn a voice profile from samples
profile = parrot.learn_voice_profile("my_voice", [sample_id1, sample_id2, sample_id3])

# Use learned voice for synthesis
audio = parrot.sing_with_learned_voice(
    lyrics="Hello world",
    melody=[60, 62, 64],
    profile_name="my_voice"
)

# Update profile with more samples
new_sample = parrot.add_voice_sample(audio4)
parrot.update_voice_profile("my_voice", [new_sample])

# List learned profiles
profiles = parrot.list_voice_profiles()
```

## Note Extraction and Instrument Conversion

```python
# Extract MIDI notes from sung audio
midi_notes = parrot.extract_notes_from_audio(recorded_audio)

# Convert to different instruments
piano_audio = parrot.notes_to_instrument(midi_notes, instrument="piano")
guitar_audio = parrot.notes_to_instrument(midi_notes, instrument="guitar")
strings_audio = parrot.notes_to_instrument(midi_notes, instrument="strings")
```

## Expression Parameters

```python
expression = {
    "vibrato_rate": 6.0,      # Vibrato frequency (Hz)
    "vibrato_depth": 0.03,    # Vibrato depth (semitones)
    "portamento_time": 0.1,    # Portamento duration (seconds)
    "dynamics": [0.5, 0.8, 1.0, 0.8, 0.5]  # Per-note dynamics
}

audio = parrot.sing(lyrics, melody, tempo_bpm=120, expression=expression)
```

## Available Instruments

- `piano` - Piano
- `guitar` - Acoustic guitar
- `strings` - String ensemble
- `flute` - Flute
- `trumpet` - Trumpet
- `violin` - Violin

## Backends

### Formant Backend (Default)
- Fast synthesis (<100ms)
- Good for previews
- No GPU required
- Demo-grade quality

### Neural Backend (Optional)
- Production-quality synthesis
- Requires DiffSinger installation
- GPU recommended
- Professional quality

## Dependencies

### Required
- `numpy>=1.20`
- `scipy>=1.7`
- `soundfile>=0.10`

### Optional (for better quality)
- `g2p_en>=2.1.0` - Better phoneme conversion
- `librosa>=0.9` - Better pitch detection
- `sounddevice` - Voice recording
- `torch>=2.0` - Neural backend

## Installation

```bash
# Basic dependencies
pip install numpy scipy soundfile

# For better phoneme conversion
pip install g2p_en

# For voice recording
pip install sounddevice

# For neural backend (optional)
# See DiffSinger installation: https://github.com/MoonInTheRiver/DiffSinger
```

## Examples

See `examples/parrot_example.py` for complete usage examples.

## Tests

Run integration tests:
```bash
python tests/test_parrot.py
```

## API Reference

### Parrot Class

#### `__init__(backend="auto", voice_model=None, device="auto", sample_rate=44100)`
Initialize Parrot synthesizer.

#### `sing(lyrics, melody, tempo_bpm=120.0, expression=None, voice_characteristics=None)`
Synthesize singing from lyrics and melody.

#### `preview(lyrics, melody, tempo_bpm=120.0)`
Quick preview (always uses formant backend).

#### `record_voice(duration_seconds=None, until_silence=False)`
Record voice from microphone.

#### `extract_voice(audio)`
Extract voice characteristics from audio.

#### `sing_with_voice(lyrics, melody, voice_characteristics, tempo_bpm=120.0, expression=None)`
Sing with mimicked voice characteristics.

#### `extract_notes_from_audio(audio, note_duration=0.25)`
Extract MIDI notes from sung audio.

#### `notes_to_instrument(midi_notes, instrument="piano", note_durations=None, velocities=None)`
Convert MIDI notes to instrument audio.

#### `save(output_path, audio)`
Save audio to file.

## License

Part of the iDAW Music Brain system.
