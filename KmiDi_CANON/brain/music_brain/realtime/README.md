# Real-time MIDI Processing

Real-time MIDI input/output processing for live performance and interactive music generation.

## Features

- **MIDI Input Capture**: Listen to MIDI input from hardware controllers or software
- **Real-time Chord Detection**: Detect chords from live MIDI input
- **Real-time Groove Analysis**: Analyze timing and velocity patterns in real-time
- **MIDI Transformation**: Transform MIDI messages (transpose, velocity scaling, etc.)
- **MIDI Output Routing**: Send processed MIDI to hardware/software destinations

## Quick Start

```python
from music_brain.realtime import RealtimeMidiProcessor, MidiProcessorConfig

# Configure processor
config = MidiProcessorConfig(
    input_port_name=None,  # Auto-select first available
    enable_chord_detection=True,
    enable_groove_analysis=True,
)

# Create processor
processor = RealtimeMidiProcessor(config)

# Set callbacks
def on_chord(chord, notes):
    if chord:
        print(f"Chord: {chord.name}")

processor.set_chord_callback(on_chord)

# Start processing
processor.start()

# Keep running...
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    processor.stop()
```

## MIDI Transformers

Transform MIDI messages in real-time:

```python
from music_brain.realtime import create_transpose_transformer

# Transpose all notes up 5 semitones
transformer = create_transpose_transformer(semitones=5)
processor.set_transform_callback(transformer)
processor.config.enable_transformation = True
```

Available transformers:
- `create_transpose_transformer()` - Transpose notes
- `create_velocity_scale_transformer()` - Scale velocities
- `create_humanize_transformer()` - Add humanization
- `create_channel_router_transformer()` - Route to specific channel
- `create_filter_transformer()` - Filter by note/velocity range

## Example

See `examples/realtime_midi_demo.py` for a complete example.

## Requirements

- `mido>=1.2.10` - MIDI I/O library
- MIDI input device or virtual MIDI port

## Platform Notes

### macOS
- Use IAC (Inter-Application Communication) driver for virtual MIDI ports
- Hardware MIDI devices appear automatically

### Linux
- Use ALSA MIDI ports
- May need to install `python-rtmidi` for better port support

### Windows
- Use Windows MIDI API
- Virtual MIDI ports available via third-party software

