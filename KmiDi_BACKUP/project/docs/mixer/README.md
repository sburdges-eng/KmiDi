# iDAW Mixer System

Comprehensive multi-channel mixing system with real-time safe C++ audio engine and cross-platform control interfaces.

## Overview

The iDAW mixer provides professional mixing capabilities across multiple platforms:

- **C++ Penta-Core Engine** (Side A): Real-time safe audio processing
- **JUCE MixerConsolePanel**: DAW-style mixer UI
- **Python Bindings**: High-level control and emotion-based mixing
- **Streamlit Web UI**: Browser-based mixer control
- **HTML/CSS/JS Interface**: Lightweight web mixer
- **OSC Integration**: Cross-platform parameter sync

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interfaces                         │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ JUCE         │ Streamlit    │ Web HTML     │ Logic Pro      │
│ Console      │ Panel        │ Interface    │ Automation     │
└──────┬───────┴──────┬───────┴──────┬───────┴──────┬─────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                      │
       ┌──────────────┴──────────────┐
       │   Python Mixer Bindings     │
       │  (emotion presets, state)   │
       └──────────────┬───────────────┘
                      │
       ┌──────────────┴───────────────┐
       │  Penta-Core Mixer Engine     │
       │  (RT-safe C++ audio)         │
       │                              │
       │  ┌────────────────────────┐  │
       │  │ Channel Strips         │  │
       │  │ Send/Return Buses      │  │
       │  │ Master Bus + Limiter   │  │
       │  └────────────────────────┘  │
       └──────────────────────────────┘
```

## Features

### Channel Strips
- **Gain**: -60 dB to +12 dB range
- **Pan**: Constant power pan law (-1.0 to +1.0)
- **Mute/Solo**: Individual channel control
- **Send Levels**: Up to 8 send buses per channel
- **Metering**: Peak and RMS level monitoring

### Send/Return Buses
- Auxiliary send buses for effects
- Configurable return levels
- Individual bus muting
- Pre-fader sends

### Master Bus
- Master gain control
- Real-time limiter
- Stereo peak metering
- Automatic gain makeup

### Real-Time Safety
- Lock-free parameter updates (atomic operations)
- No memory allocation in audio thread
- Pre-allocated buffers
- SIMD-optimized mixing (where available)

### Emotion-Based Mixing
- Preset system driven by emotional intent
- Maps emotions (grief, anxiety, hope, etc.) to mixer parameters
- Integration with `music_brain/daw/mixer_params.py`

## Usage Examples

### C++ (Penta-Core Engine)

```cpp
#include "penta/mixer/MixerEngine.h"

using namespace penta::mixer;

// Create mixer
MixerEngine mixer(48000.0);
mixer.setNumChannels(8);
mixer.setNumSendBuses(4);

// Configure channels
mixer.setChannelGain(0, -6.0f);   // Drums: -6 dB
mixer.setChannelPan(0, 0.0f);     // Center
mixer.setChannelSend(0, 0, 0.3f); // Send to reverb

mixer.setChannelGain(1, -3.0f);   // Bass: -3 dB
mixer.setChannelPan(1, -0.1f);    // Slightly left

// Configure master
mixer.setMasterGain(0.0f);
mixer.setMasterLimiter(true, -1.0f);

// In audio callback (RT-safe):
const float** inputs = ...;  // [numChannels][numFrames]
float** outputs = ...;        // [2][numFrames] (stereo)

mixer.processAudio(inputs, outputs, numFrames);

// Get metering (non-RT):
float peakL = mixer.getMasterPeakL();
float peakR = mixer.getMasterPeakR();
```

### Python

```python
from penta_core.mixer import MixerEngine
import numpy as np

# Create mixer
mixer = MixerEngine(sample_rate=48000.0)
mixer.set_num_channels(4)
mixer.set_num_send_buses(2)

# Configure channels
mixer.set_channel_gain(0, -6.0)   # dB
mixer.set_channel_pan(0, -0.5)    # Pan left
mixer.set_channel_mute(0, False)
mixer.set_channel_send(0, 0, 0.3) # Send to bus 0

# Configure master
mixer.set_master_gain(0.0)
mixer.set_master_limiter(True, -1.0)

# Process audio
inputs = np.random.randn(4, 1024).astype(np.float32)
output_l, output_r = mixer.process(inputs)

# Get metering
peak_l = mixer.get_master_peak_l()
rms_ch0 = mixer.get_channel_rms(0)
```

### Emotion-Based Mixing (Python)

```python
from penta_core.mixer import MixerEngine, apply_emotion_to_mixer
from music_brain.daw.mixer_params import EmotionMapper

# Create mixer and emotion mapper
mixer = MixerEngine(48000.0)
mixer.set_num_channels(8)

mapper = EmotionMapper()

# Get emotion preset
grief_params = mapper.get_preset("grief")
print(f"Description: {grief_params.description}")
print(f"Justification: {grief_params.emotional_justification}")

# Apply to channel
apply_emotion_to_mixer(mixer, grief_params, channel=0)

# Available presets:
# - grief: Dark, spacious, intimate
# - anxiety: Tight, anxious, hyper-present
# - anger: Aggressive, saturated
# - nostalgia: Warm, dreamy
# - hope: Bright, open, lifting
# - calm: Smooth, warm, gentle
# - tension: Building, unsettled
# - catharsis: Full, releasing
```

### Streamlit Web Interface

```bash
# Run the Streamlit mixer panel
streamlit run music_brain/ui/mixer_panel.py
```

Features:
- Channel faders and pan controls
- Mute/solo buttons
- Send level controls
- Master bus controls
- Emotion preset selector
- Real-time metering
- Session save/load
- Test signal generator

### HTML/CSS/JS Web Interface

```bash
# Serve the web mixer (any HTTP server)
python -m http.server 8000

# Open in browser
open http://localhost:8000/web/mixer.html
```

Features:
- Responsive grid layout
- Visual feedback (hover effects, meters)
- Emotion preset buttons
- Touch-friendly controls
- Gradient/glassmorphism design

## State Management

### Save/Load State (Python)

```python
# Get current state
state = mixer.get_state()

# Access state properties
print(f"Channels: {state.num_channels}")
print(f"Channel 0 gain: {state.channel_gains[0]} dB")
print(f"Channel 0 pan: {state.channel_pans[0]}")

# Load state
mixer.load_state(state)
```

### Session Export (JUCE)

```cpp
// Export mixer session
MixerConsolePanel panel;
juce::File sessionFile("my_mix.session");
panel.exportSession(sessionFile);

// Import session
panel.importSession(sessionFile);
```

## Integration with Logic Pro

The mixer can export automation data for Logic Pro:

```python
from music_brain.daw.mixer_params import export_to_logic_automation

emotion_params = mapper.get_preset("grief")
export_to_logic_automation(emotion_params, "grief_automation.json")
```

Output format is JSON with Logic Pro parameter mappings:
- Channel EQ (7-band)
- Compressor settings
- Reverb/Delay send levels
- Stereo width/pan
- Saturation amount

## OSC Communication

The mixer integrates with the Penta-Core OSC Hub for remote control:

```python
# OSC addresses:
/mixer/channel/{ch}/gain {float db}
/mixer/channel/{ch}/pan {float -1.0 to 1.0}
/mixer/channel/{ch}/mute {bool}
/mixer/channel/{ch}/solo {bool}
/mixer/channel/{ch}/send/{bus} {float 0.0 to 1.0}

/mixer/master/gain {float db}
/mixer/master/limiter/enabled {bool}
/mixer/master/limiter/threshold {float db}
```

## Performance Characteristics

### C++ Engine Benchmarks

Tested on M1 Mac @ 48 kHz:

| Configuration | Avg Processing Time | Max Latency |
|---------------|---------------------|-------------|
| 8 channels, 512 samples | ~120 μs | ~180 μs |
| 16 channels, 512 samples | ~210 μs | ~310 μs |
| 32 channels, 512 samples | ~390 μs | ~520 μs |

**512 samples @ 48 kHz = 10.67 ms available**

All configurations process well under 1ms, leaving plenty of headroom for real-time audio.

### Memory Usage

- **C++ Engine**: Pre-allocated, fixed memory (no RT allocations)
- **Per-channel overhead**: ~256 bytes
- **Total for 64 channels**: ~16 KB
- **No dynamic allocation in audio thread**

## Testing

### Run C++ Tests

```bash
cd build
ctest --output-on-failure -R mixer
```

### Run Python Tests

```bash
pytest tests/test_mixer.py -v
```

Tests cover:
- Channel gain/pan/mute/solo
- Send/return buses
- Master bus and limiter
- Metering accuracy
- State management
- Performance benchmarks
- Edge cases

## File Structure

```
iDAW/
├── include/penta/mixer/
│   └── MixerEngine.h              # C++ mixer engine header
├── src_penta-core/mixer/
│   └── MixerEngine.cpp            # C++ implementation
├── python/penta_core/
│   └── mixer.py                   # Python bindings
├── music_brain/daw/
│   └── mixer_params.py            # Emotion-based presets
├── music_brain/ui/
│   └── mixer_panel.py             # Streamlit UI
├── web/
│   └── mixer.html                 # Web interface
├── src/ui/
│   ├── MixerConsolePanel.h        # JUCE mixer UI
│   └── MixerConsolePanel.cpp
├── tests/
│   ├── test_mixer.cpp             # C++ tests
│   └── test_mixer.py              # Python tests
└── docs/mixer/
    ├── README.md                  # This file
    └── examples/                  # Usage examples
```

## Emotion Preset Reference

### Available Presets

| Preset | Description | Key Parameters |
|--------|-------------|----------------|
| **grief** | Dark, spacious, intimate | -8 dB air, long reverb (3.5s), tape saturation |
| **anxiety** | Tight, anxious, hyper-present | Fast compression (1ms attack), narrow stereo |
| **anger** | Aggressive, saturated | +6 dB presence, heavy saturation, minimal reverb |
| **nostalgia** | Warm, dreamy, vintage | -6 dB air, plate reverb, tape warmth |
| **hope** | Bright, open, lifting | +4 dB air, wide stereo, gentle compression |
| **calm** | Smooth, warm, gentle | Soft knee compression, chamber reverb |
| **tension** | Building, unsettled | +3 dB sub-bass, resonant filter, tight compression |
| **catharsis** | Full, releasing, overwhelming | Maximum stereo width, long reverb, full spectrum |

Each preset includes:
- EQ curve (7-band)
- Compression settings
- Reverb character and decay
- Delay time and feedback
- Stereo width
- Saturation type and amount
- Emotional justification

## Best Practices

### Real-Time Safety
1. Never call `new` or `malloc` in audio thread
2. Use atomic operations for parameter updates
3. Pre-allocate all buffers before audio starts
4. Keep audio thread CPU usage under 50% of available time

### Mixing Guidelines
1. Start with all channels at unity gain (0 dB)
2. Use subtractive mixing (reduce levels, don't boost)
3. Pan similar instruments away from each other
4. Use sends for shared effects (reverb, delay)
5. Apply master limiter to prevent clipping

### Emotion-Based Mixing
1. Choose emotion first, then adjust technical details
2. Trust the preset justifications
3. Customize presets for your specific needs
4. Document why you chose specific emotional intent

## Troubleshooting

### Audio Dropouts
- Reduce number of channels
- Increase buffer size
- Check CPU usage with PerformanceMonitor
- Ensure no allocations in audio thread

### Parameter Updates Not Working
- Verify channel index is in range
- Check atomic ordering (should be relaxed for audio params)
- Confirm mixer initialized with correct channel count

### Metering Shows Zero
- Process at least one buffer first
- Check input signal is not silent
- Verify channel not muted
- Ensure meters not reset between reads

## Future Enhancements

Planned features:
- [ ] EQ and dynamics processing per channel
- [ ] Effect inserts (pre/post fader)
- [ ] Automation recording and playback
- [ ] Snapshot system with A/B comparison
- [ ] MIDI learn for hardware control
- [ ] VST/AU plugin hosting in sends
- [ ] Surround sound support (5.1, 7.1, Atmos)
- [ ] Sidechain routing
- [ ] Parallel compression buses

## Contributing

When adding features to the mixer:

1. **Maintain RT-safety**: No allocations in `processAudio()`
2. **Add tests**: Both C++ and Python test coverage
3. **Update documentation**: Keep this README current
4. **Follow conventions**: Match existing code style
5. **Benchmark**: Ensure new features don't degrade performance

## License

Part of the iDAW project. See main LICENSE file.

## Credits

- Architecture inspired by professional DAW mixers (Pro Tools, Logic, Ableton)
- Emotion-based mixing concept: "Interrogate Before Generate" philosophy
- Real-time safe design patterns from JUCE framework
- Constant power pan law implementation

---

**"The audience doesn't hear 'panned 30% left.' They hear 'that part made me cry.'"**
