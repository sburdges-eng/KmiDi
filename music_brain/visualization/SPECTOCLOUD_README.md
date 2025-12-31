# Spectocloud: 3D Song Lifeline Visualization

A 3D visualization system that shows **what musical "space" a song occupies over time**, with **emotions accumulating and "sticking"** to musically meaningful regions through a snowball/memory effect.

## Overview

Spectocloud fuses concepts from:
- **MIDI track (1D)**: discrete musical events over time
- **Spectrogram (2D)**: continuous energy distribution over time and frequency
- **Combined 3D**: time + frequency + musically-derived parameters with emotional state overlays

## Core Concepts

### 1. Hard-Truth Musical Parameters = Stationary Anchors

Fixed points in 3D space representing **musical parameter subsets**. These serve as:
- A coordinate system for musical meaning
- Attractors for emotional "adhesion"
- A stable backdrop for interpretable song paths

**Key rule:** Anchors are **stationary** and **consistent across songs** (same parameter definition, same scale).

### 2. Song Lifeline = Moving Trajectory (Time-Ordered)

A time-indexed path through the space. Each time slice (frame) yields a point or small cluster.

The lifeline follows a **conical expansion**: early frames are compact, later frames have wider spread, modeling how musical and emotional complexity builds over time.

### 3. Emotional Adhesion = Sticky Accumulation

Emotional markers are emitted over time and attracted to subsets of anchors.

**Stickiness** means:
- Once emotion attaches to an anchor subset, it tends to remain there
- Attachment strengthens with repeated evidence
- Transitions are possible but resist sudden changes (hysteresis)

### 4. Electrostatic Storm Model

Anchors create an **influence field** that the emotional cloud moves through:
- **Storm energy** accumulates from proximity to activated anchors
- **Charge** builds up in a capacitor-like model
- When charge exceeds threshold, **lightning events** occur (visual discharge)

## Coordinate System

### 3D Axes

- **X = Time** (beats or seconds) - the lifeline's chronological progression
- **Y = Frequency/Pitch** (spectral centroid or pitch class)
- **Z = Musical Parameter Composite** (weighted combination of multiple features)

### Additional Visual Channels

- **Color**: Emotion valence (blue=negative, red=positive via coolwarm colormap)
- **Size**: Energy/confidence
- **Opacity**: Charge accumulation and arousal
- **Shape**: Different markers for anchors vs particles

## Musical Parameter Families

Anchors are organized into families:

1. **Harmony/Tonality**: Key proximity, mode, chord complexity, tension
2. **Rhythm/Groove**: Note density, syncopation, swing, regularity
3. **Dynamics/Energy**: RMS energy, crest factor, velocity
4. **Timbre/Spectral**: Spectral centroid, flux, roll-off
5. **Texture/Structure**: Polyphony, layer count, repetition

## Installation

```bash
# Install core dependencies
pip install numpy matplotlib

# Or install from the KmiDi project
pip install -e .
```

## Quick Start

### Basic Static Visualization

```python
from music_brain.visualization import Spectocloud

# Create visualization engine
spectocloud = Spectocloud(
    anchor_density="normal",  # sparse, normal, or dense
    n_particles=1500,
    window_size=0.2,  # seconds
)

# Process MIDI events
midi_events = [
    {'time': 0.0, 'type': 'note_on', 'note': 60, 'velocity': 64},
    {'time': 0.5, 'type': 'note_on', 'note': 64, 'velocity': 70},
    # ... more events
]

spectocloud.process_midi(
    midi_events=midi_events,
    duration=10.0,
)

# Render a frame
spectocloud.render_static_frame(
    frame_idx=25,
    output_path="spectocloud_frame.png",
    show=True,
)
```

### With Emotional Trajectory

```python
# Define emotional journey
emotion_trajectory = [
    {'valence': -0.7, 'arousal': 0.3, 'intensity': 0.6},  # Grief
    {'valence': -0.2, 'arousal': 0.7, 'intensity': 0.8},  # Longing
    {'valence': 0.6, 'arousal': 0.9, 'intensity': 0.9},   # Hope
]

spectocloud.process_midi(
    midi_events=midi_events,
    duration=10.0,
    emotion_trajectory=emotion_trajectory,
)
```

### Animation/GIF Export

```python
# Render animated GIF
spectocloud.render_animation(
    output_path="spectocloud_journey.gif",
    fps=15,
    rotate=True,  # Rotating camera view
)
```

## Examples

### Basic Example

```bash
cd examples
PYTHONPATH=/home/runner/work/KmiDi/KmiDi:$PYTHONPATH python spectocloud_example.py
```

Generates:
- `/tmp/spectocloud_frame_0.png` (beginning)
- `/tmp/spectocloud_frame_mid.png` (middle)
- `/tmp/spectocloud_frame_end.png` (end)
- `/tmp/spectocloud_data.json` (raw data)

### Animation Example

```bash
PYTHONPATH=/home/runner/work/KmiDi/KmiDi:$PYTHONPATH python spectocloud_animation.py
```

Generates:
- `/tmp/spectocloud_journey.gif` (animated emotional journey)

## Visual Interpretation Guide

### What You See

1. **Gray dots (mostly invisible)**: Stationary anchors in musical parameter space
2. **Activated anchors**: Glow when the song approaches their parameter region
3. **Colored cloud**: Emotion particles distributed around current musical state
   - Blue = negative valence (grief, sadness)
   - White/gray = neutral valence
   - Red = positive valence (joy, euphoria)
4. **Gray trail**: Song's path through musical space over time
5. **Center point**: Current frame position (black outline)

### How the Cloud Behaves

- **Early in song**: Small, compact cloud (low spread)
- **As song progresses**: Conical expansion (spread grows)
- **High arousal**: Cloud expands more, higher opacity
- **Anchor activation**: Nearby anchors glow and brighten
- **Storm charge**: Accumulates from repeated anchor proximity

### Color Coding

- **Valence** (emotional tone):
  - `-1.0` = Deep blue (very negative)
  - `0.0` = Neutral gray/white
  - `+1.0` = Deep red (very positive)
  
- **Arousal** (activation level):
  - Controls opacity and spread
  - Higher arousal = more visible, wider cloud

## Architecture

### Key Classes

- **`Spectocloud`**: Main visualization engine
- **`AnchorLibrary`**: Manages stationary anchor points
- **`MusicalParameterExtractor`**: Extracts features from MIDI/audio
- **`SpectocloudRenderer`**: CPU-based matplotlib renderer
- **`Frame`**: Time window state (position, features, emotion)
- **`StormState`**: Charge accumulator for lightning dynamics

### Processing Pipeline

```
MIDI/Audio Input
      ↓
Feature Extraction (per window)
      ↓
Anchor Similarity Matching
      ↓
Frame Generation (position + emotion)
      ↓
Storm State Update
      ↓
Particle Cloud Generation
      ↓
3D Rendering (matplotlib)
```

## Configuration Options

### Anchor Density

- **`sparse`**: 50-200 anchors (fast, less detail)
- **`normal`**: 100-150 anchors (balanced)
- **`dense`**: 150-250 anchors (detailed, slower)

### Particle Count

- **500-1000**: Fast rendering, good for prototyping
- **1500-2500**: Good balance (default)
- **5000+**: Detailed but slower

### Window Size

- **0.1s**: High temporal resolution, more frames
- **0.2s**: Balanced (default)
- **0.5s**: Coarser, fewer frames

## Performance Notes

### CPU Rendering (Current)

- Suitable for: offline rendering, GIF export, prototyping
- Frame rate: 1-5 FPS for interactive display
- Scales to: ~50 frames, 2500 particles

### Future GPU Rendering

- WebGL/three.js: Real-time in browser
- Unity/Unreal: High-fidelity, interactive exploration
- Point cloud instancing: 100K+ particles

## Design Principles

1. **Stationary anchors**: Consistent musical space across songs
2. **Conical lifeline**: Expansion models complexity accumulation
3. **Emotion as physics**: Valence/arousal modify influence field
4. **Readability first**: Avoid glitter fog; use opacity + LOD
5. **Time is sacred**: X-axis always represents chronological progression

## Integration Points

### With Chord Predictor

```python
# Use chord predictions to inform anchor activation
chord_sequence = chord_predictor.predict(midi_events)

# Segment song into blocks
blocks = segment_by_chords(chord_sequence)

# Cache and reuse block templates for repeated sections
```

### With Music Brain API

```python
# Get emotion from Music Brain
response = requests.post('http://localhost:8000/emotions', json=intent)
emotion_trajectory = response.json()['trajectory']

# Feed to Spectocloud
spectocloud.process_midi(midi_events, duration, emotion_trajectory)
```

## Troubleshooting

### "matplotlib not available"

```bash
pip install matplotlib
```

### "No frames to render"

Call `process_midi()` before rendering:

```python
spectocloud.process_midi(midi_events, duration)
spectocloud.render_static_frame(0)
```

### Animation is too large

Reduce FPS or particle count:

```python
spectocloud.render_animation(output_path, fps=10)  # Lower FPS
# or
spectocloud = Spectocloud(n_particles=800)  # Fewer particles
```

## References

- Issue spec: `Spectocloud` (3D Song Lifeline Visualization)
- Design constraint: Stationary "hard truth" parameter anchors
- Emotional model: VAD (Valence-Arousal-Dominance)
- Storm dynamics: Electrostatic field with capacitor discharge

## Future Enhancements

- [ ] GPU acceleration (WebGL, Unity)
- [ ] Real-time audio input
- [ ] Interactive controls (filter families, zoom, time scrubber)
- [ ] Block-based caching for repeated sections
- [ ] Volumetric rendering (marching cubes, isosurfaces)
- [ ] Multi-stem visualization (parallel lifelines)
- [ ] Memory map mode (where emotions ended up)

## License

Part of the KmiDi project. See main LICENSE file.
