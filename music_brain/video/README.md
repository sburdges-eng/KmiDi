# Music Video Generation Module

**Status:** Stub Implementation  
**Version:** 0.1.0  
**Integration:** Unreal Engine + Jespa

## Overview

The `music_brain.video` module provides stubs for future music video generation capabilities. It will enable automated creation of emotion-driven music videos by integrating:

- **Unreal Engine**: High-quality 3D rendering and scene composition
- **Jespa**: Video effects, transitions, and post-processing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VideoGenerator                           │
│  Main orchestrator coordinating all components              │
└────────────┬────────────────────────────────────────────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌────────────┐  ┌────────────┐
│ Unreal     │  │  Jespa     │
│ Bridge     │  │ Connector  │
│            │  │            │
│ - Scenes   │  │ - Effects  │
│ - Lighting │  │ - Filters  │
│ - Camera   │  │ - Trans.   │
└────────────┘  └────────────┘
     ▲                ▲
     │                │
     └───────┬────────┘
             │
    ┌────────┴─────────┐
    │                  │
    ▼                  ▼
┌──────────────┐  ┌──────────────┐
│ Emotion      │  │ Scene        │
│ Visual       │  │ Composer     │
│ Mapper       │  │              │
└──────────────┘  └──────────────┘
```

## Components

### 1. VideoGenerator (`video_generator.py`)

Main orchestrator that coordinates the entire video generation process.

**Key Features:**
- Generate videos from emotions
- Generate videos from complete song intents
- Real-time preview support
- Configurable quality and format settings

**Usage:**
```python
from music_brain.video import VideoGenerator, VideoConfig, VideoQuality

gen = VideoGenerator()
result = gen.generate_from_emotion(
    emotion="grief",
    intensity=0.8,
    music_path="song.wav"
)
```

### 2. UnrealBridge (`unreal_bridge.py`)

Integration layer with Unreal Engine for 3D rendering.

**Key Features:**
- Remote control API communication
- Scene parameter updates
- Camera control
- Lighting and effects management
- Frame/sequence rendering

**Future Implementation:**
- HTTP/WebSocket connection to Unreal Remote Control
- Scene asset loading
- Real-time parameter updates
- High-quality rendering via Movie Render Queue

### 3. JespaConnector (`jespa_connector.py`)

Integration with Jespa for video effects and post-processing.

**Key Features:**
- Effect pipeline management
- Color grading
- Visual effects (blur, glow, distortion, etc.)
- Transitions between scenes
- Frame and video processing

**Future Implementation:**
- REST/WebSocket API to Jespa server
- GPU-accelerated processing
- Real-time effect preview
- Batch processing support

### 4. EmotionVisualMapper (`emotion_visual_mapper.py`)

Maps emotional states to visual parameters using color psychology and film theory.

**Key Features:**
- Emotion-to-color mappings
- Motion characteristic mapping
- Visual style application
- Emotion blending
- Custom mapping support

**Emotion Mappings (Examples):**
- **Joy**: Bright yellow (1.0, 0.9, 0.3), fast motion, high brightness
- **Grief**: Deep blue-grey (0.2, 0.2, 0.3), slow motion, low contrast
- **Anger**: Intense red (0.9, 0.2, 0.1), jerky motion, high contrast
- **Peace**: Mint/aqua (0.6, 0.8, 0.7), smooth motion, balanced

### 5. SceneComposer (`scene_composer.py`)

Composes video scenes from musical structure and emotional content.

**Key Features:**
- Musical section analysis (intro, verse, chorus, etc.)
- Scene timeline generation
- Transition creation
- Beat synchronization
- Visual continuity management

**Usage:**
```python
from music_brain.video import SceneComposer

composer = SceneComposer()
scenes = composer.compose_from_structure(
    structure=[
        ("intro", 0, 8),
        ("verse", 8, 24),
        ("chorus", 24, 40),
    ],
    emotion="grief",
    intensity=0.8
)
```

## Configuration

### VideoConfig

```python
from music_brain.video import VideoConfig, VideoFormat, VideoQuality

config = VideoConfig(
    output_path=Path("output.mp4"),
    format=VideoFormat.MP4,
    quality=VideoQuality.HIGH,
    width=1920,
    height=1080,
    fps=60,
    use_unreal=True,
    use_jespa=True,
    use_gpu=True,
)
```

### UnrealConfig

```python
from music_brain.video import UnrealConfig, UnrealRenderMode

config = UnrealConfig(
    host="localhost",
    port=6969,
    render_mode=UnrealRenderMode.MOVIE_RENDER,
    enable_ray_tracing=True,
    default_scene="/Game/Scenes/EmotionDriven/Default"
)
```

### JespaConfig

```python
from music_brain.video import JespaConfig

config = JespaConfig(
    host="localhost",
    port=8080,
    use_gpu_acceleration=True,
    output_quality="high"
)
```

## Testing

Run the test suite:

```bash
pytest tests/unit/test_video_generation.py -v
```

All 35 tests currently passing with stub implementations.

## Examples

See `examples/video_generation_example.py` for comprehensive usage examples:

```bash
cd /home/runner/work/KmiDi/KmiDi
PYTHONPATH=.:$PYTHONPATH python3 examples/video_generation_example.py
```

## Future Implementation Plan

### Phase 1: Unreal Engine Integration
- [ ] Implement Unreal Remote Control API client
- [ ] Scene loading and asset management
- [ ] Parameter updating (lighting, camera, effects)
- [ ] Single frame rendering
- [ ] Sequence rendering with Movie Render Queue

### Phase 2: Jespa Integration
- [ ] Implement Jespa API client
- [ ] Effect pipeline implementation
- [ ] Color grading and filters
- [ ] Transition effects
- [ ] Video processing and export

### Phase 3: Emotion Mapping
- [ ] Sophisticated emotion-color mappings
- [ ] Cultural context support
- [ ] Emotion blending algorithms
- [ ] Dynamic parameter curves
- [ ] Custom mapping presets

### Phase 4: Scene Composition
- [ ] Musical structure analysis integration
- [ ] Beat detection and synchronization
- [ ] Intelligent transition selection
- [ ] Visual narrative flow
- [ ] Timeline export formats

### Phase 5: Integration
- [ ] Full workflow integration
- [ ] Real-time preview support
- [ ] Batch processing
- [ ] Performance optimization
- [ ] Cloud rendering support

## Dependencies (Future)

When implemented, this module will require:

- Unreal Engine 5.x with Remote Control plugin
- Jespa video processing server
- Python packages:
  - `requests` or `aiohttp` for API communication
  - `websockets` for real-time communication
  - `pillow` for image processing
  - `numpy` for numerical operations

## Design Principles

1. **Emotional Coherence**: Visual parameters must align with emotional intent
2. **Musical Sync**: Scene changes and effects sync with musical structure
3. **Cultural Awareness**: Color/visual mappings respect cultural contexts
4. **Human Imperfection**: Embrace timing drift and organic feel (KmiDi philosophy)
5. **Interrogate Before Generate**: Always question visual choices ("why this color?")

## Integration with Music Brain

The video module integrates with existing music_brain components:

- **Emotion Module**: Uses emotion classifications and thesaurus
- **Session/Intent**: Processes complete song intents for video generation
- **Production**: Aligns visual production with audio production decisions
- **Groove/Feel**: Syncs visual tempo and movement to musical feel

## Notes

- This is currently a **stub implementation**
- All methods return placeholder values or raise NotImplementedError
- The architecture and API are designed to be stable
- Future implementations will fill in the stub methods
- Tests validate the API surface and data structures

## Contact

For questions about video generation integration, see:
- Main docs: `/docs/ARCHITECTURE.md`
- KmiDi philosophy: Custom instructions in workspace
- Emotion system: `music_brain/emotion/`

---

**Last Updated**: 2026-01-28  
**Stub Version**: 0.1.0  
**Maintainer**: KmiDi Development Team
