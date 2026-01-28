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
- **Regularized embedding prediction** (NEW)
- **Fast inference with caching** (NEW)

**Regularization Benefits:**
- More accurate predictions through learned embeddings
- Prevents overfitting to training data
- Better generalization to new emotions
- Optional L1/L2/Elastic Net regularization
- Dropout for training robustness

**Emotion Mappings (Examples):**
- **Joy**: Bright yellow (1.0, 0.9, 0.3), fast motion, high brightness
- **Grief**: Deep blue-grey (0.2, 0.2, 0.3), slow motion, low contrast
- **Anger**: Intense red (0.9, 0.2, 0.1), jerky motion, high contrast
- **Peace**: Mint/aqua (0.6, 0.8, 0.7), smooth motion, balanced

**Usage with Regularization:**
```python
from music_brain.video import EmotionVisualMapper

# Enable regularization for better accuracy
mapper = EmotionVisualMapper(
    use_regularization=True,
    regularization_strength=0.001
)

params = mapper.map_emotion("grief", intensity=0.8)

# Get performance stats
stats = mapper.get_regularization_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

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

### 6. EmbeddingRegularizer (`embedding_regularization.py`)

Provides regularization for emotion-visual embeddings to improve prediction accuracy.

**Key Features:**
- L1, L2, and Elastic Net regularization
- Dropout for training robustness
- Gradient clipping
- Fast inference mode (skips regularization overhead)
- Embedding caching for 2-5x speedup
- Compatible with C++ RT-safe architecture

**Regularization Types:**
- **L2 (Ridge)**: Prevents large weights, smooth predictions
- **L1 (Lasso)**: Sparse weights, feature selection
- **Elastic Net**: Combination of L1 and L2
- **Dropout**: Random deactivation during training

**Performance Optimizations:**
- Fast inference path: 0ms overhead (skips regularization)
- Embedding cache: Instant lookup for repeated emotions
- Batch normalization for stable training
- Early stopping to prevent unnecessary computation
- Optional 8-bit quantization

**Usage:**
```python
from music_brain.video import (
    create_regularized_mapper,
    FastEmbeddingPredictor
)

# Create regularized configuration
config, regularizer = create_regularized_mapper(
    regularization_strength=0.001,
    use_dropout=False,  # Inference only
    use_fast_inference=True
)

# Create predictor
predictor = FastEmbeddingPredictor(config)

# Predict with caching
embedding = predictor.predict("grief", intensity=0.8)

# Get performance stats
stats = predictor.get_performance_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

### 7. ONNXModelExporter (`onnx_exporter.py`) **NEW**

Exports trained emotion-visual models to ONNX format for deployment in Unreal Engine.

**Key Features:**
- PyTorch → ONNX conversion
- TensorFlow/Keras → ONNX conversion
- NumPy weights → ONNX graph
- Model optimization (constant folding, operator fusion)
- INT8 quantization for faster inference
- ONNX validation and model inspection
- Compatible with UE5 NNI plugin (opset 17)

**Usage:**
```python
from music_brain.video import export_emotion_visual_model

# Export PyTorch/TensorFlow model to ONNX
export_emotion_visual_model(
    trained_model,
    output_dir=Path("Content/Models"),
    model_name="EmotionMapper"
)

# Or use exporter directly for more control
from music_brain.video import ONNXModelExporter, ONNXExportConfig

config = ONNXExportConfig(
    input_dim=128,
    output_dim=256,
    opset_version=17,
    optimize=True,
    quantize=True
)
exporter = ONNXModelExporter(config)
exporter.export_pytorch_model(model, "emotion_mapper.onnx")
```

### 8. UnrealNNIIntegration (`unreal_nni.py`) **NEW**

Integration with Unreal Engine 5's Neural Network Inference (NNI) plugin for running ONNX models.

**Based on:** https://github.com/microsoft/OnnxRuntime-UnrealEngine

**Key Features:**
- Deploy ONNX models to Unreal Engine projects
- NNI plugin configuration (CPU/DirectX12 GPU)
- Blueprint wrapper generation
- C++ wrapper generation
- Model loading and inference via Remote Control API
- Performance monitoring

**Supported Backends:**
- **CPU**: Cross-platform (Windows/Linux/Mac/Consoles)
- **DirectX 12 GPU**: Windows only
- **AUTO**: Automatic selection

**Usage:**
```python
from music_brain.video import UnrealNNIIntegration

# Initialize integration
nni = UnrealNNIIntegration(
    project_path=Path("/path/to/UE5Project")
)

# Deploy ONNX model
nni.deploy_model(
    onnx_path=Path("emotion_mapper.onnx"),
    model_name="EmotionMapper"
)

# Generate Blueprint wrapper for easy access
nni.generate_blueprint_wrapper("EmotionMapper")

# Generate C++ wrapper for performance
nni.generate_cpp_wrapper("EmotionMapper")
```

### 9. WaveNetGenerator (`wavenet_audio.py`) **NEW**

WaveNet-based audio generation with emotion conditioning for synchronized music video creation.

**Based on:** https://github.com/thakkarV/lc-wavenet

**Key Features:**
- Emotion-conditioned audio generation
- MIDI + emotion hybrid conditioning
- Time-varying emotion trajectories
- Synchronized audio/video generation
- Export to ONNX for deployment
- Musical feature extraction for visual sync

**Conditioning Modes:**
- **MUSIC**: Generate from MIDI with musical structure
- **EMOTION**: Direct emotion-to-audio generation
- **HYBRID**: Combine MIDI structure with emotion

**Usage:**
```python
from music_brain.video import create_emotion_conditioned_wavenet

# Create WaveNet generator
wavenet = create_emotion_conditioned_wavenet()

# Generate audio from emotion
audio = wavenet.generate_from_emotion(
    emotion="grief",
    intensity=0.8,
    duration=5.0  # seconds
)

# Generate with emotion trajectory
trajectory = [
    ("grief", 0.8, 0.0),   # Start with grief
    ("peace", 0.5, 5.0),   # Transition to peace
    ("joy", 0.9, 10.0),    # End with joy
]
audio = wavenet.generate_with_trajectory(trajectory, duration=10.0)

# Synchronized audio/video generation
from music_brain.video import EmotionWaveNetBridge

bridge = EmotionWaveNetBridge()
audio, video_params = bridge.generate_synchronized(
    emotion="grief",
    intensity=0.8,
    duration=10.0,
    video_fps=30
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
# All video generation tests (122 tests)
pytest tests/unit/test_video_generation.py \
       tests/unit/test_embedding_regularization.py \
       tests/unit/test_onnx_unreal.py \
       tests/unit/test_wavenet_audio.py -v

# Original video tests (35 tests)
pytest tests/unit/test_video_generation.py -v

# Regularization tests (32 tests)
pytest tests/unit/test_embedding_regularization.py -v

# ONNX/Unreal tests (29 tests)
pytest tests/unit/test_onnx_unreal.py -v

# WaveNet tests (26 tests)
pytest tests/unit/test_wavenet_audio.py -v
```

All tests currently passing with stub implementations.

## Examples

See example files for comprehensive usage:

**Video Generation:**
```bash
cd /home/runner/work/KmiDi/KmiDi
PYTHONPATH=.:$PYTHONPATH python3 examples/video_generation_example.py
```

**Regularization:**
```bash
cd /home/runner/work/KmiDi/KmiDi
PYTHONPATH=.:$PYTHONPATH python3 examples/regularization_example.py
```

## Future Implementation Plan

### Phase 1: ONNX Export & Unreal Integration
- [ ] Implement PyTorch → ONNX export with torch.onnx
- [ ] Implement TensorFlow → ONNX export with tf2onnx
- [ ] Implement ONNX model optimization pipeline
- [ ] Create Unreal Remote Control API client
- [ ] Implement model deployment to UE5 projects
- [ ] Generate Blueprint/C++ wrappers automatically

### Phase 2: WaveNet Audio Generation
- [ ] Train WaveNet on emotion-conditioned music data
- [ ] Implement MIDI reader and upsampling
- [ ] Add real-time streaming generation
- [ ] Integrate beat detection for visual sync
- [ ] Export WaveNet to ONNX for deployment

### Phase 3: Emotion Mapping Enhancement
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

### Phase 5: Complete Integration
- [ ] Full workflow integration: Emotion → Audio + Video
- [ ] Real-time preview support
- [ ] Batch processing pipeline
- [ ] Performance optimization
- [ ] Cloud rendering support
- [ ] GPU acceleration for all components

## Dependencies (Future)

When implemented, this module will require:

**For ONNX Export:**
- `onnx` - ONNX model format
- `onnxruntime` - ONNX Runtime inference
- `torch` - PyTorch for model export (optional)
- `tensorflow` - TensorFlow for model export (optional)
- `tf2onnx` - TensorFlow to ONNX conversion (optional)

**For Unreal Engine:**
- Unreal Engine 5.x with NNI (Neural Network Inference) plugin
- Remote Control API enabled in project

**For WaveNet:**
- `tensorflow` or `torch` - Deep learning framework
- `librosa` - Audio processing
- `mido` - MIDI file reading
- `soundfile` - Audio file I/O

**Common:**
- `numpy` - Numerical operations
- `pillow` - Image processing (for preview)

## External Resources

**ONNX Runtime + Unreal Engine:**
- Microsoft's OnnxRuntime-UnrealEngine: https://github.com/microsoft/OnnxRuntime-UnrealEngine
- ONNX Model Zoo: https://github.com/onnx/models
- Unreal Engine NNI Plugin Docs: https://docs.unrealengine.com/

**WaveNet:**
- lc-wavenet Repository: https://github.com/thakkarV/lc-wavenet
- WaveNet Paper: https://arxiv.org/abs/1609.03499
- MIDI Datasets: https://github.com/bytedance/GiantMIDI-Piano

**General:**
- Hugging Face Models: https://huggingface.co/models
- ONNX Runtime Docs: https://onnxruntime.ai/

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
