"""Unit tests for WaveNet audio generation."""

import numpy as np
from pathlib import Path
from music_brain.video import (
    WaveNetGenerator,
    WaveNetConfig,
    AudioGenerationConfig,
    WaveNetMode,
    EmotionWaveNetBridge,
    create_emotion_conditioned_wavenet,
)


class TestWaveNetMode:
    """Test WaveNetMode enum."""

    def test_wavenet_modes(self):
        """Test WaveNet generation modes."""
        assert WaveNetMode.MUSIC.value == "music"
        assert WaveNetMode.EMOTION.value == "emotion"
        assert WaveNetMode.HYBRID.value == "hybrid"


class TestWaveNetConfig:
    """Test WaveNetConfig dataclass."""

    def test_wavenet_config_defaults(self):
        """Test default WaveNet configuration."""
        config = WaveNetConfig()
        assert config.num_layers == 30
        assert config.num_stages == 10
        assert config.sample_rate == 16000
        assert config.use_local_conditioning is True
        assert config.local_condition_channels == 128

    def test_wavenet_config_custom(self):
        """Test custom WaveNet configuration."""
        config = WaveNetConfig(num_layers=20, sample_rate=22050, local_condition_channels=256)
        assert config.num_layers == 20
        assert config.sample_rate == 22050
        assert config.local_condition_channels == 256


class TestAudioGenerationConfig:
    """Test AudioGenerationConfig dataclass."""

    def test_audio_config_defaults(self):
        """Test default audio generation configuration."""
        config = AudioGenerationConfig()
        assert config.emotion_embedding_dim == 128
        assert config.use_emotion_trajectory is True
        assert config.output_format == "wav"
        assert config.sync_to_video is True

    def test_audio_config_custom(self):
        """Test custom audio configuration."""
        config = AudioGenerationConfig(use_midi_conditioning=True, video_fps=60, bit_depth=24)
        assert config.use_midi_conditioning is True
        assert config.video_fps == 60
        assert config.bit_depth == 24


class TestWaveNetGenerator:
    """Test WaveNetGenerator class."""

    def test_generator_creation(self):
        """Test creating WaveNet generator."""
        gen = WaveNetGenerator()
        assert gen is not None
        assert gen.wavenet_config is not None
        assert gen.audio_config is not None

    def test_generator_with_config(self):
        """Test creating generator with custom configs."""
        wavenet_config = WaveNetConfig(sample_rate=22050)
        audio_config = AudioGenerationConfig(video_fps=60)

        gen = WaveNetGenerator(wavenet_config, audio_config)
        assert gen.wavenet_config.sample_rate == 22050
        assert gen.audio_config.video_fps == 60

    def test_initialize(self):
        """Test initializing WaveNet."""
        gen = WaveNetGenerator()
        result = gen.initialize()
        # Stub returns True
        assert result is True
        assert gen._initialized is True

    def test_generate_from_emotion(self):
        """Test generating audio from emotion."""
        gen = WaveNetGenerator()
        audio = gen.generate_from_emotion(emotion="grief", intensity=0.8, duration=1.0)

        # Should return audio array
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        # Check expected length (1 second at 16kHz)
        assert len(audio) == 16000

    def test_generate_different_durations(self):
        """Test generating audio with different durations."""
        gen = WaveNetGenerator()

        audio_1s = gen.generate_from_emotion("joy", 0.5, 1.0)
        audio_5s = gen.generate_from_emotion("joy", 0.5, 5.0)

        assert len(audio_1s) == 16000
        assert len(audio_5s) == 80000  # 5 seconds

    def test_generate_from_midi_stub(self):
        """Test generating audio from MIDI (stub)."""
        gen = WaveNetGenerator()
        audio = gen.generate_from_midi(midi_path=Path("/tmp/test.mid"))

        # Stub returns silence
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 16000

    def test_generate_with_trajectory(self):
        """Test generating audio with emotion trajectory."""
        gen = WaveNetGenerator()

        trajectory = [
            ("grief", 0.8, 0.0),
            ("peace", 0.5, 5.0),
            ("joy", 0.9, 10.0),
        ]

        audio = gen.generate_with_trajectory(trajectory, duration=10.0)

        assert isinstance(audio, np.ndarray)
        assert len(audio) == 160000  # 10 seconds at 16kHz

    def test_export_to_onnx_stub(self):
        """Test exporting WaveNet to ONNX (stub)."""
        gen = WaveNetGenerator()
        result = gen.export_to_onnx(output_path=Path("/tmp/wavenet.onnx"), optimize=True)
        # Stub returns False
        assert result is False

    def test_load_pretrained_stub(self):
        """Test loading pre-trained model (stub)."""
        gen = WaveNetGenerator()
        result = gen.load_pretrained(checkpoint_path=Path("/tmp/checkpoint.pt"))
        # Stub returns False
        assert result is False

    def test_save_checkpoint_stub(self):
        """Test saving checkpoint (stub)."""
        gen = WaveNetGenerator()
        result = gen.save_checkpoint(checkpoint_path=Path("/tmp/checkpoint.pt"))
        # Stub returns False
        assert result is False


class TestEmotionWaveNetBridge:
    """Test EmotionWaveNetBridge class."""

    def test_bridge_creation(self):
        """Test creating emotion-WaveNet bridge."""
        bridge = EmotionWaveNetBridge()
        assert bridge is not None
        assert bridge.wavenet is not None

    def test_generate_synchronized(self):
        """Test generating synchronized audio and video."""
        bridge = EmotionWaveNetBridge()

        audio, video_params = bridge.generate_synchronized(
            emotion="grief", intensity=0.8, duration=2.0, video_fps=30
        )

        # Check audio
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 32000  # 2 seconds at 16kHz

        # Check video params
        assert isinstance(video_params, list)
        assert len(video_params) == 60  # 2 seconds at 30fps

        # Check video param structure
        assert "timestamp" in video_params[0]
        assert "emotion" in video_params[0]
        assert video_params[0]["emotion"] == "grief"

    def test_generate_synchronized_different_fps(self):
        """Test synchronized generation with different FPS."""
        bridge = EmotionWaveNetBridge()

        _, video_params_30 = bridge.generate_synchronized(emotion="joy", duration=1.0, video_fps=30)

        _, video_params_60 = bridge.generate_synchronized(emotion="joy", duration=1.0, video_fps=60)

        assert len(video_params_30) == 30
        assert len(video_params_60) == 60

    def test_extract_musical_features(self):
        """Test extracting musical features from audio."""
        bridge = EmotionWaveNetBridge()

        # Generate some audio
        audio = np.random.randn(16000).astype(np.float32)

        features = bridge.extract_musical_features(audio)

        assert isinstance(features, dict)
        assert "tempo" in features
        assert "energy" in features
        assert "spectral_centroid" in features


class TestCreateEmotionConditionedWaveNet:
    """Test create_emotion_conditioned_wavenet helper function."""

    def test_create_default(self):
        """Test creating with default parameters."""
        wavenet = create_emotion_conditioned_wavenet()

        assert isinstance(wavenet, WaveNetGenerator)
        assert wavenet.wavenet_config.sample_rate == 16000
        assert wavenet.wavenet_config.local_condition_channels == 128
        assert wavenet.audio_config.sync_to_video is True

    def test_create_custom_params(self):
        """Test creating with custom parameters."""
        wavenet = create_emotion_conditioned_wavenet(emotion_embedding_dim=256, sample_rate=22050)

        assert wavenet.wavenet_config.sample_rate == 22050
        assert wavenet.wavenet_config.local_condition_channels == 256


class TestWaveNetAudioOutput:
    """Test WaveNet audio output properties."""

    def test_audio_output_range(self):
        """Test that audio output is in valid range."""
        gen = WaveNetGenerator()
        audio = gen.generate_from_emotion("joy", 0.5, 0.1)

        # Check audio values are in [-1.0, 1.0] range (after normalization)
        # For now, stub returns zeros
        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)

    def test_audio_output_dtype(self):
        """Test that audio output has correct dtype."""
        gen = WaveNetGenerator()
        audio = gen.generate_from_emotion("grief", 0.8, 0.1)

        assert audio.dtype == np.float32

    def test_audio_output_shape(self):
        """Test that audio output has correct shape."""
        gen = WaveNetGenerator()
        audio = gen.generate_from_emotion("anger", 0.7, 0.5)

        # Should be 1D array
        assert len(audio.shape) == 1
        # Should be 0.5 seconds at 16kHz
        assert audio.shape[0] == 8000


class TestWaveNetConfigValidation:
    """Test WaveNet configuration validation."""

    def test_valid_sample_rates(self):
        """Test common valid sample rates."""
        for rate in [8000, 16000, 22050, 44100, 48000]:
            config = WaveNetConfig(sample_rate=rate)
            assert config.sample_rate == rate

    def test_valid_conditioning_channels(self):
        """Test valid conditioning channel dimensions."""
        for channels in [32, 64, 128, 256, 512]:
            config = WaveNetConfig(local_condition_channels=channels)
            assert config.local_condition_channels == channels
