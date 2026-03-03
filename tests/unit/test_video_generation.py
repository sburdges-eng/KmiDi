"""Unit tests for video generation module."""

import pytest
from pathlib import Path
from music_brain.video import (
    VideoGenerator,
    VideoConfig,
    VideoFormat,
    VideoQuality,
    UnrealBridge,
    UnrealConfig,
    UnrealSceneParams,
    JespaConnector,
    JespaConfig,
    JespaEffect,
    JespaEffectType,
    EmotionVisualMapper,
    VisualParams,
    VisualStyle,
    SceneComposer,
    SceneDefinition,
    SceneType,
    TransitionStyle,
)


class TestVideoGenerator:
    """Test VideoGenerator class."""
    
    def test_video_generator_creation(self):
        """Test creating a VideoGenerator."""
        gen = VideoGenerator()
        assert gen is not None
        assert gen.config is not None
    
    def test_video_generator_with_config(self):
        """Test creating a VideoGenerator with custom config."""
        config = VideoConfig(
            quality=VideoQuality.HIGH,
            fps=60,
            use_unreal=True,
            use_jespa=True
        )
        gen = VideoGenerator(config=config)
        assert gen.config.quality == VideoQuality.HIGH
        assert gen.config.fps == 60
        assert gen.config.use_unreal is True
        assert gen.config.use_jespa is True
    
    def test_video_generator_initialize(self):
        """Test initializing the video generator."""
        gen = VideoGenerator()
        result = gen.initialize()
        # Stub returns True for now
        assert result is True
    
    def test_generate_from_emotion(self):
        """Test generating video from emotion (stub)."""
        gen = VideoGenerator()
        result = gen.generate_from_emotion(emotion="grief", intensity=0.8)
        # Stub returns failure with message
        assert result.success is False
        assert "not yet implemented" in result.error_message
    
    def test_generate_from_intent(self):
        """Test generating video from intent (stub)."""
        gen = VideoGenerator()
        intent = {"emotion": "joy", "intensity": 0.7}
        result = gen.generate_from_intent(intent=intent)
        assert result.success is False
        assert "not yet implemented" in result.error_message
    
    def test_preview_scene(self):
        """Test preview scene generation (stub)."""
        gen = VideoGenerator()
        preview = gen.preview_scene(emotion="sadness", timestamp=5.0)
        # Stub returns None
        assert preview is None
    
    def test_cleanup(self):
        """Test cleanup method."""
        gen = VideoGenerator()
        gen.initialize()
        gen.cleanup()
        # Should reset initialized flag
        assert gen._initialized is False

    def test_cleanup_temp_files(self, tmp_path):
        """Test that cleanup removes only the generator-owned temp subdirectory."""
        # Base temp dir provided by the caller
        temp_dir = tmp_path / "video_temp"
        temp_dir.mkdir()

        # A file placed *outside* the generator's owned subdir (not owned by gen)
        external_file = temp_dir / "external.txt"
        external_file.write_text("do not delete me")

        config = VideoConfig(temp_dir=temp_dir)
        gen = VideoGenerator(config=config)
        gen.initialize()

        # After initialize(), the generator should have created its own subdir
        assert gen._owned_temp_dir is not None
        assert gen._owned_temp_dir.is_dir()
        sentinel = gen._owned_temp_dir / VideoGenerator._SENTINEL_FILENAME
        assert sentinel.is_file()

        # Place a file inside the owned subdir to verify it gets cleaned up
        owned_file = gen._owned_temp_dir / "frame_001.png"
        owned_file.write_text("dummy frame data")

        owned_dir = gen._owned_temp_dir
        gen.cleanup()

        # Generator's subdirectory should be gone
        assert not owned_dir.exists()
        assert not owned_file.exists()

        # The base temp_dir and external files must NOT be touched
        assert temp_dir.exists()
        assert external_file.exists()

    def test_cleanup_no_sentinel_skips_deletion(self, tmp_path):
        """Test that cleanup skips deletion when sentinel file is missing."""
        temp_dir = tmp_path / "video_temp"
        temp_dir.mkdir()

        config = VideoConfig(temp_dir=temp_dir)
        gen = VideoGenerator(config=config)
        gen.initialize()

        owned_dir = gen._owned_temp_dir
        assert owned_dir is not None

        # Remove the sentinel to simulate tampering / corruption
        sentinel = owned_dir / VideoGenerator._SENTINEL_FILENAME
        sentinel.unlink()

        gen.cleanup()

        # Owned dir should NOT have been deleted (sentinel was missing)
        assert owned_dir.exists()

    def test_cleanup_dangerous_temp_dir_rejected(self, tmp_path):
        """Test that a high-risk temp_dir is rejected during initialize()."""
        config = VideoConfig(temp_dir=Path.home())
        gen = VideoGenerator(config=config)
        result = gen.initialize()
        # initialize() should still return True but _owned_temp_dir stays None
        assert result is True
        assert gen._owned_temp_dir is None


class TestUnrealBridge:
    """Test UnrealBridge class."""
    
    def test_unreal_bridge_creation(self):
        """Test creating an UnrealBridge."""
        bridge = UnrealBridge()
        assert bridge is not None
        assert bridge.config is not None
    
    def test_unreal_bridge_with_config(self):
        """Test creating UnrealBridge with custom config."""
        config = UnrealConfig(
            host="192.168.1.100",
            port=7000,
            enable_ray_tracing=True
        )
        bridge = UnrealBridge(config=config)
        assert bridge.config.host == "192.168.1.100"
        assert bridge.config.port == 7000
        assert bridge.config.enable_ray_tracing is True
    
    def test_unreal_bridge_connect(self):
        """Test connecting to Unreal (stub)."""
        bridge = UnrealBridge()
        result = bridge.connect()
        # Stub returns False (not implemented)
        assert result is False
    
    def test_unreal_scene_params_creation(self):
        """Test creating UnrealSceneParams."""
        params = UnrealSceneParams(
            key_light_intensity=1.5,
            camera_fov=110.0,
            exposure=1.2
        )
        assert params.key_light_intensity == 1.5
        assert params.camera_fov == 110.0
        assert params.exposure == 1.2
    
    def test_update_scene(self):
        """Test updating scene parameters (stub)."""
        bridge = UnrealBridge()
        params = UnrealSceneParams()
        result = bridge.update_scene(params)
        # Not connected, should return False
        assert result is False
    
    def test_get_capabilities(self):
        """Test getting Unreal capabilities."""
        bridge = UnrealBridge()
        caps = bridge.get_capabilities()
        assert "connected" in caps
        assert caps["connected"] is False


class TestJespaConnector:
    """Test JespaConnector class."""
    
    def test_jespa_connector_creation(self):
        """Test creating a JespaConnector."""
        jespa = JespaConnector()
        assert jespa is not None
        assert jespa.config is not None
    
    def test_jespa_connector_with_config(self):
        """Test creating JespaConnector with custom config."""
        config = JespaConfig(
            host="example.com",
            port=9000,
            use_gpu_acceleration=False
        )
        jespa = JespaConnector(config=config)
        assert jespa.config.host == "example.com"
        assert jespa.config.port == 9000
        assert jespa.config.use_gpu_acceleration is False
    
    def test_jespa_effect_creation(self):
        """Test creating a JespaEffect."""
        effect = JespaEffect(
            effect_type=JespaEffectType.COLOR_GRADE,
            intensity=0.8,
            parameters={"temperature": -10, "tint": 5}
        )
        assert effect.effect_type == JespaEffectType.COLOR_GRADE
        assert effect.intensity == 0.8
        assert effect.parameters["temperature"] == -10
    
    def test_add_effect(self):
        """Test adding an effect (stub)."""
        jespa = JespaConnector()
        effect = JespaEffect(
            effect_type=JespaEffectType.GLOW,
            intensity=0.5
        )
        result = jespa.add_effect(effect)
        # Not connected, should return False
        assert result is False
    
    def test_clear_effects(self):
        """Test clearing effects."""
        jespa = JespaConnector()
        jespa._effects.append(JespaEffect(effect_type=JespaEffectType.BLUR))
        jespa.clear_effects()
        assert len(jespa._effects) == 0
    
    def test_get_processing_status(self):
        """Test getting processing status."""
        jespa = JespaConnector()
        status = jespa.get_processing_status()
        assert "connected" in status
        assert status["connected"] is False


class TestEmotionVisualMapper:
    """Test EmotionVisualMapper class."""
    
    def test_emotion_visual_mapper_creation(self):
        """Test creating an EmotionVisualMapper."""
        mapper = EmotionVisualMapper()
        assert mapper is not None
    
    def test_map_emotion_joy(self):
        """Test mapping joy emotion."""
        mapper = EmotionVisualMapper()
        params = mapper.map_emotion("joy", intensity=0.8)
        assert params.emotion_source == "joy"
        assert params.intensity == 0.8
        # Joy should map to bright yellow-ish color
        assert params.primary_color[0] > 0.8  # High red
        assert params.primary_color[1] > 0.8  # High green
    
    def test_map_emotion_grief(self):
        """Test mapping grief emotion."""
        mapper = EmotionVisualMapper()
        params = mapper.map_emotion("grief", intensity=0.7)
        assert params.emotion_source == "grief"
        assert params.intensity == 0.7
        # Grief should map to dark, desaturated color
        assert params.primary_color[0] < 0.5  # Low brightness
    
    def test_map_emotion_with_style(self):
        """Test mapping emotion with visual style."""
        mapper = EmotionVisualMapper()
        params = mapper.map_emotion("sadness", style=VisualStyle.ABSTRACT)
        assert params.visual_style == VisualStyle.ABSTRACT
    
    def test_blend_emotions(self):
        """Test blending two emotions."""
        mapper = EmotionVisualMapper()
        blended = mapper.blend_emotions("joy", "sadness", blend_factor=0.5)
        # Should contain both emotion names
        assert "joy" in blended.emotion_source
        assert "sadness" in blended.emotion_source
    
    def test_get_emotion_palette(self):
        """Test getting emotion color palette."""
        mapper = EmotionVisualMapper()
        palette = mapper.get_emotion_palette("anger")
        assert len(palette) > 0
        # Should return list of RGB tuples
        assert all(len(color) == 3 for color in palette)
    
    def test_custom_mapping(self):
        """Test adding custom emotion mapping."""
        mapper = EmotionVisualMapper()
        custom_params = VisualParams(
            primary_color=(0.9, 0.1, 0.9),
            emotion_source="custom_emotion"
        )
        mapper.add_custom_mapping("custom_emotion", custom_params)
        assert "custom_emotion" in mapper._custom_mappings


class TestSceneComposer:
    """Test SceneComposer class."""
    
    def test_scene_composer_creation(self):
        """Test creating a SceneComposer."""
        composer = SceneComposer()
        assert composer is not None
    
    def test_compose_from_structure(self):
        """Test composing scenes from musical structure."""
        composer = SceneComposer()
        structure = [
            ("intro", 0, 8),
            ("verse", 8, 24),
            ("chorus", 24, 40),
        ]
        scenes = composer.compose_from_structure(
            structure,
            emotion="joy",
            intensity=0.8
        )
        assert len(scenes) == 3
        assert scenes[0].scene_type == SceneType.INTRO
        assert scenes[1].scene_type == SceneType.VERSE
        assert scenes[2].scene_type == SceneType.CHORUS
        assert scenes[0].primary_emotion == "joy"
    
    def test_generate_transitions(self):
        """Test generating transitions between scenes."""
        composer = SceneComposer()
        structure = [
            ("verse", 0, 16),
            ("chorus", 16, 32),
        ]
        scenes = composer.compose_from_structure(structure)
        transitions = composer.generate_transitions(scenes)
        
        assert len(transitions) == 1
        assert transitions[0].from_scene_index == 0
        assert transitions[0].to_scene_index == 1
        assert transitions[0].start_time == 16.0
    
    def test_scene_definition_creation(self):
        """Test creating a SceneDefinition."""
        scene = SceneDefinition(
            start_time=0.0,
            duration=10.0,
            scene_type=SceneType.CHORUS,
            primary_emotion="excitement",
            emotion_intensity=0.9
        )
        assert scene.start_time == 0.0
        assert scene.duration == 10.0
        assert scene.scene_type == SceneType.CHORUS
        assert scene.primary_emotion == "excitement"
    
    def test_add_beat_emphasis(self):
        """Test adding beat emphasis."""
        composer = SceneComposer()
        structure = [("verse", 0, 16)]
        scenes = composer.compose_from_structure(structure)
        
        beats = [0.5, 1.0, 1.5, 2.0]
        composer.add_beat_emphasis(beats)
        
        # Beats should be added to appropriate scenes
        assert len(scenes[0].emphasis_beats) > 0
    
    def test_get_scene_at_time(self):
        """Test getting scene at specific time."""
        composer = SceneComposer()
        structure = [
            ("intro", 0, 8),
            ("verse", 8, 24),
        ]
        composer.compose_from_structure(structure)
        
        scene = composer.get_scene_at_time(5.0)
        assert scene is not None
        assert scene.scene_type == SceneType.INTRO
        
        scene = composer.get_scene_at_time(15.0)
        assert scene is not None
        assert scene.scene_type == SceneType.VERSE
        
        scene = composer.get_scene_at_time(100.0)
        assert scene is None


class TestVideoConfig:
    """Test VideoConfig dataclass."""
    
    def test_video_config_defaults(self):
        """Test VideoConfig default values."""
        config = VideoConfig()
        assert config.format == VideoFormat.MP4
        assert config.quality == VideoQuality.MEDIUM
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 30
        assert config.use_unreal is True
        assert config.use_jespa is True
    
    def test_video_config_custom(self):
        """Test VideoConfig with custom values."""
        config = VideoConfig(
            format=VideoFormat.WEBM,
            quality=VideoQuality.ULTRA,
            width=3840,
            height=2160,
            fps=60
        )
        assert config.format == VideoFormat.WEBM
        assert config.quality == VideoQuality.ULTRA
        assert config.width == 3840
        assert config.height == 2160
        assert config.fps == 60


class TestIntegration:
    """Integration tests for video generation workflow."""
    
    def test_full_workflow_stub(self):
        """Test the full video generation workflow (as stubs)."""
        # Initialize components
        gen = VideoGenerator()
        mapper = EmotionVisualMapper()
        composer = SceneComposer()
        
        # Map emotion to visuals
        visual_params = mapper.map_emotion("grief", intensity=0.8)
        assert visual_params.emotion_source == "grief"
        
        # Compose scenes
        structure = [
            ("intro", 0, 8),
            ("verse", 8, 24),
            ("chorus", 24, 40),
        ]
        scenes = composer.compose_from_structure(structure, emotion="grief")
        assert len(scenes) == 3
        
        # Generate video (stub)
        result = gen.generate_from_emotion(emotion="grief", intensity=0.8)
        # Stub should return failure
        assert result.success is False
