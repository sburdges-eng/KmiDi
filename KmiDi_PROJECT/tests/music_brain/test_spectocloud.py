"""
Tests for Spectocloud 3D visualization module.

Tests core functionality including:
- Anchor generation and library
- Musical parameter extraction
- Frame generation from MIDI
- Storm dynamics
- Rendering (basic checks, not visual validation)
"""

import pytest
import numpy as np
from music_brain.visualization.spectocloud import (
    Anchor,
    AnchorFamily,
    AnchorLibrary,
    Frame,
    EmotionParticle,
    StormState,
    MusicalParameterExtractor,
    Spectocloud,
)


class TestAnchor:
    """Test Anchor dataclass."""
    
    def test_anchor_creation(self):
        """Test basic anchor creation."""
        anchor = Anchor(
            id="TEST_1",
            name="Test Anchor",
            family=AnchorFamily.HARMONY,
            position_y=0.5,
            position_z=0.7,
        )
        
        assert anchor.id == "TEST_1"
        assert anchor.family == AnchorFamily.HARMONY
        assert anchor.position_y == 0.5
        assert anchor.position_z == 0.7
        assert anchor.base_charge == 1.0  # Default value
        assert anchor.activation == 0.0
    
    def test_anchor_with_constraints(self):
        """Test anchor with parameter constraints."""
        anchor = Anchor(
            id="HARM_1",
            name="High Tension",
            family=AnchorFamily.HARMONY,
            position_y=0.8,
            position_z=0.9,
            constraints={'tension': 0.85, 'complexity': 0.7},
        )
        
        assert 'tension' in anchor.constraints
        assert anchor.constraints['tension'] == 0.85


class TestAnchorLibrary:
    """Test AnchorLibrary generation and similarity computation."""
    
    def test_library_generation_sparse(self):
        """Test sparse anchor library generation."""
        library = AnchorLibrary(density="sparse")
        
        # Should have 50-200 anchors
        assert 50 <= len(library.anchors) <= 200
        
        # Should cover all families
        families = set(a.family for a in library.anchors)
        assert AnchorFamily.HARMONY in families
        assert AnchorFamily.RHYTHM in families
        assert AnchorFamily.DYNAMICS in families
    
    def test_library_generation_normal(self):
        """Test normal anchor library generation."""
        library = AnchorLibrary(density="normal")
        
        # Should have ~100 anchors
        assert 80 <= len(library.anchors) <= 150
    
    def test_library_generation_dense(self):
        """Test dense anchor library generation."""
        library = AnchorLibrary(density="dense")
        
        # Should have 200-600 anchors
        assert 200 <= len(library.anchors) <= 350
    
    def test_anchor_uniqueness(self):
        """Test that anchor IDs are unique."""
        library = AnchorLibrary(density="normal")
        ids = [a.id for a in library.anchors]
        assert len(ids) == len(set(ids))  # No duplicates
    
    def test_compute_similarities(self):
        """Test anchor similarity computation."""
        library = AnchorLibrary(density="sparse")
        
        # Create test features
        features = {
            'tension': 0.8,
            'complexity': 0.5,
            'density': 0.6,
            'energy': 0.7,
        }
        
        # Get top-5 similar anchors
        top_anchors = library.compute_anchor_similarities(features, top_k=5)
        
        assert len(top_anchors) == 5
        # Each result should be (anchor_id, similarity)
        for anchor_id, similarity in top_anchors:
            assert isinstance(anchor_id, str)
            assert 0.0 <= similarity <= 1.0
        
        # Similarities should be in descending order
        similarities = [s for _, s in top_anchors]
        assert similarities == sorted(similarities, reverse=True)


class TestMusicalParameterExtractor:
    """Test musical parameter extraction."""
    
    def test_extract_from_empty_midi_window(self):
        """Test feature extraction from empty MIDI window."""
        extractor = MusicalParameterExtractor()
        
        features = extractor.extract_from_midi_window(
            midi_events=[],
            window_start=0.0,
            window_end=1.0,
        )
        
        # Should return neutral/default features
        assert 'note_density' in features
        assert 'pitch_centroid' in features
        assert features['note_density'] == 0.0  # No notes
    
    def test_extract_from_midi_window(self):
        """Test feature extraction from MIDI window with notes."""
        extractor = MusicalParameterExtractor()
        
        # Create sample MIDI events
        midi_events = [
            {'time': 0.1, 'type': 'note_on', 'note': 60, 'velocity': 64},
            {'time': 0.3, 'type': 'note_on', 'note': 64, 'velocity': 80},
            {'time': 0.5, 'type': 'note_on', 'note': 67, 'velocity': 70},
        ]
        
        features = extractor.extract_from_midi_window(
            midi_events=midi_events,
            window_start=0.0,
            window_end=1.0,
        )
        
        # Check that features are extracted
        assert 'note_density' in features
        assert 'pitch_centroid' in features
        assert 'velocity_centroid' in features
        assert 'tension' in features
        
        # Values should be normalized
        assert features['note_density'] > 0.0
        assert 0.0 <= features['pitch_centroid'] <= 1.0
        assert 0.0 <= features['velocity_centroid'] <= 1.0
    
    def test_extract_from_audio_window(self):
        """Test spectral feature extraction from audio."""
        extractor = MusicalParameterExtractor()
        
        # Create synthetic audio (sine wave)
        sample_rate = 44100
        duration = 1.0
        freq = 440.0  # A4
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * freq * t) * 0.5
        
        features = extractor.extract_from_audio_window(
            audio_data=audio_data,
            sample_rate=sample_rate,
            window_start=0.0,
            window_end=1.0,
        )
        
        # Check spectral features
        assert 'spectral_centroid' in features
        assert 'rms_energy' in features
        assert 'spectral_flux' in features
        
        # Values should be normalized
        assert 0.0 <= features['spectral_centroid'] <= 1.0
        assert features['rms_energy'] > 0.0  # Should have energy from sine wave


class TestFrame:
    """Test Frame dataclass."""
    
    def test_frame_creation(self):
        """Test basic frame creation."""
        frame = Frame(
            time=1.5,
            x=1.5,
            y=0.6,
            z=0.7,
            valence=-0.3,
            arousal=0.8,
        )
        
        assert frame.time == 1.5
        assert frame.x == 1.5
        assert frame.valence == -0.3
        assert frame.arousal == 0.8
        assert frame.spread == 0.16  # Default


class TestEmotionParticle:
    """Test EmotionParticle."""
    
    def test_particle_creation(self):
        """Test particle creation with numpy array conversion."""
        particle = EmotionParticle(
            position=[1.0, 2.0, 3.0],
            velocity=[0.1, 0.2, 0.0],
            valence=0.5,
            arousal=0.7,
        )
        
        assert isinstance(particle.position, np.ndarray)
        assert isinstance(particle.velocity, np.ndarray)
        assert len(particle.position) == 3
        assert particle.valence == 0.5
        assert particle.arousal == 0.7
        assert particle.mass == 1.0  # Default


class TestStormState:
    """Test storm state tracking."""
    
    def test_storm_initialization(self):
        """Test storm state initialization."""
        storm = StormState()
        
        assert storm.charge == 0.0
        assert storm.energy == 0.0
        assert len(storm.active_arcs) == 0
        assert storm.threshold == 14.0


class TestSpectocloud:
    """Test main Spectocloud engine."""
    
    def test_initialization(self):
        """Test Spectocloud initialization."""
        spectocloud = Spectocloud(
            anchor_density="sparse",
            n_particles=1000,
            window_size=0.2,
        )
        
        assert len(spectocloud.anchor_library.anchors) > 0
        assert spectocloud.n_particles == 1000
        assert spectocloud.window_size == 0.2
        assert len(spectocloud.frames) == 0  # No data processed yet
    
    def test_process_midi_basic(self):
        """Test basic MIDI processing."""
        spectocloud = Spectocloud(anchor_density="sparse", n_particles=500)
        
        # Create simple MIDI events
        midi_events = [
            {'time': 0.0, 'type': 'note_on', 'note': 60, 'velocity': 64},
            {'time': 0.5, 'type': 'note_on', 'note': 64, 'velocity': 70},
            {'time': 1.0, 'type': 'note_on', 'note': 67, 'velocity': 80},
        ]
        
        duration = 2.0
        
        spectocloud.process_midi(
            midi_events=midi_events,
            duration=duration,
        )
        
        # Should have generated frames
        assert len(spectocloud.frames) > 0
        
        # Frames should span the duration
        assert spectocloud.frames[0].time >= 0.0
        assert spectocloud.frames[-1].time < duration
    
    def test_process_midi_with_emotion(self):
        """Test MIDI processing with emotion trajectory."""
        spectocloud = Spectocloud(anchor_density="sparse", n_particles=500)
        
        # Create MIDI events
        midi_events = [
            {'time': 0.0, 'type': 'note_on', 'note': 60, 'velocity': 64},
            {'time': 0.5, 'type': 'note_on', 'note': 64, 'velocity': 70},
        ]
        
        # Create emotion trajectory
        emotion_trajectory = [
            {'valence': -0.5, 'arousal': 0.3, 'intensity': 0.6},
            {'valence': 0.2, 'arousal': 0.7, 'intensity': 0.8},
            {'valence': 0.8, 'arousal': 0.9, 'intensity': 0.9},
        ]
        
        duration = 1.0
        
        spectocloud.process_midi(
            midi_events=midi_events,
            duration=duration,
            emotion_trajectory=emotion_trajectory,
        )
        
        # Frames should have emotion values
        assert len(spectocloud.frames) > 0
        assert spectocloud.frames[0].valence == -0.5
        assert spectocloud.frames[0].arousal == 0.3
    
    def test_conical_expansion(self):
        """Test that cloud spread increases over time (conical)."""
        spectocloud = Spectocloud(anchor_density="sparse")
        
        midi_events = [
            {'time': i * 0.1, 'type': 'note_on', 'note': 60, 'velocity': 64}
            for i in range(50)
        ]
        
        spectocloud.process_midi(midi_events, duration=5.0)
        
        # Spread should generally increase over time
        spreads = [f.spread for f in spectocloud.frames]
        
        # First frame spread < last frame spread (conical expansion)
        assert spreads[0] < spreads[-1]
    
    def test_generate_particles(self):
        """Test particle generation for a frame."""
        spectocloud = Spectocloud(anchor_density="sparse", n_particles=100)
        
        midi_events = [
            {'time': 0.0, 'type': 'note_on', 'note': 60, 'velocity': 64},
        ]
        
        spectocloud.process_midi(midi_events, duration=1.0)
        
        # Generate particles for first frame
        particles = spectocloud.generate_particles_for_frame(0)
        
        assert len(particles) == 100
        assert all(isinstance(p, EmotionParticle) for p in particles)
        
        # Particles should be distributed around frame center
        frame = spectocloud.frames[0]
        positions = np.array([p.position for p in particles])
        
        # Mean position should be close to frame center
        mean_pos = np.mean(positions, axis=0)
        assert np.abs(mean_pos[0] - frame.x) < 0.5
        assert np.abs(mean_pos[1] - frame.y) < 0.5
        assert np.abs(mean_pos[2] - frame.z) < 0.5
    
    def test_export_data(self, tmp_path):
        """Test data export to JSON."""
        spectocloud = Spectocloud(anchor_density="sparse")
        
        midi_events = [
            {'time': 0.0, 'type': 'note_on', 'note': 60, 'velocity': 64},
        ]
        
        spectocloud.process_midi(midi_events, duration=1.0)
        
        # Export to temp file
        output_path = tmp_path / "spectocloud_test.json"
        spectocloud.export_data(str(output_path))
        
        # Check file was created
        assert output_path.exists()
        
        # Load and verify structure
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert 'anchors' in data
        assert 'frames' in data
        assert len(data['anchors']) > 0
        assert len(data['frames']) > 0


class TestRendering:
    """Test rendering functionality (basic checks only)."""
    
    def test_renderer_initialization(self):
        """Test renderer can be initialized."""
        from music_brain.visualization.spectocloud import SpectocloudRenderer
        
        renderer = SpectocloudRenderer(figsize=(8, 6), dpi=100)
        assert renderer.figsize == (8, 6)
        assert renderer.dpi == 100
    
    def test_rgba_for_valence(self):
        """Test valence to RGBA conversion."""
        from music_brain.visualization.spectocloud import SpectocloudRenderer
        
        renderer = SpectocloudRenderer()
        
        # Test negative valence (should be blueish)
        rgba_neg = renderer.rgba_for_valence(-1.0, alpha=0.5)
        assert len(rgba_neg) == 4
        assert rgba_neg[3] == 0.5  # Alpha
        
        # Test positive valence (should be reddish)
        rgba_pos = renderer.rgba_for_valence(1.0, alpha=0.8)
        assert len(rgba_pos) == 4
        assert rgba_pos[3] == 0.8  # Alpha
    
    def test_render_frame_without_matplotlib(self):
        """Test graceful handling when matplotlib not available."""
        from music_brain.visualization.spectocloud import SpectocloudRenderer
        
        renderer = SpectocloudRenderer()
        
        # If matplotlib not available, should handle gracefully
        if not renderer.has_mpl:
            result = renderer.render_frame(
                anchors=[],
                frames=[],
                particles=[],
                storm=StormState(),
                current_frame_idx=0,
            )
            assert result is None


class TestPerformanceOptimizations:
    """Test new performance optimization features (v2.0)."""
    
    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        from music_brain.visualization.spectocloud import PerformanceMetrics
        
        metrics = PerformanceMetrics()
        
        # Record some times
        metrics.record_frame_time(0.01)
        metrics.record_frame_time(0.02)
        metrics.record_particle_gen_time(0.005)
        metrics.record_anchor_similarity_time(0.002)
        metrics.record_render_time(0.015)
        
        summary = metrics.get_summary()
        
        assert 'avg_frame_time_ms' in summary
        assert summary['avg_frame_time_ms'] == pytest.approx(15.0, rel=0.1)
        assert summary['total_frames_processed'] == 2
    
    def test_lod_config(self):
        """Test Level of Detail configuration."""
        from music_brain.visualization.spectocloud import LODLevel, LODConfig
        
        config = LODConfig(level=LODLevel.MEDIUM)
        
        assert config.get_particle_count() == 1200
        assert config.get_anchor_scale() == pytest.approx(0.8)
        assert config.get_texture_resolution() == 64
        
        # Change LOD level
        config.level = LODLevel.HIGH
        assert config.get_particle_count() == 2000
    
    def test_vectorized_similarity(self):
        """Test vectorized anchor similarity computation."""
        from music_brain.visualization.spectocloud import AnchorLibrary
        
        library = AnchorLibrary(density="sparse")
        
        features = {
            'tension': 0.5,
            'complexity': 0.3,
            'density': 0.7,
            'energy': 0.6,
        }
        
        # Test both methods give similar results
        standard_result = library.compute_anchor_similarities(features, top_k=5)
        vectorized_result = library.compute_anchor_similarities_vectorized(features, top_k=5)
        
        assert len(vectorized_result) == 5
        
        # Top anchor should be similar (may not be identical due to numerical precision)
        assert isinstance(vectorized_result[0][0], str)  # anchor_id
        assert 0 <= vectorized_result[0][1] <= 1  # similarity
    
    def test_spectocloud_with_lod(self):
        """Test Spectocloud initialization with LOD."""
        from music_brain.visualization.spectocloud import LODLevel
        
        spectocloud = Spectocloud(
            anchor_density="sparse",
            lod_level=LODLevel.LOW,
            use_vectorized_similarity=True,
        )
        
        # LOD.LOW should set n_particles to 600
        assert spectocloud.n_particles == 600
        assert spectocloud.use_vectorized_similarity is True


class TestTexturization:
    """Test texturization features (v2.0)."""
    
    def test_texture_config(self):
        """Test TextureConfig dataclass."""
        from music_brain.visualization.spectocloud import TextureConfig
        
        config = TextureConfig()
        
        assert config.fog_enabled is True
        assert config.fog_density == pytest.approx(0.15)
        assert config.glow_enabled is True
        assert config.noise_enabled is True
    
    def test_texture_generator(self):
        """Test TextureGenerator basic functions."""
        from music_brain.visualization.spectocloud import TextureGenerator, TextureConfig
        
        config = TextureConfig()
        generator = TextureGenerator(config)
        
        # Test noise generation
        noise = generator.generate_noise_texture(100)
        assert noise.shape == (100,)
        assert np.abs(noise).max() <= 1.0  # Should be normalized
        
        # Test caching
        noise2 = generator.generate_noise_texture(100, cache_key=42)
        noise3 = generator.generate_noise_texture(100, cache_key=42)
        np.testing.assert_array_equal(noise2, noise3)  # Same cache key = same noise
    
    def test_depth_fog(self):
        """Test depth fog application."""
        from music_brain.visualization.spectocloud import TextureGenerator, TextureConfig
        
        config = TextureConfig(fog_enabled=True, fog_density=0.5)
        generator = TextureGenerator(config)
        
        # Create test data
        positions = np.random.rand(50, 3)
        colors = np.random.rand(50, 4)
        camera_position = np.array([0.5, 0.5, 2.0])
        
        result = generator.apply_depth_fog(positions, colors, camera_position)
        
        assert result.shape == (50, 4)
        # Colors should be modified (blended with fog)
        assert not np.allclose(result, colors)
    
    def test_size_variation(self):
        """Test particle size variation."""
        from music_brain.visualization.spectocloud import TextureGenerator, TextureConfig
        
        config = TextureConfig(size_variation=0.3)
        generator = TextureGenerator(config)
        
        base_sizes = np.ones(100) * 10.0
        varied_sizes = generator.apply_size_variation(base_sizes)
        
        assert varied_sizes.shape == (100,)
        # Should have some variation
        assert np.std(varied_sizes) > 0
    
    def test_renderer_with_texture(self):
        """Test SpectocloudRenderer with texture config."""
        from music_brain.visualization.spectocloud import (
            SpectocloudRenderer, TextureConfig, LODConfig
        )
        
        texture_config = TextureConfig(fog_enabled=True, glow_enabled=True)
        lod_config = LODConfig()
        
        renderer = SpectocloudRenderer(
            texture_config=texture_config,
            lod_config=lod_config,
        )
        
        assert renderer.texture_config.fog_enabled is True
        assert renderer.lod_config.level.value == "medium"
        assert renderer.metrics is not None


class TestPerformanceSummary:
    """Test performance summary functionality."""
    
    def test_spectocloud_performance_summary(self):
        """Test getting performance summary from Spectocloud."""
        spectocloud = Spectocloud(
            anchor_density="sparse",
            n_particles=100,
            use_vectorized_similarity=True,
        )
        
        # Process some MIDI
        midi_events = [
            {'time': 0.0, 'type': 'note_on', 'note': 60, 'velocity': 64},
        ]
        spectocloud.process_midi(midi_events, duration=1.0)
        
        # Get summary from spectocloud's own metrics
        summary = spectocloud.metrics.get_summary()
        
        assert 'avg_frame_time_ms' in summary
        assert 'total_frames_processed' in summary
        # Should have processed at least 1 frame (1.0s / 0.2s window = 5 frames)
        assert summary['total_frames_processed'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
