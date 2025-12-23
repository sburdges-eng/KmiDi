"""
Test suite for Penta-Core Mixer Python bindings

Tests the Python mixer engine functionality including:
- Channel controls (gain, pan, mute, solo)
- Send buses
- Master bus
- State management
- Integration with emotion presets
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.penta_core.mixer import MixerEngine, MixerState, apply_emotion_to_mixer
from music_brain.daw.mixer_params import EmotionMapper, MixerParameters


@pytest.fixture
def mixer():
    """Create a mixer instance for testing."""
    m = MixerEngine(sample_rate=48000.0)
    m.set_num_channels(4)
    m.set_num_send_buses(2)
    return m


@pytest.fixture
def test_audio():
    """Generate test audio signals."""
    sample_rate = 48000.0
    duration = 0.1  # 100ms
    num_frames = int(duration * sample_rate)

    # Generate different frequencies for each channel
    t = np.linspace(0, duration, num_frames)
    channels = []
    for freq in [440, 880, 1320, 1760]:
        signal = np.sin(2 * np.pi * freq * t).astype(np.float32)
        channels.append(signal)

    return np.array(channels)


class TestMixerInitialization:
    """Test mixer initialization and configuration."""

    def test_default_init(self):
        mixer = MixerEngine()
        assert mixer.sample_rate == 48000.0
        assert mixer.num_channels == 0
        assert mixer.num_send_buses == 0

    def test_custom_sample_rate(self):
        mixer = MixerEngine(sample_rate=44100.0)
        assert mixer.sample_rate == 44100.0

    def test_set_channels(self, mixer):
        assert mixer.num_channels == 4
        assert len(mixer._channel_gains) == 4
        assert len(mixer._channel_pans) == 4

    def test_set_send_buses(self, mixer):
        assert mixer.num_send_buses == 2
        assert len(mixer._send_return_levels) == 2


class TestChannelControls:
    """Test channel strip controls."""

    def test_channel_gain(self, mixer):
        mixer.set_channel_gain(0, -6.0)
        assert mixer._channel_gains[0] == -6.0

    def test_channel_gain_clamping(self, mixer):
        # Test lower limit
        mixer.set_channel_gain(0, -100.0)
        assert mixer._channel_gains[0] == -60.0

        # Test upper limit
        mixer.set_channel_gain(0, 50.0)
        assert mixer._channel_gains[0] == 12.0

    def test_channel_pan(self, mixer):
        mixer.set_channel_pan(0, 0.5)
        assert mixer._channel_pans[0] == 0.5

    def test_channel_pan_clamping(self, mixer):
        mixer.set_channel_pan(0, -2.0)
        assert mixer._channel_pans[0] == -1.0

        mixer.set_channel_pan(0, 2.0)
        assert mixer._channel_pans[0] == 1.0

    def test_channel_mute(self, mixer):
        mixer.set_channel_mute(0, True)
        assert mixer._channel_mutes[0] is True

        mixer.set_channel_mute(0, False)
        assert mixer._channel_mutes[0] is False

    def test_channel_solo(self, mixer):
        mixer.set_channel_solo(0, True)
        assert mixer._channel_solos[0] is True
        assert mixer.is_any_soloed()

        mixer.set_channel_solo(0, False)
        assert mixer._channel_solos[0] is False

    def test_clear_all_solo(self, mixer):
        mixer.set_channel_solo(0, True)
        mixer.set_channel_solo(1, True)
        assert mixer.is_any_soloed()

        mixer.clear_all_solo()
        assert not mixer.is_any_soloed()


class TestSendBuses:
    """Test send/return bus functionality."""

    def test_send_level(self, mixer):
        mixer.set_channel_send(0, 0, 0.5)
        assert mixer._channel_sends[0][0] == 0.5

    def test_send_level_clamping(self, mixer):
        mixer.set_channel_send(0, 0, -0.5)
        assert mixer._channel_sends[0][0] == 0.0

        mixer.set_channel_send(0, 0, 1.5)
        assert mixer._channel_sends[0][0] == 1.0

    def test_return_level(self, mixer):
        mixer.set_send_return_level(0, 1.5)
        assert mixer._send_return_levels[0] == 1.5

    def test_send_mute(self, mixer):
        mixer.set_send_mute(0, True)
        assert mixer._send_mutes[0] is True


class TestMasterBus:
    """Test master bus controls."""

    def test_master_gain(self, mixer):
        mixer.set_master_gain(3.0)
        assert mixer._master_gain == 3.0

    def test_master_limiter(self, mixer):
        mixer.set_master_limiter(True, -2.0)
        assert mixer._master_limiter_enabled is True
        assert mixer._master_limiter_threshold == -2.0


class TestAudioProcessing:
    """Test audio processing functionality."""

    def test_basic_processing(self, mixer, test_audio):
        output_l, output_r = mixer.process(test_audio)

        assert output_l.shape[0] == test_audio.shape[1]
        assert output_r.shape[0] == test_audio.shape[1]
        assert output_l.dtype == np.float32
        assert output_r.dtype == np.float32

    def test_unity_gain_processing(self, mixer, test_audio):
        # All channels at unity gain (0 dB)
        for ch in range(4):
            mixer.set_channel_gain(ch, 0.0)
            mixer.set_channel_pan(ch, 0.0)

        output_l, output_r = mixer.process(test_audio)

        # Should have output
        assert np.max(np.abs(output_l)) > 0.0
        assert np.max(np.abs(output_r)) > 0.0

    def test_muted_channel(self, mixer, test_audio):
        # Mute all channels
        for ch in range(4):
            mixer.set_channel_mute(ch, True)

        output_l, output_r = mixer.process(test_audio)

        # Should be silent
        assert np.allclose(output_l, 0.0, atol=1e-6)
        assert np.allclose(output_r, 0.0, atol=1e-6)

    def test_solo_logic(self, mixer, test_audio):
        # Solo channel 0
        mixer.set_channel_solo(0, True)

        output_l, output_r = mixer.process(test_audio)

        # Should have output (channel 0 is soloed)
        assert np.max(np.abs(output_l)) > 0.0

    def test_pan_processing(self, mixer):
        # Generate mono signal
        sample_rate = 48000.0
        duration = 0.1
        num_frames = int(duration * sample_rate)
        t = np.linspace(0, duration, num_frames)
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Create mono input (all other channels silent)
        inputs = np.zeros((4, num_frames), dtype=np.float32)
        inputs[0] = signal

        # Pan hard left
        mixer.set_channel_pan(0, -1.0)
        output_l, output_r = mixer.process(inputs)

        peak_l = np.max(np.abs(output_l))
        peak_r = np.max(np.abs(output_r))

        # Left should be louder
        assert peak_l > peak_r

        # Pan hard right
        mixer.set_channel_pan(0, 1.0)
        output_l, output_r = mixer.process(inputs)

        peak_l = np.max(np.abs(output_l))
        peak_r = np.max(np.abs(output_r))

        # Right should be louder
        assert peak_r > peak_l

    def test_gain_processing(self, mixer):
        # Generate test signal
        sample_rate = 48000.0
        duration = 0.1
        num_frames = int(duration * sample_rate)
        t = np.linspace(0, duration, num_frames)
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        inputs = np.zeros((4, num_frames), dtype=np.float32)
        inputs[0] = signal

        # Process with unity gain
        mixer.set_channel_gain(0, 0.0)
        output_l1, _ = mixer.process(inputs)
        peak1 = np.max(np.abs(output_l1))

        # Process with +6 dB gain (should roughly double)
        mixer.set_channel_gain(0, 6.0)
        output_l2, _ = mixer.process(inputs)
        peak2 = np.max(np.abs(output_l2))

        # Peak should increase
        assert peak2 > peak1 * 1.5


class TestMetering:
    """Test metering functionality."""

    def test_channel_metering(self, mixer, test_audio):
        output_l, output_r = mixer.process(test_audio)

        # Check meters updated
        for ch in range(4):
            assert mixer.get_channel_peak(ch) > 0.0
            assert mixer.get_channel_rms(ch) > 0.0

            # RMS should be less than peak
            assert mixer.get_channel_rms(ch) <= mixer.get_channel_peak(ch)

    def test_master_metering(self, mixer, test_audio):
        output_l, output_r = mixer.process(test_audio)

        assert mixer.get_master_peak_l() > 0.0
        assert mixer.get_master_peak_r() > 0.0

    def test_reset_meters(self, mixer, test_audio):
        output_l, output_r = mixer.process(test_audio)

        # Meters should be active
        assert mixer.get_channel_peak(0) > 0.0

        # Reset
        mixer.reset_all_meters()

        # Meters should be zero
        for ch in range(4):
            assert mixer.get_channel_peak(ch) == 0.0
            assert mixer.get_channel_rms(ch) == 0.0


class TestStateManagement:
    """Test mixer state save/load functionality."""

    def test_get_state(self, mixer):
        mixer.set_channel_gain(0, -6.0)
        mixer.set_channel_pan(1, 0.5)
        mixer.set_channel_mute(2, True)
        mixer.set_master_gain(3.0)

        state = mixer.get_state()

        assert isinstance(state, MixerState)
        assert state.num_channels == 4
        assert state.channel_gains[0] == -6.0
        assert state.channel_pans[1] == 0.5
        assert state.channel_mutes[2] is True
        assert state.master_gain == 3.0

    def test_load_state(self, mixer):
        # Create a state
        state = MixerState(
            num_channels=4,
            num_send_buses=2,
            channel_gains=[-6.0, -3.0, 0.0, -9.0],
            channel_pans=[0.0, -0.5, 0.5, 0.0],
            channel_mutes=[False, False, True, False],
            channel_solos=[False, True, False, False],
            channel_peaks=[0.0] * 4,
            channel_rms=[0.0] * 4,
            send_return_levels=[1.0, 0.5],
            send_mutes=[False, False],
            master_gain=2.0,
            master_limiter_enabled=True,
            master_limiter_threshold=-2.0,
            master_peak_l=0.0,
            master_peak_r=0.0
        )

        mixer.load_state(state)

        assert mixer._channel_gains[0] == -6.0
        assert mixer._channel_pans[1] == -0.5
        assert mixer._channel_mutes[2] is True
        assert mixer._channel_solos[1] is True
        assert mixer._master_gain == 2.0


class TestHelperFunctions:
    """Test helper functions."""

    def test_db_to_linear(self):
        # 0 dB = 1.0 linear
        assert np.isclose(MixerEngine._db_to_linear(0.0), 1.0)

        # +6 dB ≈ 2.0 linear
        assert np.isclose(MixerEngine._db_to_linear(6.0), 2.0, rtol=0.01)

        # -6 dB ≈ 0.5 linear
        assert np.isclose(MixerEngine._db_to_linear(-6.0), 0.5, rtol=0.01)

    def test_pan_coefficients(self):
        # Center pan
        pan_l, pan_r = MixerEngine._calculate_pan_coefficients(0.0)
        assert np.isclose(pan_l, pan_r, atol=0.01)  # Equal power

        # Hard left
        pan_l, pan_r = MixerEngine._calculate_pan_coefficients(-1.0)
        assert pan_l > pan_r

        # Hard right
        pan_l, pan_r = MixerEngine._calculate_pan_coefficients(1.0)
        assert pan_r > pan_l


class TestEmotionIntegration:
    """Test integration with emotion-based mixer parameters."""

    def test_emotion_mapper_integration(self, mixer):
        mapper = EmotionMapper()
        grief_params = mapper.get_preset("grief")

        assert grief_params is not None
        assert isinstance(grief_params, MixerParameters)

    def test_apply_emotion_to_mixer(self, mixer):
        mapper = EmotionMapper()
        grief_params = mapper.get_preset("grief")

        apply_emotion_to_mixer(mixer, grief_params, channel=0)

        # Check that parameters were applied
        assert mixer._channel_pans[0] == grief_params.pan_position


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_channel_index(self, mixer):
        # Should not crash for invalid indices
        mixer.set_channel_gain(100, -6.0)  # Out of range
        mixer.set_channel_pan(-1, 0.5)     # Negative

    def test_invalid_send_bus_index(self, mixer):
        mixer.set_channel_send(0, 100, 0.5)  # Out of range

    def test_empty_input(self, mixer):
        # Empty audio input
        inputs = np.zeros((4, 0), dtype=np.float32)
        output_l, output_r = mixer.process(inputs)

        assert len(output_l) == 0
        assert len(output_r) == 0

    def test_silent_input(self, mixer):
        # Silent input
        silence = np.zeros((4, 1000), dtype=np.float32)
        output_l, output_r = mixer.process(silence)

        assert np.allclose(output_l, 0.0)
        assert np.allclose(output_r, 0.0)


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.benchmark
    def test_processing_performance(self, mixer, benchmark):
        # Generate 1 second of audio
        sample_rate = 48000.0
        num_frames = 48000
        t = np.linspace(0, 1.0, num_frames)

        inputs = np.array([
            np.sin(2 * np.pi * 440 * t),
            np.sin(2 * np.pi * 880 * t),
            np.sin(2 * np.pi * 1320 * t),
            np.sin(2 * np.pi * 1760 * t),
        ], dtype=np.float32)

        def process():
            return mixer.process(inputs)

        if benchmark:
            result = benchmark(process)
        else:
            # If benchmark fixture not available, just run once
            result = process()

        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
