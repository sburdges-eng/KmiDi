"""
Pitch Detection Tests - YIN algorithm and autocorrelation tests.

Tests:
- YIN algorithm accuracy
- Autocorrelation-based pitch detection
- Various waveforms (sine, square, sawtooth)
- Noise and mixed signals
- Frequency range validation
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from penta_core.dsp.parrot_dsp import detect_pitch


def generate_sine_wave(frequency: float, duration: float = 0.1, sample_rate: float = 44100.0) -> list:
    """Generate a sine wave at the given frequency."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, False)
    samples = np.sin(2 * np.pi * frequency * t).tolist()
    return samples


def generate_square_wave(frequency: float, duration: float = 0.1, sample_rate: float = 44100.0) -> list:
    """Generate a square wave at the given frequency."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, False)
    samples = np.sign(np.sin(2 * np.pi * frequency * t)).tolist()
    return samples


def generate_sawtooth_wave(frequency: float, duration: float = 0.1, sample_rate: float = 44100.0) -> list:
    """Generate a sawtooth wave at the given frequency."""
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, False)
    samples = (2 * (t * frequency - np.floor(0.5 + t * frequency))).tolist()
    return samples


def add_noise(samples: list, noise_level: float = 0.1) -> list:
    """Add white noise to samples."""
    noise = np.random.randn(len(samples)) * noise_level
    return (np.array(samples) + noise).tolist()


class TestPitchDetectionAccuracy:
    """Test pitch detection accuracy with known signals."""
    
    @pytest.mark.parametrize("frequency", [220.0, 440.0, 880.0, 1320.0])  # A3, A4, A5, E6
    def test_sine_wave_accuracy(self, frequency):
        """Test that sine waves are detected accurately."""
        samples = generate_sine_wave(frequency, duration=0.2)
        detected = detect_pitch(samples, sample_rate=44100.0)
        
        assert detected is not None, f"Should detect pitch for {frequency}Hz sine wave"
        
        # Allow 2% error tolerance
        error_percent = abs(detected - frequency) / frequency * 100
        assert error_percent < 2.0, \
            f"Pitch detection error too large: {detected}Hz vs {frequency}Hz ({error_percent:.1f}%)"
    
    @pytest.mark.parametrize("frequency", [220.0, 440.0, 880.0])
    def test_square_wave_accuracy(self, frequency):
        """Test that square waves are detected accurately (should detect fundamental)."""
        samples = generate_square_wave(frequency, duration=0.2)
        detected = detect_pitch(samples, sample_rate=44100.0)
        
        assert detected is not None, f"Should detect pitch for {frequency}Hz square wave"
        
        # Square waves have strong fundamental, should be detectable
        error_percent = abs(detected - frequency) / frequency * 100
        assert error_percent < 5.0, \
            f"Pitch detection error too large: {detected}Hz vs {frequency}Hz ({error_percent:.1f}%)"
    
    @pytest.mark.parametrize("frequency", [220.0, 440.0, 880.0])
    def test_sawtooth_wave_accuracy(self, frequency):
        """Test that sawtooth waves are detected accurately."""
        samples = generate_sawtooth_wave(frequency, duration=0.2)
        detected = detect_pitch(samples, sample_rate=44100.0)
        
        assert detected is not None, f"Should detect pitch for {frequency}Hz sawtooth wave"
        
        # Sawtooth waves should be detectable
        error_percent = abs(detected - frequency) / frequency * 100
        assert error_percent < 5.0, \
            f"Pitch detection error too large: {detected}Hz vs {frequency}Hz ({error_percent:.1f}%)"
    
    def test_frequency_range(self):
        """Test detection across frequency range."""
        frequencies = [100.0, 200.0, 400.0, 800.0, 1600.0]
        detected_count = 0
        
        for freq in frequencies:
            samples = generate_sine_wave(freq, duration=0.2)
            detected = detect_pitch(samples, sample_rate=44100.0)
            
            if detected is not None:
                error_percent = abs(detected - freq) / freq * 100
                if error_percent < 5.0:
                    detected_count += 1
        
        # Should detect at least 80% of frequencies
        assert detected_count >= len(frequencies) * 0.8, \
            f"Should detect at least 80% of frequencies, detected {detected_count}/{len(frequencies)}"


class TestPitchDetectionRobustness:
    """Test pitch detection robustness with noise and mixed signals."""
    
    def test_noise_robustness(self):
        """Test that pitch detection works with added noise."""
        frequency = 440.0
        samples = generate_sine_wave(frequency, duration=0.2)
        
        # Test with increasing noise levels
        for noise_level in [0.05, 0.1, 0.2]:
            noisy_samples = add_noise(samples, noise_level=noise_level)
            detected = detect_pitch(noisy_samples, sample_rate=44100.0)
            
            # With moderate noise, should still detect
            if noise_level <= 0.1:
                assert detected is not None, \
                    f"Should detect pitch with noise level {noise_level}"
                
                if detected is not None:
                    error_percent = abs(detected - frequency) / frequency * 100
                    assert error_percent < 10.0, \
                        f"Pitch detection with noise error too large: {error_percent:.1f}%"
    
    def test_mixed_signals(self):
        """Test detection with mixed frequencies (should detect fundamental)."""
        # Create signal with fundamental + harmonics
        fundamental = 220.0
        samples_fund = generate_sine_wave(fundamental, duration=0.2)
        samples_harm = generate_sine_wave(fundamental * 2, duration=0.2)  # Second harmonic
        
        # Mix signals (fundamental louder)
        mixed = (np.array(samples_fund) * 0.7 + np.array(samples_harm) * 0.3).tolist()
        
        detected = detect_pitch(mixed, sample_rate=44100.0)
        
        # Should detect fundamental
        if detected is not None:
            error_percent = abs(detected - fundamental) / fundamental * 100
            # Allow larger error for mixed signals
            assert error_percent < 15.0, \
                f"Should detect fundamental, got {detected}Hz vs {fundamental}Hz ({error_percent:.1f}%)"
    
    def test_silence_handling(self):
        """Test that silence returns None."""
        silence = [0.0] * 1024
        detected = detect_pitch(silence, sample_rate=44100.0)
        
        # Silence should not produce a pitch
        assert detected is None or detected == 0.0, \
            "Silence should not produce a pitch detection"
    
    def test_too_short_signal(self):
        """Test that very short signals return None."""
        # Less than 256 samples (minimum for detection)
        short_signal = generate_sine_wave(440.0, duration=0.001)  # ~44 samples at 44.1kHz
        
        detected = detect_pitch(short_signal, sample_rate=44100.0)
        assert detected is None, "Very short signals should return None"


class TestPitchDetectionParameters:
    """Test pitch detection with different parameters."""
    
    def test_min_max_frequency_range(self):
        """Test that min/max frequency parameters are respected."""
        frequency = 440.0
        samples = generate_sine_wave(frequency, duration=0.2)
        
        # Should detect within range
        detected = detect_pitch(samples, sample_rate=44100.0, min_hz=100.0, max_hz=2000.0)
        assert detected is not None, "Should detect within valid range"
        
        # Should not detect outside range
        detected_outside = detect_pitch(samples, sample_rate=44100.0, min_hz=1000.0, max_hz=2000.0)
        # 440Hz is outside this range, so detection may fail
        if detected_outside is not None:
            assert detected_outside >= 1000.0, "Detected frequency should respect max range"
    
    def test_different_sample_rates(self):
        """Test pitch detection at different sample rates."""
        frequency = 440.0
        
        for sample_rate in [22050.0, 44100.0, 48000.0]:
            samples = generate_sine_wave(frequency, duration=0.2, sample_rate=sample_rate)
            detected = detect_pitch(samples, sample_rate=sample_rate)
            
            if detected is not None:
                error_percent = abs(detected - frequency) / frequency * 100
                assert error_percent < 5.0, \
                    f"Should detect accurately at {sample_rate}Hz sample rate"


class TestPitchDetectionEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_low_frequency(self):
        """Test detection of very low frequencies."""
        frequency = 50.0  # Very low frequency
        samples = generate_sine_wave(frequency, duration=0.5)  # Longer duration needed
        
        detected = detect_pitch(samples, sample_rate=44100.0, min_hz=30.0, max_hz=2000.0)
        
        # Low frequencies are harder to detect accurately
        if detected is not None:
            error_percent = abs(detected - frequency) / frequency * 100
            assert error_percent < 20.0, \
                f"Low frequency detection should be within 20%: {error_percent:.1f}%"
    
    def test_very_high_frequency(self):
        """Test detection of very high frequencies."""
        frequency = 2000.0  # High frequency
        samples = generate_sine_wave(frequency, duration=0.2)
        
        detected = detect_pitch(samples, sample_rate=44100.0, min_hz=50.0, max_hz=3000.0)
        
        if detected is not None:
            error_percent = abs(detected - frequency) / frequency * 100
            assert error_percent < 10.0, \
                f"High frequency detection should be accurate: {error_percent:.1f}%"
    
    def test_perfect_octave_detection(self):
        """Test that octaves are detected accurately."""
        # Test multiple octaves of A
        octaves = [110.0, 220.0, 440.0, 880.0]  # A2, A3, A4, A5
        
        for freq in octaves:
            samples = generate_sine_wave(freq, duration=0.2)
            detected = detect_pitch(samples, sample_rate=44100.0)
            
            if detected is not None:
                error_percent = abs(detected - freq) / freq * 100
                assert error_percent < 3.0, \
                    f"Octave {freq}Hz should be detected accurately: {error_percent:.1f}% error"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

