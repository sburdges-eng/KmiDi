"""
Unit tests for Tier 1 Audio Generator
Tests audio synthesis and emotion-based timbre control
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from music_brain.tier1.audio_generator import AudioGenerator


class TestAudioGenerator(unittest.TestCase):
    """Test Tier 1 audio synthesis"""

    def setUp(self):
        """Initialize audio generator"""
        self.audio_gen = AudioGenerator(sample_rate=44100)

    def test_audio_synthesis(self):
        """Test basic audio synthesis from MIDI notes"""
        midi_notes = [60, 64, 67, 72]  # C, E, G, C notes
        audio = self.audio_gen.synthesize_texture(
            midi_notes=midi_notes,
            duration_sec=2.0,
            emotion="neutral"
        )
        
        self.assertIsNotNone(audio)
        expected_samples = int(2.0 * 44100)
        self.assertGreater(len(audio), expected_samples * 0.9)  # Allow some tolerance

    def test_emotion_timbre_control(self):
        """Test that emotion affects timbre generation"""
        midi_notes = [60, 64, 67]
        
        audio_grief = self.audio_gen.synthesize_texture(
            midi_notes=midi_notes,
            duration_sec=1.0,
            emotion="grief"
        )
        
        audio_joy = self.audio_gen.synthesize_texture(
            midi_notes=midi_notes,
            duration_sec=1.0,
            emotion="joy"
        )
        
        # Different emotions should produce different audio
        self.assertFalse(np.allclose(audio_grief, audio_joy))

    def test_audio_amplitude_normalization(self):
        """Test that audio is properly normalized"""
        midi_notes = [60, 64, 67]
        audio = self.audio_gen.synthesize_texture(
            midi_notes=midi_notes,
            duration_sec=1.0
        )
        
        # Audio should be in valid range
        self.assertLessEqual(np.max(np.abs(audio)), 1.0)
        self.assertGreater(np.max(np.abs(audio)), 0.1)  # Should have meaningful signal

    def test_adsr_envelope(self):
        """Test ADSR envelope application"""
        # Create a simple test signal
        signal = np.ones(44100)  # 1 second of ones
        
        envelope = self.audio_gen._adsr_envelope(
            signal_length=44100,
            attack_ms=10,
            decay_ms=50,
            sustain_level=0.8,
            release_ms=100
        )
        
        self.assertEqual(len(envelope), 44100)
        # Attack should start at 0
        self.assertLess(envelope[0], 0.5)
        # Should reach sustain level
        self.assertTrue(any(np.abs(envelope[1000:2000] - 0.8) < 0.1))

    def test_different_instruments(self):
        """Test synthesis with different instruments"""
        midi_notes = [60, 64, 67]
        instruments = ["piano", "strings", "pad"]
        
        for instrument in instruments:
            audio = self.audio_gen.synthesize_texture(
                midi_notes=midi_notes,
                duration_sec=0.5,
                instrument=instrument
            )
            self.assertIsNotNone(audio)
            self.assertGreater(len(audio), 0)


if __name__ == '__main__':
    unittest.main()
