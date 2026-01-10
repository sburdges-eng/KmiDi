"""
Unit tests for Tier 1 MIDI Generator
Tests baseline MIDI generation without fine-tuning
"""

import unittest
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from music_brain.tier1.midi_generator import MelodyTransformer, HarmonyPredictor, GroovePredictor


class TestMIDIGenerator(unittest.TestCase):
    """Test Tier 1 MIDI generation pipeline"""

    def setUp(self):
        """Initialize generators"""
        self.melody_gen = MelodyTransformer()
        self.harmony_gen = HarmonyPredictor()
        self.groove_gen = GroovePredictor()

    def test_melody_generation(self):
        """Test melody generation with emotion input"""
        emotion = "grief"
        melody = self.melody_gen.generate_melody(
            emotion=emotion,
            num_notes=16,
            temperature=0.7
        )
        
        self.assertIsNotNone(melody)
        self.assertEqual(len(melody), 16)
        # Verify note values are in valid MIDI range (0-127)
        self.assertTrue(all(0 <= note <= 127 for note in melody if note is not None))

    def test_harmony_prediction(self):
        """Test harmony/chord prediction"""
        emotion = "hope"
        chords = self.harmony_gen.predict_harmony(emotion=emotion, num_chords=8)
        
        self.assertIsNotNone(chords)
        self.assertEqual(len(chords), 8)

    def test_groove_generation(self):
        """Test groove pattern generation"""
        groove = self.groove_gen.generate_groove(
            emotion="joy",
            tempo=120,
            num_bars=4
        )
        
        self.assertIsNotNone(groove)
        # Verify groove has valid structure
        self.assertTrue(hasattr(groove, '__len__'))

    def test_full_pipeline(self):
        """Test complete MIDI generation pipeline"""
        emotion = "neutral"
        midi_data = self.melody_gen.full_pipeline(
            emotion=emotion,
            duration_bars=8
        )
        
        self.assertIsNotNone(midi_data)
        # Verify MIDI data structure
        self.assertTrue(hasattr(midi_data, '__len__'))

    def test_emotion_embedding(self):
        """Test emotion embedding generation"""
        emotions = ["grief", "joy", "anger", "fear", "neutral"]
        
        for emotion in emotions:
            embedding = self.melody_gen.emotion_to_embedding(emotion)
            self.assertIsNotNone(embedding)
            # Verify embedding dimensions
            self.assertEqual(len(embedding.shape), 1)

    def test_reproducibility(self):
        """Test that same emotion produces similar melodies with same seed"""
        emotion = "sadness"
        
        melody1 = self.melody_gen.generate_melody(emotion=emotion, seed=42, num_notes=8)
        melody2 = self.melody_gen.generate_melody(emotion=emotion, seed=42, num_notes=8)
        
        # With same seed, should be identical
        np.testing.assert_array_equal(melody1, melody2)


if __name__ == '__main__':
    unittest.main()
