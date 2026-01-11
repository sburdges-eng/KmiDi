import numpy as np

from music_brain.tier1.midi_generator import HarmonyPredictor


def test_harmony_predictor_output_shape_and_bounds():
    hp = HarmonyPredictor(device="cpu")
    chords = hp.predict_harmony(emotion="calm", num_chords=4)
    assert len(chords) == 4
    for triad in chords:
        assert len(triad) == 3
        # MIDI pitch bounds (reasonable mid-register)
        assert all(0 <= p <= 127 for p in triad)


def test_harmony_predictor_deterministic_on_emotion():
    hp = HarmonyPredictor(device="cpu")
    chords_a = hp.predict_harmony(emotion="calm", num_chords=3)
    chords_b = hp.predict_harmony(emotion="calm", num_chords=3)
    assert chords_a == chords_b
