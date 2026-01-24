"""
Tests for lightweight audio analysis utilities.
"""

from __future__ import annotations

import numpy as np

from music_brain.audio.analyzer import AudioAnalyzer
from music_brain.audio.frequency import FrequencyAnalyzer
from music_brain.audio.chord_detection import ChordDetector


def sine_wave(freq: float, duration: float = 1.0, sr: int = 22050) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_frequency_pitch_detection():
    sr = 22050
    wave = sine_wave(440.0, sr=sr)
    analyzer = FrequencyAnalyzer()
    freq = analyzer.pitch_detection(wave, sr)
    assert abs(freq - 440.0) < 5.0


def test_chord_detector_summarize():
    sr = 22050
    wave = sine_wave(261.63, sr=sr) + sine_wave(329.63, sr=sr) + sine_wave(392.0, sr=sr)
    detector = ChordDetector()
    chords = detector.detect_chords(wave, sr)
    progression = detector.summarize_progression(chords)
    assert progression  # ensures some chord detected


def test_audio_analyzer_waveform():
    sr = 22050
    wave = sine_wave(440.0, sr=sr)
    analyzer = AudioAnalyzer(sample_rate=sr)
    analysis = analyzer.analyze_waveform(wave, sr)
    assert analysis.key == "A"
    assert analysis.chords

