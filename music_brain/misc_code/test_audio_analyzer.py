import numpy as np

from music_brain.audio import analyzer as analyzer_module
from music_brain.audio.analyzer import AudioAnalyzer, extract_features


class DummyLibrosa:
    class feature:
        @staticmethod
        def chroma_cqt(y, sr):
            return np.zeros((12, 16))

    class beat:
        @staticmethod
        def tempo(*args, **kwargs):
            return 120.0

        @staticmethod
        def beat_track(*args, **kwargs):
            return 120.0, np.array([0, 100, 200])

    class onset:
        @staticmethod
        def onset_strength(*args, **kwargs):
            return np.ones(16)

    class segment:
        @staticmethod
        def agglomerative(*args, **kwargs):
            return np.zeros(4)


def test_detect_key_confidence_handles_nan(monkeypatch):
    monkeypatch.setattr(analyzer_module, "librosa", DummyLibrosa())
    analyzer = AudioAnalyzer()
    result = analyzer.detect_key_details(np.zeros(2048), 22050)
    assert np.isfinite(result.confidence)
    assert 0.0 <= result.confidence <= 1.0


def test_extract_features_handles_scalar_tempo(monkeypatch):
    monkeypatch.setattr(analyzer_module, "librosa", DummyLibrosa())
    data = np.zeros(2048)
    features = extract_features(data, 22050)
    assert "tempo_curve" in features
    assert isinstance(features["tempo_curve"], list)
    assert "tempo_curve_tail" in features
    assert isinstance(features["tempo_curve_tail"], list)

