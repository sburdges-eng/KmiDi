"""
High-level audio analysis utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import wave

import numpy as np

try:  # Optional dependency
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    librosa = None

from music_brain.audio.chord_detection import ChordDetector, DetectedChord
from music_brain.audio.frequency import FrequencyAnalyzer, frequency_to_note_name


@dataclass
class RhythmAnalysis:
    bpm: float
    confidence: float


@dataclass
class SpectralAnalysis:
    energy: float
    spectral_centroid: float
    harmonic_content: Dict[str, float]


@dataclass
class KeyDetectionResult:
    root: str
    mode: str
    confidence: float
    correlations: Dict[str, float]


@dataclass
class AudioAnalysis:
    bpm: float
    key: str
    mode: str
    rhythm: RhythmAnalysis
    spectral: SpectralAnalysis
    chords: List[str]
    key_confidence: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "bpm": self.bpm,
            "key": self.key,
            "mode": self.mode,
            "key_confidence": self.key_confidence,
            "rhythm": asdict(self.rhythm),
            "spectral": {
                **asdict(self.spectral),
            },
            "chords": self.chords,
        }


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class AudioAnalyzer:
    """
    Provide tempo, key, and chord insights for an audio file or waveform.
    """

    def __init__(self, sample_rate: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self.freq_analyzer = FrequencyAnalyzer()
        self.chord_detector = ChordDetector()

    def analyze_file(self, filepath: str) -> AudioAnalysis:
        data, sr = self._load_audio(filepath)
        return self.analyze_waveform(data, sr)

    def analyze_waveform(self, audio_data: np.ndarray, sample_rate: int) -> AudioAnalysis:
        bpm, bpm_conf = self.detect_bpm(audio_data, sample_rate)
        key_result = self.detect_key_details(audio_data, sample_rate)
        spectral = self._spectral_analysis(audio_data, sample_rate)
        chords = self.chord_detector.summarize_progression(
            self.chord_detector.detect_chords(audio_data, sample_rate)
        )
        rhythm = RhythmAnalysis(bpm=bpm, confidence=bpm_conf)
        return AudioAnalysis(
            bpm=bpm,
            key=key_result.root,
            mode=key_result.mode,
            rhythm=rhythm,
            spectral=spectral,
            chords=chords,
            key_confidence=key_result.confidence,
        )

    def detect_bpm(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[float, float]:
        if librosa:
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            confidence = float(len(beats) / (audio_data.size / sample_rate) if beats.size else 0.0)
            return float(tempo), confidence

        autocorr = np.correlate(audio_data, audio_data, mode="full")[audio_data.size - 1 :]
        autocorr[: sample_rate // 2] = 0
        peak_index = int(np.argmax(autocorr[: sample_rate * 2]) or 1)
        tempo_seconds = peak_index / sample_rate
        bpm = 60.0 / tempo_seconds if tempo_seconds else 60.0
        return bpm, 0.3

    def detect_key(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, str]:
        result = self.detect_key_details(audio_data, sample_rate)
        return result.root, result.mode

    def detect_key_details(self, audio_data: np.ndarray, sample_rate: int) -> KeyDetectionResult:
        if librosa:
            chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
            if chroma.size == 0:
                return KeyDetectionResult("C", "major", 0.0, {})
            avg = np.mean(chroma, axis=1)
            if np.allclose(avg, 0):
                return KeyDetectionResult("C", "major", 0.0, {})
            avg = avg / (np.sum(avg) + 1e-12)
            correlations: Dict[str, float] = {}
            best_root = "C"
            best_mode = "major"
            best_corr = -1.0

            for root in range(12):
                for mode_name, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
                    rotated = np.roll(profile, root)
                    rotated = rotated / (np.sum(rotated) + 1e-12)
                    corr = self._safe_correlation(avg, rotated)
                    key = f"{NOTE_NAMES[root]}_{mode_name}"
                    correlations[key] = corr
                    if np.isfinite(corr) and corr > best_corr:
                        best_corr = corr
                        best_root = NOTE_NAMES[root]
                        best_mode = mode_name

            confidence = self._confidence_from_correlations(list(correlations.values()), best_corr)
            return KeyDetectionResult(best_root, best_mode, confidence, correlations)

        frequency = self.freq_analyzer.pitch_detection(audio_data, sample_rate)
        note = frequency_to_note_name(frequency)
        return KeyDetectionResult(note, "major", 0.2, {})

    def _spectral_analysis(self, audio_data: np.ndarray, sample_rate: int) -> SpectralAnalysis:
        energy = float(np.sqrt(np.mean(np.square(audio_data))))
        spectrum = self.freq_analyzer.fft_analysis(audio_data, sample_rate)
        denom = float(np.sum(spectrum.magnitudes))
        if denom <= 0:
            denom = 1e-9
        centroid = float(np.sum(spectrum.frequencies * spectrum.magnitudes) / denom)
        harmonic = self.freq_analyzer.harmonic_content(audio_data, sample_rate)
        return SpectralAnalysis(
            energy=energy,
            spectral_centroid=centroid,
            harmonic_content=harmonic,
        )

    @staticmethod
    def _safe_correlation(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0
        if np.allclose(vec_a, vec_a[0]) or np.allclose(vec_b, vec_b[0]):
            return 0.0
        corr = np.corrcoef(vec_a, vec_b)[0, 1]
        if not np.isfinite(corr):
            return 0.0
        return float(corr)

    @staticmethod
    def _confidence_from_correlations(corrs: List[float], best_corr: float) -> float:
        valid = [c for c in corrs if np.isfinite(c)]
        if not valid:
            return 0.0
        if len(valid) == 1:
            return float(np.clip((valid[0] + 1.0) / 2.0, 0.0, 1.0))
        baseline = np.percentile(valid, 85)
        if not np.isfinite(baseline):
            baseline = float(np.nanmean(valid)) if np.any(np.isfinite(valid)) else 0.0
        denom = max(1e-3, 1.0 - baseline)
        confidence = (best_corr - baseline) / denom
        return float(np.clip(confidence, 0.0, 1.0))

    def _load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        if librosa:
            data, sr = librosa.load(filepath, sr=self.sample_rate)
            return data, int(sr)

        with wave.open(filepath, "rb") as wav_file:
            sr = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio /= np.iinfo(np.int16).max
        return audio, sr


def extract_features(
    audio_data: np.ndarray,
    sample_rate: int,
    include_segments: bool = False,
    segment_count: int = 8,
) -> Dict[str, object]:
    """
    Extract high-level audio features for offline analysis.
    """
    analyzer = AudioAnalyzer()
    bpm, bpm_conf = analyzer.detect_bpm(audio_data, sample_rate)
    key_result = analyzer.detect_key_details(audio_data, sample_rate)
    spectral = analyzer._spectral_analysis(audio_data, sample_rate)

    features: Dict[str, object] = {
        "bpm": bpm,
        "bpm_confidence": bpm_conf,
        "key": key_result.root,
        "mode": key_result.mode,
        "key_confidence": key_result.confidence,
        "spectral_centroid": spectral.spectral_centroid,
        "energy": spectral.energy,
        "harmonic_content": spectral.harmonic_content,
    }

    tempo_curve_tail: List[float] = []
    if librosa:
        try:
            tempo_curve = librosa.beat.tempo(y=audio_data, sr=sample_rate, aggregate=None)
        except TypeError:
            tempo_curve = librosa.beat.tempo(y=audio_data, sr=sample_rate)

        tempo_array = np.atleast_1d(np.asarray(tempo_curve, dtype=float))
        tempo_array = tempo_array[np.isfinite(tempo_array)]
        tempo_curve_tail = tempo_array[-5:].tolist() if tempo_array.size else []
        features["tempo_curve"] = tempo_array.tolist()
    else:
        features["tempo_curve"] = []

    features["tempo_curve_tail"] = tempo_curve_tail

    if include_segments and librosa:
        segments = np.array_split(audio_data, max(1, segment_count))
        features["segments"] = [
            {
                "index": idx,
                "energy": float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0,
            }
            for idx, seg in enumerate(segments)
        ]
    return features


__all__ = [
    "AudioAnalyzer",
    "AudioAnalysis",
    "RhythmAnalysis",
    "SpectralAnalysis",
    "KeyDetectionResult",
    "extract_features",
]

