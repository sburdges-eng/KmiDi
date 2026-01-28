"""
Audio Analyzer - Main audio analysis interface for DAiW.

Provides unified access to:
- BPM/tempo detection
- Key detection
- Feature extraction
- Audio segmentation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum
import warnings

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - degrade gracefully
    SCIPY_AVAILABLE = False

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    LIBROSA_AVAILABLE = False
    np = None  # type: ignore

from music_brain.audio.feel import AudioFeatures, analyze_feel

DEFAULT_TARGET_SR = 44100
DEFAULT_HOP_LENGTH = 512
ENERGY_EPS = 1e-9


# =================================================================
# DATA CLASSES
# =================================================================

class KeyMode(Enum):
    """Musical key mode."""
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class KeyDetectionResult:
    """Result of key detection."""
    key: str  # e.g., "C", "F#", "Bb"
    mode: KeyMode
    confidence: float  # 0.0 - 1.0
    correlation_vector: List[float] = field(default_factory=list)  # 12 values
    
    @property
    def full_key(self) -> str:
        """Full key name (e.g., 'C major', 'A minor')."""
        return f"{self.key} {self.mode.value}"
    
    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "mode": self.mode.value,
            "full_key": self.full_key,
            "confidence": self.confidence,
        }


@dataclass
class BPMDetectionResult:
    """Result of BPM/tempo detection."""
    bpm: float
    confidence: float  # 0.0 - 1.0
    beat_frames: List[int] = field(default_factory=list)
    beat_times: List[float] = field(default_factory=list)  # In seconds
    tempo_alternatives: List[float] = field(default_factory=list)  # Other likely tempos
    
    def to_dict(self) -> Dict:
        return {
            "bpm": self.bpm,
            "confidence": self.confidence,
            "beat_count": len(self.beat_times),
            "alternatives": self.tempo_alternatives[:3],
        }


@dataclass
class AudioSegment:
    """A segment of audio with detected characteristics."""
    start_time: float  # Seconds
    end_time: float
    energy: float
    key: Optional[str] = None
    bpm: Optional[float] = None
    label: str = ""  # e.g., "intro", "verse", "chorus"
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class AudioAnalysis:
    """Complete audio analysis result."""
    filepath: str
    duration_seconds: float
    sample_rate: int
    
    # Detected features
    bpm_result: Optional[BPMDetectionResult] = None
    key_result: Optional[KeyDetectionResult] = None
    
    # Audio features
    features: Optional[AudioFeatures] = None
    feature_summary: Dict[str, float] = field(default_factory=dict)
    
    # Segmentation
    segments: List[AudioSegment] = field(default_factory=list)
    
    # Raw features (for advanced use)
    chroma: Optional[List[List[float]]] = None
    mfcc: Optional[List[List[float]]] = None
    
    def to_dict(self) -> Dict:
        result = {
            "filepath": self.filepath,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
        }
        if self.bpm_result:
            result["bpm"] = self.bpm_result.to_dict()
        if self.key_result:
            result["key"] = self.key_result.to_dict()
        if self.features:
            result["features"] = self.features.to_dict()
        if self.feature_summary:
            result["feature_summary"] = self.feature_summary
        if self.segments:
            result["segments"] = [
                {"start": s.start_time, "end": s.end_time, "label": s.label}
                for s in self.segments
            ]
        return result


# =================================================================
# KEY DETECTION
# =================================================================

# Krumhansl-Schmuckler key profiles (stored as Python lists so module loads even without NumPy)
MAJOR_PROFILE_VALUES = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE_VALUES = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_key(
    audio_data: "np.ndarray",
    sr: int,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> KeyDetectionResult:
    """
    Detect the musical key of audio using the Krumhansl-Schmuckler algorithm.
    
    Args:
        audio_data: Audio samples (mono)
        sr: Sample rate
        hop_length: Analysis hop length
    
    Returns:
        KeyDetectionResult with detected key and confidence
    """
    if not LIBROSA_AVAILABLE or np is None:
        raise ImportError("librosa required for key detection")
    
    # Preference harmonic content for tonal estimation
    harmonic = librosa.effects.harmonic(audio_data)
    chroma_cqt = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop_length)
    chroma_cens = librosa.feature.chroma_cens(y=harmonic, sr=sr, hop_length=hop_length)
    chroma = 0.6 * chroma_cqt + 0.4 * chroma_cens
    chroma_mean = np.mean(chroma, axis=1)
    
    # Normalize
    chroma_sum = np.sum(chroma_mean)
    if chroma_sum > 0:
        chroma_mean = chroma_mean / chroma_sum
    
    correlations = []
    major_profile = np.array(MAJOR_PROFILE_VALUES)
    minor_profile = np.array(MINOR_PROFILE_VALUES)

    for shift in range(12):
        shifted_major = np.roll(major_profile, shift)
        shifted_minor = np.roll(minor_profile, shift)
        shifted_major /= np.sum(shifted_major)
        shifted_minor /= np.sum(shifted_minor)
        corr_major = float(np.corrcoef(chroma_mean, shifted_major)[0, 1])
        corr_minor = float(np.corrcoef(chroma_mean, shifted_minor)[0, 1])
        correlations.append({"key": KEY_NAMES[shift], "mode": "major", "correlation": corr_major})
        correlations.append({"key": KEY_NAMES[shift], "mode": "minor", "correlation": corr_minor})
    
    best = max(correlations, key=lambda item: item["correlation"])
    all_corrs = np.array([c["correlation"] for c in correlations])
    percentile = np.percentile(all_corrs, 75)
    max_corr = best["correlation"]
    confidence = (max_corr - percentile) / (abs(max_corr) + 1e-6)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    
    return KeyDetectionResult(
        key=best["key"],
        mode=KeyMode.MAJOR if best["mode"] == "major" else KeyMode.MINOR,
        confidence=confidence,
        correlation_vector=chroma_mean.tolist(),
    )


# =================================================================
# BPM DETECTION
# =================================================================

def detect_bpm(
    audio_data: "np.ndarray",
    sr: int,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> BPMDetectionResult:
    """
    Detect tempo/BPM from audio.
    
    Args:
        audio_data: Audio samples (mono)
        sr: Sample rate
        hop_length: Analysis hop length
    
    Returns:
        BPMDetectionResult with detected BPM and confidence
    """
    if not LIBROSA_AVAILABLE or np is None:
        raise ImportError("librosa required for BPM detection")
    
    percussive = librosa.effects.percussive(audio_data)
    onset_env = librosa.onset.onset_strength(y=percussive, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    
    # Get alternative tempo candidates from tempogram autocorrelation
    alternatives = _extract_tempo_candidates(onset_env, sr, hop_length)
    
    # Calculate confidence from beat interval consistency
    temp_array = np.asarray(tempo)
    primary_bpm = float(temp_array.item()) if temp_array.size == 1 else float(temp_array[0])
    
    if len(beat_frames) > 2:
        intervals = np.diff(beat_times)
        expected = 60.0 / primary_bpm
        deviations = np.abs(intervals - expected) / (expected + ENERGY_EPS)
        confidence = float(np.clip(1.0 - np.mean(deviations), 0.0, 1.0))
    else:
        confidence = 0.4
    
    # Filter alternatives to exclude the primary tempo and keep reasonable range
    alternatives = [t for t in alternatives if abs(t - primary_bpm) > 5 and 60 <= t <= 200][:4]
    
    return BPMDetectionResult(
        bpm=primary_bpm,
        confidence=confidence,
        beat_frames=beat_frames.tolist(),
        beat_times=beat_times.tolist(),
        tempo_alternatives=alternatives,
    )


def _extract_tempo_candidates(
    onset_env: "np.ndarray",
    sr: int,
    hop_length: int,
    n_candidates: int = 5,
) -> List[float]:
    """
    Extract tempo candidates from tempogram autocorrelation peaks.
    
    Uses the tempogram to find peaks in the tempo distribution,
    providing meaningful alternative tempo estimates.
    
    Args:
        onset_env: Onset strength envelope
        sr: Sample rate
        hop_length: Analysis hop length
        n_candidates: Maximum number of candidates to return
    
    Returns:
        List of tempo candidates in BPM, sorted by strength
    """
    if not LIBROSA_AVAILABLE or np is None:
        return []
    
    try:
        # Compute tempogram (autocorrelation of onset envelope)
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
        )
        
        # Aggregate tempogram across time to get global tempo distribution
        # Shape: (n_tempo_bins,)
        tempo_distribution = np.mean(tempogram, axis=1)
        
        # Get the tempo axis (BPM values for each bin)
        # librosa's tempogram uses lag-based representation
        # Convert lag indices to BPM: BPM = 60 * sr / (lag * hop_length)
        n_bins = tempogram.shape[0]
        lag_to_bpm = np.zeros(n_bins)
        for lag in range(1, n_bins):
            lag_to_bpm[lag] = 60.0 * sr / (lag * hop_length)
        
        # Find peaks in the tempo distribution (local maxima)
        candidates = []
        for i in range(2, n_bins - 2):
            bpm = lag_to_bpm[i]
            # Only consider reasonable tempo range
            if 60 <= bpm <= 200:
                # Check if this is a local maximum
                if (tempo_distribution[i] > tempo_distribution[i-1] and
                    tempo_distribution[i] > tempo_distribution[i+1] and
                    tempo_distribution[i] > tempo_distribution[i-2] and
                    tempo_distribution[i] > tempo_distribution[i+2]):
                    candidates.append((bpm, tempo_distribution[i]))
        
        # Sort by strength (distribution value) descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N BPM values
        return [round(c[0], 1) for c in candidates[:n_candidates]]
    
    except Exception:
        # Fallback: return empty list if tempogram analysis fails
        return []


# =================================================================
# FEATURE EXTRACTION
# =================================================================

def extract_features(
    audio_data: "np.ndarray",
    sr: int,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> Dict:
    """
    Extract comprehensive audio features.
    
    Args:
        audio_data: Audio samples (mono)
        sr: Sample rate
        hop_length: Analysis hop length
    
    Returns:
        Dictionary of extracted features
    """
    if not LIBROSA_AVAILABLE or np is None:
        raise ImportError("librosa required for feature extraction")
    
    harmonic = librosa.effects.harmonic(audio_data)
    percussive = librosa.effects.percussive(audio_data)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, hop_length=hop_length)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20, hop_length=hop_length)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop_length)
    tempo_func = getattr(getattr(librosa, "feature", None), "rhythm", None)
    onset_strength = librosa.onset.onset_strength(y=percussive, sr=sr)
    if tempo_func and hasattr(tempo_func, "tempo"):
        tempo_curve = tempo_func.tempo(onset_envelope=onset_strength, sr=sr, aggregate=None)
    else:
        tempo_curve = librosa.beat.tempo(onset_envelope=onset_strength, sr=sr, aggregate=None)
    
    return {
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_contrast_mean": float(np.mean(spectral_contrast)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "harmonic_energy": float(np.mean(np.square(harmonic))),
        "percussive_energy": float(np.mean(np.square(percussive))),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "mfcc_means": [float(np.mean(mfcc[i])) for i in range(mfcc.shape[0])],
        "chroma_means": [float(np.mean(chroma[i])) for i in range(chroma.shape[0])],
        "tempo_curve_top": [float(t) for t in tempo_curve[-5:]] if tempo_curve is not None else [],
    }


# =================================================================
# AUDIO ANALYZER CLASS
# =================================================================

class AudioAnalyzer:
    """
    Main audio analysis interface for DAiW.
    
    Provides unified access to BPM detection, key detection, structural segmentation,
    and feature extraction tuned for emotional workflows.
    """
    
    def __init__(self, hop_length: int = DEFAULT_HOP_LENGTH, target_sr: int = DEFAULT_TARGET_SR):
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa required for AudioAnalyzer. "
                "Install with: pip install librosa numpy scipy"
            )
        self.hop_length = hop_length
        self.target_sr = target_sr
    
    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze_file(
        self,
        filepath: str,
        detect_key: bool = True,
        detect_bpm: bool = True,
        extract_features_flag: bool = True,
        analyze_segments: bool = True,
        num_segments: int = 4,
        max_duration: Optional[float] = None,
    ) -> AudioAnalysis:
        """Perform complete analysis of an audio file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        audio_data, sr, duration = self._load_audio(filepath, max_duration=max_duration)
        result = AudioAnalysis(
            filepath=str(filepath),
            duration_seconds=duration,
            sample_rate=sr,
        )
        
        harmonic = librosa.effects.harmonic(audio_data)
        percussive = librosa.effects.percussive(audio_data)
        
        if detect_key:
            result.key_result = self.detect_key(harmonic, sr)
        if detect_bpm:
            result.bpm_result = self.detect_bpm(percussive, sr)
        if extract_features_flag:
            result.feature_summary = self.extract_features(audio_data, sr)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result.features = analyze_feel(str(filepath), hop_length=self.hop_length)
            except Exception as exc:  # pragma: no cover - depends on external deps
                warnings.warn(f"Feel analysis unavailable: {exc}", RuntimeWarning)
        if analyze_segments:
            result.segments = self._segment_from_data(audio_data, sr, num_segments=num_segments)
        
        return result
    
    def detect_key(self, audio_data: "np.ndarray", sr: int) -> KeyDetectionResult:
        return detect_key(audio_data, sr, self.hop_length)
    
    def detect_bpm(self, audio_data: "np.ndarray", sr: int) -> BPMDetectionResult:
        return detect_bpm(audio_data, sr, self.hop_length)
    
    def extract_features(self, audio_data: "np.ndarray", sr: int) -> Dict:
        return extract_features(audio_data, sr, self.hop_length)
    
    def segment_audio(
        self,
        filepath: str,
        num_segments: int = 4,
    ) -> List[AudioSegment]:
        """Segment audio file into labeled regions."""
        audio_data, sr, _ = self._load_audio(Path(filepath))
        return self._segment_from_data(audio_data, sr, num_segments=num_segments)
    
    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _load_audio(
        self,
        filepath: Path,
        max_duration: Optional[float] = None,
    ):
        y, sr = librosa.load(
            str(filepath),
            sr=self.target_sr,
            mono=True,
            duration=max_duration,
        )
        duration = librosa.get_duration(y=y, sr=sr)
        peak = np.max(np.abs(y)) if np is not None else 0
        if peak > 0:
            y = y / peak
        return y, sr, duration
    
    def _segment_from_data(
        self,
        audio_data: "np.ndarray",
        sr: int,
        num_segments: int,
    ) -> List[AudioSegment]:
        if not SCIPY_AVAILABLE or np is None:
            warnings.warn(
                "scipy not installed; advanced segmentation disabled.",
                RuntimeWarning,
            )
            return []
        
        novelty = librosa.onset.onset_strength(y=audio_data, sr=sr, hop_length=self.hop_length)
        height = np.percentile(novelty, 75)
        peaks, _ = find_peaks(novelty, height=height, distance=int(0.5 * sr / self.hop_length))
        
        if len(peaks) < num_segments - 1:
            extra = np.linspace(0, len(novelty) - 1, num_segments + 1, dtype=int)[1:-1]
            peaks = np.unique(np.concatenate([peaks, extra]))
        else:
            idx = np.argsort(novelty[peaks])[-(num_segments - 1):]
            peaks = np.sort(peaks[idx])
        
        boundary_frames = np.concatenate([[0], peaks, [len(novelty) - 1]])
        boundary_times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=self.hop_length)
        
        segments: List[AudioSegment] = []
        for start, end in zip(boundary_times[:-1], boundary_times[1:]):
            start_sample = int(start * sr)
            end_sample = max(int(end * sr), start_sample + 1)
            segment_audio = audio_data[start_sample:end_sample]
            energy = float(np.mean(np.abs(segment_audio)))
            segments.append(
                AudioSegment(
                    start_time=float(start),
                    end_time=float(end),
                    energy=energy,
                )
            )
        
        self._label_segments(segments)
        return segments
    
    def _label_segments(self, segments: List[AudioSegment]) -> None:
        labels = ["intro", "verse", "pre-chorus", "chorus", "bridge", "break", "outro"]
        if not segments:
            return
        # Default chronological assignment
        for idx, segment in enumerate(segments):
            label_idx = min(idx, len(labels) - 1)
            segment.label = labels[label_idx]
        # Override with energy heuristics
        max_energy_idx = max(range(len(segments)), key=lambda i: segments[i].energy)
        segments[max_energy_idx].label = "chorus"
        segments[-1].label = "outro"

