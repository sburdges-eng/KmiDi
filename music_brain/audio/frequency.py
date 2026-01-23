"""
Frequency Analysis - FFT, pitch detection, and harmonic analysis utilities.

Provides low-level frequency domain analysis tools for audio processing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import math

try:
    import numpy as np
    from scipy import signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# =================================================================
# DATA CLASSES
# =================================================================

@dataclass
class PitchDetection:
    """Result of pitch detection."""
    frequency_hz: float
    midi_note: int
    note_name: str
    cents_deviation: float  # Deviation from perfect pitch (-50 to +50)
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "frequency_hz": self.frequency_hz,
            "midi_note": self.midi_note,
            "note_name": self.note_name,
            "cents_deviation": self.cents_deviation,
            "confidence": self.confidence,
        }


@dataclass
class HarmonicContent:
    """Analysis of harmonic content."""
    fundamental_freq: float
    harmonics: List[Tuple[float, float]]  # List of (frequency, amplitude) pairs
    harmonic_ratio: float  # Ratio of harmonic to inharmonic content
    spectral_centroid: float
    spectral_spread: float
    
    def to_dict(self) -> Dict:
        return {
            "fundamental_freq": self.fundamental_freq,
            "num_harmonics": len(self.harmonics),
            "harmonic_ratio": self.harmonic_ratio,
            "spectral_centroid": self.spectral_centroid,
            "spectral_spread": self.spectral_spread,
        }


@dataclass
class FFTAnalysis:
    """Result of FFT analysis."""
    frequencies: List[float]
    magnitudes: List[float]
    peak_frequencies: List[float]
    peak_magnitudes: List[float]
    dominant_frequency: float
    
    def to_dict(self) -> Dict:
        return {
            "dominant_frequency": self.dominant_frequency,
            "num_peaks": len(self.peak_frequencies),
            "peak_frequencies": self.peak_frequencies[:10],  # Top 10
            "peak_magnitudes": self.peak_magnitudes[:10],
        }


# =================================================================
# PITCH UTILITIES
# =================================================================

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
A4_FREQ = 440.0
A4_MIDI = 69


def freq_to_midi(freq: float) -> float:
    """Convert frequency (Hz) to MIDI note number (float)."""
    if freq <= 0:
        return 0.0
    return 12 * math.log2(freq / A4_FREQ) + A4_MIDI


def midi_to_freq(midi_note: float) -> float:
    """Convert MIDI note number to frequency (Hz)."""
    return A4_FREQ * (2 ** ((midi_note - A4_MIDI) / 12))


def midi_to_note_name(midi_note: int) -> str:
    """Convert MIDI note number to note name (e.g., 'C4', 'A#5')."""
    octave = (midi_note // 12) - 1
    note_idx = midi_note % 12
    return f"{NOTE_NAMES[note_idx]}{octave}"


def note_name_to_midi(note_name: str) -> int:
    """Convert note name to MIDI note number."""
    # Parse note name (e.g., 'C4', 'A#5')
    import re
    match = re.match(r'([A-G]#?)(-?\d+)', note_name)
    if not match:
        raise ValueError(f"Invalid note name: {note_name}")
    
    note = match.group(1)
    octave = int(match.group(2))
    
    note_idx = NOTE_NAMES.index(note)
    return (octave + 1) * 12 + note_idx


def cents_from_freq(freq: float, target_midi: int) -> float:
    """Calculate cents deviation from a target MIDI note."""
    target_freq = midi_to_freq(target_midi)
    if target_freq <= 0 or freq <= 0:
        return 0.0
    return 1200 * math.log2(freq / target_freq)


# =================================================================
# FREQUENCY ANALYZER CLASS
# =================================================================

class FrequencyAnalyzer:
    """
    Frequency domain analysis utilities.
    
    Provides FFT analysis, pitch detection, and harmonic content analysis.
    """
    
    def __init__(self, default_sr: int = 44100):
        """
        Initialize frequency analyzer.
        
        Args:
            default_sr: Default sample rate if not specified
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy required for FrequencyAnalyzer. "
                "Install with: pip install scipy numpy"
            )
        self.default_sr = default_sr
    
    def fft_analysis(
        self,
        audio_data: np.ndarray,
        sr: Optional[int] = None,
        window: str = 'hann',
        min_freq: float = 20.0,
        max_freq: float = 20000.0,
    ) -> FFTAnalysis:
        """
        Perform FFT analysis on audio data.
        
        Args:
            audio_data: Audio samples (mono)
            sr: Sample rate (uses default if None)
            window: Window function name
            min_freq: Minimum frequency to consider
            max_freq: Maximum frequency to consider
        
        Returns:
            FFTAnalysis with frequency spectrum data
        """
        sr = sr or self.default_sr
        n = len(audio_data)
        
        # Apply window
        win = signal.get_window(window, n)
        windowed = audio_data * win
        
        # FFT
        spectrum = fft(windowed)
        frequencies = fftfreq(n, 1/sr)
        
        # Take positive frequencies only
        pos_mask = frequencies >= 0
        frequencies = frequencies[pos_mask]
        magnitudes = np.abs(spectrum[pos_mask])
        
        # Normalize
        magnitudes = magnitudes / np.max(magnitudes) if np.max(magnitudes) > 0 else magnitudes
        
        # Filter by frequency range
        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        frequencies = frequencies[freq_mask]
        magnitudes = magnitudes[freq_mask]
        
        # Find peaks
        peaks, _ = signal.find_peaks(magnitudes, height=0.1, distance=int(n * 10 / sr))
        
        peak_freqs = frequencies[peaks].tolist()
        peak_mags = magnitudes[peaks].tolist()
        
        # Sort by magnitude
        sorted_peaks = sorted(zip(peak_mags, peak_freqs), reverse=True)
        peak_mags = [p[0] for p in sorted_peaks]
        peak_freqs = [p[1] for p in sorted_peaks]
        
        dominant = peak_freqs[0] if peak_freqs else 0.0
        
        return FFTAnalysis(
            frequencies=frequencies.tolist(),
            magnitudes=magnitudes.tolist(),
            peak_frequencies=peak_freqs,
            peak_magnitudes=peak_mags,
            dominant_frequency=dominant,
        )
    
    def pitch_detection(
        self,
        audio_data: np.ndarray,
        sr: Optional[int] = None,
        method: str = 'yin',
        min_freq: float = 50.0,
        max_freq: float = 2000.0,
    ) -> Optional[PitchDetection]:
        """
        Detect fundamental pitch from audio.
        
        Args:
            audio_data: Audio samples (mono)
            sr: Sample rate
            method: Detection method ('yin', 'autocorrelation', 'fft')
            min_freq: Minimum frequency to detect
            max_freq: Maximum frequency to detect
        
        Returns:
            PitchDetection or None if no pitch detected
        """
        sr = sr or self.default_sr
        
        if method == 'yin' and LIBROSA_AVAILABLE:
            return self._pitch_yin(audio_data, sr, min_freq, max_freq)
        elif method == 'autocorrelation':
            return self._pitch_autocorrelation(audio_data, sr, min_freq, max_freq)
        else:
            return self._pitch_fft(audio_data, sr, min_freq, max_freq)
    
    def _pitch_yin(
        self,
        audio_data: np.ndarray,
        sr: int,
        min_freq: float,
        max_freq: float,
    ) -> Optional[PitchDetection]:
        """Pitch detection using YIN algorithm (librosa)."""
        f0 = librosa.yin(
            audio_data, fmin=min_freq, fmax=max_freq, sr=sr
        )
        
        # Get median frequency (most stable estimate)
        f0_valid = f0[f0 > 0]
        if len(f0_valid) == 0:
            return None
        
        freq = float(np.median(f0_valid))
        
        # Convert to MIDI and note
        midi_float = freq_to_midi(freq)
        midi_note = int(round(midi_float))
        cents = cents_from_freq(freq, midi_note)
        
        # Estimate confidence from consistency
        confidence = 1.0 - (np.std(f0_valid) / np.mean(f0_valid)) if np.mean(f0_valid) > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))
        
        return PitchDetection(
            frequency_hz=freq,
            midi_note=midi_note,
            note_name=midi_to_note_name(midi_note),
            cents_deviation=cents,
            confidence=confidence,
        )
    
    def _pitch_autocorrelation(
        self,
        audio_data: np.ndarray,
        sr: int,
        min_freq: float,
        max_freq: float,
    ) -> Optional[PitchDetection]:
        """Pitch detection using autocorrelation."""
        # Compute autocorrelation
        corr = np.correlate(audio_data, audio_data, mode='full')
        corr = corr[len(corr)//2:]  # Take positive lags only
        
        # Find first peak (after initial maximum)
        min_lag = int(sr / max_freq)
        max_lag = int(sr / min_freq)
        
        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        
        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            return None
        
        # Find peak
        peak_idx = np.argmax(search_region) + min_lag
        
        # Convert lag to frequency
        freq = sr / peak_idx if peak_idx > 0 else 0.0
        
        if freq < min_freq or freq > max_freq:
            return None
        
        # Confidence from peak strength
        confidence = corr[peak_idx] / corr[0] if corr[0] > 0 else 0.0
        
        midi_float = freq_to_midi(freq)
        midi_note = int(round(midi_float))
        cents = cents_from_freq(freq, midi_note)
        
        return PitchDetection(
            frequency_hz=freq,
            midi_note=midi_note,
            note_name=midi_to_note_name(midi_note),
            cents_deviation=cents,
            confidence=float(confidence),
        )
    
    def _pitch_fft(
        self,
        audio_data: np.ndarray,
        sr: int,
        min_freq: float,
        max_freq: float,
    ) -> Optional[PitchDetection]:
        """Pitch detection using FFT peak finding."""
        fft_result = self.fft_analysis(
            audio_data, sr, min_freq=min_freq, max_freq=max_freq
        )
        
        if not fft_result.peak_frequencies:
            return None
        
        # Take dominant frequency
        freq = fft_result.dominant_frequency
        
        if freq < min_freq or freq > max_freq:
            return None
        
        midi_float = freq_to_midi(freq)
        midi_note = int(round(midi_float))
        cents = cents_from_freq(freq, midi_note)
        
        # Confidence from peak prominence
        confidence = fft_result.peak_magnitudes[0] if fft_result.peak_magnitudes else 0.0
        
        return PitchDetection(
            frequency_hz=freq,
            midi_note=midi_note,
            note_name=midi_to_note_name(midi_note),
            cents_deviation=cents,
            confidence=float(confidence),
        )
    
    def harmonic_content(
        self,
        audio_data: np.ndarray,
        sr: Optional[int] = None,
        fundamental_freq: Optional[float] = None,
        num_harmonics: int = 10,
    ) -> HarmonicContent:
        """
        Analyze harmonic content of audio.
        
        Args:
            audio_data: Audio samples (mono)
            sr: Sample rate
            fundamental_freq: Known fundamental (auto-detected if None)
            num_harmonics: Number of harmonics to analyze
        
        Returns:
            HarmonicContent analysis
        """
        sr = sr or self.default_sr
        
        # Detect fundamental if not provided
        if fundamental_freq is None:
            pitch = self.pitch_detection(audio_data, sr)
            fundamental_freq = pitch.frequency_hz if pitch else 100.0
        
        # Get spectrum
        fft_result = self.fft_analysis(audio_data, sr)
        frequencies = np.array(fft_result.frequencies)
        magnitudes = np.array(fft_result.magnitudes)
        
        # Find harmonic peaks
        harmonics = []
        harmonic_energy = 0.0
        
        for h in range(1, num_harmonics + 1):
            target_freq = fundamental_freq * h
            
            # Find closest frequency bin
            idx = np.argmin(np.abs(frequencies - target_freq))
            
            # Search for peak in neighborhood
            window = 5
            start = max(0, idx - window)
            end = min(len(magnitudes), idx + window)
            
            local_peak_idx = start + np.argmax(magnitudes[start:end])
            peak_freq = frequencies[local_peak_idx]
            peak_mag = magnitudes[local_peak_idx]
            
            harmonics.append((float(peak_freq), float(peak_mag)))
            harmonic_energy += peak_mag
        
        # Calculate harmonic ratio
        total_energy = np.sum(magnitudes)
        harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.0
        
        # Spectral centroid and spread
        centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0.0
        spread = np.sqrt(np.sum(magnitudes * (frequencies - centroid)**2) / np.sum(magnitudes)) if np.sum(magnitudes) > 0 else 0.0
        
        return HarmonicContent(
            fundamental_freq=fundamental_freq,
            harmonics=harmonics,
            harmonic_ratio=float(harmonic_ratio),
            spectral_centroid=float(centroid),
            spectral_spread=float(spread),
        )


# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def analyze_frequency_spectrum(
    filepath: str,
    window_size: float = 0.1,
    sr: Optional[int] = None,
) -> FFTAnalysis:
    """
    Analyze frequency spectrum of an audio file.
    
    Args:
        filepath: Path to audio file
        window_size: Analysis window in seconds
        sr: Sample rate (None = use file's native rate)
    
    Returns:
        FFTAnalysis result
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required for audio file loading")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    # Load audio
    y, file_sr = librosa.load(str(filepath), sr=sr, mono=True, duration=window_size)
    
    analyzer = FrequencyAnalyzer(default_sr=file_sr)
    return analyzer.fft_analysis(y, file_sr)


def detect_pitch_from_audio(
    filepath: str,
    method: str = 'yin',
    max_duration: float = 5.0,
) -> Optional[PitchDetection]:
    """
    Detect pitch from an audio file.
    
    Args:
        filepath: Path to audio file
        method: Detection method
        max_duration: Maximum duration to analyze
    
    Returns:
        PitchDetection or None
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa required for audio file loading")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    
    y, sr = librosa.load(str(filepath), sr=None, mono=True, duration=max_duration)
    
    analyzer = FrequencyAnalyzer(default_sr=sr)
    return analyzer.pitch_detection(y, sr, method=method)

