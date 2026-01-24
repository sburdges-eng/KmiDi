"""
Parrot Vocal Synthesizer - Voice Learning and Mimicry System

The Parrot function learns voice characteristics from uploaded audio and can
mimic voices after prolonged exposure. It learns vowels, accents, pitch contours,
and timbre to create realistic voice synthesis.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import numpy as np
from enum import Enum

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
    sf = None

# Fallback to numpy if librosa not available
try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False
    np = None


class VowelType(Enum):
    """Vowel classification"""
    A = "a"  # "ah" as in "father"
    E = "e"  # "eh" as in "bed"
    I = "i"  # "ee" as in "see"
    O = "o"  # "oh" as in "go"
    U = "u"  # "oo" as in "food"
    SCHWA = "ə"  # "uh" as in "about"
    UNKNOWN = "unknown"


@dataclass
class FormantData:
    """Formant frequencies (F1, F2, F3) for vowel analysis"""
    f1: float  # First formant (vowel height)
    f2: float  # Second formant (vowel frontness)
    f3: float  # Third formant (vowel rounding)
    confidence: float = 0.0  # Detection confidence (0.0-1.0)


@dataclass
class VowelSample:
    """Individual vowel sample with formant data"""
    vowel_type: VowelType
    formants: FormantData
    duration: float  # Duration in seconds
    pitch_contour: List[float] = field(default_factory=list)  # Pitch over time
    spectral_centroid: float = 0.0  # Timbre characteristic
    spectral_rolloff: float = 0.0  # Brightness


@dataclass
class AccentCharacteristics:
    """Learned accent characteristics"""
    vowel_shifts: Dict[str, FormantData] = field(default_factory=dict)  # Vowel formant shifts
    pitch_range: Tuple[float, float] = (0.0, 0.0)  # Typical pitch range (Hz)
    intonation_pattern: List[float] = field(default_factory=list)  # Typical intonation curve
    rhythm_timing: Dict[str, float] = field(default_factory=dict)  # Timing characteristics
    consonant_emphasis: Dict[str, float] = field(default_factory=dict)  # Consonant strength


@dataclass
class VoiceCharacteristics:
    """Complete voice characteristics learned from audio"""
    # Formant data
    vowel_formants: Dict[VowelType, List[FormantData]] = field(default_factory=dict)
    
    # Pitch characteristics
    average_pitch: float = 0.0  # Hz
    pitch_range: Tuple[float, float] = (0.0, 0.0)  # Min/max Hz
    vibrato_rate: float = 0.0  # Vibrato frequency (Hz)
    vibrato_depth: float = 0.0  # Vibrato depth (cents)
    
    # Timbre characteristics
    spectral_centroid_mean: float = 0.0  # Brightness
    spectral_rolloff_mean: float = 0.0  # High-frequency content
    spectral_bandwidth_mean: float = 0.0  # Timbre width
    
    # Accent characteristics
    accent: AccentCharacteristics = field(default_factory=AccentCharacteristics)
    
    # Speaking/singing style
    attack_time: float = 0.0  # Note attack speed
    release_time: float = 0.0  # Note release speed
    breathiness: float = 0.0  # Breath noise level (0.0-1.0)
    nasality: float = 0.0  # Nasal resonance (0.0-1.0)
    
    # Voice quality (jitter/shimmer)
    jitter: float = 0.0  # Pitch period variation (%)
    shimmer: float = 0.0  # Amplitude variation (%)
    hnr: float = 0.0  # Harmonic-to-noise ratio (dB)
    
    # Prosody (rhythm and intonation)
    speaking_rate: float = 0.0  # Syllables per second
    pause_frequency: float = 0.0  # Pauses per second
    average_pause_duration: float = 0.0  # Average pause length (seconds)
    intonation_range: float = 0.0  # Pitch range for intonation (semitones)
    pitch_contour_template: List[float] = field(default_factory=list)  # Typical pitch contour
    
    # Consonant characteristics
    consonant_strength: Dict[str, float] = field(default_factory=dict)  # Consonant emphasis
    
    # Learning metadata
    exposure_time: float = 0.0  # Total audio analyzed (seconds)
    sample_count: int = 0  # Number of samples analyzed
    confidence: float = 0.0  # Overall model confidence (0.0-1.0)
    training_files: List[str] = field(default_factory=list)  # Source files used for training


@dataclass
class VoiceModel:
    """Complete voice model for synthesis"""
    name: str
    characteristics: VoiceCharacteristics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        data = asdict(self)
        # Convert enums to strings
        if 'vowel_formants' in data['characteristics']:
            vowel_data = {}
            for vowel, formants in data['characteristics']['vowel_formants'].items():
                vowel_data[vowel.value if hasattr(vowel, 'value') else str(vowel)] = [
                    asdict(f) for f in formants
                ]
            data['characteristics']['vowel_formants'] = vowel_data
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VoiceModel':
        """Deserialize from dictionary"""
        # Convert string keys back to VowelType enums
        if 'characteristics' in data and 'vowel_formants' in data['characteristics']:
            vowel_data = {}
            for vowel_str, formants in data['characteristics']['vowel_formants'].items():
                try:
                    vowel = VowelType(vowel_str)
                    vowel_data[vowel] = [
                        FormantData(**f) if isinstance(f, dict) else f
                        for f in formants
                    ]
                except ValueError:
                    continue
            data['characteristics']['vowel_formants'] = vowel_data
        
        return cls(**data)


@dataclass
class ParrotConfig:
    """Configuration for Parrot vocal synthesizer"""
    # Learning parameters
    min_exposure_time: float = 30.0  # Minimum seconds of audio to learn from
    learning_rate: float = 0.1  # How quickly to adapt (0.0-1.0)
    confidence_threshold: float = 0.7  # Minimum confidence to use model
    batch_learning: bool = True  # Enable batch processing for multiple files
    
    # Analysis parameters
    formant_window_size: float = 0.025  # Window size for formant analysis (seconds)
    pitch_hop_length: int = 512  # Hop length for pitch detection
    vowel_detection_threshold: float = 0.6  # Confidence threshold for vowel detection
    analyze_jitter_shimmer: bool = True  # Analyze voice quality parameters
    analyze_prosody: bool = True  # Analyze rhythm and intonation
    
    # Synthesis parameters
    synthesis_sample_rate: int = 44100
    formant_shift_range: Tuple[float, float] = (0.8, 1.2)  # Formant shift range for variation
    pitch_variation: float = 0.05  # Pitch variation amount (5%)
    use_formant_synthesis: bool = True  # Use proper formant synthesis
    add_breathiness: bool = True  # Add breath noise
    add_jitter_shimmer: bool = True  # Add natural voice variation
    prosody_strength: float = 1.0  # Prosody application strength (0.0-1.0)
    
    # Emotion/expression
    emotion: Optional[str] = None  # Emotion preset (happy, sad, angry, etc.)
    expression_intensity: float = 0.5  # Expression intensity (0.0-1.0)
    
    # Analysis enhancements
    analyze_jitter_shimmer: bool = True  # Analyze voice quality parameters
    analyze_prosody: bool = True  # Analyze rhythm and intonation
    batch_learning: bool = True  # Enable batch processing for multiple files
    use_formant_synthesis: bool = True  # Use proper formant synthesis
    add_breathiness: bool = True  # Add breath noise
    add_jitter_shimmer: bool = True  # Add natural voice variation
    prosody_strength: float = 1.0  # Prosody application strength (0.0-1.0)
    
    # Emotion/expression
    emotion: Optional[str] = None  # Emotion preset (happy, sad, angry, etc.)
    expression_intensity: float = 0.5  # Expression intensity (0.0-1.0)


class ParrotVocalSynthesizer:
    """
    Parrot Vocal Synthesizer - Learns and mimics voices from audio.
    
    The Parrot function analyzes uploaded audio to learn:
    - Vowel formants and transitions
    - Accent characteristics
    - Pitch contours and vibrato
    - Timbre and spectral characteristics
    - Speaking/singing style
    
    With prolonged exposure, it improves mimicry accuracy.
    """
    
    def __init__(self, config: Optional[ParrotConfig] = None):
        """
        Initialize Parrot synthesizer.
        
        Args:
            config: Configuration for learning and synthesis
        """
        self.config = config or ParrotConfig()
        self.voice_models: Dict[str, VoiceModel] = {}
        self.current_model: Optional[VoiceModel] = None
        
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "Parrot requires librosa for voice analysis. "
                "Install with: pip install librosa soundfile numpy scipy"
            )
    
    def analyze_voice(
        self,
        audio_file: str,
        voice_name: Optional[str] = None,
        update_existing: bool = True
    ) -> VoiceCharacteristics:
        """
        Analyze voice characteristics from audio file.
        
        Args:
            audio_file: Path to audio file
            voice_name: Name for the voice model (auto-generated if None)
            update_existing: If True, update existing model; if False, create new
        
        Returns:
            VoiceCharacteristics learned from audio
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for voice analysis")
        
        # Load audio
        audio_data, sr = librosa.load(audio_file, sr=None)
        duration = len(audio_data) / sr
        
        # Initialize or load existing characteristics
        if voice_name and voice_name in self.voice_models and update_existing:
            characteristics = self.voice_models[voice_name].characteristics
            existing_exposure = characteristics.exposure_time
            existing_samples = characteristics.sample_count
        else:
            characteristics = VoiceCharacteristics()
            existing_exposure = 0.0
            existing_samples = 0
        
        # Analyze formants and vowels
        vowel_samples = self._detect_vowels(audio_data, sr)
        
        # Analyze pitch characteristics
        pitch_data = self._analyze_pitch(audio_data, sr)
        
        # Analyze timbre
        timbre_data = self._analyze_timbre(audio_data, sr)
        
        # Analyze accent characteristics
        accent_data = self._analyze_accent(audio_data, sr, vowel_samples)
        
        # Update characteristics with weighted average (learning)
        learning_weight = self.config.learning_rate
        
        # Update vowel formants
        for vowel_sample in vowel_samples:
            vowel_type = vowel_sample.vowel_type
            if vowel_type not in characteristics.vowel_formants:
                characteristics.vowel_formants[vowel_type] = []
            characteristics.vowel_formants[vowel_type].append(vowel_sample.formants)
        
        # Update pitch characteristics
        if pitch_data:
            if characteristics.average_pitch == 0.0:
                characteristics.average_pitch = pitch_data['mean']
            else:
                characteristics.average_pitch = (
                    (1 - learning_weight) * characteristics.average_pitch +
                    learning_weight * pitch_data['mean']
                )
            
            if characteristics.pitch_range == (0.0, 0.0):
                characteristics.pitch_range = (pitch_data['min'], pitch_data['max'])
            else:
                old_min, old_max = characteristics.pitch_range
                characteristics.pitch_range = (
                    (1 - learning_weight) * old_min + learning_weight * pitch_data['min'],
                    (1 - learning_weight) * old_max + learning_weight * pitch_data['max']
                )
            
            characteristics.vibrato_rate = (
                (1 - learning_weight) * characteristics.vibrato_rate +
                learning_weight * pitch_data.get('vibrato_rate', 0.0)
            )
            characteristics.vibrato_depth = (
                (1 - learning_weight) * characteristics.vibrato_depth +
                learning_weight * pitch_data.get('vibrato_depth', 0.0)
            )
        
        # Update timbre
        if timbre_data:
            characteristics.spectral_centroid_mean = (
                (1 - learning_weight) * characteristics.spectral_centroid_mean +
                learning_weight * timbre_data['centroid']
            )
            characteristics.spectral_rolloff_mean = (
                (1 - learning_weight) * characteristics.spectral_rolloff_mean +
                learning_weight * timbre_data['rolloff']
            )
            characteristics.spectral_bandwidth_mean = (
                (1 - learning_weight) * characteristics.spectral_bandwidth_mean +
                learning_weight * timbre_data['bandwidth']
            )
        
        # Update accent
        if accent_data:
            characteristics.accent = accent_data
        
        # Update metadata
        characteristics.exposure_time = existing_exposure + duration
        characteristics.sample_count = existing_samples + len(vowel_samples)
        
        # Calculate confidence based on exposure time and sample count
        min_exposure = self.config.min_exposure_time
        if characteristics.exposure_time >= min_exposure:
            characteristics.confidence = min(1.0, characteristics.exposure_time / (min_exposure * 2))
        else:
            characteristics.confidence = characteristics.exposure_time / min_exposure
        
        return characteristics
    
    def _detect_vowels(
        self,
        audio_data: np.ndarray,
        sr: int
    ) -> List[VowelSample]:
        """Detect vowels and extract formant data."""
        vowel_samples = []
        
        # Use short-time Fourier transform for formant analysis
        hop_length = int(self.config.formant_window_size * sr)
        frame_length = hop_length * 2
        
        # Compute spectrogram
        stft = librosa.stft(audio_data, hop_length=hop_length, n_fft=frame_length)
        magnitude = np.abs(stft)
        
        # Find formants (peaks in spectrum)
        for i in range(magnitude.shape[1]):
            frame = magnitude[:, i]
            time = i * hop_length / sr
            
            # Find spectral peaks (formants)
            peaks = self._find_formant_peaks(frame, sr, frame_length)
            
            if len(peaks) >= 2:  # Need at least F1 and F2
                f1, f2, f3 = peaks[0], peaks[1], peaks[2] if len(peaks) > 2 else peaks[1] * 1.5
                
                formant_data = FormantData(
                    f1=f1,
                    f2=f2,
                    f3=f3,
                    confidence=self._calculate_formant_confidence(frame, peaks)
                )
                
                # Classify vowel based on formant positions
                vowel_type = self._classify_vowel(formant_data)
                
                if vowel_type != VowelType.UNKNOWN:
                    # Get pitch contour for this frame
                    pitch = self._get_pitch_at_time(audio_data, sr, time)
                    
                    # Get spectral characteristics
                    centroid = librosa.feature.spectral_centroid(
                        y=audio_data[int(time*sr):int((time+0.025)*sr)],
                        sr=sr
                    )[0, 0] if int((time+0.025)*sr) < len(audio_data) else 0.0
                    
                    rolloff = librosa.feature.spectral_rolloff(
                        y=audio_data[int(time*sr):int((time+0.025)*sr)],
                        sr=sr
                    )[0, 0] if int((time+0.025)*sr) < len(audio_data) else 0.0
                    
                    vowel_samples.append(VowelSample(
                        vowel_type=vowel_type,
                        formants=formant_data,
                        duration=0.025,  # Frame duration
                        pitch_contour=[pitch] if pitch > 0 else [],
                        spectral_centroid=centroid,
                        spectral_rolloff=rolloff
                    ))
        
        return vowel_samples
    
    def _find_formant_peaks(
        self,
        spectrum: np.ndarray,
        sr: int,
        n_fft: int
    ) -> List[float]:
        """Find formant peaks in spectrum."""
        # Convert bin indices to frequencies
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Focus on speech range (50-4000 Hz)
        speech_range = (freqs >= 50) & (freqs <= 4000)
        speech_spectrum = spectrum[speech_range]
        speech_freqs = freqs[speech_range]
        
        # Find peaks
        peaks = []
        for i in range(1, len(speech_spectrum) - 1):
            if (speech_spectrum[i] > speech_spectrum[i-1] and
                speech_spectrum[i] > speech_spectrum[i+1] and
                speech_spectrum[i] > np.max(speech_spectrum) * 0.3):  # Threshold
                peaks.append(speech_freqs[i])
        
        # Sort by magnitude and take top 3
        if peaks:
            peak_magnitudes = [spectrum[np.argmin(np.abs(freqs - p))] for p in peaks]
            sorted_peaks = sorted(zip(peaks, peak_magnitudes), key=lambda x: x[1], reverse=True)
            return [p[0] for p in sorted_peaks[:3]]
        
        return []
    
    def _classify_vowel(self, formants: FormantData) -> VowelType:
        """Classify vowel based on formant positions."""
        f1, f2 = formants.f1, formants.f2
        
        # Vowel classification based on F1/F2 space
        # F1: height (low = high F1, high = low F1)
        # F2: frontness (front = high F2, back = low F2)
        
        if f1 < 500 and f2 > 2000:  # High front
            return VowelType.I  # "ee"
        elif f1 < 500 and f2 < 1500:  # High back
            return VowelType.U  # "oo"
        elif 500 < f1 < 700 and f2 > 1800:  # Mid front
            return VowelType.E  # "eh"
        elif 500 < f1 < 700 and 1000 < f2 < 1500:  # Mid central
            return VowelType.SCHWA  # "uh"
        elif f1 > 700 and f2 > 1200:  # Low front
            return VowelType.A  # "ah"
        elif f1 > 700 and f2 < 1200:  # Low back
            return VowelType.O  # "oh"
        else:
            return VowelType.UNKNOWN
    
    def _analyze_pitch(
        self,
        audio_data: np.ndarray,
        sr: int
    ) -> Optional[Dict[str, float]]:
        """Analyze pitch characteristics."""
        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(
            y=audio_data,
            sr=sr,
            hop_length=self.config.pitch_hop_length
        )
        
        # Get pitch values (non-zero)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if not pitch_values:
            return None
        
        pitch_array = np.array(pitch_values)
        
        # Analyze vibrato (pitch modulation)
        if len(pitch_values) > 10:
            # Detect periodic pitch variation
            pitch_diff = np.diff(pitch_array)
            # Simple vibrato detection (could be improved)
            vibrato_rate = 0.0
            vibrato_depth = 0.0
            
            # Calculate pitch variation
            pitch_std = np.std(pitch_array)
            vibrato_depth = pitch_std / np.mean(pitch_array) * 1200  # Convert to cents
        else:
            vibrato_rate = 0.0
            vibrato_depth = 0.0
        
        return {
            'mean': float(np.mean(pitch_array)),
            'min': float(np.min(pitch_array)),
            'max': float(np.max(pitch_array)),
            'std': float(np.std(pitch_array)),
            'vibrato_rate': vibrato_rate,
            'vibrato_depth': vibrato_depth
        }
    
    def _analyze_timbre(
        self,
        audio_data: np.ndarray,
        sr: int
    ) -> Optional[Dict[str, float]]:
        """Analyze timbre characteristics."""
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        
        # Spectral rolloff (high-frequency content)
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
        
        # Spectral bandwidth (timbre width)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
        
        return {
            'centroid': float(np.mean(centroid)),
            'rolloff': float(np.mean(rolloff)),
            'bandwidth': float(np.mean(bandwidth))
        }
    
    def _analyze_accent(
        self,
        audio_data: np.ndarray,
        sr: int,
        vowel_samples: List[VowelSample]
    ) -> AccentCharacteristics:
        """Analyze accent characteristics."""
        accent = AccentCharacteristics()
        
        # Analyze vowel formant shifts (compared to standard)
        standard_formants = {
            VowelType.A: FormantData(f1=730, f2=1090, f3=2440),
            VowelType.E: FormantData(f1=570, f2=1980, f3=2440),
            VowelType.I: FormantData(f1=270, f2=2290, f3=3010),
            VowelType.O: FormantData(f1=570, f2=840, f3=2410),
            VowelType.U: FormantData(f1=300, f2=870, f3=2240),
        }
        
        for vowel_type, standard in standard_formants.items():
            samples = [v for v in vowel_samples if v.vowel_type == vowel_type]
            if samples:
                avg_f1 = np.mean([s.formants.f1 for s in samples])
                avg_f2 = np.mean([s.formants.f2 for s in samples])
                avg_f3 = np.mean([s.formants.f3 for s in samples])
                
                # Calculate shift from standard
                shift = FormantData(
                    f1=avg_f1 - standard.f1,
                    f2=avg_f2 - standard.f2,
                    f3=avg_f3 - standard.f3
                )
                accent.vowel_shifts[vowel_type.value] = shift
        
        # Analyze pitch range and intonation
        pitch_data = self._analyze_pitch(audio_data, sr)
        if pitch_data:
            accent.pitch_range = (pitch_data['min'], pitch_data['max'])
        
        return accent
    
    def _analyze_jitter_shimmer(
        self,
        audio_data: np.ndarray,
        sr: int,
        pitch_data: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """Analyze jitter (pitch period variation) and shimmer (amplitude variation)."""
        if not pitch_data or pitch_data['mean'] == 0:
            return None
        
        # Extract pitch periods
        fundamental_freq = pitch_data['mean']
        period_samples = int(sr / fundamental_freq)
        
        if period_samples < 10 or period_samples > len(audio_data) // 2:
            return None
        
        # Find pitch periods (zero-crossing detection)
        periods = []
        amplitudes = []
        
        # Simple period detection
        for i in range(0, len(audio_data) - period_samples * 2, period_samples):
            segment = audio_data[i:i + period_samples]
            if len(segment) == period_samples:
                # Find period length (zero crossings)
                zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
                if len(zero_crossings) >= 2:
                    period_length = zero_crossings[1] - zero_crossings[0]
                    if 10 < period_length < period_samples * 2:
                        periods.append(period_length)
                        amplitudes.append(np.max(np.abs(segment)))
        
        if len(periods) < 3:
            return None
        
        # Calculate jitter (period variation)
        periods_array = np.array(periods)
        period_mean = np.mean(periods_array)
        period_std = np.std(periods_array)
        jitter = (period_std / period_mean * 100.0) if period_mean > 0 else 0.0
        
        # Calculate shimmer (amplitude variation)
        amplitudes_array = np.array(amplitudes)
        amp_mean = np.mean(amplitudes_array)
        amp_std = np.std(amplitudes_array)
        shimmer = (amp_std / amp_mean * 100.0) if amp_mean > 0 else 0.0
        
        # Calculate HNR (harmonic-to-noise ratio) - simplified
        hnr = 20.0  # Placeholder
        
        return {
            'jitter': float(jitter),
            'shimmer': float(shimmer),
            'hnr': float(hnr)
        }
    
    def _analyze_prosody(
        self,
        audio_data: np.ndarray,
        sr: int,
        pitch_data: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze prosody (rhythm and intonation)."""
        duration = len(audio_data) / sr
        
        # Estimate speaking rate (syllables per second)
        hop_length = 512
        frame_length = 2048
        energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find energy peaks (potential syllables)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(energy, height=np.mean(energy) * 1.2, distance=int(sr / hop_length / 4))
        syllable_count = len(peaks)
        speaking_rate = syllable_count / duration if duration > 0 else 0.0
        
        # Analyze pauses
        energy_threshold = np.percentile(energy, 20)
        pause_frames = energy < energy_threshold
        pause_count = 0
        pause_durations = []
        
        in_pause = False
        pause_start = 0
        for i, is_pause in enumerate(pause_frames):
            if is_pause and not in_pause:
                pause_start = i
                in_pause = True
            elif not is_pause and in_pause:
                pause_duration = (i - pause_start) * hop_length / sr
                if pause_duration > 0.1:
                    pause_count += 1
                    pause_durations.append(pause_duration)
                in_pause = False
        
        pause_frequency = pause_count / duration if duration > 0 else 0.0
        average_pause_duration = np.mean(pause_durations) if pause_durations else 0.0
        
        # Analyze intonation
        if pitch_data:
            intonation_range = (pitch_data['max'] - pitch_data['min']) / pitch_data['mean'] * 12
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr, hop_length=hop_length)
            pitch_contour = []
            for t in range(min(20, pitches.shape[1])):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    semitones = 12 * np.log2(pitch / pitch_data['mean']) if pitch_data['mean'] > 0 else 0.0
                    pitch_contour.append(float(semitones))
        else:
            intonation_range = 0.0
            pitch_contour = []
        
        return {
            'speaking_rate': float(speaking_rate),
            'pause_frequency': float(pause_frequency),
            'average_pause_duration': float(average_pause_duration),
            'intonation_range': float(intonation_range),
            'pitch_contour': pitch_contour
        }
    
    def _get_pitch_at_time(
        self,
        audio_data: np.ndarray,
        sr: int,
        time: float
    ) -> float:
        """Get pitch at specific time."""
        start_idx = int(time * sr)
        end_idx = min(start_idx + self.config.pitch_hop_length, len(audio_data))
        
        if end_idx <= start_idx:
            return 0.0
        
        segment = audio_data[start_idx:end_idx]
        pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
        
        if pitches.size > 0:
            index = magnitudes.argmax()
            pitch = pitches.flatten()[index] if index < pitches.size else 0.0
            return float(pitch) if pitch > 0 else 0.0
        
        return 0.0
    
    def _calculate_formant_confidence(
        self,
        spectrum: np.ndarray,
        peaks: List[float]
    ) -> float:
        """Calculate confidence in formant detection."""
        if not peaks:
            return 0.0
        
        # Confidence based on peak prominence
        peak_magnitudes = [spectrum[np.argmin(np.abs(np.arange(len(spectrum)) * 22050 / len(spectrum) - p))] 
                          for p in peaks if p > 0]
        
        if peak_magnitudes:
            max_mag = np.max(spectrum)
            avg_peak_mag = np.mean(peak_magnitudes)
            confidence = min(1.0, avg_peak_mag / max_mag if max_mag > 0 else 0.0)
            return float(confidence)
        
        return 0.0
    
    def train_parrot_batch(
        self,
        audio_files: List[str],
        voice_name: str,
        update_existing: bool = True
    ) -> VoiceModel:
        """
        Train Parrot on multiple audio files (batch learning).
        
        Args:
            audio_files: List of audio file paths
            voice_name: Name for the voice model
            update_existing: If True, update existing model
        
        Returns:
            Trained VoiceModel
        """
        print(f"Training Parrot on {len(audio_files)} audio files...")
        
        for i, audio_file in enumerate(audio_files, 1):
            if not Path(audio_file).exists():
                print(f"Warning: Skipping {audio_file} (not found)")
                continue
            
            print(f"  [{i}/{len(audio_files)}] Processing {Path(audio_file).name}...")
            self.analyze_voice(audio_file, voice_name, update_existing=True)
        
        # Get final model
        if voice_name in self.voice_models:
            model = self.voice_models[voice_name]
        else:
            raise ValueError(f"Failed to create voice model '{voice_name}'")
        
        print(f"✓ Batch training complete")
        print(f"  Total exposure: {model.characteristics.exposure_time:.1f}s")
        print(f"  Confidence: {model.characteristics.confidence:.2%}")
        
        return model
    
    def train_parrot(
        self,
        audio_file: str,
        voice_name: str,
        update_existing: bool = True
    ) -> VoiceModel:
        """
        Train Parrot on a voice from audio file.
        
        Args:
            audio_file: Path to audio file
            voice_name: Name for the voice model
            update_existing: If True, update existing model
        
        Returns:
            Trained VoiceModel
        """
        characteristics = self.analyze_voice(audio_file, voice_name, update_existing)
        
        # Create or update voice model
        if voice_name in self.voice_models and update_existing:
            model = self.voice_models[voice_name]
            model.characteristics = characteristics
        else:
            model = VoiceModel(
                name=voice_name,
                characteristics=characteristics,
                metadata={
                    'source_file': audio_file,
                    'trained_at': str(Path(audio_file).stat().st_mtime) if Path(audio_file).exists() else None
                }
            )
        
        self.voice_models[voice_name] = model
        return model
    
    def synthesize_vocal(
        self,
        text: str,
        voice_name: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> np.ndarray:
        """
        Synthesize vocal audio using learned voice model.
        
        Args:
            text: Text to synthesize
            voice_name: Name of voice model to use (uses current if None)
            output_file: Optional path to save audio
        
        Returns:
            Synthesized audio as numpy array
        """
        if voice_name:
            if voice_name not in self.voice_models:
                raise ValueError(f"Voice model '{voice_name}' not found")
            model = self.voice_models[voice_name]
        elif self.current_model:
            model = self.current_model
        else:
            raise ValueError("No voice model selected. Train a model first or specify voice_name.")
        
        if model.characteristics.confidence < self.config.confidence_threshold:
            raise ValueError(
                f"Voice model confidence ({model.characteristics.confidence:.2f}) "
                f"below threshold ({self.config.confidence_threshold}). "
                f"Train with more audio (minimum {self.config.min_exposure_time}s)."
            )
        
        # Enhanced formant synthesis
        from music_brain.vocal.synthesis import formant_synthesize
        from music_brain.vocal.phonemes import text_to_phonemes
        
        # Convert text to phonemes
        phonemes = text_to_phonemes(text)
        
        # Apply prosody (rhythm and intonation)
        if self.config.prosody_strength > 0 and model.characteristics.speaking_rate > 0:
            phonemes = self._apply_prosody(phonemes, model.characteristics)
        
        # Synthesize using formant synthesis
        if self.config.use_formant_synthesis:
            audio = formant_synthesize(
                phonemes,
                model,
                sample_rate=self.config.synthesis_sample_rate,
                emotion=self.config.emotion,
                expression_intensity=self.config.expression_intensity
            )
        else:
            # Fallback to simple synthesis
            audio = self._simple_synthesize(phonemes, model)
        
        if output_file:
            sf.write(output_file, audio, self.config.synthesis_sample_rate)
        
        return audio
    
    def _simple_synthesize(
        self,
        phonemes: List,
        model: VoiceModel
    ) -> np.ndarray:
        """Simple synthesis fallback."""
        sample_rate = self.config.synthesis_sample_rate
        total_duration = sum(p.duration for p in phonemes)
        t = np.linspace(0, total_duration, int(sample_rate * total_duration))
        
        base_freq = model.characteristics.average_pitch if model.characteristics.average_pitch > 0 else 200.0
        
        # Generate with vibrato
        vibrato = model.characteristics.vibrato_rate
        vibrato_depth = model.characteristics.vibrato_depth / 1200.0
        
        if vibrato > 0:
            vibrato_signal = np.sin(2 * np.pi * vibrato * t) * vibrato_depth
            freq_contour = base_freq * (1 + vibrato_signal)
        else:
            freq_contour = np.full_like(t, base_freq)
        
        audio = np.sin(2 * np.pi * freq_contour * t) * 0.3
        return audio
    
    def _apply_prosody(
        self,
        phonemes: List,
        char: VoiceCharacteristics
    ) -> List:
        """Apply prosody (rhythm and intonation) to phonemes."""
        from music_brain.vocal.phonemes import Phoneme
        
        # Adjust durations based on speaking rate
        if char.speaking_rate > 0:
            target_rate = char.speaking_rate
            current_rate = len([p for p in phonemes if p.phoneme_type.value == 'vowel']) / sum(p.duration for p in phonemes)
            if current_rate > 0:
                duration_scale = target_rate / current_rate
                for phoneme in phonemes:
                    phoneme.duration *= duration_scale
        
        # Apply intonation (pitch contour)
        if char.intonation_range > 0 and char.pitch_contour_template:
            # Map template to phonemes
            vowel_indices = [i for i, p in enumerate(phonemes) if p.phoneme_type.value == 'vowel']
            if vowel_indices and char.pitch_contour_template:
                for idx, vowel_idx in enumerate(vowel_indices):
                    if idx < len(char.pitch_contour_template):
                        pitch_shift = char.pitch_contour_template[idx] * char.intonation_range
                        base_pitch = char.average_pitch if char.average_pitch > 0 else 200.0
                        phonemes[vowel_idx].pitch = base_pitch * (2 ** (pitch_shift / 12))
        
        return phonemes
    
    def set_current_voice(self, voice_name: str):
        """Set the current voice model for synthesis."""
        if voice_name not in self.voice_models:
            raise ValueError(f"Voice model '{voice_name}' not found")
        self.current_model = self.voice_models[voice_name]
    
    def list_voices(self) -> List[str]:
        """List all trained voice models."""
        return list(self.voice_models.keys())
    
    def get_voice_info(self, voice_name: str) -> Dict[str, Any]:
        """Get information about a voice model."""
        if voice_name not in self.voice_models:
            raise ValueError(f"Voice model '{voice_name}' not found")
        
        model = self.voice_models[voice_name]
        char = model.characteristics
        return {
            'name': model.name,
            'exposure_time': char.exposure_time,
            'sample_count': char.sample_count,
            'confidence': char.confidence,
            'average_pitch': char.average_pitch,
            'pitch_range': char.pitch_range,
            'vibrato_rate': char.vibrato_rate,
            'vibrato_depth': char.vibrato_depth,
            'vowels_learned': [v.value if hasattr(v, 'value') else str(v) for v in char.vowel_formants.keys()],
            'jitter': char.jitter,
            'shimmer': char.shimmer,
            'hnr': char.hnr,
            'speaking_rate': char.speaking_rate,
            'breathiness': char.breathiness,
            'nasality': char.nasality,
            'training_files': char.training_files,
            'metadata': model.metadata
        }
    
    def blend_voices(
        self,
        voice1_name: str,
        voice2_name: str,
        blend_ratio: float = 0.5,
        output_name: Optional[str] = None
    ) -> VoiceModel:
        """
        Blend two voice models together.
        
        Args:
            voice1_name: First voice model name
            voice2_name: Second voice model name
            blend_ratio: Blend ratio (0.0 = voice1, 1.0 = voice2, 0.5 = equal)
            output_name: Name for blended model (auto-generated if None)
        
        Returns:
            Blended VoiceModel
        """
        if voice1_name not in self.voice_models:
            raise ValueError(f"Voice model '{voice1_name}' not found")
        if voice2_name not in self.voice_models:
            raise ValueError(f"Voice model '{voice2_name}' not found")
        
        model1 = self.voice_models[voice1_name]
        model2 = self.voice_models[voice2_name]
        char1 = model1.characteristics
        char2 = model2.characteristics
        
        # Create blended characteristics
        blended = VoiceCharacteristics()
        
        # Blend formants
        all_vowels = set(char1.vowel_formants.keys()) | set(char2.vowel_formants.keys())
        for vowel in all_vowels:
            formants1 = char1.vowel_formants.get(vowel, [])
            formants2 = char2.vowel_formants.get(vowel, [])
            
            if formants1 and formants2:
                # Blend average formants
                avg1 = _average_formants(formants1)
                avg2 = _average_formants(formants2)
                
                blended_formant = FormantData(
                    f1=avg1.f1 * (1 - blend_ratio) + avg2.f1 * blend_ratio,
                    f2=avg1.f2 * (1 - blend_ratio) + avg2.f2 * blend_ratio,
                    f3=avg1.f3 * (1 - blend_ratio) + avg2.f3 * blend_ratio,
                    confidence=(avg1.confidence + avg2.confidence) / 2
                )
                blended.vowel_formants[vowel] = [blended_formant]
            elif formants1:
                blended.vowel_formants[vowel] = formants1
            elif formants2:
                blended.vowel_formants[vowel] = formants2
        
        # Blend pitch
        blended.average_pitch = char1.average_pitch * (1 - blend_ratio) + char2.average_pitch * blend_ratio
        blended.pitch_range = (
            char1.pitch_range[0] * (1 - blend_ratio) + char2.pitch_range[0] * blend_ratio,
            char1.pitch_range[1] * (1 - blend_ratio) + char2.pitch_range[1] * blend_ratio
        )
        blended.vibrato_rate = char1.vibrato_rate * (1 - blend_ratio) + char2.vibrato_rate * blend_ratio
        blended.vibrato_depth = char1.vibrato_depth * (1 - blend_ratio) + char2.vibrato_depth * blend_ratio
        
        # Blend timbre
        blended.spectral_centroid_mean = char1.spectral_centroid_mean * (1 - blend_ratio) + char2.spectral_centroid_mean * blend_ratio
        blended.spectral_rolloff_mean = char1.spectral_rolloff_mean * (1 - blend_ratio) + char2.spectral_rolloff_mean * blend_ratio
        blended.spectral_bandwidth_mean = char1.spectral_bandwidth_mean * (1 - blend_ratio) + char2.spectral_bandwidth_mean * blend_ratio
        
        # Blend other characteristics
        blended.breathiness = char1.breathiness * (1 - blend_ratio) + char2.breathiness * blend_ratio
        blended.nasality = char1.nasality * (1 - blend_ratio) + char2.nasality * blend_ratio
        blended.jitter = char1.jitter * (1 - blend_ratio) + char2.jitter * blend_ratio
        blended.shimmer = char1.shimmer * (1 - blend_ratio) + char2.shimmer * blend_ratio
        blended.speaking_rate = char1.speaking_rate * (1 - blend_ratio) + char2.speaking_rate * blend_ratio
        
        # Blend confidence
        blended.confidence = (char1.confidence + char2.confidence) / 2
        blended.exposure_time = char1.exposure_time + char2.exposure_time
        blended.sample_count = char1.sample_count + char2.sample_count
        blended.training_files = list(set(char1.training_files + char2.training_files))
        
        # Create blended model
        output_name = output_name or f"{voice1_name}_{voice2_name}_blend"
        blended_model = VoiceModel(
            name=output_name,
            characteristics=blended,
            metadata={
                'blended_from': [voice1_name, voice2_name],
                'blend_ratio': blend_ratio
            }
        )
        
        self.voice_models[output_name] = blended_model
        return blended_model


def _average_formants(formants_list: List[FormantData]) -> FormantData:
    """Calculate average formants from a list."""
    if not formants_list:
        return FormantData(f1=0, f2=0, f3=0)
    
    avg_f1 = np.mean([f.f1 for f in formants_list])
    avg_f2 = np.mean([f.f2 for f in formants_list])
    avg_f3 = np.mean([f.f3 for f in formants_list])
    avg_conf = np.mean([f.confidence for f in formants_list])
    
    return FormantData(
        f1=float(avg_f1),
        f2=float(avg_f2),
        f3=float(avg_f_f3),
        confidence=float(avg_conf)
    )


# Convenience functions
def analyze_voice(audio_file: str, voice_name: Optional[str] = None) -> VoiceCharacteristics:
    """Analyze voice characteristics from audio file."""
    parrot = ParrotVocalSynthesizer()
    return parrot.analyze_voice(audio_file, voice_name)


def train_parrot(audio_file: str, voice_name: str) -> VoiceModel:
    """Train Parrot on a voice from audio file."""
    parrot = ParrotVocalSynthesizer()
    return parrot.train_parrot(audio_file, voice_name)


def synthesize_vocal(
    text: str,
    voice_name: str,
    output_file: Optional[str] = None
) -> np.ndarray:
    """Synthesize vocal audio using learned voice model."""
    parrot = ParrotVocalSynthesizer()
    return parrot.synthesize_vocal(text, voice_name, output_file)


def load_voice_model(file_path: str) -> VoiceModel:
    """Load a voice model from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return VoiceModel.from_dict(data)


def save_voice_model(model: VoiceModel, file_path: str):
    """Save a voice model to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(model.to_dict(), f, indent=2)

