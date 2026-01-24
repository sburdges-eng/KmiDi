"""
Formant Synthesis Engine for Parrot

Implements proper formant synthesis using learned voice characteristics.
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy import signal

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

from music_brain.vocal.parrot import VoiceModel, FormantData, VowelType
from music_brain.vocal.phonemes import Phoneme, text_to_phonemes, phoneme_to_vowel_type


def formant_synthesize(
    phonemes: List[Phoneme],
    voice_model: VoiceModel,
    sample_rate: int = 44100,
    emotion: Optional[str] = None,
    expression_intensity: float = 0.5
) -> np.ndarray:
    """
    Synthesize audio using formant synthesis.
    
    Args:
        phonemes: List of phonemes to synthesize
        voice_model: Learned voice model
        sample_rate: Sample rate for output
        emotion: Emotion preset (happy, sad, angry, etc.)
        expression_intensity: Expression intensity (0.0-1.0)
    
    Returns:
        Synthesized audio as numpy array
    """
    audio_segments = []
    char = voice_model.characteristics
    
    # Emotion modifications
    emotion_mods = _get_emotion_modifications(emotion, expression_intensity) if emotion else {}
    
    for phoneme in phonemes:
        if phoneme.phoneme_type.value == 'silence':
            # Generate silence
            silence_samples = int(phoneme.duration * sample_rate)
            audio_segments.append(np.zeros(silence_samples))
            continue
        
        # Get formant data for vowel
        if phoneme.phoneme_type.value == 'vowel':
            vowel_type_str = phoneme_to_vowel_type(phoneme)
            if vowel_type_str:
                try:
                    vowel_type = VowelType[vowel_type_str]
                    formants_list = char.vowel_formants.get(vowel_type, [])
                    
                    if formants_list:
                        # Use average formants for this vowel
                        avg_formants = _average_formants(formants_list)
                    else:
                        # Fallback to standard formants
                        avg_formants = _get_standard_formants(vowel_type_str)
                except (KeyError, ValueError):
                    avg_formants = _get_standard_formants('A')
            else:
                avg_formants = _get_standard_formants('A')
            
            # Apply emotion modifications
            if emotion_mods:
                avg_formants = _apply_emotion_to_formants(avg_formants, emotion_mods)
            
            # Synthesize vowel with formants
            segment = _synthesize_vowel(
                avg_formants,
                phoneme,
                char,
                sample_rate,
                emotion_mods
            )
        else:
            # Synthesize consonant
            segment = _synthesize_consonant(
                phoneme,
                char,
                sample_rate
            )
        
        audio_segments.append(segment)
    
    # Concatenate all segments
    if audio_segments:
        audio = np.concatenate(audio_segments)
    else:
        audio = np.array([])
    
    # Apply global voice characteristics
    audio = _apply_voice_characteristics(audio, char, sample_rate, emotion_mods)
    
    # Normalize
    if len(audio) > 0 and np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio


def _synthesize_vowel(
    formants: FormantData,
    phoneme: Phoneme,
    char,
    sample_rate: int,
    emotion_mods: dict
) -> np.ndarray:
    """Synthesize a vowel using formant synthesis."""
    duration = phoneme.duration
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    
    # Get pitch
    base_pitch = char.average_pitch if char.average_pitch > 0 else 200.0
    if phoneme.pitch:
        base_pitch = phoneme.pitch
    
    # Apply stress (higher pitch for stressed syllables)
    if phoneme.stress > 0:
        base_pitch *= (1.0 + phoneme.stress * 0.1)
    
    # Apply emotion pitch modifications
    if emotion_mods.get('pitch_shift', 0) != 0:
        base_pitch *= (1.0 + emotion_mods['pitch_shift'])
    
    # Generate pitch contour with vibrato
    if char.vibrato_rate > 0 and char.vibrato_depth > 0:
        vibrato = np.sin(2 * np.pi * char.vibrato_rate * t)
        vibrato_depth_ratio = char.vibrato_depth / 1200.0
        pitch_contour = base_pitch * (1.0 + vibrato * vibrato_depth_ratio)
    else:
        pitch_contour = np.full_like(t, base_pitch)
    
    # Add jitter (pitch period variation)
    if char.jitter > 0:
        jitter_amount = char.jitter / 100.0
        jitter_noise = np.random.normal(0, jitter_amount, len(t))
        pitch_contour *= (1.0 + jitter_noise)
    
    # Generate excitation signal (glottal pulse)
    excitation = _generate_glottal_pulse(pitch_contour, t, sample_rate, char)
    
    # Apply formant filters
    audio = excitation.copy()
    
    # F1 filter (first formant)
    if formants.f1 > 0:
        audio = _apply_formant_filter(audio, formants.f1, 50, sample_rate)
    
    # F2 filter (second formant)
    if formants.f2 > 0:
        audio = _apply_formant_filter(audio, formants.f2, 70, sample_rate)
    
    # F3 filter (third formant)
    if formants.f3 > 0:
        audio = _apply_formant_filter(audio, formants.f3, 90, sample_rate)
    
    # Apply timbre shaping
    audio = _apply_timbre_shaping(audio, char, sample_rate)
    
    # Add breathiness
    if char.breathiness > 0:
        breath_noise = np.random.normal(0, char.breathiness * 0.1, len(audio))
        audio = audio * (1.0 - char.breathiness * 0.3) + breath_noise * char.breathiness
    
    # Add shimmer (amplitude variation)
    if char.shimmer > 0:
        shimmer_amount = char.shimmer / 100.0
        shimmer_noise = np.random.normal(1.0, shimmer_amount, len(audio))
        audio *= shimmer_noise
    
    # Apply attack and release envelopes
    attack_samples = int(char.attack_time * sample_rate) if char.attack_time > 0 else int(0.01 * sample_rate)
    release_samples = int(char.release_time * sample_rate) if char.release_time > 0 else int(0.05 * sample_rate)
    
    if attack_samples > 0:
        attack_env = np.linspace(0, 1, attack_samples)
        audio[:attack_samples] *= attack_env
    
    if release_samples > 0:
        release_env = np.linspace(1, 0, release_samples)
        audio[-release_samples:] *= release_env
    
    return audio


def _synthesize_consonant(
    phoneme: Phoneme,
    char,
    sample_rate: int
) -> np.ndarray:
    """Synthesize a consonant."""
    duration = phoneme.duration
    num_samples = int(duration * sample_rate)
    
    # Consonants are noise-based or brief transitions
    consonant_type = phoneme.symbol
    
    # Fricatives (noise-based)
    fricatives = ['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h']
    if consonant_type in fricatives:
        # Generate noise
        noise = np.random.normal(0, 0.3, num_samples)
        # Apply high-pass filter for fricatives
        if LIBROSA_AVAILABLE:
            # Simple high-pass using spectral filtering
            fft = np.fft.fft(noise)
            freqs = np.fft.fftfreq(len(noise), 1/sample_rate)
            # Attenuate low frequencies
            fft[freqs < 1000] *= 0.1
            noise = np.real(np.fft.ifft(fft))
        return noise
    
    # Plosives (brief burst)
    plosives = ['p', 'b', 't', 'd', 'k', 'g']
    if consonant_type in plosives:
        # Brief noise burst
        burst_samples = min(int(0.01 * sample_rate), num_samples)
        burst = np.random.normal(0, 0.5, burst_samples)
        # Exponential decay
        decay = np.exp(-np.linspace(0, 5, burst_samples))
        burst *= decay
        # Pad with silence
        if num_samples > burst_samples:
            silence = np.zeros(num_samples - burst_samples)
            return np.concatenate([burst, silence])
        return burst[:num_samples]
    
    # Nasals and liquids (formant-like)
    nasals_liquids = ['m', 'n', 'ŋ', 'l', 'r', 'w', 'j']
    if consonant_type in nasals_liquids:
        # Generate brief formant-like sound
        t = np.linspace(0, duration, num_samples)
        freq = 200.0  # Low frequency for nasals
        signal = np.sin(2 * np.pi * freq * t) * 0.2
        # Apply envelope
        envelope = np.exp(-t * 10)  # Quick decay
        signal *= envelope
        return signal
    
    # Default: brief silence
    return np.zeros(num_samples)


def _generate_glottal_pulse(
    pitch_contour: np.ndarray,
    t: np.ndarray,
    sample_rate: int,
    char
) -> np.ndarray:
    """Generate glottal pulse excitation signal."""
    excitation = np.zeros_like(t)
    
    for i, pitch in enumerate(pitch_contour):
        if pitch > 0:
            period = 1.0 / pitch
            phase = (t[i] % period) / period
            
            # Simple glottal pulse model (Rosenberg model)
            if phase < 0.5:
                pulse = np.sin(np.pi * phase)
            else:
                pulse = 0.0
            
            excitation[i] = pulse
    
    # Normalize
    if np.max(np.abs(excitation)) > 0:
        excitation = excitation / np.max(np.abs(excitation))
    
    return excitation * 0.5


def _apply_formant_filter(
    audio: np.ndarray,
    formant_freq: float,
    bandwidth: float,
    sample_rate: int
) -> np.ndarray:
    """Apply a formant filter (resonator)."""
    # Design IIR bandpass filter for formant
    nyquist = sample_rate / 2.0
    low = max(1.0, (formant_freq - bandwidth) / nyquist)
    high = min(0.99, (formant_freq + bandwidth) / nyquist)
    
    if low < high:
        b, a = signal.butter(2, [low, high], btype='band')
        audio = signal.filtfilt(b, a, audio)
    
    return audio


def _apply_timbre_shaping(
    audio: np.ndarray,
    char,
    sample_rate: int
) -> np.ndarray:
    """Apply timbre characteristics using spectral shaping."""
    if not LIBROSA_AVAILABLE:
        return audio
    
    # Apply spectral centroid (brightness)
    if char.spectral_centroid_mean > 0:
        # Simple brightness adjustment using high-pass
        nyquist = sample_rate / 2.0
        cutoff = min(char.spectral_centroid_mean / nyquist, 0.9)
        if cutoff > 0.01:
            b, a = signal.butter(2, cutoff, btype='high')
            audio = signal.filtfilt(b, a, audio)
    
    return audio


def _apply_voice_characteristics(
    audio: np.ndarray,
    char,
    sample_rate: int,
    emotion_mods: dict
) -> np.ndarray:
    """Apply global voice characteristics."""
    # Apply nasality if present
    if char.nasality > 0:
        # Add nasal formant around 1000 Hz
        nyquist = sample_rate / 2.0
        nasal_freq = 1000.0 / nyquist
        if nasal_freq < 0.9:
            b, a = signal.butter(2, [nasal_freq * 0.8, nasal_freq * 1.2], btype='band')
            nasal_component = signal.filtfilt(b, a, audio) * char.nasality * 0.3
            audio = audio + nasal_component
    
    return audio


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
        f3=float(avg_f3),
        confidence=float(avg_conf)
    )


def _get_standard_formants(vowel_type: str) -> FormantData:
    """Get standard formant values for a vowel."""
    standard = {
        'A': FormantData(f1=730, f2=1090, f3=2440),
        'E': FormantData(f1=570, f2=1980, f3=2440),
        'I': FormantData(f1=270, f2=2290, f3=3010),
        'O': FormantData(f1=570, f2=840, f3=2410),
        'U': FormantData(f1=300, f2=870, f3=2240),
        'SCHWA': FormantData(f1=500, f2=1500, f3=2500),
    }
    return standard.get(vowel_type, standard['A'])


def _get_emotion_modifications(emotion: str, intensity: float) -> dict:
    """Get formant and pitch modifications for emotion."""
    mods = {
        'happy': {
            'pitch_shift': 0.15 * intensity,  # Higher pitch
            'formant_shift_f1': -0.1 * intensity,  # Slightly brighter
            'formant_shift_f2': 0.1 * intensity,
        },
        'sad': {
            'pitch_shift': -0.1 * intensity,  # Lower pitch
            'formant_shift_f1': 0.1 * intensity,  # Darker
            'formant_shift_f2': -0.1 * intensity,
        },
        'angry': {
            'pitch_shift': 0.2 * intensity,  # Higher, tense
            'formant_shift_f1': -0.15 * intensity,
            'formant_shift_f2': 0.15 * intensity,
        },
        'excited': {
            'pitch_shift': 0.25 * intensity,
            'formant_shift_f1': -0.2 * intensity,
            'formant_shift_f2': 0.2 * intensity,
        },
        'calm': {
            'pitch_shift': -0.05 * intensity,
            'formant_shift_f1': 0.05 * intensity,
            'formant_shift_f2': -0.05 * intensity,
        },
    }
    return mods.get(emotion.lower(), {})


def _apply_emotion_to_formants(
    formants: FormantData,
    emotion_mods: dict
) -> FormantData:
    """Apply emotion modifications to formants."""
    f1_shift = emotion_mods.get('formant_shift_f1', 0.0)
    f2_shift = emotion_mods.get('formant_shift_f2', 0.0)
    f3_shift = emotion_mods.get('formant_shift_f3', 0.0)
    
    return FormantData(
        f1=formants.f1 * (1.0 + f1_shift),
        f2=formants.f2 * (1.0 + f2_shift),
        f3=formants.f3 * (1.0 + f3_shift),
        confidence=formants.confidence
    )

