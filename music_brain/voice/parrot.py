"""
Parrot - Unified Singing Voice Synthesizer

Kelly = Emotional side (intent/affect inputs). Dee = Music/theory side (melody/harmony/arrangement).
Main API for singing voice synthesis with voice mimicking and
instrument conversion capabilities.

Features:
- Sing from lyrics and melody
- Record and mimic user voice
- Convert sung notes to different instruments
- Support for formant (preview) and neural (production) backends
"""

import numpy as np
from typing import List, Optional, Dict, Union
from pathlib import Path
import soundfile as sf

from music_brain.voice.phoneme_processor import PhonemeProcessor, PhonemeSequence
from music_brain.voice.pitch_controller import PitchController, ExpressionParams
from music_brain.voice.singing_synthesizer import SingingSynthesizer, FormantConfig
from music_brain.voice.neural_backend import NeuralBackend, create_neural_backend
from music_brain.voice.voice_input import VoiceRecorder, VoiceMimic
from music_brain.voice.instrument_synth import InstrumentSynthesizer, get_instrument_preset
from music_brain.voice.voice_learning import VoiceLearningManager, LearnedVoiceProfile
from music_brain.groove.fan_feedback import FanProfile


class Parrot:
    """
    Parrot - Unified singing voice synthesizer.

    Example:
        >>> parrot = Parrot(backend="auto")
        >>>
        >>> # Sing from lyrics and melody
        >>> audio = parrot.sing(
        ...     lyrics="Hello world",
        ...     melody=[60, 62, 64, 65, 64, 62, 60],
        ...     tempo_bpm=120
        ... )
        >>> parrot.save("output.wav", audio)
        >>>
        >>> # Record and mimic user voice
        >>> recorded = parrot.record_voice(duration_seconds=3.0)
        >>> characteristics = parrot.extract_voice(recorded)
        >>> mimicked = parrot.sing_with_voice(
        ...     lyrics="Hello world",
        ...     melody=[60, 62, 64],
        ...     voice_characteristics=characteristics
        ... )
        >>>
        >>> # Convert sung notes to instrument
        >>> midi_notes = parrot.extract_notes_from_audio(recorded)
        >>> instrument_audio = parrot.notes_to_instrument(midi_notes, instrument="piano")
    """

    def __init__(
        self,
        backend: str = "auto",  # "formant", "neural", or "auto"
        voice_model: Optional[str] = None,
        device: str = "auto",
        sample_rate: int = 44100
    ):
        """
        Initialize Parrot.

        Args:
            backend: Synthesis backend ("formant", "neural", or "auto")
            voice_model: Optional voice model path (for neural backend)
            device: Device for neural backend ("auto", "cuda", "mps", "cpu")
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.backend_type = backend

        # Initialize components
        self.phoneme_processor = PhonemeProcessor()
        self.pitch_controller = PitchController(sample_rate)

        # Formant backend (always available)
        formant_config = FormantConfig(sample_rate=sample_rate)
        self.formant_synth = SingingSynthesizer(formant_config)

        # Neural backend (optional)
        self.neural_backend = None
        if backend in ["neural", "auto"]:
            self.neural_backend = create_neural_backend(
                model_path=voice_model,
                device=device
            )
            if not self.neural_backend.is_available():
                if backend == "neural":
                    print(
                        "Warning: Neural backend requested but not available. Using formant backend.")
                self.backend_type = "formant"

        # Voice input/mimicking
        self.voice_recorder = VoiceRecorder(sample_rate=sample_rate)
        self.voice_mimic = VoiceMimic(sample_rate=sample_rate)

        # Instrument synthesizer
        self.instrument_synth = InstrumentSynthesizer("piano", sample_rate)

        # Voice learning
        self.learning_manager = VoiceLearningManager(sample_rate=sample_rate)

        # State
        self.has_the_a = False

    def give_the_a(self):
        """
        The fans give the 'A'. This unlocks the Parrot's full potential.
        """
        print("PRRT: *SQUAWK* AAAAAAA! *Clears throat*")
        print("Parrot: Thank you. I am ready to sing.")
        print("Parrot: *Vibrating intensely* I feel the power of the A!")
        self.has_the_a = True

    def sing(
        self,
        lyrics: str,
        melody: List[int],
        tempo_bpm: float = 120.0,
        expression: Optional[Dict] = None,
        voice_characteristics: Optional[Dict] = None,
        fan_profile: Optional[FanProfile] = None
    ) -> np.ndarray:
        """
        Synthesize singing from lyrics and melody.

        Args:
            lyrics: Lyrics text
            melody: List of MIDI note numbers
            tempo_bpm: Tempo in BPM
            expression: Optional expression parameters (vibrato, dynamics, etc.)
            voice_characteristics: Optional voice characteristics for mimicking
            fan_profile: Optional FanProfile to influence performance

        Returns:
            Audio signal
        """
        # Process lyrics to phonemes
        phoneme_sequence = self.phoneme_processor.process_lyrics(
            lyrics, melody, tempo_bpm
        )

        # Create pitch curve
        note_durations = [phoneme_sequence.total_duration_ms /
                          1000.0 / len(melody)] * len(melody)

        # Default expression - CURED LARYNGITIS DEFAULTS
        vib_rate = 5.5
        vib_depth = 0.15  # Increased from 0.02 (2 cents) to 0.15 (15 cents)
        dynamics = None

        if expression:
            vib_rate = expression.get("vibrato_rate", 5.5)
            vib_depth = expression.get("vibrato_depth", 0.15)
            dynamics = expression.get("dynamics")

        # Apply Fan Profile Influence
        if fan_profile:
            # Metalhead: More intensity (dynamics), faster vibrato?
            if fan_profile.velocity_multiplier > 1.1:
                vib_rate *= 1.2
                vib_depth *= 1.5
                # Simulate louder dynamics if not provided
                if dynamics is None:
                    dynamics = np.ones(len(melody)) * 1.2

            # Reggae/Chill: Slower vibrato, smoother
            elif fan_profile.velocity_multiplier < 0.9:
                vib_rate *= 0.8
                vib_depth *= 0.8

        # Check for "The A" - if missing, maybe add random squawks or reduce quality?
        if not self.has_the_a:
            # For now, just print a warning, or maybe inject a squawk into the lyrics?
            # (Injecting into lyrics would require re-processing phonemes, let's just warn)
            print("PRRT: *Squawk* Where is my A? (Performance may be lackluster)")
            vib_depth *= 0.5  # Still penalize, but base is higher now (0.075)
        else:
            # WITH THE A: FULL POTENTIAL
            vib_depth *= 1.5  # Boost to 0.225+ (Professional level)
            if dynamics is None:
                dynamics = np.ones(len(melody)) * 1.1  # Louder/Confident

        expression_params = ExpressionParams(
            vibrato_rate=vib_rate,
            vibrato_depth=vib_depth,
            portamento_time=expression.get(
                "portamento_time", 0.05) if expression else 0.05,
            dynamics=dynamics
        )

        pitch_curve = self.pitch_controller.create_pitch_curve(
            melody, note_durations, expression_params
        )

        # Choose backend
        use_neural = (
            self.backend_type == "neural" or
            (self.backend_type ==
             "auto" and self.neural_backend and self.neural_backend.is_available())
        )

        if use_neural:
            # Try neural backend
            audio = self.neural_backend.synthesize(
                phoneme_sequence, pitch_curve, expression)
            if audio is None:
                # Fallback to formant
                use_neural = False

        if not use_neural:
            # Use formant backend
            audio = self.formant_synth.synthesize(
                phoneme_sequence, pitch_curve, expression)

        # Apply voice characteristics if provided (mimicking)
        if voice_characteristics:
            audio = self.voice_mimic.apply_voice_characteristics(
                audio, voice_characteristics)

        return audio

    def preview(
        self,
        lyrics: str,
        melody: List[int],
        tempo_bpm: float = 120.0
    ) -> np.ndarray:
        """
        Quick preview (always uses formant backend).

        Args:
            lyrics: Lyrics text
            melody: List of MIDI note numbers
            tempo_bpm: Tempo in BPM

        Returns:
            Audio signal
        """
        return self.sing(lyrics, melody, tempo_bpm)

    def record_voice(
        self,
        duration_seconds: Optional[float] = None,
        until_silence: bool = False
    ) -> Optional[np.ndarray]:
        """
        Record voice from microphone.

        Args:
            duration_seconds: Duration to record (if None and until_silence=False, uses 3 seconds)
            until_silence: Record until silence is detected

        Returns:
            Audio signal or None if recording failed
        """
        if until_silence:
            return self.voice_recorder.record_until_silence()
        else:
            if duration_seconds is None:
                duration_seconds = 3.0
            return self.voice_recorder.record(duration_seconds)

    def extract_voice(self, audio: np.ndarray) -> Dict:
        """
        Extract voice characteristics from audio.

        Args:
            audio: Audio signal

        Returns:
            Dictionary of voice characteristics
        """
        return self.voice_mimic.extract_voice_characteristics(audio)

    def sing_with_voice(
        self,
        lyrics: str,
        melody: List[int],
        voice_characteristics: Dict,
        tempo_bpm: float = 120.0,
        expression: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Sing with mimicked voice characteristics.

        Args:
            lyrics: Lyrics text
            melody: List of MIDI note numbers
            voice_characteristics: Voice characteristics to mimic
            tempo_bpm: Tempo in BPM
            expression: Optional expression parameters

        Returns:
            Audio signal with mimicked voice
        """
        return self.sing(lyrics, melody, tempo_bpm, expression, voice_characteristics)

    def extract_notes_from_audio(
        self,
        audio: np.ndarray,
        note_duration: float = 0.25
    ) -> List[int]:
        """
        Extract MIDI notes from sung audio.

        Args:
            audio: Audio signal (sung voice)
            note_duration: Expected note duration in seconds

        Returns:
            List of MIDI note numbers
        """
        return self.pitch_controller.audio_to_midi_notes(audio, self.sample_rate, note_duration)

    def notes_to_instrument(
        self,
        midi_notes: List[int],
        instrument: str = "piano",
        note_durations: Optional[List[float]] = None,
        velocities: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Convert MIDI notes to instrument audio.

        Args:
            midi_notes: List of MIDI note numbers
            instrument: Instrument name (piano, guitar, strings, flute, trumpet, violin)
            note_durations: Optional durations for each note (default: 0.5 seconds)
            velocities: Optional velocities for each note (default: 0.8)

        Returns:
            Audio signal
        """
        # Create instrument synthesizer
        synth = InstrumentSynthesizer(instrument, self.sample_rate)

        if note_durations is None:
            note_durations = [0.5] * len(midi_notes)

        return synth.synthesize_notes(midi_notes, note_durations, velocities)

    def save(self, output_path: str, audio: np.ndarray) -> bool:
        """
        Save audio to file.

        Args:
            output_path: Output file path
            audio: Audio signal

        Returns:
            True if successful
        """
        try:
            sf.write(output_path, audio, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False

    def save_recording(self, audio: np.ndarray, output_path: str) -> bool:
        """
        Save recorded audio to file.

        Args:
            audio: Audio signal
            output_path: Output file path

        Returns:
            True if successful
        """
        return self.voice_recorder.save_recording(audio, output_path)

    def add_voice_sample(
        self,
        audio: np.ndarray,
        sample_id: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a voice sample for learning.

        Args:
            audio: Audio signal (voice recording)
            sample_id: Optional sample ID (auto-generated if None)
            text: Optional transcript of what was said/sung
            metadata: Optional metadata (emotion, style, etc.)

        Returns:
            Sample ID
        """
        return self.learning_manager.add_sample(audio, sample_id, text, metadata)

    def learn_voice_profile(
        self,
        profile_name: str,
        sample_ids: Optional[List[str]] = None
    ) -> LearnedVoiceProfile:
        """
        Learn a voice profile from stored samples.

        Args:
            profile_name: Name for the learned profile
            sample_ids: Optional list of sample IDs (uses all if None)

        Returns:
            LearnedVoiceProfile
        """
        return self.learning_manager.learn_profile(profile_name, sample_ids)

    def load_voice_profile(self, profile_name: str) -> Optional[Dict]:
        """
        Load a learned voice profile for use in synthesis.

        Args:
            profile_name: Profile name

        Returns:
            Voice characteristics dict (for use with sing_with_voice) or None
        """
        return self.learning_manager.get_profile_characteristics(profile_name)

    def list_voice_profiles(self) -> List[str]:
        """List all learned voice profiles."""
        return self.learning_manager.list_profiles()

    def update_voice_profile(
        self,
        profile_name: str,
        new_sample_ids: List[str]
    ) -> LearnedVoiceProfile:
        """
        Update an existing voice profile with new samples.

        Args:
            profile_name: Profile name
            new_sample_ids: List of new sample IDs

        Returns:
            Updated LearnedVoiceProfile
        """
        return self.learning_manager.update_profile_from_samples(profile_name, new_sample_ids)

    def sing_with_learned_voice(
        self,
        lyrics: str,
        melody: List[int],
        profile_name: str,
        tempo_bpm: float = 120.0,
        expression: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Sing using a learned voice profile.

        Args:
            lyrics: Lyrics text
            melody: List of MIDI note numbers
            profile_name: Name of learned voice profile
            tempo_bpm: Tempo in BPM
            expression: Optional expression parameters

        Returns:
            Audio signal
        """
        characteristics = self.load_voice_profile(profile_name)
        if not characteristics:
            raise ValueError(f"Voice profile '{profile_name}' not found")

        return self.sing_with_voice(lyrics, melody, characteristics, tempo_bpm, expression)


# Convenience function
def create_parrot(
    backend: str = "auto",
    voice_model: Optional[str] = None,
    device: str = "auto"
) -> Parrot:
    """
    Create Parrot instance.

    Args:
        backend: Synthesis backend
        voice_model: Optional voice model path
        device: Device for neural backend

    Returns:
        Parrot instance
    """
    return Parrot(backend=backend, voice_model=voice_model, device=device)
