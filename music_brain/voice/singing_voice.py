"""
SingingVoice - Unified API over formant (preview) and neural (production) backends.

Provides a simple interface that follows the plan:
- Phase 1: Enhanced formant synthesizer for quick previews.
- Phase 2: Neural backend (DiffSinger placeholder) for production.
- Phase 3: Unified API to pick the best backend.
- Phase 4: Phoneme + pitch processing helpers.
"""

from typing import List, Optional, Union, Dict
from pathlib import Path
import numpy as np
import soundfile as sf

from music_brain.voice.phoneme_processor import PhonemeProcessor
from music_brain.voice.pitch_controller import PitchController, ExpressionParams, PitchCurve
from music_brain.voice.singing_synthesizer import SingingSynthesizer, FormantConfig
from music_brain.voice.neural_backend import NeuralBackend, create_neural_backend


class SingingVoice:
    """
    Unified singing voice API with formant (preview) and neural (production) backends.
    """

    def __init__(
        self,
        backend: str = "auto",  # "formant", "neural", or "auto"
        voice_model: Optional[str] = None,
        device: str = "auto",
        formant_config: Optional[FormantConfig] = None,
        sample_rate: int = 44100,
    ):
        self.backend_preference = backend
        self.sample_rate = sample_rate
        self.phoneme_processor = PhonemeProcessor()
        self.pitch_controller = PitchController(sample_rate=sample_rate)
        self.formant = SingingSynthesizer(formant_config or FormantConfig(sample_rate=sample_rate))
        self.neural: Optional[NeuralBackend] = None

        if backend in ("neural", "auto"):
            self.neural = create_neural_backend(model_path=voice_model, device=device)

    def _choose_backend(self) -> str:
        if self.backend_preference == "formant":
            return "formant"
        if self.backend_preference == "neural":
            return "neural" if self.neural and self.neural.is_available() else "formant"
        # auto: prefer neural if available
        if self.neural and self.neural.is_available():
            return "neural"
        return "formant"

    def _build_pitch_curve(
        self,
        melody: List[int],
        total_duration_ms: float,
        expression: Optional[Union[ExpressionParams, Dict]] = None,
    ) -> PitchCurve:
        note_count = max(1, len(melody))
        note_duration_s = (total_duration_ms / 1000.0) / note_count
        note_durations = [note_duration_s for _ in range(note_count)]

        if expression is None:
            expr_params = ExpressionParams()
        elif isinstance(expression, ExpressionParams):
            expr_params = expression
        else:
            expr_params = ExpressionParams(**expression)

        return self.pitch_controller.create_pitch_curve(
            melody_notes=melody,
            note_durations=note_durations,
            expression=expr_params,
        )

    def sing(
        self,
        lyrics: str,
        melody: List[int],
        tempo_bpm: float = 120.0,
        expression: Optional[Union[ExpressionParams, Dict]] = None,
        backend: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize singing from lyrics and melody.
        """
        phonemes = self.phoneme_processor.text_to_phonemes(lyrics)
        phoneme_seq = self.phoneme_processor.align_to_melody(phonemes, melody, tempo_bpm)
        pitch_curve = self._build_pitch_curve(melody, phoneme_seq.total_duration_ms, expression)

        chosen_backend = backend or self._choose_backend()
        if chosen_backend == "neural" and self.neural and self.neural.is_available():
            audio = self.neural.synthesize(phoneme_seq, pitch_curve, expression=expression)
            if audio is not None:
                return audio

        # Fallback to formant (preview) backend
        return self.formant.synthesize(phoneme_seq, pitch_curve, expression=expression or {})

    def preview(
        self,
        lyrics: str,
        melody: List[int],
        tempo_bpm: float = 120.0,
        expression: Optional[Union[ExpressionParams, Dict]] = None,
    ) -> np.ndarray:
        """
        Always uses the formant backend for a quick preview.
        """
        phonemes = self.phoneme_processor.text_to_phonemes(lyrics)
        phoneme_seq = self.phoneme_processor.align_to_melody(phonemes, melody, tempo_bpm)
        pitch_curve = self._build_pitch_curve(melody, phoneme_seq.total_duration_ms, expression)
        return self.formant.synthesize(phoneme_seq, pitch_curve, expression=expression or {})

    def save(self, path: Union[str, Path], audio: np.ndarray):
        """Save audio to file."""
        sf.write(str(path), audio, self.sample_rate)


def create_singing_voice(
    backend: str = "auto",
    voice_model: Optional[str] = None,
    device: str = "auto",
    formant_config: Optional[FormantConfig] = None,
    sample_rate: int = 44100,
) -> SingingVoice:
    """
    Factory helper to create SingingVoice with sensible defaults.
    """
    return SingingVoice(
        backend=backend,
        voice_model=voice_model,
        device=device,
        formant_config=formant_config,
        sample_rate=sample_rate,
    )
