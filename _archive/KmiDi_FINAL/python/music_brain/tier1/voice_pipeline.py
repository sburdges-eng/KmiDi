"""
VoiceGenerationPipeline
-----------------------
Thin Python wrapper enforcing clone (VoiceIdentity) vs performer (VoicePerformer)
separation while delegating synthesis to existing voice components.

This module does NOT decide structure; it only renders given controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict
import numpy as np
import soundfile as sf

from music_brain.voice.singing_voice import SingingVoice


@dataclass
class VoiceIdentity:
    """Speaker identity / timbre (clone)."""

    name: str
    embedding_path: Optional[Path] = None
    formant_profile: Optional[Dict[str, float]] = None

    @classmethod
    def load_speaker_embedding(cls, path: Path, name: Optional[str] = None) -> "VoiceIdentity":
        return cls(name=name or path.stem, embedding_path=path)


@dataclass
class VocalNote:
    pitch: int
    start_beat: float
    duration_beats: float
    lyric: str = ""
    expression: Dict[str, float] = None  # vibrato, breathiness, intensity, etc.


@dataclass
class VocalExpression:
    intensity_curve: List[float]
    phrasing_offsets: List[float]
    breathiness: float = 0.2
    brightness: float = 0.5
    emotion_hint: str = ""


@dataclass
class VoiceRenderResult:
    audio_path: Optional[Path]
    notes: List[VocalNote]
    metadata: Dict[str, Any]


class VoicePerformer:
    """
    Performer controls that override clone defaults (prosody, timing, phrasing).
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._voice = SingingVoice(backend="auto", sample_rate=sample_rate)

    def render(
        self,
        notes: List[VocalNote],
        identity: VoiceIdentity,
        expression: VocalExpression,
        output_path: Path,
    ) -> VoiceRenderResult:
        """
        Render vocals using the SingingVoice backend.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_notes = sorted(notes, key=lambda n: (n.start_beat, n.pitch))
        melody = [note.pitch for note in sorted_notes]
        lyrics = " ".join([note.lyric or "la" for note in sorted_notes]) or "la"

        # Use the average duration to estimate tempo alignment
        tempo_bpm = 120.0
        audio = self._voice.sing(
            lyrics=lyrics,
            melody=melody,
            tempo_bpm=tempo_bpm,
            expression=expression.__dict__,
        )
        sf.write(str(output_path), audio, self.sample_rate)
        metadata = {
            "identity": identity.name,
            "embedding": str(identity.embedding_path) if identity.embedding_path else None,
            "formant_profile": identity.formant_profile,
            "sample_rate": self.sample_rate,
            "expression": expression.__dict__,
        }
        return VoiceRenderResult(audio_path=output_path, notes=notes, metadata=metadata)


class VoiceGenerationPipeline:
    """
    Pipeline enforcing clone vs performer separation for vocals.
    """

    def __init__(self, sample_rate: int = 44100):
        self.performer = VoicePerformer(sample_rate=sample_rate)

    def generate(
        self,
        identity: VoiceIdentity,
        notes: List[VocalNote],
        expression: Optional[VocalExpression] = None,
        output_path: Optional[Path] = None,
    ) -> VoiceRenderResult:
        expr = expression or self._default_expression(len(notes))
        out_path = output_path or Path.home() / "Music" / "iDAW_Output" / "voice" / "voice_render.wav"
        return self.performer.render(notes=notes, identity=identity, expression=expr, output_path=out_path)

    @staticmethod
    def _default_expression(n: int) -> VocalExpression:
        return VocalExpression(
            intensity_curve=[0.6 for _ in range(max(1, n))],
            phrasing_offsets=[0.0 for _ in range(max(1, n))],
            breathiness=0.2,
            brightness=0.5,
            emotion_hint="neutral",
        )


# Convenience factory
def create_voice_pipeline(sample_rate: int = 44100) -> VoiceGenerationPipeline:
    return VoiceGenerationPipeline(sample_rate=sample_rate)

