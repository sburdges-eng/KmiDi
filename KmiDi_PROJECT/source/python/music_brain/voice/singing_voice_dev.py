"""
SingingVoiceDev - Copy of SingingVoice with an embedded development prompt.

Use this inside a dev container to keep improving the singing stack while
leaving behavior identical to SingingVoice. The prompt below captures
improvement/learning tasks; no runtime behavior changes are introduced.
"""

from typing import Optional

from music_brain.voice.singing_voice import SingingVoice

# Development prompt to guide iterative improvements when left running in a dev container.
DEV_PROMPT = """
You are maintaining SingingVoice (formant preview + neural production).
Keep behavior compatible with the existing API. Priorities:
1) Integrate a real neural backend (DiffSinger or similar):
   - Install dependencies.
   - Wire model loading in NeuralBackend._load_model.
   - Implement NeuralBackend.synthesize with acoustic feature generation + vocoder.
   - Add simple caching for repeated phrases.
2) Improve formant quality:
   - Better consonant noise and coarticulation.
   - Per-phoneme envelopes; refine aspiration/breathiness.
3) Phoneme/duration:
   - Prefer g2p_en; add custom dictionary hook.
   - Add optional neural duration predictor; keep rule-based fallback.
4) Expression and pitch:
   - Support per-note velocity → amplitude.
   - Add per-note vibrato depth/rate; refine portamento curve.
5) Learning hooks:
   - Use VoiceLearningManager to store user samples and learned profiles.
   - Allow selecting learned voice profiles in SingingVoiceDev.
6) Evaluation:
   - Add pytest/regression tests for phoneme alignment, pitch curves, and formant output sanity.
   - Add a quality checklist (noise floor, clipping, latency).
7) Examples/docs:
   - Update examples to show formant vs neural outputs and learned voices.
Constraints: Do not break existing public APIs. Keep formant path stable.
"""


class SingingVoiceDev(SingingVoice):
    """
    Development-focused copy of SingingVoice. Same runtime behavior; includes
    DEV_PROMPT for iterative improvements inside dev containers.
    """

    prompt: str = DEV_PROMPT

    def __init__(
        self,
        backend: str = "auto",
        voice_model: Optional[str] = None,
        device: str = "auto",
        formant_config=None,
        sample_rate: int = 44100,
    ):
        super().__init__(
            backend=backend,
            voice_model=voice_model,
            device=device,
            formant_config=formant_config,
            sample_rate=sample_rate,
        )


def create_singing_voice_dev(
    backend: str = "auto",
    voice_model: Optional[str] = None,
    device: str = "auto",
    formant_config=None,
    sample_rate: int = 44100,
) -> SingingVoiceDev:
    """Factory helper mirroring create_singing_voice."""
    return SingingVoiceDev(
        backend=backend,
        voice_model=voice_model,
        device=device,
        formant_config=formant_config,
        sample_rate=sample_rate,
    )
