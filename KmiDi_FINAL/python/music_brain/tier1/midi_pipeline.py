"""
MidiGenerationPipeline
----------------------
Deterministic MIDI generation path using existing Tier1MIDIGenerator components:
- HarmonyPredictor
- MelodyTransformer
- GroovePredictor
- DynamicsEngine (optional)

Inputs:
  - CompleteSongIntent (from BrainController)

Outputs:
  - MIDI content (melody/harmony/groove) as arrays and optional MIDI file
  - Metadata: tempo, key, density, groove profile

Constraints:
  - Local execution only
  - Deterministic (explicit seed)
  - No diffusion / hallucinated audio
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

from music_brain.session.intent_schema import CompleteSongIntent
from music_brain.tier1.midi_generator import Tier1MIDIGenerator


@dataclass
class MidiGenerationResult:
    melody: np.ndarray
    harmony: Dict[int, list]
    groove: Dict[str, Any]
    tempo_bpm: int
    key: str
    mode: str
    metadata: Dict[str, Any]
    midi_path: Optional[Path] = None


class MidiGenerationPipeline:
    """
    High-level deterministic wrapper around Tier1MIDIGenerator.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        device: str = "auto",
        seed: Optional[int] = 17,
    ):
        self._seed = seed
        self._gen = Tier1MIDIGenerator(
            device=device,
            checkpoint_dir=checkpoint_dir,
            verbose=False,
        )

    def generate(self, intent: CompleteSongIntent, length: int = 32) -> MidiGenerationResult:
        """
        Run the deterministic MIDI pipeline from intent.
        """
        tempo_bpm = self._choose_tempo(intent)
        key, mode = self._choose_key_mode(intent)

        # We currently ignore key/mode in Tier1MIDIGenerator stubs, but keep metadata explicit
        emotion_embedding = self._emotion_to_embedding(intent)

        melody = self._gen.generate_melody(
            emotion_embedding=emotion_embedding,
            length=length,
            temperature=0.8,  # modest creativity but bounded
            nucleus_p=0.9,
            seed=self._seed,
        )
        harmony = self._gen.generate_harmony(
            melody_notes=melody,
            emotion_embedding=emotion_embedding,
        )
        groove = self._gen.generate_groove(
            emotion_embedding=emotion_embedding,
            base_tempo_bpm=tempo_bpm,
        )

        metadata = {
            "model": "Tier1",
            "tempo_bpm": tempo_bpm,
            "key": key,
            "mode": mode,
            "seed": self._seed,
            "length": length,
        }

        return MidiGenerationResult(
            melody=melody,
            harmony=harmony,
            groove=groove,
            tempo_bpm=tempo_bpm,
            key=key,
            mode=mode,
            metadata=metadata,
        )

    def export_midi(
        self,
        result: MidiGenerationResult,
        output_path: Path,
    ) -> Path:
        """
        Convert melody + groove to a MIDI file using Tier1 helper.
        """
        self._gen.melody_to_midi_file(
            notes=result.melody,
            groove_params=result.groove,
            output_path=str(output_path),
            tempo_bpm=result.tempo_bpm,
        )
        result.midi_path = output_path
        return output_path

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _emotion_to_embedding(self, intent: CompleteSongIntent) -> np.ndarray:
        # Tier1 stubs expect a 64-dim vector. Use deterministic seed.
        rng = np.random.default_rng(self._seed or 0)
        return rng.standard_normal(64).astype(np.float32)

    def _choose_tempo(self, intent: CompleteSongIntent) -> int:
        lo, hi = intent.technical_constraints.technical_tempo_range
        try:
            lo_i, hi_i = int(lo), int(hi)
        except Exception:
            lo_i, hi_i = 80, 120
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        # deterministic midpoint
        return int((lo_i + hi_i) / 2)

    def _choose_key_mode(self, intent: CompleteSongIntent) -> tuple[str, str]:
        key = intent.technical_constraints.technical_key or "C"
        mode = intent.technical_constraints.technical_mode or "major"
        return key, mode


# Convenience
def create_midi_pipeline(
    checkpoint_dir: Optional[str] = None, device: str = "auto", seed: Optional[int] = 17
) -> MidiGenerationPipeline:
    return MidiGenerationPipeline(
        checkpoint_dir=checkpoint_dir,
        device=device,
        seed=seed,
    )

