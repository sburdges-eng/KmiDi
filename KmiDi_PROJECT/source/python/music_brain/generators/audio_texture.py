"""
AudioTextureJob (Experimental)
------------------------------
Optional offline/batch audio texture generation. Primary audio path remains
MIDI → DAW; this is decorative only.

Current implementation is a stub placeholder to keep orchestration explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AudioTextureConfig:
    output_dir: Path = Path.home() / "Music" / "iDAW_Output" / "audio_textures"
    seed: Optional[int] = 123


class AudioTextureJob:
    def __init__(self, config: Optional[AudioTextureConfig] = None):
        self.config = config or AudioTextureConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_texture(self, prompt: str, duration_seconds: float = 10.0) -> Path:
        """
        Experimental placeholder: writes an empty file to mark a completed job.
        Real diffusion should be plugged in here when available.
        """
        safe_name = prompt.replace(" ", "_")[:48] or "texture"
        out_path = self.config.output_dir / f"{safe_name}.wav"
        out_path.write_bytes(b"")  # No audio synthesis
        return out_path


def create_audio_texture_job(config: Optional[AudioTextureConfig] = None) -> AudioTextureJob:
    return AudioTextureJob(config=config)

