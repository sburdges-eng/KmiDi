"""
Utility to extract OpenL3 audio embeddings (optional dependency).

Usage:
    from training.audio_embeddings import extract_openl3
    emb = extract_openl3("audio.wav")
"""

from pathlib import Path
from typing import Optional

import numpy as np


def extract_openl3(
    audio_path: str,
    sr: int = 48000,
    content_type: str = "music",
    embedding_size: int = 512,
) -> Optional[np.ndarray]:
    """
    Return an OpenL3 embedding or None if openl3 is unavailable.
    """
    try:
        import openl3  # type: ignore
        import soundfile as sf
    except ImportError:
        return None

    path = Path(audio_path)
    if not path.exists():
        return None

    audio, file_sr = sf.read(str(path))
    if file_sr != sr:
        try:
            import librosa

            audio = librosa.resample(audio.astype(float), orig_sr=file_sr, target_sr=sr)
            file_sr = sr
        except Exception:
            return None

    try:
        emb, _ = openl3.get_audio_embedding(
            audio,
            file_sr,
            content_type=content_type,
            embedding_size=embedding_size,
        )
        # Pool across time dimension
        if emb is None or len(emb) == 0:
            return None
        return emb.mean(axis=0)
    except Exception:
        return None


__all__ = ["extract_openl3"]
