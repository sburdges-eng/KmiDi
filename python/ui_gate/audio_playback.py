"""
Minimal audio playback — stdlib only, no pip deps.
Plays a short tone when user clicks Play. Platform: macOS afplay, Windows winsound, Linux aplay.

LISTENING_GUARDRAIL:
This audio is a probe, not musical output.
It must never be derived from intent or value states.
Fixed tones only. If pitch is tied to harmony here, CI should scream.
"""

from __future__ import annotations

import math

# Non-musical probes. Fixed Hz. Not derived from intent, value states, or harmony.
PROBE_TONE_A = 440.0
PROBE_TONE_B = 554.37
PROBE_TONE_C = 659.25
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


def _generate_tone_hz(freq: float = 440.0, duration_ms: int = 300, sample_rate: int = 44100) -> bytes:
    """Generate mono 16-bit WAV sine tone."""
    n_samples = int(sample_rate * duration_ms / 1000)
    max_val = 32767 * 0.3  # avoid clipping
    frames = []
    for i in range(n_samples):
        t = i / sample_rate
        val = max_val * math.sin(2 * math.pi * freq * t)
        frames.append(struct.pack("<h", int(val)))
    return b"".join(frames)


def _write_wav(data: bytes, path: Path, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(data)


def play_tone(freq: float = PROBE_TONE_A, duration_ms: int = 300) -> bool:
    """
    Play a probe tone. Returns True if playback succeeded.
    Uses: macOS afplay, Windows winsound, Linux aplay (if available).
    Probe only — not musical output. Use PROBE_TONE_* constants.
    """
    sample_rate = 44100
    data = _generate_tone_hz(freq, duration_ms, sample_rate)

    try:
        if sys.platform == "darwin":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = Path(f.name)
            try:
                _write_wav(data, path, sample_rate)
                subprocess.run(["afplay", str(path)], check=True, capture_output=True, timeout=2)
            finally:
                path.unlink(missing_ok=True)
            return True
        elif sys.platform == "win32":
            import winsound
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = Path(f.name)
            try:
                _write_wav(data, path, sample_rate)
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            finally:
                path.unlink(missing_ok=True)
            return True
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = Path(f.name)
            try:
                _write_wav(data, path, sample_rate)
                subprocess.run(["aplay", "-q", str(path)], check=True, capture_output=True, timeout=2)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
            finally:
                path.unlink(missing_ok=True)
            return True
    except Exception:
        return False
