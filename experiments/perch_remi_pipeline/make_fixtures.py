#!/usr/bin/env python
"""Create WAV and real MIDI fixtures for pipeline tests. Run from repo root or this dir."""
from __future__ import annotations

import struct
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Small real piano MIDI from MidiTok tests (REMI-tokenizable)
REAL_MIDI_URL = (
    "https://raw.githubusercontent.com/Natooz/MidiTok/main/"
    "tests/MIDIs_one_track/6338816_Etude%20No.%204.mid"
)


def write_minimal_wav(path: Path, duration_s: float = 1.0, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)
    waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), waveform, sr)


def write_minimal_midi(path: Path) -> None:
    """Write a minimal valid SMF (one track, one note then end-of-track)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b"MThd" + struct.pack(">IHH", 6, 0, 1) + struct.pack(">H", 480)
    track_events = (
        b"\x00\x90\x3C\x40"
        b"\x82\x03\xC0\x80\x3C\x00"
        b"\x00\xFF\x2F\x00"
    )
    track = b"MTrk" + struct.pack(">I", len(track_events)) + track_events
    path.write_bytes(header + track)


def fetch_real_midi(path: Path) -> bool:
    """Download a small real piano MIDI (MidiTok test file) for tokenizer tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(REAL_MIDI_URL, headers={"User-Agent": "KmiDi-fixtures/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            path.write_bytes(resp.read())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[make_fixtures] Could not fetch real MIDI: {e}")
        return False


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    write_minimal_wav(FIXTURES_DIR / "sample.wav")
    # Use real piano MIDI for symbolic pipeline (tokenizer produces non-empty tokens)
    real_mid = FIXTURES_DIR / "sample.mid"
    if fetch_real_midi(real_mid):
        print(f"[make_fixtures] Wrote real MIDI to {real_mid}")
    else:
        write_minimal_midi(real_mid)
        print(f"[make_fixtures] Wrote minimal MIDI to {real_mid} (tokenize may yield 0 tokens)")
    print(f"[make_fixtures] Wrote {FIXTURES_DIR / 'sample.wav'}")


if __name__ == "__main__":
    main()
