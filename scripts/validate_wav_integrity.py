#!/usr/bin/env python3
"""
Validate WAV files under a directory: read with soundfile, check duration > 0 and no NaN/Inf.
Exit 0 if all OK, 1 if any corrupt or unreadable.
Usage: python scripts/validate_wav_integrity.py --root /path/to/maestro-v3.0.0 [--pattern "**/*.wav"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    print("soundfile and numpy required: pip install soundfile numpy", file=sys.stderr)
    sys.exit(2)


def validate_one(path: Path) -> tuple[bool, str]:
    """Return (ok, message)."""
    try:
        info = sf.info(str(path))
        if info.frames <= 0:
            return False, "zero frames"
        if info.samplerate <= 0:
            return False, "invalid samplerate"
        data, sr = sf.read(str(path), always_2d=False)
        if data.size == 0:
            return False, "empty array"
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False, "NaN/Inf in samples"
        return True, f"ok ({info.frames} frames, {info.samplerate} Hz)"
    except Exception as e:
        return False, str(e)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate WAV integrity under a directory.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory to scan")
    parser.add_argument("--pattern", default="**/*.wav", help="Glob for audio files")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print failures")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1
    paths = sorted(root.glob(args.pattern))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        print("No WAV files found.", file=sys.stderr)
        return 1
    failed: list[tuple[Path, str]] = []
    for p in paths:
        ok, msg = validate_one(p)
        if not ok:
            failed.append((p, msg))
        elif not args.quiet:
            print(f"OK {p.relative_to(root)}: {msg}")
    if failed:
        for p, msg in failed:
            print(f"FAIL {p.relative_to(root)}: {msg}", file=sys.stderr)
        print(f"\n{len(failed)}/{len(paths)} files failed validation.", file=sys.stderr)
        return 1
    print(f"All {len(paths)} WAVs passed validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
