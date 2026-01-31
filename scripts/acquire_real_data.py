#!/usr/bin/env python3
"""
Acquire real JEPA data when none exists locally.

1. Downloads MAESTRO v3.0.0 midi-only zip (~56 MB) to ~/Datasets/kmidi_jepa/maestro_midi/
2. Unpacks MIDI files
3. Optionally renders MIDI -> WAV with fluidsynth (requires fluidsynth + SF2 soundfont)
4. Writes ~/Datasets/kmidi_jepa/manifests/aligned.jsonl

Usage:
  python scripts/acquire_real_data.py
      Full run: download, unpack, render (if fluidsynth available), build manifest.
  python scripts/acquire_real_data.py --no-render
      Skip MIDI->WAV render; manifest will have empty audio_path (or midi path only).
  python scripts/acquire_real_data.py --limit 50
      Only process first 50 MIDI files (for quick test).

Requires: fluidsynth on PATH and FLUIDSYNTH_SF2 set to a .sf2 path (or script skips render).
Install: brew install fluidsynth. Soundfont: e.g. FluidR3_GM.sf2 (search online or use system path).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import urllib.request

MAESTRO_MIDI_ZIP = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"


def get_base_dir() -> Path:
    home = Path(os.environ.get("HOME", os.path.expanduser("~")))
    return home / "Datasets" / "kmidi_jepa"


def download_midi_zip(base: Path) -> Path:
    midi_dir = base / "maestro_midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    zip_path = midi_dir / "maestro-v3.0.0-midi.zip"
    if zip_path.exists():
        print(f"Already exists: {zip_path}")
        return zip_path
    print(f"Downloading {MAESTRO_MIDI_ZIP} -> {zip_path}")
    urllib.request.urlretrieve(MAESTRO_MIDI_ZIP, zip_path)
    print(f"Saved {zip_path}")
    return zip_path


def unpack_zip(zip_path: Path) -> Path:
    extract_dir = zip_path.parent
    if any(extract_dir.rglob("*.mid")) or any(extract_dir.rglob("*.midi")):
        print(f"Already unpacked under {extract_dir}")
        return extract_dir
    print(f"Unpacking {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    return extract_dir


def find_midi_files(root: Path, limit: int | None) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*.midi")) + sorted(root.rglob("*.mid")):
        if p.is_file():
            out.append(p)
            if limit and len(out) >= limit:
                break
    return out


def render_midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    soundfont: str,
) -> bool:
    try:
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                soundfont,
                str(midi_path),
                "-F",
                str(wav_path),
                "-r",
                "44100",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return wav_path.exists()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run(
    base: Path,
    no_render: bool = False,
    limit: int | None = None,
) -> None:
    base.mkdir(parents=True, exist_ok=True)
    zip_path = download_midi_zip(base)
    root = unpack_zip(zip_path)
    midi_files = find_midi_files(root, limit)
    if not midi_files:
        print("No MIDI files found under", root, file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(midi_files)} MIDI files")

    soundfont = os.environ.get("FLUIDSYNTH_SF2", "")
    do_render = not no_render and shutil.which("fluidsynth") and soundfont and Path(soundfont).exists()
    if no_render:
        print("Skipping MIDI->WAV render (--no-render)")
    elif not do_render:
        print("Skipping render: set FLUIDSYNTH_SF2 to a .sf2 path and have fluidsynth on PATH for WAV output.")

    audio_dir = base / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = base / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "aligned.jsonl"

    rows: list[dict] = []
    for i, midi_path in enumerate(midi_files):
        rel = midi_path.relative_to(root)
        wav_name = rel.with_suffix(".wav").name
        wav_path = audio_dir / wav_name
        audio_path_str = ""
        if do_render:
            if not wav_path.exists():
                if render_midi_to_wav(midi_path, wav_path, soundfont):
                    audio_path_str = str(wav_path.resolve())
                else:
                    wav_path = None
            else:
                audio_path_str = str(wav_path.resolve())
        rows.append({
            "audio_path": audio_path_str,
            "midi_path": str(midi_path.resolve()),
            "specto_path": "",
            "start_offset": 0.0,
            "tempo": 120,
            "timebase": 480,
        })
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(midi_files)}")

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    with_audio = sum(1 for r in rows if r.get("audio_path"))
    print(f"Wrote {len(rows)} rows to {manifest_path} (with audio: {with_audio})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Acquire real JEPA data (MAESTRO midi + optional WAV)")
    ap.add_argument("--no-render", action="store_true", help="Skip MIDI->WAV render")
    ap.add_argument("--limit", type=int, default=None, help="Max MIDI files to process")
    args = ap.parse_args()
    base = get_base_dir()
    run(base, no_render=args.no_render, limit=args.limit)


if __name__ == "__main__":
    main()
