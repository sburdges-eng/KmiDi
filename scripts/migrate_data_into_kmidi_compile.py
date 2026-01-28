#!/usr/bin/env python3
"""
Migrate data from sibling AUDIO_MIDI_DATA (and optional dirs) into KmiDi-compile
so you can delete everything outside KmiDi-compile and keep scripts working.

Usage:
    python scripts/migrate_data_into_kmidi_compile.py --dry-run
    python scripts/migrate_data_into_kmidi_compile.py --copy
    python scripts/migrate_data_into_kmidi_compile.py --link
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Sibling AUDIO_MIDI_DATA relative to workspace root (parent of KmiDi-compile)
WORKSPACE = ROOT.parent
AUDIO_MIDI_SIBLING = WORKSPACE / "AUDIO_MIDI_DATA"
TARGET_AUDIO = ROOT / "datasets" / "audio"


def main() -> int:
    ap = argparse.ArgumentParser(description="Move or link sibling data into KmiDi-compile")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    ap.add_argument("--copy", action="store_true", help="Copy AUDIO_MIDI_DATA into datasets/audio")
    ap.add_argument("--link", action="store_true", help="Symlink AUDIO_MIDI_DATA into datasets/audio")
    args = ap.parse_args()

    if not args.dry_run and not args.copy and not args.link:
        ap.print_help()
        print("\nProvide one of: --dry-run, --copy, --link")
        return 1

    if args.dry_run:
        print("DRY RUN – no changes made\n")
        print("Would ensure:", TARGET_AUDIO)
        if AUDIO_MIDI_SIBLING.exists():
            for name in ("kelly-audio-data", "SSD_Transfer"):
                src = AUDIO_MIDI_SIBLING / name
                if src.exists():
                    if args.copy:
                        print("  COPY:", src, "->", TARGET_AUDIO)
                    elif args.link:
                        print("  LINK:", src, "->", TARGET_AUDIO / name)
        else:
            print("  (AUDIO_MIDI_DATA sibling not found:", AUDIO_MIDI_SIBLING, ")")
        return 0

    TARGET_AUDIO.mkdir(parents=True, exist_ok=True)

    if args.copy:
        for name in ("kelly-audio-data", "SSD_Transfer"):
            src = AUDIO_MIDI_SIBLING / name
            if not src.exists():
                continue
            dest = TARGET_AUDIO / name
            if dest.exists():
                print("Skip (exists):", dest)
                continue
            print("Copy:", src, "->", dest)
            shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)

    if args.link:
        for name in ("kelly-audio-data", "SSD_Transfer"):
            src = AUDIO_MIDI_SIBLING / name
            if not src.exists():
                continue
            dest = TARGET_AUDIO / name
            if dest.exists():
                print("Skip (exists):", dest)
                continue
            print("Link:", src, "->", dest)
            dest.symlink_to(src)

    print("\nNext: set KELLY_AUDIO_DATA_ROOT and AUDIO_DATA_ROOT to:", TARGET_AUDIO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
