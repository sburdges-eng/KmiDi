"""
Categorize existing files under Datasets/ by type (audio, midi, chord, rhythm, emotion, multimodal, voice).
Scans --scan-dir and writes manifest.json; optionally copies/moves files into category subdirs.
Run from repo root: PYTHONPATH=. python scripts/categorize_datasets.py [--scan-dir Datasets] [--out Dir] [--dry-run]
"""

import os
import sys
import json
import argparse
import shutil
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.storage_paths import DATASETS_DIR, ensure_storage_dirs

DEFAULT_SCAN = DATASETS_DIR
DEFAULT_OUT = DATASETS_DIR

# Extension -> category (match MUSIC_DATASETS)
EXT_TO_CATEGORY = {
    "wav": "audio", "mp3": "audio", "flac": "audio", "ogg": "audio", "m4a": "audio",
    "mid": "midi", "midi": "midi",
    "csv": "chord", "chord": "chord", "lab": "chord",  # chord/harmony annotations
    "txt": "multimodal",  # tabs, lyrics
    "png": "multimodal", "jpg": "multimodal", "jpeg": "multimodal",
    "mp4": "multimodal", "mov": "multimodal", "avi": "multimodal", "webm": "multimodal",
    "npy": "multimodal",
}
DEFAULT_CAT = "audio"
CATEGORIES = ["audio", "midi", "chord", "rhythm", "emotion", "multimodal", "voice"]


def ensure_category_dirs(base):
    for c in CATEGORIES:
        os.makedirs(os.path.join(base, c), exist_ok=True)


def scan(scan_dir):
    """Return (manifest_dict, list of (abs_path, category, basename))."""
    manifest = defaultdict(list)
    by_ext = defaultdict(int)
    todo = []
    scan_dir = os.path.abspath(scan_dir)
    for root, _dirs, files in os.walk(scan_dir):
        for f in files:
            if f.startswith(".") or f == "manifest.json":
                continue
            ext = f.lower().split(".")[-1] if "." in f else ""
            cat = EXT_TO_CATEGORY.get(ext, DEFAULT_CAT)
            by_ext[ext] += 1
            abs_path = os.path.join(root, f)
            try:
                rel = os.path.relpath(abs_path, scan_dir)
            except ValueError:
                rel = abs_path
            manifest[cat].append(rel)
            todo.append((abs_path, cat, f))
    manifest["by_ext"] = dict(by_ext)
    return dict(manifest), todo


def main():
    ensure_storage_dirs()

    p = argparse.ArgumentParser(description="Categorize datasets by type (audio, midi, chord, etc.)")
    p.add_argument("--scan-dir", default=DEFAULT_SCAN, help="Directory to scan")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output base for category dirs and manifest")
    p.add_argument("--organize", action="store_true", help="Copy files into out/<category>/ (default: only manifest)")
    p.add_argument("--move", action="store_true", help="Move instead of copy (use with --organize)")
    p.add_argument("--dry-run", action="store_true", help="Print actions only")
    p.add_argument("--no-manifest", action="store_true", help="Do not write manifest.json")
    args = p.parse_args()
    ensure_category_dirs(args.out)
    scan_dir = os.path.abspath(args.scan_dir)
    out_dir = os.path.abspath(args.out)
    manifest, todo = scan(scan_dir)
    total = sum(len(v) for k, v in manifest.items() if k != "by_ext" and isinstance(v, list))
    print("Scanned {} files.".format(total))
    for cat in CATEGORIES:
        n = len(manifest.get(cat, []))
        if n:
            print("  {}: {}".format(cat, n))
    if manifest.get("by_ext"):
        print("  by_ext:", manifest["by_ext"])
    if args.organize and not args.dry_run:
        for abs_path, cat, basename in todo:
            if not os.path.isfile(abs_path):
                continue
            dest_dir = os.path.join(out_dir, cat)
            dest_path = os.path.join(dest_dir, basename)
            if os.path.normpath(abs_path) == os.path.normpath(dest_path):
                continue
            if os.path.isfile(dest_path):
                base, ext = os.path.splitext(basename)
                dest_path = os.path.join(dest_dir, base + "_" + os.path.basename(os.path.dirname(abs_path))[:20] + ext)
            os.makedirs(dest_dir, exist_ok=True)
            if args.move:
                shutil.move(abs_path, dest_path)
            else:
                shutil.copy2(abs_path, dest_path)
    elif args.organize and args.dry_run:
        for abs_path, cat, basename in todo:
            print("[dry-run] {} -> {}/{}".format(abs_path, cat, basename))
    if not args.dry_run and not args.no_manifest:
        manifest_path = os.path.join(args.out, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print("Wrote", manifest_path)


if __name__ == "__main__":
    main()
