#!/usr/bin/env python3
"""
Download/cache audio files from a manifest, labeling filenames for clarity.

Input manifest format (jsonl):
{"url": "https://example.com/audio.wav", "label": "happy"}
{"path": "/abs/local/audio.wav", "label": "sad"}

Output:
- Files are cached to the target directory (default: /Volumes/sbdrive/kmidi_audio_cache)
- Filenames include the label: <label>_<idx>_<basename>.<ext>
- A new manifest jsonl is written with updated "path" entries pointing to the cache.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict

import requests


def load_manifest(path: Path) -> List[Dict]:
    entries: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "label" in obj and ("url" in obj or "path" in obj):
                    entries.append(obj)
            except json.JSONDecodeError:
                continue
    if not entries:
        raise ValueError(f"No valid entries found in {path}")
    return entries


def safe_ext(name: str) -> str:
    ext = Path(name).suffix
    return ext if ext else ".wav"


def cache_entry(idx: int, entry: Dict, out_dir: Path, timeout: int = 15) -> Dict:
    label = str(entry["label"]).replace(" ", "_")
    src = entry.get("url") or entry.get("path")
    if not src:
        raise ValueError("Entry missing url/path")

    ext = safe_ext(src)
    stem = Path(src).stem
    target_name = f"{label}_{idx:05d}_{stem}{ext}"
    target_path = out_dir / target_name

    if target_path.exists():
        return {"path": str(target_path), "label": label, "source": src}

    out_dir.mkdir(parents=True, exist_ok=True)

    if entry.get("url"):
        resp = requests.get(src, stream=True, timeout=timeout)
        resp.raise_for_status()
        with target_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        source_path = Path(src)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        shutil.copyfile(source_path, target_path)

    return {"path": str(target_path), "label": label, "source": src}


def main():
    parser = argparse.ArgumentParser(description="Cache/download audio manifest")
    parser.add_argument("--manifest", required=True, help="Input manifest jsonl")
    parser.add_argument(
        "--out-dir",
        default="/Volumes/sbdrive/kmidi_audio_cache",
        help="Cache directory (default: /Volumes/sbdrive/kmidi_audio_cache)",
    )
    parser.add_argument(
        "--output-manifest",
        help="Path for cached manifest (default: <out-dir>/cached_manifest.jsonl)",
    )
    parser.add_argument("--limit", type=int, help="Optional limit on number of items")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_manifest = (
        Path(args.output_manifest)
        if args.output_manifest
        else out_dir / "cached_manifest.jsonl"
    )

    entries = load_manifest(manifest_path)
    if args.limit:
        entries = entries[: args.limit]

    cached: List[Dict] = []
    for idx, entry in enumerate(entries):
        try:
            cached.append(cache_entry(idx, entry, out_dir))
            print(f"[{idx+1}/{len(entries)}] cached -> {cached[-1]['path']}")
        except Exception as exc:  # pragma: no cover
            print(f"[{idx+1}/{len(entries)}] failed: {exc}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w") as f:
        for row in cached:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote cached manifest: {out_manifest}")


if __name__ == "__main__":
    main()
