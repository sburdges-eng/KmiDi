#!/usr/bin/env python3
"""
MAESTRO v3.0.0: download metadata CSV and/or build aligned.jsonl from unpacked MAESTRO.

Usage:
  python scripts/acquire_maestro.py --csv-only
      Downloads maestro-v3.0.0.csv to ~/Datasets/kmidi_jepa/maestro/
  python scripts/acquire_maestro.py --build-manifest
      Builds ~/Datasets/kmidi_jepa/manifests/aligned.jsonl from unpacked MAESTRO.
  python scripts/acquire_maestro.py --download [--throttle-mbps N] [--no-resume]
      Downloads full zip (~101 GB) with progress, optional rate limit, resume support.
  python scripts/acquire_maestro.py --unzip
      Unzips maestro-v3.0.0.zip in maestro/ with progress.
  python scripts/acquire_maestro.py --analyze
      Reports stats from CSV/manifest: rows, duration, split, disk usage, optional validation.
"""

import argparse
import csv
import json
import os
import sys
import time
import zipfile
from pathlib import Path

try:
    import urllib.request
except ImportError:
    urllib = None

MAESTRO_BASE = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0"
CSV_URL = f"{MAESTRO_BASE}/maestro-v3.0.0.csv"
ZIP_URL = f"{MAESTRO_BASE}/maestro-v3.0.0.zip"
ZIP_NAME = "maestro-v3.0.0.zip"


def get_base_dir() -> Path:
    home = Path(os.environ.get("HOME", os.path.expanduser("~")))
    return home / "Datasets" / "kmidi_jepa"


def download_csv_only(base: Path) -> None:
    maestro_dir = base / "maestro"
    maestro_dir.mkdir(parents=True, exist_ok=True)
    csv_path = maestro_dir / "maestro-v3.0.0.csv"
    print(f"Downloading {CSV_URL} -> {csv_path}")
    urllib.request.urlretrieve(CSV_URL, csv_path)
    print(f"Saved {csv_path}")


def download_full_zip(
    base: Path,
    throttle_mbps: float | None = None,
    resume: bool = True,
) -> Path:
    """Download full MAESTRO zip with progress, optional throttle, resume. Returns path to zip."""
    maestro_dir = base / "maestro"
    maestro_dir.mkdir(parents=True, exist_ok=True)
    zip_path = maestro_dir / ZIP_NAME
    start_byte = zip_path.stat().st_size if (resume and zip_path.exists()) else 0
    mode = "ab" if start_byte else "wb"
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "KmiDi-acquire/1.0"})
    if start_byte:
        req.add_header("Range", f"bytes={start_byte}-")
    try:
        resp = urllib.request.urlopen(req)
        cl = resp.headers.get("Content-Length")
        total = start_byte + (int(cl) if cl else 0)
        if start_byte and not cl:
            total = 108_000_000_000  # ~101 GB fallback for ETA
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)
    if total and start_byte >= total:
        print(f"Already complete: {zip_path}")
        return zip_path
    chunk = 2 * 1024 * 1024  # 2 MB
    throttle_sleep = 0.0
    if throttle_mbps and throttle_mbps > 0:
        throttle_sleep = (chunk / (1024 * 1024)) / throttle_mbps
    done = start_byte
    start_time = time.monotonic()
    last_print = start_time
    with open(zip_path, mode) as f:
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)
            done += len(data)
            if throttle_sleep:
                time.sleep(throttle_sleep)
            now = time.monotonic()
            if now - last_print >= 5.0 and total:
                elapsed = now - start_time
                speed = (done - start_byte) / (1024 * 1024) / elapsed if elapsed > 0 else 0
                eta = (total - done) / (1024 * 1024) / speed if speed > 0 else 0
                print(f"  {done / (1024**3):.2f} / {total / (1024**3):.2f} GB  {speed:.2f} MB/s  ETA {eta / 60:.0f} min")
                last_print = now
    print(f"Saved {zip_path} ({done / (1024**3):.2f} GB)")
    return zip_path


def unzip_with_progress(base: Path) -> None:
    """Unzip maestro-v3.0.0.zip in maestro/ with progress (file count)."""
    maestro_dir = base / "maestro"
    zip_path = maestro_dir / ZIP_NAME
    if not zip_path.exists():
        print(f"Missing {zip_path}. Run --download first.", file=sys.stderr)
        sys.exit(1)
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        total = len(names)
        for i, name in enumerate(names):
            z.extract(name, maestro_dir)
            if (i + 1) % 500 == 0 or i + 1 == total:
                print(f"  Extracted {i + 1}/{total} files")
    print(f"Unpacked to {maestro_dir}")


def analyze(base: Path, validate_samples: int = 5) -> None:
    """Report stats from CSV/manifest: rows, duration, split, disk usage; optionally validate paths."""
    maestro_dir = base / "maestro"
    csv_path = maestro_dir / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        csv_path = maestro_dir / "maestro-v3.0.0" / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        print(f"Missing CSV. Run --csv-only or unzip full MAESTRO first.", file=sys.stderr)
        sys.exit(1)
    by_split: dict[str, list[dict]] = {}
    total_duration = 0.0
    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            split_name = r.get("split", "unknown")
            by_split.setdefault(split_name, []).append(r)
            try:
                total_duration += float(r.get("duration", 0))
            except (TypeError, ValueError):
                pass
    print("--- MAESTRO analysis ---")
    print(f"Total rows (CSV): {len(rows)}")
    print(f"Total duration: {total_duration / 3600:.2f} hours")
    for k, v in sorted(by_split.items(), key=lambda x: -len(x[1])):
        print(f"  split={k}: {len(v)} rows")
    # Disk usage of maestro dir
    try:
        total_bytes = sum(
            p.stat().st_size for p in maestro_dir.rglob("*") if p.is_file()
        )
        print(f"Disk usage (maestro/): {total_bytes / (1024**3):.2f} GB")
    except OSError:
        print("Disk usage: (unable to compute)")
    # Validate first N paths exist
    content_root = maestro_dir if (maestro_dir / "2004").exists() or (maestro_dir / "2018").exists() else maestro_dir / "maestro-v3.0.0"
    if validate_samples > 0 and rows:
        missing = 0
        for r in rows[:validate_samples]:
            midi = content_root / r.get("midi_filename", "").strip()
            audio = content_root / r.get("audio_filename", "").strip()
            if not midi.exists():
                missing += 1
            if not audio.exists():
                missing += 1
        if missing:
            print(f"Sample validation: {missing} of {validate_samples * 2} paths missing (first {validate_samples} rows)")
        else:
            print(f"Sample validation: first {validate_samples} rows — all paths present")
    print("---")


def build_manifest(base: Path, split: str | None = None, limit: int | None = None) -> None:
    """
    Build aligned.jsonl from unpacked MAESTRO.
    CSV has: canonical_composer, canonical_title, split, year, midi_filename, audio_filename, duration.
    Files live under maestro/ (or maestro/maestro-v3.0.0/ if zip had top-level folder).
    """
    maestro_dir = base / "maestro"
    csv_path = maestro_dir / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        csv_path = maestro_dir / "maestro-v3.0.0" / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        print(f"Missing CSV. Run --csv-only or unzip full MAESTRO first.", file=sys.stderr)
        sys.exit(1)
    content_root = maestro_dir if (maestro_dir / "2004").exists() or (maestro_dir / "2018").exists() else maestro_dir / "maestro-v3.0.0"

    manifest_dir = base / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "aligned.jsonl"

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if split and r.get("split") != split:
                continue
            midi_filename = r.get("midi_filename", "").strip()
            audio_filename = r.get("audio_filename", "").strip()
            if not midi_filename or not audio_filename:
                continue
            midi_abs = content_root / midi_filename
            audio_abs = content_root / audio_filename
            if not midi_abs.exists() or not audio_abs.exists():
                continue
            rows.append({
                "audio_path": str(audio_abs.resolve()),
                "midi_path": str(midi_abs.resolve()),
                "specto_path": "",
                "start_offset": 0.0,
                "tempo": 120,
                "timebase": 480,
            })
            if limit and len(rows) >= limit:
                break

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} rows to {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="MAESTRO: download CSV/zip, unzip, build manifest, analyze")
    ap.add_argument("--csv-only", action="store_true", help="Download CSV only")
    ap.add_argument("--download", action="store_true", help="Download full zip (~101 GB) with progress, optional throttle/resume")
    ap.add_argument("--throttle-mbps", type=float, default=None, metavar="N", help="Cap download at N MB/s (e.g. 10)")
    ap.add_argument("--no-resume", action="store_true", help="Do not resume; overwrite partial zip")
    ap.add_argument("--unzip", action="store_true", help="Unzip maestro-v3.0.0.zip in maestro/ with progress")
    ap.add_argument("--build-manifest", action="store_true", help="Build aligned.jsonl from unpacked MAESTRO")
    ap.add_argument("--analyze", action="store_true", help="Report stats from CSV: rows, duration, split, disk usage, validation")
    ap.add_argument("--analyze-samples", type=int, default=5, metavar="N", help="Validate first N rows exist (default 5)")
    ap.add_argument("--split", type=str, default=None, help="Filter by split: train, validation, test (for --build-manifest)")
    ap.add_argument("--limit", type=int, default=None, help="Max rows to write (for --build-manifest)")
    args = ap.parse_args()

    base = get_base_dir()
    base.mkdir(parents=True, exist_ok=True)

    if args.csv_only:
        download_csv_only(base)
    if args.download:
        download_full_zip(base, throttle_mbps=args.throttle_mbps, resume=not args.no_resume)
    if args.unzip:
        unzip_with_progress(base)
    if args.build_manifest:
        build_manifest(base, split=args.split, limit=args.limit)
    if args.analyze:
        analyze(base, validate_samples=args.analyze_samples)
    if not any([args.csv_only, args.download, args.unzip, args.build_manifest, args.analyze]):
        ap.print_help()
        print("\nOption B (full MAESTRO): --csv-only then --download [--throttle-mbps 10] then --unzip then --build-manifest then --analyze.")


if __name__ == "__main__":
    main()
