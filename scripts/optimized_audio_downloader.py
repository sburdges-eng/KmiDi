#!/usr/bin/env python3
"""
Optimized Audio Downloader

Thin orchestration layer around
penta_core.ml.datasets.AudioDownloader with sane,
performance-minded defaults for current hardware profiles.

Key defaults:
- Uses storage config from penta_core (external SSD if configured), otherwise falls back to ~/.kelly/audio/{downloads,raw}.
- Parallelizes multiple URL downloads with a small thread pool to keep disks busy without thrashing.
- Supports direct URLs, Freesound packs, and Hugging Face datasets.

Examples:
    # Direct URLs (parallel)
    python3 scripts/optimized_audio_downloader.py --urls https://example.com/a.wav https://example.com/b.wav

    # URLs from file plus custom subdir and no extraction
    python3 scripts/optimized_audio_downloader.py \
        --urls-file urls.txt --output-subdir foley_raw --no-extract

    # Force root storage to an external drive (e.g., /Volumes/sbdrive/audio)
    python3 scripts/optimized_audio_downloader.py --root /Volumes/sbdrive/audio --urls-file urls.txt

    # Freesound pack (requires FREESOUND_API_KEY)
    python3 scripts/optimized_audio_downloader.py --freesound-pack 12345 --output-subdir freesound_emotion

    # Hugging Face dataset
    python3 scripts/optimized_audio_downloader.py --hf speech_commands --subset v0.02 --split train
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from penta_core.ml.datasets import (
        AudioDownloader,
        ensure_audio_directories,
    )
except ImportError as exc:  # pragma: no cover - dependency guard for standalone use
    sys.stderr.write(
        "penta_core not installed or not on PYTHONPATH. "
        "Ensure project dependencies are installed before running this script.\n"
    )
    raise


def read_urls(path: Path) -> List[str]:
    """Load URLs from a text file, ignoring blanks/comments."""
    lines = path.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def download_urls(
    downloader: AudioDownloader,
    urls: Iterable[str],
    extract: bool,
    output_subdir: Optional[str],
    max_workers: int,
) -> List[dict]:
    """Download multiple URLs in parallel and return result dicts."""
    results = []
    urls_list = list(urls)
    if not urls_list:
        return results

    def _job(u: str):
        target_dir = (
            downloader.output_dir / output_subdir if output_subdir else None
        )
        res = downloader.download_url(
            u,
            extract=extract,
            extract_to=target_dir,
        )
        return {
            "url": u,
            "success": res.success,
            "path": str(res.path) if res.path else None,
            "files": res.files_count,
            "size_mb": res.total_size_mb,
            "error": res.error,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for out in pool.map(_job, urls_list):
            results.append(out)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized audio downloader")
    src = parser.add_argument_group("sources")
    src.add_argument(
        "--urls", nargs="+", help="One or more direct URLs to download"
    )
    src.add_argument(
        "--urls-file",
        type=Path,
        help="File containing URLs (one per line, # for comments)",
    )
    src.add_argument(
        "--freesound-pack",
        type=str,
        help="Freesound pack ID to download (requires FREESOUND_API_KEY)",
    )
    src.add_argument(
        "--hf",
        dest="hf_dataset",
        type=str,
        help="Hugging Face dataset name (e.g., speech_commands)",
    )
    src.add_argument(
        "--subset", type=str, help="Hugging Face subset/config name"
    )
    src.add_argument(
        "--split",
        type=str,
        default="train",
        help="Hugging Face split (train/test/validation)",
    )

    parser.add_argument(
        "--root",
        type=Path,
        help="Override storage root (downloads/raw/cache under this path)",
    )
    parser.add_argument(
        "--output-subdir", type=str, help="Optional subdirectory under output_dir"
    )
    parser.add_argument("--download-dir", type=Path, help="Override download dir")
    parser.add_argument("--output-dir", type=Path, help="Override output dir")
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip archive extraction for URL downloads",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers for URL downloads",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved storage config and exit",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Establish storage config (prefers explicit --root, otherwise penta_core defaults).
    if args.root:
        root = args.root.expanduser().resolve()
        (root / "downloads").mkdir(parents=True, exist_ok=True)
        (root / "raw").mkdir(parents=True, exist_ok=True)
        (root / "cache").mkdir(parents=True, exist_ok=True)
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        dirs = {
            "root": root,
            "downloads": root / "downloads",
            "raw": root / "raw",
            "cache": root / "cache",
            "manifests": root / "manifests",
        }
    else:
        dirs = ensure_audio_directories()

    download_dir = args.download_dir or dirs["downloads"]
    output_dir = args.output_dir or dirs["raw"]

    downloader = AudioDownloader(
        download_dir=download_dir,
        output_dir=output_dir,
        freesound_api_key=None,  # Uses FREESOUND_API_KEY env by default
    )

    if args.print_config:
        config = {
            "download_dir": str(download_dir),
            "output_dir": str(output_dir),
            "cache_dir": str(dirs.get("cache", "")),
            "manifests_dir": str(dirs.get("manifests", "")),
        }
        print(json.dumps(config, indent=2))
        return 0

    urls: List[str] = []
    if args.urls:
        urls.extend(args.urls)
    if args.urls_file:
        urls.extend(read_urls(args.urls_file))

    work_performed = False
    summary = {}

    if urls:
        work_performed = True
        url_results = download_urls(
            downloader=downloader,
            urls=urls,
            extract=not args.no_extract,
            output_subdir=args.output_subdir,
            max_workers=max(1, args.max_workers),
        )
        summary["urls"] = url_results

    if args.freesound_pack:
        work_performed = True
        res = downloader.download_freesound_pack(
            args.freesound_pack,
            output_subdir=args.output_subdir,
        )
        summary["freesound"] = {
            "success": res.success,
            "path": str(res.path) if res.path else None,
            "files": res.files_count,
            "size_mb": res.total_size_mb,
            "error": res.error,
        }

    if args.hf_dataset:
        work_performed = True
        res = downloader.download_huggingface_dataset(
            dataset_name=args.hf_dataset,
            subset=args.subset,
            split=args.split,
            output_subdir=args.output_subdir,
        )
        summary["huggingface"] = {
            "success": res.success,
            "path": str(res.path) if res.path else None,
            "files": res.files_count,
            "size_mb": res.total_size_mb,
            "error": res.error,
        }

    if not work_performed:
        print("No work performed. Provide --urls/--urls-file, --freesound-pack, or --hf.", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
