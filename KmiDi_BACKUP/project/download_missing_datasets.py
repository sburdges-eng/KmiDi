#!/usr/bin/env python3
"""Download missing audio datasets identified in the audit."""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import tarfile

# Target directory for datasets
TARGET_ROOT = Path("/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data")

# Missing datasets to download
DATASETS = {
    "MAESTRO": {
        "url": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "output_dir": TARGET_ROOT / "raw" / "melodies" / "maestro-v3.0.0",
        "priority": "high",
        "description": "Piano MIDI with dynamics and timing (200+ hours)",
    },
    # Note: RAVDESS and TESS require Kaggle API - handled separately
    # CREMA-D can be downloaded via GitHub
    "CREMA-D": {
        "url": "https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip",
        "output_dir": TARGET_ROOT / "raw" / "emotions" / "cremad",
        "priority": "medium",
        "description": "Crowd-sourced Emotional Multimodal Actors Dataset",
    },
    "GTZAN": {
        "url": "https://mirg.city.ac.uk/datasets/gtzan/genres.tar.gz",
        "output_dir": TARGET_ROOT / "raw" / "emotions" / "gtzan",
        "priority": "medium",
        "description": "Music genre classification (10 genres)",
    },
}


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    if output_path.exists():
        print(f"✓ Already exists: {output_path}")
        return True

    print(f"\n{'='*60}")
    print(f"Downloading: {description}")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    try:

        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
            print(
                f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)",
                end="",
                flush=True,
            )

        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        urllib.request.urlretrieve(url, temp_path, reporthook=show_progress)
        temp_path.rename(output_path)
        print(f"\n✓ Download complete: {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract ZIP file."""
    print(f"\nExtracting {zip_path.name} to {extract_to}...")
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extraction complete")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def extract_tar(tar_path: Path, extract_to: Path) -> bool:
    """Extract TAR.GZ file."""
    print(f"\nExtracting {tar_path.name} to {extract_to}...")
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
        print(f"✓ Extraction complete")
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def download_dataset(name: str, config: dict, download_only: bool = False) -> bool:
    """Download and extract a dataset."""
    url = config["url"]
    output_dir = config["output_dir"]
    description = config["description"]

    # Determine file extension
    if url.endswith(".zip"):
        ext = ".zip"
        extract_func = extract_zip
    elif url.endswith(".tar.gz") or url.endswith(".tgz"):
        ext = ".tar.gz"
        extract_func = extract_tar
    else:
        ext = Path(url).suffix
        extract_func = None

    # Download to downloads directory first
    downloads_dir = TARGET_ROOT / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(url).name
    if not filename or filename == Path(url).name:
        filename = f"{name.lower().replace('-', '_')}{ext}"

    download_path = downloads_dir / filename

    # Download
    if not download_file(url, download_path, description):
        return False

    # Extract if not download-only
    if not download_only and extract_func:
        return extract_func(download_path, output_dir)

    return True


def main():
    """Main function."""
    print("=" * 60)
    print("DOWNLOADING MISSING AUDIO DATASETS")
    print("=" * 60)
    print("\nTarget directory:", TARGET_ROOT)
    print("\nThis script will download:")
    print("  - MAESTRO v3.0 (High Priority - Required for melody training)")
    print("  - CREMA-D (Medium Priority - Emotion dataset)")
    print("  - GTZAN (Medium Priority - Genre classification)")
    print("\nNote: RAVDESS and TESS require Kaggle API - see instructions below")
    print("\n" + "=" * 60 + "\n")

    # Check if target directory exists
    if not TARGET_ROOT.exists():
        print(f"⚠ Target directory does not exist: {TARGET_ROOT}")
        response = input("Create it? (y/n): ").lower()
        if response != "y":
            print("Exiting.")
            return
        TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    # Download high priority datasets first
    high_priority = {k: v for k, v in DATASETS.items() if v["priority"] == "high"}
    medium_priority = {k: v for k, v in DATASETS.items() if v["priority"] == "medium"}

    success_count = 0
    fail_count = 0

    # Download high priority
    print("\n" + "=" * 60)
    print("HIGH PRIORITY DATASETS")
    print("=" * 60)
    for name, config in high_priority.items():
        if download_dataset(name, config):
            success_count += 1
        else:
            fail_count += 1

    # Ask before downloading medium priority
    if medium_priority:
        print("\n" + "=" * 60)
        print("MEDIUM PRIORITY DATASETS")
        print("=" * 60)
        response = input("\nDownload medium priority datasets (CREMA-D, GTZAN)? (y/n): ").lower()
        if response == "y":
            for name, config in medium_priority.items():
                if download_dataset(name, config):
                    success_count += 1
                else:
                    fail_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")

    # Instructions for Kaggle datasets
    print("\n" + "=" * 60)
    print("ADDITIONAL DATASETS REQUIRING KAGGLE API")
    print("=" * 60)
    print("\nThe following datasets require Kaggle API setup:")
    print("\n1. RAVDESS:")
    print("   kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio")
    print(
        "   unzip ravdess-emotional-speech-audio.zip -d",
        TARGET_ROOT / "raw" / "emotions" / "ravdess",
    )
    print("\n2. TESS:")
    print("   kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess")
    print(
        "   unzip toronto-emotional-speech-set-tess.zip -d",
        TARGET_ROOT / "raw" / "emotions" / "tess",
    )
    print("\nSetup Kaggle API:")
    print("  1. pip install kaggle")
    print("  2. Get API token from https://www.kaggle.com/settings")
    print("  3. Save to ~/.kaggle/kaggle.json")


if __name__ == "__main__":
    main()
