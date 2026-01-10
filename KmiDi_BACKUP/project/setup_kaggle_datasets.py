#!/usr/bin/env python3
"""Setup and download Kaggle datasets (RAVDESS and TESS)."""

import os
import sys
import subprocess
from pathlib import Path

TARGET_ROOT = Path("/Users/seanburdges/RECOVERY_OPS/AUDIO_MIDI_DATA/kelly-audio-data")

KAGGLE_DATASETS = {
    "RAVDESS": {
        "dataset": "uwrfkaggler/ravdess-emotional-speech-audio",
        "output_dir": TARGET_ROOT / "raw" / "emotions" / "ravdess",
        "priority": "high",
        "description": "Ryerson Audio-Visual Database of Emotional Speech and Song"
    },
    "TESS": {
        "dataset": "ejlok1/toronto-emotional-speech-set-tess",
        "output_dir": TARGET_ROOT / "raw" / "emotions" / "tess",
        "priority": "medium",
        "description": "Toronto Emotional Speech Set"
    },
}

def check_kaggle_installed() -> bool:
    """Check if kaggle package is installed."""
    try:
        import kaggle
        return True
    except ImportError:
        return False

def install_kaggle() -> bool:
    """Install kaggle package."""
    print("Installing kaggle package...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "--quiet"],
                      check=True, capture_output=True)
        print("✓ Kaggle package installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install kaggle: {e}")
        return False

def check_kaggle_config() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if not kaggle_dir.exists():
        print(f"\n⚠ Kaggle config directory not found: {kaggle_dir}")
        return False

    if not kaggle_json.exists():
        print(f"\n⚠ Kaggle credentials not found: {kaggle_json}")
        print("\nTo setup Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token'")
        print("4. Download kaggle.json file")
        print(f"5. Move it to: {kaggle_json}")
        print("6. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False

    # Check permissions
    stat = kaggle_json.stat()
    mode = stat.st_mode
    if mode & 0o077:
        print(f"\n⚠ Warning: kaggle.json has insecure permissions")
        print(f"Run: chmod 600 {kaggle_json}")
        return False

    print(f"✓ Kaggle credentials found: {kaggle_json}")
    return True

def download_kaggle_dataset(name: str, dataset_id: str, output_dir: Path) -> bool:
    """Download dataset from Kaggle."""
    try:
        import kaggle
    except ImportError:
        print("✗ Kaggle package not available")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading {name} from Kaggle")
    print(f"Dataset: {dataset_id}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    try:
        kaggle.api.dataset_download_files(
            dataset_id,
            path=str(output_dir),
            unzip=True,
        )
        print(f"\n✓ {name} downloaded successfully")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {name}: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\nCheck your Kaggle API credentials:")
            print("1. Verify ~/.kaggle/kaggle.json exists")
            print("2. Get new token from https://www.kaggle.com/settings")
            print("3. Ensure username and key are correct")
        return False

def main():
    """Main setup function."""
    print("="*60)
    print("KAGGLE DATASET SETUP")
    print("="*60)
    print("\nThis script will:")
    print("  1. Check/install Kaggle API package")
    print("  2. Verify Kaggle credentials")
    print("  3. Download RAVDESS and TESS datasets")
    print("\n" + "="*60 + "\n")

    # Step 1: Check/Install Kaggle
    if not check_kaggle_installed():
        print("Kaggle package not found.")
        response = input("Install kaggle package? (y/n): ").lower()
        if response == 'y':
            if not install_kaggle():
                print("Failed to install kaggle. Exiting.")
                return
        else:
            print("Cannot proceed without kaggle package. Exiting.")
            return
    else:
        print("✓ Kaggle package is installed")

    # Step 2: Check credentials
    if not check_kaggle_config():
        print("\n⚠ Kaggle API credentials not configured.")
        print("Please setup credentials first (see instructions above).")
        print("\nAfter setting up credentials, run this script again.")
        return

    # Step 3: Download datasets
    high_priority = {k: v for k, v in KAGGLE_DATASETS.items() if v["priority"] == "high"}
    medium_priority = {k: v for k, v in KAGGLE_DATASETS.items() if v["priority"] == "medium"}

    success_count = 0
    fail_count = 0

    # Download high priority first
    print("\n" + "="*60)
    print("HIGH PRIORITY DATASETS")
    print("="*60)
    for name, config in high_priority.items():
        if download_kaggle_dataset(name, config["dataset"], config["output_dir"]):
            success_count += 1
        else:
            fail_count += 1

    # Ask about medium priority
    if medium_priority:
        print("\n" + "="*60)
        print("MEDIUM PRIORITY DATASETS")
        print("="*60)
        response = input("\nDownload medium priority dataset (TESS)? (y/n): ").lower()
        if response == 'y':
            for name, config in medium_priority.items():
                if download_kaggle_dataset(name, config["dataset"], config["output_dir"]):
                    success_count += 1
                else:
                    fail_count += 1

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {fail_count}")

    if success_count > 0:
        print("\n✓ Datasets are ready for training!")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
