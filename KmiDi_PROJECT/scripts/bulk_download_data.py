#!/usr/bin/env python3
"""
Bulk download training datasets.

Downloads:
- RAVDESS (emotion audio)
- Lakh MIDI (melody training)
- FMA (groove datasets)
"""

import sys
import os
import requests
from pathlib import Path
from typing import Optional
import hashlib
import zipfile
import tarfile

project_root = Path(__file__).parent.parent


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  Error downloading: {e}")
        return False


def verify_checksum(file_path: Path, expected_hash: Optional[str] = None) -> bool:
    """Verify file checksum if provided."""
    if not expected_hash:
        return True
    
    print(f"  Verifying checksum...")
    with open(file_path, 'rb') as f:
        sha256 = hashlib.sha256()
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
        file_hash = sha256.hexdigest()
    
    if file_hash.lower() != expected_hash.lower():
        print(f"  ⚠ Checksum mismatch! Expected {expected_hash[:16]}..., got {file_hash[:16]}...")
        return False
    
    print(f"  ✓ Checksum verified")
    return True


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract archive (zip or tar)."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            print(f"  ⚠ Unknown archive format: {archive_path.suffix}")
            return False
        
        print(f"  ✓ Extracted to {output_dir}")
        return True
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def download_ravdess(data_dir: Path) -> bool:
    """Download RAVDESS emotion audio dataset."""
    print("\n" + "="*70)
    print("RAVDESS Dataset")
    print("="*70)
    
    # RAVDESS requires manual download from:
    # https://zenodo.org/record/1188976
    print("  RAVDESS dataset requires manual download from:")
    print("  https://zenodo.org/record/1188976")
    print("  Please download and extract to: data/ravdess/")
    
    return True


def download_lakh_midi(data_dir: Path) -> bool:
    """Download Lakh MIDI dataset (subset)."""
    print("\n" + "="*70)
    print("Lakh MIDI Dataset")
    print("="*70)
    
    # Lakh MIDI dataset info:
    # https://colinraffel.com/projects/lmd/
    print("  Lakh MIDI dataset requires manual download from:")
    print("  http://colinraffel.com/projects/lmd/")
    print("  Please download and extract to: data/lakh_midi/")
    
    return True


def download_fma(data_dir: Path) -> bool:
    """Download FMA (Free Music Archive) dataset."""
    print("\n" + "="*70)
    print("FMA Dataset")
    print("="*70)
    
    # FMA dataset info:
    # https://github.com/mdeff/fma
    print("  FMA dataset requires manual download:")
    print("  https://github.com/mdeff/fma")
    print("  Please download and extract to: data/fma/")
    
    return True


def main():
    data_dir = project_root / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Training Data Download Script")
    print("="*70)
    print(f"\nData will be downloaded to: {data_dir}")
    print("\nNote: Some datasets require manual download due to size/licensing.")
    
    # Download datasets
    download_ravdess(data_dir / 'ravdess')
    download_lakh_midi(data_dir / 'lakh_midi')
    download_fma(data_dir / 'fma')
    
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("  1. Manually download required datasets (see URLs above)")
    print("  2. Extract to the specified directories")
    print("  3. Run scripts/augment_training_data.py to prepare data")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
