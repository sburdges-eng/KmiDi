#!/usr/bin/env python3
import os
import hashlib
from collections import defaultdict
from pathlib import Path
import csv
import sys

def get_file_hash(filepath, chunk_size=8192):
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    root_dir = Path("/Volumes/Sean's SSD")
    if not root_dir.exists():
        print(f"Error: {root_dir} not found.")
        sys.exit(1)

    # Directories we consider "official" for datasets
    official_dirs = [
        root_dir / "Datasets",
        root_dir / "kmidi_audio_data",
        root_dir / "My Mac" / "KmiDi MIDI Companion" / "AUDIO_MIDI_DATA"
    ]
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.aiff', '.aif', '.ogg', '.m4a', '.mid', '.midi'}
    
    print("Scanning drive for audio/MIDI files (this may take a while)...")
    
    # size_bytes -> list of file paths
    size_map = defaultdict(list)
    outliers = []
    total_files = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories and Trashes to save time/permissions
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ['_FORENSIC_READONLY_KMIDI', 'backup']]
        
        curr_path = Path(dirpath)
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in audio_extensions and not f.startswith('.'):
                full_path = curr_path / f
                
                try:
                    size = os.path.getsize(full_path)
                    size_map[size].append(full_path)
                    total_files += 1
                    
                    # Check if outlier
                    is_official = any(str(full_path).startswith(str(off_dir)) for off_dir in official_dirs)
                    if not is_official:
                        outliers.append(full_path)
                except OSError:
                    pass

    print(f"Found {total_files} total audio/MIDI files.")
    
    # Filter size_map to only sizes with > 1 file (potential duplicates)
    potential_dupes = {size: paths for size, paths in size_map.items() if len(paths) > 1}
    
    print(f"Hashing {sum(len(paths) for paths in potential_dupes.values())} potential duplicates to confirm...")
    
    # hash -> list of file paths
    exact_dupes = defaultdict(list)
    
    for size, paths in potential_dupes.items():
        for path in paths:
            file_hash = get_file_hash(path)
            if file_hash:
                exact_dupes[file_hash].append(path)
                
    # Filter exact_dupes to only hashes with > 1 file
    exact_dupes = {h: paths for h, paths in exact_dupes.items() if len(paths) > 1}
    
    # Write Duplicates Report
    dupe_report_path = root_dir / "duplicate_audio_report.csv"
    wasted_bytes = 0
    
    with open(dupe_report_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Hash', 'File Size (Bytes)', 'File Path', 'Keep/Delete Suggestion'])
        
        for h, paths in exact_dupes.items():
            file_size = os.path.getsize(paths[0])
            wasted_bytes += file_size * (len(paths) - 1)
            
            # Sort so the one in an "official" dir is kept, others suggested for deletion
            paths.sort(key=lambda p: 0 if any(str(p).startswith(str(off)) for off in official_dirs) else 1)
            
            writer.writerow([h, file_size, str(paths[0]), 'KEEP (Canonical)'])
            for p in paths[1:]:
                writer.writerow([h, file_size, str(p), 'DELETE (Duplicate)'])
                
    # Write Outliers Report
    outlier_report_path = root_dir / "outlier_audio_report.csv"
    with open(outlier_report_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File Size (Bytes)', 'File Path', 'Suggestion'])
        for p in outliers:
            writer.writerow([os.path.getsize(p), str(p), 'REVIEW (Outside official dataset folders)'])

    print("\n" + "="*50)
    print("CLEANUP SCAN COMPLETE")
    print("="*50)
    print(f"Exact Duplicates Found: {sum(len(paths) - 1 for paths in exact_dupes.values())}")
    print(f"Potential Space Savings: {wasted_bytes / (1024*1024*1024):.2f} GB")
    print(f"Outlying Audio Files: {len(outliers)}")
    print(f"\nReports generated at:")
    print(f"1. {dupe_report_path}")
    print(f"2. {outlier_report_path}")
    print("\nReview these CSV files. If you want to automatically delete the marked duplicates, let the AI know!")

if __name__ == '__main__':
    main()
