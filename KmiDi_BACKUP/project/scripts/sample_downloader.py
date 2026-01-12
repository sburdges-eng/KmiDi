#!/usr/bin/env python3
"""
iDAW Sample Library Downloader
Downloads samples from free libraries and syncs to Google Drive

Supports:
- Freesound
- BBC Sound Effects
- SampleSwap
- SampleRadar
- Looperman
- Bedroom Producers Blog

Features:
- Downloads samples to local staging area
- Auto-syncs to Google Drive (1TB available)
- Smart local cache: keeps most recent 5GB per source for offline sampling
- Automatically uploads older files and removes them from local disk
- Tracks storage usage across both local and cloud

How it works:
1. Downloads samples to local staging (~/.idaw_sample_staging)
2. Syncs to Google Drive automatically
3. Keeps newest 5GB per source locally for offline use
4. Deletes older files from local disk (still in Google Drive)

Usage:
    # Download with default 5GB local cache per source
    python3 sample_downloader.py --source bbc --category cinema_fx --urls-file urls.txt

    # Download with custom 10GB local cache per source
    python3 sample_downloader.py --source freesound --category cinema_fx --urls https://example.com/sample.wav --local-cache-gb 10

    # Keep all files locally (no cleanup)
    python3 sample_downloader.py --source bbc --category cinema_fx --urls-file urls.txt --keep-local-files

    # Check storage
    python3 sample_downloader.py --check-storage
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import urllib.request
import urllib.parse
import shutil

# Google Drive path
GDRIVE_ROOT = Path.home() / "sburdges@gmail.com - Google Drive" / "My Drive"
GDRIVE_SAMPLES = GDRIVE_ROOT / "iDAW_Samples"

# Local staging area (for batch downloads before sync)
LOCAL_STAGING = Path.home() / ".idaw_sample_staging"

# iDAW categories matching your library
IDAW_CATEGORIES = {
    "velvet_noir": "Velvet Noir",
    "rhythm_core": "Rhythm Core",
    "cinema_fx": "Cinema FX & Foley",
    "lo_fi_dreams": "Lo-Fi Dreams",
    "brass_soul": "Brass & Soul",
    "organic_textures": "Organic Textures"
}

# Sample sources
SAMPLE_SOURCES = {
    "freesound": {
        "name": "Freesound",
        "url": "https://freesound.org",
        "description": "Public domain sound effects and field recordings",
        "api_required": True,
        "best_for": ["cinema_fx", "organic_textures"]
    },
    "bbc": {
        "name": "BBC Sound Effects",
        "url": "http://bbcsfx.acropolis.org.uk",
        "description": "30,000+ BBC sound effects",
        "api_required": False,
        "best_for": ["cinema_fx", "organic_textures"]
    },
    "sampleswap": {
        "name": "SampleSwap",
        "url": "https://sampleswap.org",
        "description": "Royalty-free music loops and samples",
        "api_required": False,
        "best_for": ["rhythm_core", "lo_fi_dreams"]
    },
    "sampleradar": {
        "name": "SampleRadar (MusicRadar)",
        "url": "https://www.musicradar.com/news/tech/free-music-samples-royalty-free-loops-hits-and-multis-to-download",
        "description": "64,000+ royalty-free samples",
        "api_required": False,
        "best_for": ["rhythm_core", "brass_soul", "lo_fi_dreams"]
    },
    "looperman": {
        "name": "Looperman",
        "url": "https://www.looperman.com",
        "description": "Community loops, acapellas, and vocals",
        "api_required": False,
        "best_for": ["rhythm_core", "velvet_noir"]
    },
    "bedroom_producers": {
        "name": "Bedroom Producers Blog",
        "url": "https://bedroomproducersblog.com/free-samples/",
        "description": "Curated free sample packs",
        "api_required": False,
        "best_for": ["lo_fi_dreams", "rhythm_core"]
    }
}


class SampleDownloader:
    """Manages downloading and organizing samples to Google Drive"""

    def __init__(self, max_storage_gb: float = 1000):
        self.max_storage_bytes = max_storage_gb * 1024 * 1024 * 1024
        self.gdrive_samples = GDRIVE_SAMPLES
        self.local_staging = LOCAL_STAGING
        self.download_log = self.local_staging / "download_log.json"

        # Create directories
        self.local_staging.mkdir(parents=True, exist_ok=True)

        # Load download history
        self.load_history()

    def load_history(self):
        """Load download history to avoid re-downloading"""
        if self.download_log.exists():
            with open(self.download_log, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                "downloads": [],
                "total_bytes": 0,
                "last_update": None
            }

    def save_history(self):
        """Save download history"""
        self.history["last_update"] = datetime.now().isoformat()
        with open(self.download_log, 'w') as f:
            json.dump(self.history, f, indent=2)

    def check_storage(self) -> Dict:
        """Check current storage usage"""
        try:
            # Calculate Google Drive samples folder size
            gdrive_size = 0
            if self.gdrive_samples.exists():
                for path in self.gdrive_samples.rglob('*'):
                    if path.is_file():
                        try:
                            gdrive_size += path.stat().st_size
                        except (OSError, PermissionError):
                            pass

            # Calculate local staging size
            local_size = 0
            for path in self.local_staging.rglob('*'):
                if path.is_file():
                    try:
                        local_size += path.stat().st_size
                    except (OSError, PermissionError):
                        pass

            total_size = gdrive_size + local_size
            available = self.max_storage_bytes - total_size

            return {
                "gdrive_gb": gdrive_size / (1024**3),
                "local_staging_gb": local_size / (1024**3),
                "total_used_gb": total_size / (1024**3),
                "available_gb": available / (1024**3),
                "max_storage_gb": self.max_storage_bytes / (1024**3),
                "percent_used": (total_size / self.max_storage_bytes) * 100
            }
        except Exception as e:
            return {"error": str(e)}

    def download_file(self, url: str, destination: Path, source: str, category: str) -> bool:
        """Download a single file"""
        try:
            print(f"Downloading: {url}")
            print(f"  → {destination}")

            # Create parent directory
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    sys.stdout.write(f"\r  Progress: {percent}%")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, destination, reporthook)
            print()  # New line after progress

            # Record in history
            file_size = destination.stat().st_size
            self.history["downloads"].append({
                "url": url,
                "path": str(destination),
                "source": source,
                "category": category,
                "size_bytes": file_size,
                "timestamp": datetime.now().isoformat()
            })
            self.history["total_bytes"] += file_size
            self.save_history()

            print(f"✓ Downloaded: {destination.name} ({file_size / (1024**2):.2f} MB)")
            return True

        except Exception as e:
            print(f"✗ Error downloading {url}: {e}")
            return False

    def cleanup_old_files_per_source(self, max_gb_per_source: float = 5.0):
        """Keep only the most recent max_gb_per_source GB per source, delete older files"""
        print(f"\n=== Managing local cache (keeping {max_gb_per_source}GB per source) ===")

        max_bytes = max_gb_per_source * 1024 * 1024 * 1024

        # Group files by source from download history
        files_by_source = {}

        for download in self.history.get("downloads", []):
            source = download.get("source", "unknown")
            file_path = Path(download.get("path"))

            if file_path.exists():
                if source not in files_by_source:
                    files_by_source[source] = []

                files_by_source[source].append({
                    "path": file_path,
                    "size": file_path.stat().st_size,
                    "timestamp": download.get("timestamp")
                })

        total_deleted = 0

        # For each source, keep only newest files up to max_gb
        for source, files in files_by_source.items():
            # Sort by timestamp (newest first)
            files.sort(key=lambda x: x["timestamp"], reverse=True)

            current_size = 0
            files_to_delete = []

            for file_info in files:
                if current_size + file_info["size"] <= max_bytes:
                    # Keep this file
                    current_size += file_info["size"]
                else:
                    # Delete this file (exceeds quota)
                    files_to_delete.append(file_info)

            # Delete old files
            if files_to_delete:
                print(f"\n{source}:")
                print(f"  Keeping: {current_size / (1024**3):.2f} GB ({len(files) - len(files_to_delete)} files)")
                print(f"  Deleting: {sum(f['size'] for f in files_to_delete) / (1024**3):.2f} GB ({len(files_to_delete)} files)")

                for file_info in files_to_delete:
                    try:
                        file_info["path"].unlink()
                        total_deleted += file_info["size"]
                    except Exception as e:
                        print(f"    ✗ Could not delete {file_info['path'].name}: {e}")

        if total_deleted > 0:
            print(f"\n✓ Freed {total_deleted / (1024**3):.2f} GB of local disk space")
            print(f"  Most recent {max_gb_per_source}GB per source kept for offline use")

    def sync_to_gdrive(self, category: str = None, cleanup_local: bool = True, keep_gb_per_source: float = 5.0):
        """Sync staged files to Google Drive and manage local cache"""
        print("\n=== Syncing to Google Drive ===")

        try:
            # Ensure Google Drive folder exists
            self.gdrive_samples.mkdir(parents=True, exist_ok=True)

            # Get files to sync
            if category:
                category_name = IDAW_CATEGORIES.get(category, category)
                source_dir = self.local_staging / category_name
            else:
                source_dir = self.local_staging

            if not source_dir.exists():
                print(f"No files to sync in {source_dir}")
                return

            synced_files = []

            # Copy files to Google Drive
            for source_file in source_dir.rglob('*'):
                if source_file.is_file() and source_file.name != 'download_log.json':
                    # Determine relative path
                    rel_path = source_file.relative_to(self.local_staging)
                    dest_file = self.gdrive_samples / rel_path

                    # Create parent dir
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    # Copy if not exists or different
                    if not dest_file.exists() or source_file.stat().st_size != dest_file.stat().st_size:
                        print(f"Syncing: {rel_path}")
                        shutil.copy2(source_file, dest_file)
                        print(f"  ✓ {dest_file}")

                    # Verify copy was successful
                    if dest_file.exists() and dest_file.stat().st_size == source_file.stat().st_size:
                        synced_files.append(source_file)

            print("\n✓ Sync complete! Files will upload to Google Drive automatically.")

            # Manage local cache - keep recent files per source, delete old ones
            if cleanup_local:
                self.cleanup_old_files_per_source(max_gb_per_source=keep_gb_per_source)

        except Exception as e:
            print(f"✗ Sync error: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Google Drive is running")
            print("2. Check that you have write permissions")
            print("3. Verify the Google Drive folder path:")
            print(f"   {self.gdrive_samples}")

    def download_from_direct_links(self, urls: List[str], source: str, category: str, cleanup_local: bool = True, keep_gb_per_source: float = 5.0):
        """Download from a list of direct download URLs"""
        category_name = IDAW_CATEGORIES.get(category, category)
        dest_dir = self.local_staging / category_name

        print(f"\n=== Downloading {len(urls)} files ===")
        print(f"Source: {source}")
        print(f"Category: {category_name}")

        success_count = 0
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}]")

            # Generate filename from URL
            filename = Path(urllib.parse.urlparse(url).path).name
            if not filename:
                filename = f"sample_{hashlib.md5(url.encode()).hexdigest()[:8]}.wav"

            dest_file = dest_dir / filename

            # Skip if already downloaded
            if dest_file.exists():
                print(f"  Skipping (already exists): {filename}")
                continue

            # Check storage
            storage = self.check_storage()
            if storage.get("available_gb", 0) < 0.1:  # Less than 100MB available
                print("✗ Storage limit reached!")
                break

            # Download
            if self.download_file(url, dest_file, source, category):
                success_count += 1

        print(f"\n=== Download Complete ===")
        print(f"Successfully downloaded: {success_count}/{len(urls)} files")

        # Auto-sync to Google Drive
        if success_count > 0:
            self.sync_to_gdrive(category, cleanup_local=cleanup_local, keep_gb_per_source=keep_gb_per_source)


def main():
    parser = argparse.ArgumentParser(description="iDAW Sample Library Downloader")
    parser.add_argument("--list-sources", action="store_true", help="List available sample sources")
    parser.add_argument("--list-categories", action="store_true", help="List iDAW categories")
    parser.add_argument("--check-storage", action="store_true", help="Check storage usage")
    parser.add_argument("--source", help="Sample source (e.g., freesound, bbc)")
    parser.add_argument("--category", help="iDAW category (e.g., cinema_fx, rhythm_core)")
    parser.add_argument("--urls", nargs="+", help="Direct download URLs")
    parser.add_argument("--urls-file", help="File containing URLs (one per line)")
    parser.add_argument("--max-storage-gb", type=float, default=1000, help="Max storage in GB (default: 1000)")
    parser.add_argument("--keep-local-files", action="store_true", help="Keep ALL local files after syncing (default: keep only 5GB per source)")
    parser.add_argument("--local-cache-gb", type=float, default=5.0, help="GB to keep locally per source for offline use (default: 5GB)")

    args = parser.parse_args()

    downloader = SampleDownloader(max_storage_gb=args.max_storage_gb)

    # List sources
    if args.list_sources:
        print("\n=== Available Sample Sources ===\n")
        for key, info in SAMPLE_SOURCES.items():
            print(f"{key}:")
            print(f"  Name: {info['name']}")
            print(f"  URL: {info['url']}")
            print(f"  Description: {info['description']}")
            print(f"  Best for: {', '.join(info['best_for'])}")
            print()
        return

    # List categories
    if args.list_categories:
        print("\n=== iDAW Categories ===\n")
        for key, name in IDAW_CATEGORIES.items():
            print(f"{key}: {name}")
        return

    # Check storage
    if args.check_storage:
        storage = downloader.check_storage()
        if "error" in storage:
            print(f"Error checking storage: {storage['error']}")
        else:
            print("\n=== Storage Usage ===")
            print(f"Google Drive: {storage['gdrive_gb']:.2f} GB")
            print(f"Local Staging: {storage['local_staging_gb']:.2f} GB")
            print(f"Total Used: {storage['total_used_gb']:.2f} GB / {storage['max_storage_gb']:.2f} GB")
            print(f"Available: {storage['available_gb']:.2f} GB")
            print(f"Usage: {storage['percent_used']:.1f}%")
        return

    # Download from URLs
    if args.urls or args.urls_file:
        if not args.source or not args.category:
            print("Error: --source and --category are required when downloading")
            return

        urls = []
        if args.urls:
            urls.extend(args.urls)
        if args.urls_file:
            with open(args.urls_file, 'r') as f:
                urls.extend([line.strip() for line in f if line.strip()])

        # Cleanup local files by default (unless --keep-local-files is specified)
        # If --keep-local-files is used, keep everything; otherwise keep only --local-cache-gb per source
        cleanup_local = not args.keep_local_files
        keep_gb = args.local_cache_gb if cleanup_local else float('inf')  # Keep everything if --keep-local-files

        downloader.download_from_direct_links(
            urls,
            args.source,
            args.category,
            cleanup_local=cleanup_local,
            keep_gb_per_source=keep_gb
        )
        return

    # Show help if no action
    print("\niDAW Sample Downloader")
    print("=" * 50)
    print("\nQuick start:")
    print("  1. List available sources: python3 sample_downloader.py --list-sources")
    print("  2. Check storage: python3 sample_downloader.py --check-storage")
    print("  3. Download samples:")
    print("     python3 sample_downloader.py --source bbc --category cinema_fx --urls https://example.com/sample.wav")
    print("\nFor more options: python3 sample_downloader.py --help")
    print()


if __name__ == "__main__":
    main()
