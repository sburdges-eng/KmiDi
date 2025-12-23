#!/usr/bin/env python3
"""
Emotion-Scale Sample Fetcher
Downloads free .wav samples organized by 74 emotions × 52 scales
Syncs to Google Drive with 25MB limit per combination
"""

import json
import os
import sys
import requests
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import urllib.parse

# Paths
SCRIPT_DIR = Path(__file__).parent
MUSIC_BRAIN_DIR = SCRIPT_DIR / "music_brain"
SCALES_DB_PATH = MUSIC_BRAIN_DIR / "scales_database.json"
GDRIVE_ROOT = Path.home() / "sburdges@gmail.com - Google Drive" / "My Drive"
GDRIVE_SAMPLES = GDRIVE_ROOT / "iDAW_Samples" / "Emotion_Scale_Library"

# Local staging (temporary)
LOCAL_STAGING = SCRIPT_DIR / "emotion_scale_staging"

# Config
CONFIG_FILE = SCRIPT_DIR / "freesound_config.json"
DOWNLOAD_LOG = SCRIPT_DIR / "emotion_scale_downloads.json"

# Size limits
MAX_SIZE_PER_COMBO_MB = 25
MAX_SIZE_PER_COMBO_BYTES = MAX_SIZE_PER_COMBO_MB * 1024 * 1024

class FreesoundFetcher:
    """Fetch samples from Freesound.org API"""

    def __init__(self, api_key=None):
        self.api_key = api_key or self.load_api_key()
        self.base_url = "https://freesound.org/apiv2"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Token {self.api_key}"})

    def load_api_key(self):
        """Load API key from config"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get('freesound_api_key')
        return None

    def save_api_key(self, api_key):
        """Save API key to config"""
        config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

        config['freesound_api_key'] = api_key

        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        self.api_key = api_key
        self.session.headers.update({"Authorization": f"Token {api_key}"})

    def search(self, query, filter_params=None, page_size=15):
        """Search for sounds"""
        if not self.api_key:
            raise Exception("Freesound API key required. Run: ./emotion_scale_sampler.py setup")

        params = {
            'query': query,
            'page_size': page_size,
            'fields': 'id,name,tags,duration,filesize,type,previews,download',
            'filter': 'type:wav'  # Only .wav files
        }

        if filter_params:
            params.update(filter_params)

        try:
            response = self.session.get(f"{self.base_url}/search/text/", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None

    def download_sound(self, sound_id, output_path):
        """Download a sound file"""
        if not self.api_key:
            raise Exception("Freesound API key required")

        try:
            # Get sound details
            response = self.session.get(f"{self.base_url}/sounds/{sound_id}/")
            response.raise_for_status()
            sound_data = response.json()

            # Get download URL (requires OAuth, use preview for now)
            preview_url = sound_data['previews']['preview-hq-mp3']  # High quality preview

            # Download file
            download_response = requests.get(preview_url, stream=True)
            download_response.raise_for_status()

            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in download_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return output_path.stat().st_size

        except requests.exceptions.RequestException as e:
            print(f"Download error: {e}")
            return 0

class EmotionScaleSampler:
    """Main sampler class"""

    def __init__(self):
        self.scales_db = self.load_scales_db()
        self.emotions = self.extract_emotions()
        self.base_scales = self.extract_base_scales()
        self.fetcher = FreesoundFetcher()
        self.download_log = self.load_download_log()

        # Create directories
        LOCAL_STAGING.mkdir(parents=True, exist_ok=True)
        GDRIVE_SAMPLES.mkdir(parents=True, exist_ok=True)

    def load_scales_db(self):
        """Load scales database"""
        if not SCALES_DB_PATH.exists():
            print(f"Error: Scales database not found at {SCALES_DB_PATH}")
            return None

        with open(SCALES_DB_PATH, 'r') as f:
            return json.load(f)

    def extract_emotions(self):
        """Extract all unique emotions from scales database"""
        emotions = set()

        if not self.scales_db:
            return list(emotions)

        for scale in self.scales_db.get('scales', []):
            for emotion in scale.get('emotional_quality', []):
                emotions.add(emotion.lower())

            if scale.get('music_brain_emotion'):
                emotions.add(scale['music_brain_emotion'].lower())

        return sorted(list(emotions))

    def extract_base_scales(self):
        """Extract unique base scale names"""
        scales = set()

        if not self.scales_db:
            return list(scales)

        for scale in self.scales_db.get('scales', []):
            scales.add(scale['scale_type'])

        return sorted(list(scales))

    def load_download_log(self):
        """Load download history"""
        if DOWNLOAD_LOG.exists():
            with open(DOWNLOAD_LOG, 'r') as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "combinations": {},
            "total_size_mb": 0,
            "total_files": 0
        }

    def save_download_log(self):
        """Save download history"""
        with open(DOWNLOAD_LOG, 'w') as f:
            json.dump(self.download_log, f, indent=2)

    def get_combo_key(self, emotion, scale):
        """Get combination key for tracking"""
        return f"{emotion}_{scale}"

    def get_combo_size(self, emotion, scale):
        """Get current size for emotion-scale combination"""
        key = self.get_combo_key(emotion, scale)
        combo_data = self.download_log['combinations'].get(key, {})
        return combo_data.get('total_size_bytes', 0)

    def can_download_more(self, emotion, scale, file_size):
        """Check if we can download more for this combination"""
        current_size = self.get_combo_size(emotion, scale)
        return (current_size + file_size) <= MAX_SIZE_PER_COMBO_BYTES

    def create_search_query(self, emotion, scale):
        """Create smart search query from emotion and scale"""
        # Combine emotion with musical terms
        queries = [
            f"{emotion} {scale.lower()}",
            f"{emotion} music",
            f"{emotion} ambient",
            f"{scale.lower()} scale",
        ]
        return queries

    def download_for_combination(self, emotion, scale, max_files=10):
        """Download samples for specific emotion-scale combination"""
        key = self.get_combo_key(emotion, scale)

        # Initialize combo tracking
        if key not in self.download_log['combinations']:
            self.download_log['combinations'][key] = {
                'emotion': emotion,
                'scale': scale,
                'total_size_bytes': 0,
                'files': [],
                'last_updated': datetime.now().isoformat()
            }

        combo_data = self.download_log['combinations'][key]
        current_size = combo_data['total_size_bytes']

        if current_size >= MAX_SIZE_PER_COMBO_BYTES:
            print(f"  ✓ {emotion}/{scale} already at 25MB limit")
            return

        print(f"\n{'='*70}")
        print(f"Fetching: {emotion.upper()} + {scale}")
        print(f"Current: {current_size / 1024 / 1024:.2f}MB / {MAX_SIZE_PER_COMBO_MB}MB")
        print(f"{'='*70}")

        # Create output directory
        output_dir = LOCAL_STAGING / emotion / scale
        output_dir.mkdir(parents=True, exist_ok=True)

        # Try different search queries
        queries = self.create_search_query(emotion, scale)
        downloaded_count = 0

        for query in queries:
            if current_size >= MAX_SIZE_PER_COMBO_BYTES:
                break

            print(f"\nSearching: {query}")
            results = self.fetcher.search(query, page_size=5)

            if not results or 'results' not in results:
                continue

            for sound in results['results']:
                if current_size >= MAX_SIZE_PER_COMBO_BYTES:
                    break

                if downloaded_count >= max_files:
                    break

                sound_id = sound['id']
                sound_name = sound['name']
                file_size = sound.get('filesize', 0)

                # Check if we can download
                if not self.can_download_more(emotion, scale, file_size):
                    print(f"  ⚠ Skipping {sound_name} (would exceed 25MB)")
                    continue

                # Download
                filename = f"{sound_id}_{sound_name[:50]}.wav"
                filename = "".join(c for c in filename if c.isalnum() or c in '._- ')
                output_path = output_dir / filename

                if output_path.exists():
                    print(f"  ⏭ Skip (exists): {filename}")
                    continue

                print(f"  ⬇ Downloading: {sound_name[:50]}...")
                downloaded_size = self.fetcher.download_sound(sound_id, output_path)

                if downloaded_size > 0:
                    # Update tracking
                    combo_data['files'].append({
                        'id': sound_id,
                        'name': sound_name,
                        'filename': filename,
                        'size_bytes': downloaded_size,
                        'downloaded': datetime.now().isoformat()
                    })
                    current_size += downloaded_size
                    combo_data['total_size_bytes'] = current_size
                    combo_data['last_updated'] = datetime.now().isoformat()

                    downloaded_count += 1
                    print(f"    ✓ Downloaded {downloaded_size / 1024 / 1024:.2f}MB")
                    print(f"    Total: {current_size / 1024 / 1024:.2f}MB / {MAX_SIZE_PER_COMBO_MB}MB")

                    # Save progress
                    self.save_download_log()

                    # Rate limiting
                    time.sleep(1)
                else:
                    print(f"    ✗ Download failed")

        print(f"\n✓ Completed {emotion}/{scale}: {downloaded_count} files, {current_size / 1024 / 1024:.2f}MB")

    def sync_to_gdrive(self):
        """Sync local staging to Google Drive"""
        print(f"\n{'='*70}")
        print("Syncing to Google Drive...")
        print(f"{'='*70}")

        import shutil

        total_synced = 0
        for emotion_dir in LOCAL_STAGING.iterdir():
            if not emotion_dir.is_dir():
                continue

            for scale_dir in emotion_dir.iterdir():
                if not scale_dir.is_dir():
                    continue

                # Create target directory in GDrive
                target_dir = GDRIVE_SAMPLES / emotion_dir.name / scale_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)

                # Copy files
                for file in scale_dir.glob("*.wav"):
                    target_file = target_dir / file.name

                    if not target_file.exists():
                        shutil.copy2(file, target_file)
                        total_synced += 1
                        print(f"  ✓ Synced: {emotion_dir.name}/{scale_dir.name}/{file.name}")

        print(f"\n✓ Synced {total_synced} files to Google Drive")
        print(f"Location: {GDRIVE_SAMPLES}")

    def show_stats(self):
        """Show download statistics"""
        print(f"\n{'='*70}")
        print("EMOTION-SCALE SAMPLE LIBRARY STATISTICS")
        print(f"{'='*70}")

        total_combos = len(self.download_log['combinations'])
        total_size = sum(c['total_size_bytes'] for c in self.download_log['combinations'].values())
        total_files = sum(len(c['files']) for c in self.download_log['combinations'].values())

        print(f"\nTotal Combinations: {total_combos}")
        print(f"Total Files: {total_files}")
        print(f"Total Size: {total_size / 1024 / 1024:.2f}MB")
        print(f"\nAvailable Emotions: {len(self.emotions)}")
        print(f"Available Scales: {len(self.base_scales)}")
        print(f"Max Combinations: {len(self.emotions)} × {len(self.base_scales)} = {len(self.emotions) * len(self.base_scales)}")

        if total_combos > 0:
            print(f"\nTop 10 Downloaded Combinations:")
            sorted_combos = sorted(
                self.download_log['combinations'].items(),
                key=lambda x: x[1]['total_size_bytes'],
                reverse=True
            )

            for i, (key, data) in enumerate(sorted_combos[:10], 1):
                size_mb = data['total_size_bytes'] / 1024 / 1024
                file_count = len(data['files'])
                print(f"  {i}. {data['emotion']}/{data['scale']}: {file_count} files, {size_mb:.2f}MB")

def main():
    if len(sys.argv) < 2:
        print("="*70)
        print("EMOTION-SCALE SAMPLE FETCHER")
        print("="*70)
        print("\nDownload free .wav samples organized by emotion and scale")
        print("74 emotions × 52 scales = 3,848 combinations")
        print("25MB limit per combination")
        print("\nUSAGE:")
        print("  ./emotion_scale_sampler.py setup          # Configure Freesound API key")
        print("  ./emotion_scale_sampler.py fetch <emotion> <scale>")
        print("  ./emotion_scale_sampler.py batch <count>  # Fetch random combinations")
        print("  ./emotion_scale_sampler.py sync           # Sync to Google Drive")
        print("  ./emotion_scale_sampler.py stats          # Show statistics")
        print("  ./emotion_scale_sampler.py list           # List emotions and scales")
        print("\nEXAMPLES:")
        print("  ./emotion_scale_sampler.py fetch melancholy dorian")
        print("  ./emotion_scale_sampler.py batch 10")
        print("")
        return

    command = sys.argv[1].lower()
    sampler = EmotionScaleSampler()

    if command == 'setup':
        print("="*70)
        print("FREESOUND API SETUP")
        print("="*70)
        print("\n1. Go to https://freesound.org/")
        print("2. Create a free account")
        print("3. Go to https://freesound.org/apiv2/apply/")
        print("4. Create an API key")
        print("\nEnter your Freesound API key:")
        api_key = input("> ").strip()

        if api_key:
            sampler.fetcher.save_api_key(api_key)
            print(f"\n✓ API key saved to {CONFIG_FILE}")
        else:
            print("\n✗ No API key provided")

    elif command == 'list':
        print(f"\nAvailable Emotions ({len(sampler.emotions)}):")
        for i in range(0, len(sampler.emotions), 6):
            print("  " + ", ".join(sampler.emotions[i:i+6]))

        print(f"\nAvailable Scales ({len(sampler.base_scales)}):")
        for i in range(0, len(sampler.base_scales), 5):
            print("  " + ", ".join(sampler.base_scales[i:i+5]))

    elif command == 'fetch':
        if len(sys.argv) < 4:
            print("Error: Emotion and scale required")
            print("Usage: ./emotion_scale_sampler.py fetch <emotion> <scale>")
            return

        emotion = sys.argv[2].lower()
        scale = sys.argv[3]

        if emotion not in sampler.emotions:
            print(f"Error: Unknown emotion '{emotion}'")
            print(f"Run './emotion_scale_sampler.py list' to see available emotions")
            return

        if scale not in sampler.base_scales:
            print(f"Error: Unknown scale '{scale}'")
            print(f"Run './emotion_scale_sampler.py list' to see available scales")
            return

        sampler.download_for_combination(emotion, scale)
        sampler.sync_to_gdrive()

    elif command == 'batch':
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 10

        import random

        print(f"\nFetching {count} random emotion-scale combinations...")

        for i in range(count):
            emotion = random.choice(sampler.emotions)
            scale = random.choice(sampler.base_scales)

            try:
                sampler.download_for_combination(emotion, scale, max_files=5)
            except Exception as e:
                print(f"Error: {e}")
                continue

        sampler.sync_to_gdrive()
        sampler.show_stats()

    elif command == 'sync':
        sampler.sync_to_gdrive()

    elif command == 'stats':
        sampler.show_stats()

    else:
        print(f"Error: Unknown command '{command}'")

if __name__ == "__main__":
    main()
