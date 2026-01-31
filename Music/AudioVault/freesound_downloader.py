#!/usr/bin/env python3
"""
Freesound Pack Downloader for AudioVault
Downloads curated packs directly to AudioVault/raw/
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

# AudioVault structure
VAULT_DIR = Path.home() / "Music" / "AudioVault"
RAW_DIR = VAULT_DIR / "raw"
API_KEY_FILE = Path.home() / ".freesound_api_key"

# Curated pack IDs for lo-fi/bedroom emo production
PACK_LIST = {
    "drums_acoustic": {
        "pack_id": 14856,
        "name": "Acoustic_Drum_Kit_Clean",
        "user": "afleetingspeck",
    },
    "drums_vintage": {
        "pack_id": 6,
        "name": "Vintage_Drum_Machine",
        "user": "suburban_grilla",
    },
    "drums_brush": {
        "pack_id": 28587,
        "name": "Brush_Drums",
        "user": "Mhistorically",
    },
    "room_ambience": {
        "pack_id": 10855,
        "name": "Room_Tones",
        "user": "klankbeeld",
    },
    "guitar_notes": {
        "pack_id": 8647,
        "name": "Acoustic_Guitar_Notes",
        "user": "MTG",
    },
    "bass_jazz": {
        "pack_id": 13115,
        "name": "Jazz_Bass",
        "user": "FrusciMike",
    },
    "vinyl_fx": {
        "pack_id": 17926,
        "name": "Vinyl_Crackle",
        "user": "OldFritz",
    },
}


def load_api_key():
    """Load Freesound API key"""
    if not API_KEY_FILE.exists():
        print(f"❌ API key not found: {API_KEY_FILE}")
        print("\nGet free API key: https://freesound.org/apiv2/apply/")
        print(f"Then: echo 'YOUR_KEY' > {API_KEY_FILE}")
        print(f"      chmod 600 {API_KEY_FILE}")
        sys.exit(1)

    with open(API_KEY_FILE) as f:
        api_key = f.read().strip()

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print(f"❌ Update {API_KEY_FILE} with real API key")
        sys.exit(1)

    return api_key


def get_pack_sounds(pack_id, api_key):
    """Get sounds from a pack"""
    url = f"https://freesound.org/apiv2/packs/{pack_id}/sounds/?token={api_key}&fields=id,name,previews"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        return data.get("results", [])
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []


def download_sound(sound_id, sound_name, preview_url, pack_dir):
    """Download sound preview (HQ MP3)"""
    # Sanitize filename
    safe_name = "".join(c for c in sound_name if c.isalnum() or c in (' ', '-', '_')).strip()
    output_file = pack_dir / f"{safe_name}.mp3"

    if output_file.exists():
        print(f"    ⏭️  {safe_name}")
        return True

    try:
        urllib.request.urlretrieve(preview_url, output_file)
        print(f"    ✅ {safe_name}")
        return True
    except Exception as e:
        print(f"    ❌ {safe_name} - {e}")
        return False


def download_pack(pack_key, pack_info, api_key):
    """Download all sounds from pack"""
    pack_id = pack_info["pack_id"]
    pack_name = pack_info["name"]

    print(f"\n📦 {pack_name}")

    # Create pack directory
    pack_dir = RAW_DIR / pack_name
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Get sounds
    sounds = get_pack_sounds(pack_id, api_key)
    if not sounds:
        print(f"   ⚠️  No sounds found")
        return

    print(f"   {len(sounds)} sounds")

    # Download each
    success = 0
    for sound in sounds:
        if download_sound(sound["id"], sound["name"],
                         sound["previews"]["preview-hq-mp3"], pack_dir):
            success += 1

    print(f"   ✅ {success}/{len(sounds)} complete")


def main():
    """Download all packs to AudioVault/raw/"""
    print("=" * 60)
    print("FREESOUND → AUDIOVAULT DOWNLOADER")
    print("=" * 60)

    # Load API key
    api_key = load_api_key()
    print(f"✅ API key loaded\n")

    # Create raw directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Downloading to: {RAW_DIR}\n")

    # Download packs
    print(f"📥 {len(PACK_LIST)} packs queued...\n")

    for pack_key, pack_info in PACK_LIST.items():
        download_pack(pack_key, pack_info, api_key)

    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nRaw samples: {RAW_DIR}")
    print(f"\nNext: python3 audio_refinery.py")
    print()


if __name__ == "__main__":
    main()
