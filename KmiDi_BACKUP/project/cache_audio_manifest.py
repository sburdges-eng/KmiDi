# Cache/Download and Relabel Audio
# Downloads audio datasets and relabels them to a specified directory.

import os
import shutil
import requests


def download_audio(url, save_path):
    """Download audio file from a URL."""
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print(f"Downloaded: {save_path}")


def cache_audio(dataset_manifest, output_dir):
    """Cache audio files to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    for entry in dataset_manifest:
        audio_url = entry['audio_url']
        audio_filename = os.path.basename(audio_url)
        save_path = os.path.join(output_dir, audio_filename)
        download_audio(audio_url, save_path)


if __name__ == "__main__":
    # Example usage
    dataset_manifest = [
        {"audio_url": "https://example.com/audio1.wav", "emotion": "happy"},
        {"audio_url": "https://example.com/audio2.wav", "emotion": "sad"}
    ]
    cache_audio(dataset_manifest, "/Volumes/sbdrive/audio_cache")
