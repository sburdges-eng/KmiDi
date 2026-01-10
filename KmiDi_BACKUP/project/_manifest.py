# CREMA-D/RAVDESS Label Parsing
# Parses metadata for audio datasets like CREMA-D and RAVDESS.

import os
import json


def parse_crema_d_manifest(manifest_path):
    """Parse CREMA-D dataset manifest."""
    with open(manifest_path, 'r') as f:
        data = json.load(f)

    parsed = {}
    for entry in data:
        audio_path = entry['audio_filepath']
        label = entry['emotion']
        parsed[audio_path] = label
    return parsed


def parse_ravdess_manifest(manifest_path):
    """Parse RAVDESS dataset manifest."""
    with open(manifest_path, 'r') as f:
        data = json.load(f)

    parsed = {}
    for entry in data:
        audio_path = entry['audio_filepath']
        label = entry['emotion']
        parsed[audio_path] = label
    return parsed


if __name__ == "__main__":
    crema_d_manifest = parse_crema_d_manifest("crema_d_manifest.json")
    ravdess_manifest = parse_ravdess_manifest("ravdess_manifest.json")
    print("CREMA-D Manifest:", crema_d_manifest)
    print("RAVDESS Manifest:", ravdess_manifest)
