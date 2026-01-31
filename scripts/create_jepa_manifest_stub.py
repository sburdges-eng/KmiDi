#!/usr/bin/env python3
"""Create a placeholder aligned.jsonl at ~/Datasets/kmidi_jepa/manifests/aligned.jsonl for JEPA config validation."""

import json
import os
from pathlib import Path

def main() -> None:
    home = Path(os.environ.get("HOME", os.path.expanduser("~")))
    manifest_dir = home / "Datasets" / "kmidi_jepa" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "aligned.jsonl"

    # One example row with placeholder paths; replace with real paths when you have data
    example = {
        "audio_path": "",
        "midi_path": "",
        "specto_path": "",
        "start_offset": 0.0,
        "tempo": 120,
        "timebase": 480,
    }

    with open(manifest_path, "w") as f:
        f.write(json.dumps(example) + "\n")

    print(manifest_path)
    print("Placeholder manifest written (one empty row). Add real paths or more rows for training.")


if __name__ == "__main__":
    main()
