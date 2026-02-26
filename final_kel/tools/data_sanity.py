"""
Data sanity: validate dataset folder (audio/MIDI) and pipeline output (npy) before training.
Run: PYTHONPATH=. python tools/data_sanity.py [--datasets-dir datasets] [--npy processed/kmidi_multimodal.npy]
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.storage_paths import DATASETS_DIR, PROCESSED_DATASET_FILE, ensure_storage_dirs


def scan_datasets_dir(path, exts_audio=("wav", "mp3", "flac"), exts_midi=("mid", "midi")):
    audio = []
    midi = []
    if not os.path.isdir(path):
        return audio, midi
    for root, _dirs, files in os.walk(path):
        for f in files:
            ext = f.lower().split(".")[-1]
            full = os.path.join(root, f)
            if ext in exts_audio:
                audio.append(full)
            elif ext in exts_midi:
                midi.append(full)
    return audio, midi


def check_npy(npy_path):
    import numpy as np
    if not os.path.isfile(npy_path):
        print("Npy not found:", npy_path)
        return
    data = np.load(npy_path, allow_pickle=True).item()
    keys = list(data.keys())
    print("Npy keys count:", len(keys))
    sample = data.get(keys[0], {}) if keys else {}
    print("Sample keys:", list(sample.keys()))
    for k in list(sample.keys())[:5]:
        v = sample[k]
        if hasattr(v, "shape"):
            print("  ", k, "shape:", v.shape if hasattr(v, "shape") else type(v))
        elif isinstance(v, (list, tuple)):
            print("  ", k, "len:", len(v))
        else:
            print("  ", k, "type:", type(v).__name__)


def main():
    ensure_storage_dirs()

    p = argparse.ArgumentParser(description="KmiDi data sanity check")
    p.add_argument("--datasets-dir", default=DATASETS_DIR, help="Folder to scan for audio/MIDI")
    p.add_argument("--npy", default=PROCESSED_DATASET_FILE, help="Path to kmidi_multimodal.npy (pipeline output)")
    args = p.parse_args()

    print("=== Datasets dir:", args.datasets_dir)
    audio, midi = scan_datasets_dir(args.datasets_dir)
    print("Audio files:", len(audio))
    print("MIDI files:", len(midi))
    if audio:
        print("First audio:", audio[0])
    if midi:
        print("First MIDI:", midi[0])

    if args.npy:
        print("\n=== Npy:", args.npy)
        check_npy(args.npy)


if __name__ == "__main__":
    main()
