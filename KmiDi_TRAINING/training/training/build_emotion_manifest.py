#!/usr/bin/env python3
"""
Build a labeled audio manifest from common emotion datasets.

Supports:
- CREMA-D (Cheyney): parses emotion code from filename (e.g., 1012_IEO_HAP_HI.wav)
- RAVDESS: parses emotion code from filename (e.g., 03-01-05-01-01-01-01.wav)

Outputs a JSONL manifest with fields:
{"path": "/abs/path.wav", "label": "happy"}
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


CREMA_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

RAVDESS_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised",
}


def parse_cremad(path: Path) -> Optional[str]:
    """
    CREMA-D filename format: ID_PROMPT_EMOTION_LEVEL.wav
    Example: 1012_IEO_HAP_HI.wav -> HAP -> happy
    """
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    emotion_code = parts[2].upper()
    return CREMA_MAP.get(emotion_code)


def parse_ravdess(path: Path) -> Optional[str]:
    """
    RAVDESS filename format: MM-VO-EMO-INT-STY-XX-YY.wav
    Example: 03-01-05-01-01-01-01.wav -> EMO=05 -> angry
    """
    parts = path.stem.split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return RAVDESS_MAP.get(emotion_code)


def build_manifest(
    dataset: str, data_dir: Path, out_path: Path, limit: Optional[int] = None
) -> None:
    data_dir = data_dir.expanduser().resolve()
    wavs = sorted(list(data_dir.rglob("*.wav")))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under {data_dir}")

    manifest: List[Dict[str, str]] = []
    for path in wavs:
        label = None
        if dataset == "cremad":
            label = parse_cremad(path)
        elif dataset == "ravdess":
            label = parse_ravdess(path)
        else:
            label = None

        if label is None:
            continue

        manifest.append({"path": str(path), "label": label})
        if limit and len(manifest) >= limit:
            break

    if not manifest:
        raise ValueError(f"No labeled files parsed for dataset={dataset} in {data_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote manifest with {len(manifest)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build audio emotion manifest")
    parser.add_argument("--dataset", choices=["cremad", "ravdess"], required=True)
    parser.add_argument("--data-dir", required=True, help="Root directory of extracted audio")
    parser.add_argument("--out", required=True, help="Output manifest jsonl")
    parser.add_argument("--limit", type=int, help="Optional limit for quick tests")
    args = parser.parse_args()

    build_manifest(
        dataset=args.dataset,
        data_dir=Path(args.data_dir),
        out_path=Path(args.out),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
