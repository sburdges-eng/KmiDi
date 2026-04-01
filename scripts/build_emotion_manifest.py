#!/usr/bin/env python3
"""
Build emotion manifest from DEAM and PMEmo datasets.

Scans audio files and annotation CSVs from DEAM and PMEmo, normalises
valence/arousal to [-1, 1], assigns deterministic 70/15/15 train/val/test
splits, and writes data/emotion_manifest.json.

Usage:
    python scripts/build_emotion_manifest.py
    python scripts/build_emotion_manifest.py \\
        --datasets-root ~/Datasets --output data/emotion_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# 1-9 annotation scale → [-1, 1]
_SCALE_MIN = 1.0
_SCALE_MAX = 9.0


def _normalise(value: float) -> float:
    """Normalise a [1, 9] annotation value to [-1, 1]."""
    return 2.0 * (value - _SCALE_MIN) / (_SCALE_MAX - _SCALE_MIN) - 1.0


def _assign_splits(entries: list[dict]) -> list[dict]:
    """Assign deterministic 70/15/15 train/val/test splits by sorted ID."""
    entries_sorted = sorted(entries, key=lambda e: e["id"])
    n = len(entries_sorted)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    for i, entry in enumerate(entries_sorted):
        if i < n_train:
            entry["split"] = "train"
        elif i < n_train + n_val:
            entry["split"] = "val"
        else:
            entry["split"] = "test"
    return entries_sorted


def _find_audio(audio_dir: Path, stem: str) -> Path | None:
    """Find an audio file by stem, checking .wav then .mp3."""
    for ext in (".wav", ".mp3"):
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_deam(deam_root: Path) -> list[dict]:
    """Load entries from the DEAM dataset.

    Expected layout:
        <deam_root>/audio/*.{wav,mp3}
        <deam_root>/annotations/valence.csv   — columns: song_id, mean
        <deam_root>/annotations/arousal.csv   — columns: song_id, mean
    """
    try:
        import csv
    except ImportError:
        import csv  # noqa: F811 (always stdlib)

    audio_dir = deam_root / "audio"
    annotations_dir = deam_root / "annotations"
    valence_csv = annotations_dir / "valence.csv"
    arousal_csv = annotations_dir / "arousal.csv"

    if not audio_dir.is_dir():
        logger.warning("DEAM audio dir not found: %s — skipping", audio_dir)
        return []
    if not valence_csv.exists():
        logger.warning("DEAM valence.csv not found: %s — skipping", valence_csv)
        return []
    if not arousal_csv.exists():
        logger.warning("DEAM arousal.csv not found: %s — skipping", arousal_csv)
        return []

    # Load valence
    valence_map: dict[str, float] = {}
    with valence_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys/values for robustness
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            song_id = row.get("song_id") or row.get("Song_id") or row.get("songId") or ""
            mean_val = None
            for key in ("mean", "mean_all", "valence_mean", "mean_valence"):
                if key in row:
                    mean_val = row[key]
                    break
            if song_id and mean_val:
                try:
                    valence_map[song_id.strip()] = float(mean_val)
                except ValueError:
                    pass

    # Load arousal
    arousal_map: dict[str, float] = {}
    with arousal_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            song_id = row.get("song_id") or row.get("Song_id") or row.get("songId") or ""
            mean_val = None
            for key in ("mean", "mean_all", "arousal_mean", "mean_arousal"):
                if key in row:
                    mean_val = row[key]
                    break
            if song_id and mean_val:
                try:
                    arousal_map[song_id.strip()] = float(mean_val)
                except ValueError:
                    pass

    common_ids = set(valence_map) & set(arousal_map)
    logger.info(
        "DEAM: %d valence, %d arousal, %d common IDs",
        len(valence_map), len(arousal_map), len(common_ids),
    )

    entries: list[dict] = []
    missing_audio = 0
    for song_id in sorted(common_ids):
        audio_path = _find_audio(audio_dir, song_id)
        if audio_path is None:
            missing_audio += 1
            continue
        entries.append({
            "id": f"deam_{song_id}",
            "dataset": "DEAM",
            "audio_path": str(audio_path),
            "valence": _normalise(valence_map[song_id]),
            "arousal": _normalise(arousal_map[song_id]),
            "split": None,  # assigned later
        })

    if missing_audio:
        logger.warning("DEAM: %d entries skipped (audio file not found)", missing_audio)
    logger.info("DEAM: loaded %d entries", len(entries))
    return entries


def load_pmemo(pmemo_root: Path) -> list[dict]:
    """Load entries from the PMEmo dataset.

    Expected layout:
        <pmemo_root>/audio/*.{wav,mp3}
        <pmemo_root>/annotations/static_annotations.csv
            — columns: musicId, mean_valence, mean_arousal (scale 1-9)
    """
    try:
        import csv
    except ImportError:
        import csv  # noqa: F811

    audio_dir = pmemo_root / "audio"
    annotations_csv = pmemo_root / "annotations" / "static_annotations.csv"

    if not audio_dir.is_dir():
        logger.warning("PMEmo audio dir not found: %s — skipping", audio_dir)
        return []
    if not annotations_csv.exists():
        logger.warning("PMEmo static_annotations.csv not found: %s — skipping", annotations_csv)
        return []

    entries: list[dict] = []
    missing_audio = 0

    with annotations_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            music_id = (
                row.get("musicId")
                or row.get("music_id")
                or row.get("MusicId")
                or ""
            ).strip()
            if not music_id:
                continue

            val_str = row.get("mean_valence") or row.get("valence") or ""
            aro_str = row.get("mean_arousal") or row.get("arousal") or ""
            try:
                valence = float(val_str)
                arousal = float(aro_str)
            except ValueError:
                continue

            audio_path = _find_audio(audio_dir, music_id)
            if audio_path is None:
                missing_audio += 1
                continue

            entries.append({
                "id": f"pmemo_{music_id}",
                "dataset": "PMEmo",
                "audio_path": str(audio_path),
                "valence": _normalise(valence),
                "arousal": _normalise(arousal),
                "split": None,
            })

    if missing_audio:
        logger.warning("PMEmo: %d entries skipped (audio file not found)", missing_audio)
    logger.info("PMEmo: loaded %d entries", len(entries))
    return entries


def print_summary(entries: list[dict]) -> None:
    """Print summary statistics for the manifest."""
    from collections import Counter

    datasets = Counter(e["dataset"] for e in entries)
    splits = Counter(e["split"] for e in entries)

    print("\n=== Emotion Manifest Summary ===")
    print(f"Total entries: {len(entries)}")
    print("\nBy dataset:")
    for ds, count in sorted(datasets.items()):
        print(f"  {ds}: {count}")
    print("\nBy split:")
    for split, count in sorted(splits.items()):
        print(f"  {split}: {count}")

    if entries:
        valences = [e["valence"] for e in entries]
        arousals = [e["arousal"] for e in entries]
        print(f"\nValence: min={min(valences):.3f} max={max(valences):.3f} "
              f"mean={sum(valences)/len(valences):.3f}")
        print(f"Arousal: min={min(arousals):.3f} max={max(arousals):.3f} "
              f"mean={sum(arousals)/len(arousals):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build emotion manifest from DEAM and PMEmo datasets."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path.home() / "Datasets",
        help="Root directory containing datasets (default: ~/Datasets)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/emotion_manifest.json"),
        help="Output JSON manifest path (default: data/emotion_manifest.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    datasets_root = args.datasets_root.expanduser().resolve()
    emotions_root = datasets_root / "by_domain" / "emotions"

    deam_root = emotions_root / "DEAM"
    pmemo_root = emotions_root / "PMEmo"

    all_entries: list[dict] = []

    # Load DEAM
    if deam_root.is_dir():
        all_entries.extend(load_deam(deam_root))
    else:
        logger.warning("DEAM dataset not found at %s — skipping", deam_root)

    # Load PMEmo
    if pmemo_root.is_dir():
        all_entries.extend(load_pmemo(pmemo_root))
    else:
        logger.warning("PMEmo dataset not found at %s — skipping", pmemo_root)

    if not all_entries:
        logger.error("No entries loaded. Check dataset paths and annotation files.")
        return 1

    # Assign deterministic splits
    all_entries = _assign_splits(all_entries)

    # Write output
    output_path = args.output
    if not output_path.is_absolute():
        output_path = Path(_PROJECT_ROOT) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(all_entries, f, indent=2)

    logger.info("Manifest written to %s", output_path)
    print_summary(all_entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
