#!/usr/bin/env python3
"""
Build emotion manifest from RAVDESS, CREMA-D, and TESS datasets.

Scans each dataset's metadata.csv, maps categorical emotion labels to
valence/arousal coordinates, assigns deterministic 70/15/15 train/val/test
splits, and writes data/emotion_manifest.json.

Usage:
    python scripts/build_emotion_manifest.py
    python scripts/build_emotion_manifest.py \\
        --datasets-root "/Volumes/Sean's SSD/Datasets/processed/emotions" \\
        --output data/emotion_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# Categorical emotion → (valence, arousal) mapping
# "calm" is a RAVDESS-specific label; mapped to a low-arousal, near-neutral position.
EMOTION_VA_MAP = {
    "angry":    {"valence": -0.6, "arousal":  0.8},
    "calm":     {"valence":  0.2, "arousal": -0.3},
    "disgust":  {"valence": -0.7, "arousal":  0.3},
    "fear":     {"valence": -0.6, "arousal":  0.7},
    "happy":    {"valence":  0.8, "arousal":  0.6},
    "neutral":  {"valence":  0.0, "arousal":  0.3},
    "sad":      {"valence": -0.7, "arousal": -0.3},
    "surprise": {"valence":  0.3, "arousal":  0.8},
}

# TESS prefix patterns to strip, with special-case normalisation
# "pleasant_surprise" and "pleasant_surprised" both map to "surprise"
_TESS_PREFIX_RE = re.compile(r"^(?:yaf|oaf)_(.+)$", re.IGNORECASE)
_TESS_ALIAS = {
    "pleasant_surprise":  "surprise",
    "pleasant_surprised": "surprise",
}


def _normalise_tess_emotion(raw: str) -> str | None:
    """Strip yaf_/oaf_ prefix from a TESS emotion label and return base name.

    Returns None if the label can't be mapped to EMOTION_VA_MAP.
    """
    raw = raw.strip().lower()
    m = _TESS_PREFIX_RE.match(raw)
    base = m.group(1) if m else raw
    base = _TESS_ALIAS.get(base, base)
    return base if base in EMOTION_VA_MAP else None


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


def load_dataset(
    dataset_dir: Path,
    dataset_name: str,
    is_tess: bool = False,
) -> tuple[list[dict], int]:
    """Load entries from a single dataset directory using its metadata.csv.

    Returns (entries, skip_count).
    """
    metadata_csv = dataset_dir / "metadata.csv"
    if not metadata_csv.exists():
        logger.warning("%s metadata.csv not found at %s — skipping", dataset_name, metadata_csv)
        return [], 0

    entries: list[dict] = []
    skipped_missing = 0
    skipped_unknown = 0

    with metadata_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_file = row.get("file", "").strip()
            raw_emotion = row.get("emotion", "").strip()

            if not rel_file or not raw_emotion:
                skipped_unknown += 1
                continue

            # Resolve audio path
            audio_path = dataset_dir / rel_file
            if not audio_path.exists():
                logger.debug("Missing audio: %s", audio_path)
                skipped_missing += 1
                continue

            # Map emotion label to valence/arousal
            if is_tess:
                emotion = _normalise_tess_emotion(raw_emotion)
            else:
                emotion = raw_emotion.strip().lower()
                if emotion not in EMOTION_VA_MAP:
                    emotion = None

            if emotion is None:
                logger.debug("Unknown emotion %r in %s — skipping", raw_emotion, dataset_name)
                skipped_unknown += 1
                continue

            va = EMOTION_VA_MAP[emotion]
            entry_id = f"{dataset_name.lower()}_{Path(rel_file).stem}"

            entries.append({
                "id": entry_id,
                "dataset": dataset_name,
                "audio_path": str(audio_path),
                "valence": va["valence"],
                "arousal": va["arousal"],
                "emotion": emotion,
                "split": None,  # assigned later
            })

    total_skipped = skipped_missing + skipped_unknown
    if skipped_missing:
        logger.warning(
            "%s: %d entries skipped (audio file not found)", dataset_name, skipped_missing
        )
    if skipped_unknown:
        logger.warning(
            "%s: %d entries skipped (unknown/unmappable emotion)", dataset_name, skipped_unknown
        )
    logger.info(
        "%s: loaded %d entries, skipped %d total", dataset_name, len(entries), total_skipped
    )
    return entries, total_skipped


def print_summary(entries: list[dict], total_skipped: int) -> None:
    """Print manifest summary statistics."""
    from collections import Counter

    datasets = Counter(e["dataset"] for e in entries)
    splits = Counter(e["split"] for e in entries)
    emotions = Counter(e["emotion"] for e in entries)

    print("\n=== Emotion Manifest Summary ===")
    print(f"Total entries: {len(entries)}")
    print(f"Total skipped: {total_skipped}")

    print("\nBy dataset:")
    for ds, count in sorted(datasets.items()):
        print(f"  {ds}: {count}")

    print("\nBy split:")
    for split, count in sorted(splits.items()):
        print(f"  {split}: {count}")

    print("\nBy emotion:")
    for emo, count in sorted(emotions.items()):
        print(f"  {emo}: {count}")

    if entries:
        valences = [e["valence"] for e in entries]
        arousals = [e["arousal"] for e in entries]
        print(f"\nValence: min={min(valences):.3f}  max={max(valences):.3f}"
              f"  mean={sum(valences)/len(valences):.3f}")
        print(f"Arousal: min={min(arousals):.3f}  max={max(arousals):.3f}"
              f"  mean={sum(arousals)/len(arousals):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build emotion manifest from RAVDESS, CREMA-D, and TESS datasets."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/Volumes/Sean's SSD/Datasets/processed/emotions"),
        help=(
            "Root directory containing ravdess/, cremad/, tess/ subdirs "
            "(default: /Volumes/Sean's SSD/Datasets/processed/emotions)"
        ),
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
    if not datasets_root.is_dir():
        logger.error("Datasets root not found: %s", datasets_root)
        return 1

    all_entries: list[dict] = []
    grand_skipped = 0

    # RAVDESS
    ravdess_dir = datasets_root / "ravdess"
    entries, skipped = load_dataset(ravdess_dir, "RAVDESS", is_tess=False)
    all_entries.extend(entries)
    grand_skipped += skipped

    # CREMA-D
    cremad_dir = datasets_root / "cremad"
    entries, skipped = load_dataset(cremad_dir, "CREMAD", is_tess=False)
    all_entries.extend(entries)
    grand_skipped += skipped

    # TESS
    tess_dir = datasets_root / "tess"
    entries, skipped = load_dataset(tess_dir, "TESS", is_tess=True)
    all_entries.extend(entries)
    grand_skipped += skipped

    if not all_entries:
        logger.error(
            "No entries loaded. Check that datasets-root contains ravdess/, cremad/, tess/."
        )
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

    logger.info("Manifest written to %s (%d entries)", output_path, len(all_entries))
    print_summary(all_entries, grand_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
