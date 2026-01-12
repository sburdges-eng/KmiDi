"""
Normalize DEAM + EMO-Music into a single manifest with reproducible splits.

Expected layout (workspace-relative):
datasets/
├── DEAM/
│   ├── audio/
│   ├── annotations/
│   └── metadata.csv
└── EMO_Music/
    ├── audio/
    ├── annotations/
    └── metadata.csv

Metadata CSV must include at least: id, file (audio filename), valence, arousal.
All audio paths are stored workspace-relative in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import contextlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


TARGET_SR = 22050
MIN_DURATION = 30.0  # seconds


@dataclass
class AudioIssue:
    path: Path
    code: str
    detail: str


def read_metadata_csv(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows


def validate_audio(path: Path, target_sr: int = TARGET_SR, min_duration: float = MIN_DURATION) -> List[AudioIssue]:
    issues: List[AudioIssue] = []
    try:
        # Use built-in wave module to avoid external deps like librosa.
        # This supports .wav files, which are the expected format.
        if path.suffix.lower() != ".wav":
            raise ValueError(f"Unsupported audio format without librosa: {path.suffix}")

        with contextlib.closing(wave.open(str(path), "rb")) as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
        duration = n_frames / float(sr) if sr else 0.0
    except Exception as exc:  # pragma: no cover - runtime guard
        issues.append(AudioIssue(path, "LOAD_FAIL", str(exc)))
        return issues
    if sr != target_sr:
        issues.append(AudioIssue(path, "BAD_SR", f"{sr}"))
    if duration < min_duration:
        issues.append(AudioIssue(path, "TOO_SHORT", f"{duration:.2f}s"))
    return issues


def build_entries(
    dataset_name: str,
    dataset_root: Path,
    rows: List[Dict[str, str]],
    audio_dir: Path,
) -> Tuple[List[Dict[str, object]], List[AudioIssue]]:
    entries: List[Dict[str, object]] = []
    issues: List[AudioIssue] = []

    for row in rows:
        file_rel = row.get("file") or row.get("audio_path") or ""
        if not file_rel:
            issues.append(AudioIssue(dataset_root, "MISSING_FILE_FIELD", "row missing file/audio_path"))
            continue

        audio_path = (audio_dir / file_rel).resolve()
        if not audio_path.exists():
            issues.append(AudioIssue(audio_path, "MISSING_FILE", "not found"))
            continue

        issues.extend(validate_audio(audio_path))

        try:
            valence = float(row["valence"])
            arousal = float(row["arousal"])
        except (KeyError, ValueError) as exc:
            issues.append(AudioIssue(audio_path, "BAD_LABELS", f"{exc}"))
            continue

        entry = {
            "id": row.get("id") or audio_path.stem,
            "dataset": dataset_name,
            "audio_path": str(audio_path.relative_to(Path.cwd())),
            "valence": valence,
            "arousal": arousal,
            "split": "train",  # filled later
        }
        entries.append(entry)

    return entries, issues


def assign_splits(entries: List[Dict[str, object]], train: float, val: float, seed: int) -> None:
    rng = random.Random(seed)
    rng.shuffle(entries)
    n = len(entries)
    n_train = int(train * n)
    n_val = int(val * n)
    for i, e in enumerate(entries):
        if i < n_train:
            e["split"] = "train"
        elif i < n_train + n_val:
            e["split"] = "val"
        else:
            e["split"] = "test"


def load_dataset(dataset_dir: Path, dataset_name: str) -> Tuple[List[Dict[str, object]], List[AudioIssue]]:
    csv_path = dataset_dir / "metadata.csv"
    audio_dir = dataset_dir / "audio"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv at {csv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio/ directory at {audio_dir}")

    rows = read_metadata_csv(csv_path)
    return build_entries(dataset_name, dataset_dir, rows, audio_dir)


def run(root: Path, output_manifest: Path, train_ratio: float, val_ratio: float, seed: int) -> None:
    combined: List[Dict[str, object]] = []
    all_issues: List[AudioIssue] = []

    for name in ("DEAM", "EMO_Music"):
        dataset_dir = root / name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Expected dataset directory missing: {dataset_dir}")
        entries, issues = load_dataset(dataset_dir, name)
        combined.extend(entries)
        all_issues.extend(issues)

    assign_splits(combined, train_ratio, val_ratio, seed)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    if all_issues:
        print("Validation issues detected:")
        for issue in all_issues:
            print(f"- {issue.code}: {issue.path} ({issue.detail})")
    else:
        print("No validation issues found.")

    print(f"Wrote manifest with {len(combined)} entries -> {output_manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize DEAM + EMO-Music into a unified manifest.")
    parser.add_argument("--root", type=Path, default=Path("datasets"), help="Root datasets directory.")
    parser.add_argument("--output", type=Path, default=Path("datasets/validation/emotion_manifest.json"), help="Output manifest path.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for split assignment.")
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be < 1.0 to leave room for test.")

    run(args.root, args.output, args.train_ratio, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
