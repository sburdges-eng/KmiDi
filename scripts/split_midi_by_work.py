#!/usr/bin/env python3
"""Split MIDI dataset by work ID (train/val/test) so the same piece never leaks across splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.training_common import DATASET_SPLITS, choose_split


def get_work_id_from_path(path: Path, depth: int = 2) -> str:
    """Derive a stable work ID from path. Default: parent names up to depth (e.g. composer/piece)."""
    parts = path.resolve().parts
    if len(parts) <= depth:
        return str(path)
    return "/".join(parts[-(depth + 1) : -1] + [path.stem])


def split_midi_paths_by_work(
    midi_paths: list[Path],
    seed: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    work_id_depth: int = 2,
) -> dict[str, list[Path]]:
    """Assign each path to train/val/test by work ID (deterministic)."""
    work_to_paths: dict[str, list[Path]] = {}
    for p in midi_paths:
        if not p.suffix.lower() in (".mid", ".midi"):
            continue
        wid = get_work_id_from_path(p, depth=work_id_depth)
        work_to_paths.setdefault(wid, []).append(p)

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    split_map: dict[str, list[Path]] = {s: [] for s in DATASET_SPLITS}
    for work_id, paths in work_to_paths.items():
        split_name = choose_split(work_id, seed, ratios)
        split_map[split_name].extend(paths)

    for paths in split_map.values():
        paths.sort(key=lambda p: (p.parent, p.name))
    return split_map


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split MIDI files into train/val/test by work ID (no same work across splits)."
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Root directory containing MIDI files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/midi_splits"),
        help="Directory to write train/val/test file lists.",
    )
    parser.add_argument("--seed", default="remi-bpe-v1", help="Stable split seed.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--work-id-depth",
        type=int,
        default=2,
        help="Path depth for work ID (e.g. 2 => composer/piece from path).",
    )
    parser.add_argument(
        "--ext",
        default=".mid,.midi",
        help="Comma-separated extensions to include.",
    )
    args = parser.parse_args()

    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}
    midi_paths = []
    for p in args.data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            midi_paths.append(p)

    if not midi_paths:
        print(f"No MIDI files under {args.data_dir}")
        return 1

    split_map = split_midi_paths_by_work(
        midi_paths,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        work_id_depth=args.work_id_depth,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in DATASET_SPLITS:
        out_file = args.output_dir / f"{split_name}.txt"
        paths = split_map[split_name]
        out_file.write_text("\n".join(str(p.resolve()) for p in paths) + ("\n" if paths else ""))
        print(f"{split_name}: {len(paths)} files -> {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
