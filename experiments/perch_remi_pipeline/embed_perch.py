#!/usr/bin/env python
"""
Embed audio with Perch for KmiDi experiments.

CLI example:

    python embed_perch.py \
        --audio-root ~/Datasets/maestro/audio \
        --pattern "**/*.wav" \
        --window-seconds 8.0 \
        --stride-seconds 4.0 \
        --output embeddings_maestro_perch.jsonl

This script is intentionally conservative:
- Pure Python, no repo-specific imports.
- Leaves Perch model loading as a clearly marked integration point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import soundfile as sf


@dataclass
class EmbeddingWindow:
    id: str
    audio_path: str
    start: float
    duration: float
    embedding_path: str
    sha1_audio: str


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_perch_model() -> object:
    """
    Placeholder Perch loader.

    Integration point:
    - Clone google-research/perch and follow its install docs.
    - Replace this stub with real model loading (e.g. TF / JAX / PyTorch wrapper).
    - Expose a simple `embed_waveform(waveform: np.ndarray, sr: int) -> np.ndarray`
      that returns a 1D vector (e.g. 1536-d).
    """
    raise RuntimeError(
        "Perch model loading is not wired yet. "
        "See google-research/perch and update load_perch_model()."
    )


def embed_waveform_segment(
    model: object,
    waveform: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Integration hook: call into Perch.

    Currently returns a deterministic dummy vector so the rest of the
    pipeline (JSONL + .npy writing) can be exercised without a model.
    """
    # Replace this block with a real call into Perch once integrated.
    # For now, use a fixed-size pseudo-embedding based on simple stats.
    mean = float(np.mean(waveform))
    std = float(np.std(waveform) + 1e-8)
    vec = np.array(
        [mean, std, float(sr), float(len(waveform))],
        dtype=np.float32,
    )
    return vec


def iter_audio_files(root: Path, pattern: str) -> Iterable[Path]:
    # e.g. "**/*.wav" -> rglob("*.wav"); "*.wav" -> rglob("*.wav")
    suffix = pattern.split("/")[-1] if "/" in pattern else pattern
    if "*" in suffix or "?" in suffix or "[" in suffix:
        return root.rglob(suffix)
    return root.glob("**/*")


def window_indices(num_samples: int, sr: int, window_s: float, stride_s: float) -> List[tuple[int, int]]:
    win = int(round(window_s * sr))
    stride = int(round(stride_s * sr))
    if win <= 0 or stride <= 0:
        return []
    indices: List[tuple[int, int]] = []
    start = 0
    while start + win <= num_samples:
        indices.append((start, start + win))
        start += stride
    return indices


def process_file(
    model: object,
    audio_path: Path,
    window_s: float,
    stride_s: float,
    out_dir: Path,
    dataset_root: Optional[Path],
) -> List[EmbeddingWindow]:
    audio_rel = (
        str(audio_path.relative_to(dataset_root))
        if dataset_root and audio_path.is_relative_to(dataset_root)
        else str(audio_path)
    )

    data, sr = sf.read(str(audio_path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    num_samples = data.shape[0]
    indices = window_indices(num_samples, sr, window_s, stride_s)

    sha1 = sha1_file(audio_path)
    windows: List[EmbeddingWindow] = []

    for idx, (s, e) in enumerate(indices):
        segment = data[s:e]
        emb = embed_waveform_segment(model, segment, sr)

        emb_id = f"{audio_path.stem}_w{idx:06d}"
        emb_filename = f"{emb_id}.npy"
        emb_path = out_dir / emb_filename
        np.save(emb_path, emb)

        win = EmbeddingWindow(
            id=emb_id,
            audio_path=audio_rel,
            start=float(s) / float(sr),
            duration=float(e - s) / float(sr),
            embedding_path=str(emb_path),
            sha1_audio=sha1,
        )
        windows.append(win)

    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute (stub) Perch-style embeddings for audio.")
    parser.add_argument("--audio-root", type=str, required=True, help="Root directory of audio files.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.wav",
        help="Glob pattern under audio-root (default: **/*.wav).",
    )
    parser.add_argument("--window-seconds", type=float, default=8.0, help="Window size in seconds.")
    parser.add_argument("--stride-seconds", type=float, default=4.0, help="Stride in seconds.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file listing embedding windows.",
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default=None,
        help="Directory for .npy embeddings (default: alongside output JSONL).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional path used to make audio_path relative.",
    )

    args = parser.parse_args()

    audio_root = Path(args.audio_root).expanduser().resolve()
    out_jsonl = Path(args.output).expanduser().resolve()
    emb_dir = Path(args.embedding_dir).expanduser().resolve() if args.embedding_dir else out_jsonl.parent / "embeddings"
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None

    emb_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Placeholder model; currently unused in stub embed_waveform_segment.
    try:
        model = load_perch_model()
    except RuntimeError as exc:
        print(f"[embed_perch] WARNING: {exc}")
        print("[embed_perch] Proceeding with stub embeddings for pipeline testing.")
        model = object()

    all_windows: List[EmbeddingWindow] = []
    for path in sorted(iter_audio_files(audio_root, args.pattern)):
        if not path.is_file():
            continue
        try:
            windows = process_file(
                model=model,
                audio_path=path,
                window_s=args.window_seconds,
                stride_s=args.stride_seconds,
                out_dir=emb_dir,
                dataset_root=dataset_root,
            )
            all_windows.extend(windows)
        except Exception as exc:  # noqa: BLE001
            print(f"[embed_perch] ERROR processing {path}: {exc}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for win in all_windows:
            f.write(json.dumps(asdict(win), ensure_ascii=False) + "\n")

    print(f"[embed_perch] Wrote {len(all_windows)} windows to {out_jsonl}")


if __name__ == "__main__":
    main()

