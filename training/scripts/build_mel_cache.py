#!/usr/bin/env python3
"""Pre-decode FMA mp3s to mel-spectrogram .npy files.

One-time materialization that eliminates the per-batch librosa+audioread
bottleneck at train time. Output layout mirrors the audio tree:

    gs://<bucket>/fma/mel_cache/fma_medium/<sub>/<trackid>.npy
    (float16 mel dB, shape [n_mels, T])

Training script gets ~50-100× faster per epoch once the cache exists.
Cache is shared across every FMA task (genre/subgenre/tags/pretrain).

Usage::

    # Local (from SSD to local cache):
    python training/scripts/build_mel_cache.py \\
        --manifest ~/Datasets/fma_metadata/fma_medium_tags_full_local.csv \\
        --output-root ~/Datasets/fma_mel_cache --num-workers 8

    # Vertex (GCS Fuse → GCS Fuse):
    python training/scripts/build_mel_cache.py \\
        --manifest /gcs/kmidi-train-us-central1/fma/manifest/fma_medium_tags_full.csv \\
        --gcs-audio-prefix /gcs/kmidi-train-us-central1/fma/audio \\
        --output-root /gcs/kmidi-train-us-central1/fma/mel_cache \\
        --num-workers 16 --vertex
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_mel_cache")


def resolve(raw: str, gcs_prefix: str | None) -> str:
    if gcs_prefix and raw.startswith("/Volumes/"):
        tail = raw.split("/audio/fma_medium/", 1)
        if len(tail) == 2:
            return f"{gcs_prefix.rstrip('/')}/{tail[1]}"
    return raw


def decode_one(args_tuple):
    """Worker: decode one mp3 → mel_db float16 → save .npy. Returns
    (track_id, status_str)."""
    (src_path, dst_path, sample_rate, n_mels, n_fft, hop_length, max_dur) = args_tuple
    try:
        if Path(dst_path).exists():
            return (Path(src_path).stem, "skip-exists")
        import librosa
        import warnings
        warnings.filterwarnings("ignore")
        y, _ = librosa.load(src_path, sr=sample_rate, mono=True, duration=max_dur)
        if len(y) < sample_rate:  # under 1s — probably bad
            return (Path(src_path).stem, f"too-short:{len(y)}")
        mel = librosa.feature.melspectrogram(
            y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=2.0)
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80.0).astype(np.float16)
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(dst_path, mel_db)
        return (Path(src_path).stem, "ok")
    except Exception as e:
        return (Path(src_path).stem, f"fail:{type(e).__name__}:{str(e)[:120]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gcs-audio-prefix", default=None)
    ap.add_argument("--output-root", required=True,
                    help="Cache root. e.g. ~/Datasets/fma_mel_cache or "
                         "/gcs/<bucket>/fma/mel_cache")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop-length", type=int, default=512)
    ap.add_argument("--max-duration", type=float, default=6.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--name", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--vertex", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    # Build (src, dst) pairs
    tasks = []
    for _, row in df.iterrows():
        src = resolve(row["file_path"], args.gcs_audio_prefix)
        tid = Path(src).stem
        sub = tid[:3]
        dst = output_root / "fma_medium" / sub / f"{tid}.npy"
        tasks.append((
            src, str(dst),
            args.sample_rate, args.n_mels, args.n_fft,
            args.hop_length, args.max_duration,
        ))

    logger.info("cache root: %s", output_root)
    logger.info("tracks to process: %d  workers: %d", len(tasks), args.num_workers)

    t0 = time.perf_counter()
    ok = 0
    skipped = 0
    failed = 0
    fail_samples = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(decode_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures)):
            tid, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip-exists":
                skipped += 1
            else:
                failed += 1
                if len(fail_samples) < 10:
                    fail_samples.append(f"{tid}: {status}")
            if (i + 1) % 500 == 0:
                dt = time.perf_counter() - t0
                rate = (i + 1) / dt
                eta = (len(tasks) - i - 1) / max(rate, 1e-6)
                logger.info("  %d/%d  rate=%.1f/s  ETA=%.0fs  ok=%d skip=%d fail=%d",
                            i + 1, len(tasks), rate, eta, ok, skipped, failed)

    elapsed = time.perf_counter() - t0
    logger.info("Done in %.0fs. ok=%d skip=%d fail=%d", elapsed, ok, skipped, failed)
    if fail_samples:
        logger.info("first failures:")
        for s in fail_samples:
            logger.info("  %s", s)

    # Persist summary
    if args.vertex and "AIP_MODEL_DIR" in os.environ:
        raw = os.environ["AIP_MODEL_DIR"]
        summary_dir = (Path("/gcs") / raw[len("gs://"):]) if raw.startswith("gs://") else Path(raw)
    else:
        summary_dir = output_root
    summary_dir.mkdir(parents=True, exist_ok=True)
    import json
    (summary_dir / "build_summary.json").write_text(json.dumps({
        "ok": ok, "skip": skipped, "fail": failed,
        "total": len(tasks), "wall_seconds": round(elapsed, 2),
        "manifest": args.manifest,
        "output_root": str(output_root),
        "hparams": {
            "sample_rate": args.sample_rate, "n_mels": args.n_mels,
            "n_fft": args.n_fft, "hop_length": args.hop_length,
            "max_duration": args.max_duration,
        },
        "fail_samples": fail_samples,
    }, indent=2))
    # Allow up to 1% decode failures (corrupt mp3s in the wild). Real
    # corpus-level breakage shows up as >>1% and fails loudly.
    fail_ratio = failed / max(len(tasks), 1)
    return 0 if fail_ratio < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
