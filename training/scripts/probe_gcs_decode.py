#!/usr/bin/env python3
"""Diagnostic: probe librosa mp3 decode success rate against GCS Fuse paths.

Pulls the FMA manifest, samples N tracks per genre, attempts decode, prints
per-genre and overall success/fail rates plus timing histograms. No model,
no training — just decode.

Usage::

    # Local with /Volumes mp3s
    python training/scripts/probe_gcs_decode.py --manifest <local-csv>

    # Vertex container with /gcs/ Fuse mount
    python training/scripts/probe_gcs_decode.py \\
        --manifest /gcs/kmidi-train-us-central1/fma/manifest/fma_medium_manifest_uploaded.csv \\
        --gcs-audio-prefix /gcs/kmidi-train-us-central1/fma/audio/fma_medium \\
        --per-genre 20 --vertex
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("probe_gcs_decode")


def resolve(raw_path: str, gcs_prefix: str | None) -> str:
    if gcs_prefix and raw_path.startswith("/Volumes/"):
        tail = raw_path.split("/audio/fma_medium/", 1)
        if len(tail) == 2:
            return f"{gcs_prefix.rstrip('/')}/{tail[1]}"
    return raw_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--gcs-audio-prefix", default=None)
    ap.add_argument("--per-genre", type=int, default=20)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-duration", type=float, default=6.0)
    ap.add_argument("--vertex", action="store_true",
                    help="Write report to AIP_MODEL_DIR if set")
    ap.add_argument("--report-dir", default=None,
                    help="Where to write probe_report.json. Overrides AIP_MODEL_DIR.")
    # Accept-and-ignore — the dispatcher always injects these for trainer jobs.
    ap.add_argument("--name", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--num-workers", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    sample = (df.groupby("genre_top").head(args.per_genre)
              .reset_index(drop=True))
    logger.info("probing %d files across %d genres",
                len(sample), sample["genre_top"].nunique())

    import librosa
    import warnings
    warnings.filterwarnings("ignore")

    per_genre_ok = defaultdict(int)
    per_genre_total = defaultdict(int)
    per_genre_decode_ms = defaultdict(list)
    failures = []

    t_start = time.perf_counter()
    for i, row in sample.iterrows():
        path = resolve(row["file_path"], args.gcs_audio_prefix)
        g = row["genre_top"]
        per_genre_total[g] += 1
        t0 = time.perf_counter()
        try:
            y, _ = librosa.load(path, sr=args.sample_rate, mono=True,
                                duration=args.max_duration)
            dt = (time.perf_counter() - t0) * 1000
            n_zero = int(np.sum(y == 0))
            if len(y) < 1000:
                failures.append({"path": path, "genre": g,
                                 "reason": f"too_short:{len(y)}"})
                continue
            if n_zero / len(y) > 0.95:
                failures.append({"path": path, "genre": g,
                                 "reason": f"silent:{n_zero}/{len(y)}"})
                continue
            per_genre_ok[g] += 1
            per_genre_decode_ms[g].append(dt)
        except Exception as e:
            failures.append({"path": path, "genre": g,
                             "reason": f"{type(e).__name__}: {str(e)[:200]}"})
        if (i + 1) % 25 == 0:
            logger.info("  %d/%d done", i + 1, len(sample))

    elapsed = time.perf_counter() - t_start

    # Aggregate
    report = {
        "total": int(len(sample)),
        "ok": int(sum(per_genre_ok.values())),
        "fail": int(sum(per_genre_total.values()) - sum(per_genre_ok.values())),
        "wall_seconds": round(elapsed, 2),
        "per_genre": {
            g: {
                "total": per_genre_total[g],
                "ok": per_genre_ok[g],
                "fail_pct": round(100 * (1 - per_genre_ok[g] / per_genre_total[g]), 1),
                "decode_ms_p50": round(float(np.median(per_genre_decode_ms[g])), 1) if per_genre_decode_ms[g] else None,
                "decode_ms_p95": round(float(np.percentile(per_genre_decode_ms[g], 95)), 1) if per_genre_decode_ms[g] else None,
            } for g in sorted(per_genre_total)
        },
        "first_10_failures": failures[:10],
        "manifest": args.manifest,
        "gcs_audio_prefix": args.gcs_audio_prefix,
    }

    print("\n=== PROBE REPORT ===")
    print(json.dumps(report, indent=2))

    # Persist
    if args.report_dir:
        out_dir = Path(args.report_dir)
    elif args.vertex and "AIP_MODEL_DIR" in os.environ:
        raw = os.environ["AIP_MODEL_DIR"]
        if raw.startswith("gs://"):
            out_dir = Path("/gcs") / raw[len("gs://"):]
        else:
            out_dir = Path(raw)
    else:
        out_dir = Path.home() / "Models" / "checkpoints" / "probe_local"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s/probe_report.json", out_dir)
    return 0 if report["ok"] / max(report["total"], 1) > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
