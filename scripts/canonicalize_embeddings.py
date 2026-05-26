#!/usr/bin/env python3
"""
Fit and save an orthogonal map between two embedding spaces (Procrustes + mean shift).

Use when upgrading encoders: embed ~500 anchor pairs with old and new models,
then run this script to produce a .npz map. At query time, apply the map so
new queries match the existing vector DB without re-embedding.

Usage:
  python scripts/canonicalize_embeddings.py \\
    --old-embeddings path/to/anchor_old.npy \\
    --new-embeddings path/to/anchor_new.npy \\
    --output path/to/map.npz

Anchor matrices must be (N, D) with the same N and D (same items, same dimension).
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit orthogonal map between two embedding spaces.")
    parser.add_argument(
        "--old-embeddings", type=str, required=True, help="Path to (N, D) old-space anchors."
    )
    parser.add_argument(
        "--new-embeddings", type=str, required=True, help="Path to (N, D) new-space anchors."
    )
    parser.add_argument("--output", type=str, required=True, help="Output path for .npz map.")
    parser.add_argument("--no-center", action="store_true", help="Disable mean centering.")
    args = parser.parse_args()

    try:
        from penta_core.ml.canonicalize_embeddings import fit_orthogonal_map, save_map
    except ModuleNotFoundError:
        from music_brain.penta_core.ml.canonicalize_embeddings import fit_orthogonal_map, save_map

    import numpy as np

    anchor_old = np.load(args.old_embeddings)
    anchor_new = np.load(args.new_embeddings)
    R, center_old, center_new = fit_orthogonal_map(
        anchor_old, anchor_new, center=not args.no_center
    )
    save_map(args.output, R, center_old, center_new)
    print(f"Saved map to {args.output}")
    print(
        "At query time: apply_map(query_new, R, center_old, center_new, normalize=True) for retrieval."  # noqa: E501
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
