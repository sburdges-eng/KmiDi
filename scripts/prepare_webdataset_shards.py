#!/usr/bin/env python3
"""
Convert aligned manifest to WebDataset shards for optimal cloud IO.

Target shard size: 500MB–1GB. GPU starvation from slow disk reads is avoidable.

Usage:
  python scripts/prepare_webdataset_shards.py \\
    --manifest ~/Datasets/kmidi_jepa/manifests/aligned.jsonl \\
    --output ~/Datasets/kmidi_jepa/shards \\
    --shard_size_mb 750

Reference: docs/CLOUD_TRAINING.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare WebDataset shards from aligned manifest")
    parser.add_argument("--manifest", required=True, help="Path to aligned.jsonl")
    parser.add_argument("--output", required=True, help="Output dir for .tar shards")
    parser.add_argument("--shard_size_mb", type=float, default=750, help="Target shard size MB")
    parser.add_argument("--dry_run", action="store_true", help="Only count samples, don't write")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return 1

    lines = manifest_path.read_text().strip().split("\n")
    samples = [json.loads(ln) for ln in lines if ln.strip()]
    print(f"Loaded {len(samples)} samples from {manifest_path}")

    if args.dry_run:
        print("Dry run. Install webdataset and run without --dry_run to create shards.")
        return 0

    try:
        import webdataset as wds
    except ImportError:
        print("Install webdataset: pip install webdataset")
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "train-%06d.tar")
    max_size = int(args.shard_size_mb * 1e6)

    with wds.ShardWriter(pattern, maxcount=10000, maxsize=max_size) as sink:
        for i, s in enumerate(samples):
            sink.write({"__key__": f"sample{i:08d}", "sample.json": s})

    print(f"Shards written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
